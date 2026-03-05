# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Curriculum-GRPO Dataset for loading teacher traces.
"""

import copy
import json
import logging
import os
from typing import Callable, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class CurriculumGRPODataset(Dataset):
    """
    Dataset for Curriculum-GRPO training.
    
    Loads teacher traces from JSONL files and supports dynamic prompt
    construction based on curriculum level k.
    
    Data format (JSONL):
    {
        "question": "Math problem...",
        "steps": ["Step 1...", "Step 2...", ...],
        "teacher_answer": "42",
        "ground_truth": "42",
        "index": 0
    }
    """
    
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        max_samples: int = -1,
    ):
        """
        Initialize Curriculum-GRPO Dataset.
        
        Args:
            data_files: Path(s) to JSONL file(s) containing teacher traces.
            tokenizer: Tokenizer for text processing.
            config: Dataset configuration.
            max_samples: Maximum number of samples to load (-1 for all).
        """
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        
        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/cgrpo"))
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.max_response_length = config.get("max_response_length", 1024)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        
        self.thinking_start = config.get("thinking_start", "<think>")
        self.thinking_end = config.get("thinking_end", "</think>")
        self.answer_start = config.get("answer_start", "<answer>")
        self.answer_end = config.get("answer_end", "</answer>")
        
        self.use_chat_template = config.get("use_chat_template", True)
        self.thought_start_phrase = config.get("thought_start_phrase", "Okay, let's think step by step.")
        
        self._load_data()
        
        logger.info(f"CurriculumGRPODataset loaded {len(self.data)} samples")
    
    def _load_data(self):
        """Load data from JSONL files."""
        all_data = []
        
        for data_file in self.data_files:
            data_file = os.path.expanduser(data_file)
            
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            logger.info(f"Loading data from {data_file}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        item['_source_file'] = data_file
                        item['_line_num'] = line_num
                        all_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {data_file}: {e}")
                        continue
        
        if self.max_samples > 0 and len(all_data) > self.max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(all_data), self.max_samples, replace=False)
            all_data = [all_data[i] for i in indices]
            logger.info(f"Sampled {self.max_samples} from {len(all_data)} total samples")
        
        self.data = all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single data item.
        
        Returns raw data that will be processed during rollout with
        curriculum-aware prompt construction.
        
        Args:
            idx: Data index.
        
        Returns:
            Dictionary containing:
            - raw_prompt: The question
            - steps: Teacher's reasoning steps
            - teacher_answer: Teacher's final answer
            - ground_truth: Ground truth answer
            - data_source: Data source identifier
            - reward_model: Reward configuration
        """
        item = self.data[idx]
        
        question = item.get("question", "")
        steps = item.get("steps", [])
        teacher_answer = item.get("teacher_answer", "")
        ground_truth = item.get("ground_truth", teacher_answer)
        index = item.get("index", idx)
        
        if isinstance(ground_truth, str):
            gt_answer = self._extract_answer(ground_truth)
        else:
            gt_answer = str(ground_truth)

        # Return raw messages to be formatted by AgentLoop
        raw_prompt = self._build_prompt_messages(question)
        
        return {
            "raw_prompt": raw_prompt,
            "steps": steps,
            "teacher_answer": teacher_answer,
            "ground_truth": gt_answer,
            "data_source": "cgrpo/gsm8k",
            "reward_model": {
                "style": "cgrpo",
                "ground_truth": gt_answer,
            },
            "extra_info": {
                "index": index,
                "num_steps": len(steps),
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
    
    def _extract_answer(self, text: str) -> str:
        return text.strip()
    
    def _build_prompt_messages(self, question: str) -> list[dict]:
        """Build prompt messages without applying chat template."""
        
    system_prompt = "You are an expert mathematician. You should think step-by-step."
    prompt_instruction = (
        "Please reason step by step to solve this problem.\n"
        f"Put your detailed reasoning process within {self.thinking_start} and {self.thinking_end} tags.\n"
        "Inside these tags, if some reasoning steps are already provided, continue the reasoning from them instead of starting from scratch.\n"
        "After your reasoning, briefly summarize the key steps in several short sentences and then put your final answer within \\boxed{} tags. For example: \\boxed{42}.\n"
    )
        
        content = f"{question}\n\n{prompt_instruction}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        return messages

def create_curriculum_collate_fn(
    tokenizer: PreTrainedTokenizer,
    current_k: int = 1,
    config: Optional[DictConfig] = None,
) -> Callable:
    """
    Create collate function for curriculum dataset.
    
    Args:
        tokenizer: Tokenizer for processing.
        current_k: Current curriculum level.
        config: Configuration.
    
    Returns:
        Collate function.
    """
    def collate_fn(data_list: list[dict]) -> dict:
        """Collate batch with curriculum-aware prompt construction."""
        from collections import defaultdict
        
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)
        
        for data in data_list:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    tensors[key].append(val)
                else:
                    non_tensors[key].append(val)
        
        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)
        
        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)
        
        return {**tensors, **non_tensors}
    
    return collate_fn
