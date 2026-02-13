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
        
        return {
            "raw_prompt": [{"role": "user", "content": question}],
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
        """
        Extract final answer from text.
        
        Handles formats like:
        - "#### 42"
        - "<answer>42</answer>"
        - Plain number
        """
        import re
        
        if "####" in text:
            match = re.search(r'####\s*(-?[\d,\.]+)', text)
            if match:
                return match.group(1).replace(',', '')
        
        if "<answer>" in text:
            match = re.search(r'<answer>\s*(-?[\d,\.]+)\s*</answer>', text)
            if match:
                return match.group(1).replace(',', '')
        
        numbers = re.findall(r'-?[\d,\.]+', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return text.strip()
    
    def build_curriculum_prompt(
        self,
        item: dict,
        current_k: int,
    ) -> tuple[str, str, int]:
        """
        Build curriculum-aware prompt for a given item.
        
        Args:
            item: Data item containing question and steps.
            current_k: Current curriculum level (number of steps student should generate).
        
        Returns:
            Tuple of (full_prompt, teacher_prefix, cut_index).
        """
        question = item.get("question", "")
        steps = item.get("steps", [])
        
        if not steps:
            return self._build_simple_prompt(question), "", 0
        
        num_steps = len(steps)
        cut_index = max(0, num_steps - current_k)
        
        teacher_prefix_steps = steps[:cut_index]
        student_target_steps = steps[cut_index:]
        
        teacher_prefix = "\n".join(teacher_prefix_steps) if teacher_prefix_steps else ""
        
        full_prompt = self._build_chatml_prompt(
            question=question,
            teacher_prefix=teacher_prefix,
        )
        
        return full_prompt, teacher_prefix, cut_index
    
    def _build_chatml_prompt(
        self,
        question: str,
        teacher_prefix: str = "",
    ) -> str:
        """
        Build ChatML format prompt.
        
        Args:
            question: User question.
            teacher_prefix: Teacher's reasoning prefix (steps student doesn't need to generate).
        
        Returns:
            Formatted prompt string.
        """
        if self.use_chat_template:
            messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            if teacher_prefix:
                prompt += f"{self.thinking_start}\n{teacher_prefix}"
        else:
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            if teacher_prefix:
                prompt += f"{self.thinking_start}\n{teacher_prefix}"
        
        return prompt
    
    def _build_simple_prompt(self, question: str) -> str:
        """Build simple prompt without chat template."""
        if self.use_chat_template:
            messages = [{"role": "user", "content": question}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

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
