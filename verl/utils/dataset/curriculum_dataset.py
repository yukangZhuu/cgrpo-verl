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
Dataset for CGRPO training.
Supports standard GRPO (no guidance) and static mixture (with per-sample guidance).
"""

import copy
import json
import logging
import os
from typing import Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert mathematician with strong problem-solving skills. "
    "Think step by step."
)

FORMAT_BLOCK = (
    "Use this format:\n"
    "<think>\n"
    "[Your reasoning process here, showing how YOU would reach the solution]\n"
    "</think>\n"
    "\\boxed{answer}"
)


class CurriculumGRPODataset(Dataset):
    """
    Dataset for CGRPO / standard GRPO training.

    Each JSONL record may contain:
      - question, ground_truth, steps, teacher_answer  (always present)
      - g_level, guidance_steps                        (present in expanded data)
      - pass_rate                                      (optional, for analysis)

    When g_level / guidance_steps are absent the sample is treated as a
    standard GRPO sample (g_level=0, no guidance).

    Three guidance_mode values control prompt construction:
      "none"   – plain question (standard GRPO)
      "prefix" – guidance_steps passed to AgentLoop for prefix injection
      "hint"   – guidance_steps formatted as hints in the user message
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples

        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.max_response_length = config.get("max_response_length", 1024)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.guidance_mode = config.get("guidance_mode", "none")
        assert self.guidance_mode in ("none", "prefix", "hint"), (
            f"Invalid guidance_mode: {self.guidance_mode}"
        )

        self.curriculum_method = config.get("curriculum_method", "none")

        self._load_data()
        logger.info(
            f"CurriculumGRPODataset loaded {len(self.data)} samples, "
            f"guidance_mode={self.guidance_mode}"
        )

    def _load_data(self):
        all_data = []
        for data_file in self.data_files:
            data_file = os.path.expanduser(data_file)
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file not found: {data_file}")
            logger.info(f"Loading data from {data_file}")
            with open(data_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        all_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse line {line_num} in {data_file}: {e}"
                        )

        if self.max_samples > 0 and len(all_data) > self.max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(all_data), self.max_samples, replace=False)
            all_data = [all_data[i] for i in indices]

        self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        question = item.get("question", "")
        steps = item.get("steps", [])
        teacher_answer = item.get("teacher_answer", "")
        ground_truth = item.get("ground_truth", teacher_answer)
        index = item.get("index", idx)

        g_level = float(item.get("g_level", 0.0))
        guidance_steps = item.get("guidance_steps", [])
        pass_rate = item.get("pass_rate", -1.0)

        # --- Adaptive curriculum support ---
        # frozen_g_level: if present, this sample has a fixed guidance level
        # (used for anchor samples in adaptive mode)
        frozen_g_level = item.get("frozen_g_level", None)
        adaptive_id = item.get("adaptive_id", str(index))

        if frozen_g_level is not None:
            # Anchor sample: compute guidance_steps from frozen level
            from verl.utils.curriculum import PerSampleCurriculumState

            g_level = float(frozen_g_level)
            guidance_steps = PerSampleCurriculumState.compute_guidance_steps(
                steps, g_level
            )
        elif self.curriculum_method == "adaptive" and not guidance_steps:
            # Adaptive non-frozen sample: trainer will fill guidance dynamically.
            # Use -1 as sentinel so the trainer knows to compute it.
            g_level = -1.0

        gt_answer = str(ground_truth).strip()

        raw_prompt = self._build_prompt_messages(question, guidance_steps)

        # For prefix mode the guidance_steps are forwarded separately;
        # the AgentLoop will inject them into the assistant prefix.
        guidance_for_agentloop = (
            guidance_steps if self.guidance_mode == "prefix" else []
        )

        return {
            "question": question,
            "raw_prompt": raw_prompt,
            "steps": steps,
            "guidance_steps": guidance_for_agentloop,
            "g_level": g_level,
            "pass_rate": pass_rate,
            "teacher_answer": teacher_answer,
            "ground_truth": gt_answer,
            "adaptive_id": adaptive_id,
            "frozen_g_level": frozen_g_level if frozen_g_level is not None else -1.0,
            "data_source": "cgrpo",
            "reward_model": {
                "style": "cgrpo",
                "ground_truth": gt_answer,
            },
            "extra_info": {
                "index": index,
                "num_steps": len(steps),
                "g_level": g_level,
                "guidance_steps_count": len(guidance_steps),
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt_messages(
        self, question: str, guidance_steps: list[str]
    ) -> list[dict]:
        """Build chat messages.  The AgentLoop will apply_chat_template."""

        if self.guidance_mode == "hint" and len(guidance_steps) > 0:
            user_content = self._build_hint_user_content(question, guidance_steps)
        else:
            user_content = self._build_standard_user_content(question)

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _build_standard_user_content(question: str) -> str:
        """Used for none / prefix modes, and hint mode when g_level=0."""
        return (
            f"{question}\n"
            f"Please reason step by step to solve this problem.\n"
            f"{FORMAT_BLOCK}"
        )

    @staticmethod
    def _build_hint_user_content(
        question: str, guidance_steps: list[str]
    ) -> str:
        steps_text = "\n".join(guidance_steps)
        return (
            f"{question}\n\n"
            f"Below are some initial reasoning steps that may help you:\n"
            f"{steps_text}\n\n"
            f"Please solve the problem step by step. You should use the provided "
            f"steps as a reference, but do NOT just copy them. Instead, reconstruct "
            f"the complete reasoning process in your own words, starting from the "
            f"beginning, and continue the reasoning to find the final answer.\n"
            f"{FORMAT_BLOCK}"
        )


def create_curriculum_collate_fn(
    tokenizer: PreTrainedTokenizer,
    config: Optional[DictConfig] = None,
) -> Callable:
    def collate_fn(data_list: list[dict]) -> dict:
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
