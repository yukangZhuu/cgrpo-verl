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
Reward Manager for Curriculum-GRPO.
"""

import logging
import re
from typing import Any, Optional

import numpy as np
import torch

from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


class CurriculumGRPORewardManager(AbstractRewardManager):
    """
    Reward Manager for Curriculum-GRPO.
    
    Evaluates student responses based on:
    1. Final answer correctness (primary reward)
    2. Optional: Format compliance
    
    The reward is based on the final answer extracted from the response,
    regardless of how many steps the student generated.
    """
    
    def __init__(
        self,
        tokenizer: Any = None,
        num_examine: int = 0,
        format_score: float = 0.0,
        correct_score: float = 1.0,
        answer_start: str = "<answer>",
        answer_end: str = "</answer>",
        thinking_start: str = "<think>",
        thinking_end: str = "</think>",
        strict_format: bool = False,
        **kwargs,
    ):
        """
        Initialize Curriculum-GRPO Reward Manager.
        
        Args:
            tokenizer: Tokenizer (for decoding if needed).
            num_examine: Number of samples to examine for debugging.
            format_score: Score for correct format but wrong answer.
            correct_score: Score for correct answer.
            answer_start: Start tag for answer.
            answer_end: End tag for answer.
            thinking_start: Start tag for thinking.
            thinking_end: End tag for thinking.
            strict_format: Whether to require strict format compliance.
            **kwargs: Additional arguments.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.correct_score = correct_score
        self.answer_start = answer_start
        self.answer_end = answer_end
        self.thinking_start = thinking_start
        self.thinking_end = thinking_end
        self.strict_format = strict_format
    
    def __call__(self, data: Any, return_dict: bool = False, **kwargs) -> Any:
        """
        Compute rewards for a batch of data.
        
        Args:
            data: DataProto containing batch data.
            return_dict: Whether to return dict with extra info.
            **kwargs: Additional arguments.
        
        Returns:
            Reward tensor and optionally extra info dict.
        """
        if hasattr(data, 'batch'):
            return self._compute_batch_reward(data, return_dict=return_dict)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _compute_batch_reward(
        self,
        data: Any,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor, dict] | torch.Tensor:
        """
        Compute rewards for a batch.
        
        Args:
            data: DataProto with batch data.
            return_dict: Whether to return extra info.
        
        Returns:
            Reward tensor or tuple of (reward_tensor, extra_info_dict).
        """
        batch = data.batch
        non_tensor_batch = data.non_tensor_batch
        
        responses = batch.get("responses")
        attention_mask = batch.get("attention_mask")
        
        if responses is None:
            raise ValueError("responses not found in batch")
        
        batch_size = responses.shape[0]
        response_length = responses.shape[1]
        
        response_texts = self._decode_responses(responses, attention_mask)
        
        ground_truths = self._extract_ground_truths(non_tensor_batch)
        
        rewards = []
        extra_info = {
            "extracted_answers": [],
            "ground_truths": [],
            "is_correct": [],
            "has_format": [],
        }
        
        for i in range(batch_size):
            response_text = response_texts[i]
            ground_truth = ground_truths[i]
            
            extracted_answer = self._extract_answer(response_text)
            is_correct = self._check_answer(extracted_answer, ground_truth)
            has_format = self._check_format(response_text)
            
            if is_correct:
                reward = self.correct_score
            elif has_format and not self.strict_format:
                reward = self.format_score
            else:
                reward = 0.0
            
            rewards.append(reward)
            extra_info["extracted_answers"].append(extracted_answer)
            extra_info["ground_truths"].append(ground_truth)
            extra_info["is_correct"].append(is_correct)
            extra_info["has_format"].append(has_format)
        
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)
        for i, reward in enumerate(rewards):
            response_mask = self._get_response_end_mask(
                responses[i],
                attention_mask[i] if attention_mask is not None else None
            )
            last_valid_idx = response_mask.sum().item() - 1
            if last_valid_idx >= 0:
                reward_tensor[i, last_valid_idx] = reward
        
        if self.num_examine > 0:
            self._examine_samples(
                response_texts[:self.num_examine],
                extra_info["extracted_answers"][:self.num_examine],
                ground_truths[:self.num_examine],
                rewards[:self.num_examine],
            )
        
        if return_dict:
            return reward_tensor, extra_info
        else:
            return reward_tensor, extra_info
    
    def _decode_responses(
        self,
        responses: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[str]:
        """
        Decode response token IDs to text.
        
        Args:
            responses: Response token IDs [batch_size, response_len].
            attention_mask: Attention mask [batch_size, seq_len].
        
        Returns:
            List of decoded response strings.
        """
        response_texts = []
        
        for i in range(responses.shape[0]):
            tokens = responses[i].tolist()
            
            if attention_mask is not None:
                prompt_len = attention_mask.shape[1] - responses.shape[1]
                mask = attention_mask[i, prompt_len:].tolist()
                valid_tokens = [t for t, m in zip(tokens, mask) if m == 1]
            else:
                valid_tokens = [t for t in tokens if t != 0]
            
            text = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
            response_texts.append(text)
        
        return response_texts
    
    def _extract_ground_truths(self, non_tensor_batch: dict) -> list[str]:
        """
        Extract ground truths from non_tensor_batch.
        
        Args:
            non_tensor_batch: Non-tensor batch data.
        
        Returns:
            List of ground truth strings.
        """
        ground_truths = []
        
        if "reward_model" in non_tensor_batch:
            reward_models = non_tensor_batch["reward_model"]
            for rm in reward_models:
                if isinstance(rm, dict) and "ground_truth" in rm:
                    ground_truths.append(str(rm["ground_truth"]))
                else:
                    ground_truths.append("")
        elif "ground_truth" in non_tensor_batch:
            for gt in non_tensor_batch["ground_truth"]:
                ground_truths.append(str(gt))
        else:
            ground_truths = [""] * len(non_tensor_batch.get("raw_prompt", [""]))
        
        return ground_truths
    
    def _extract_answer(self, text: str) -> str:
        """
        Extract final answer from response text.
        
        Supports multiple formats:
        1. <answer>42</answer>
        2. #### 42
        3. The answer is 42.
        4. Plain number at the end
        
        Args:
            text: Response text.
        
        Returns:
            Extracted answer string.
        """
        if self.answer_start in text and self.answer_end in text:
            pattern = re.escape(self.answer_start) + r'\s*(-?[\d,\.]+)\s*' + re.escape(self.answer_end)
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(',', '')
        
        if "####" in text:
            match = re.search(r'####\s*(-?[\d,\.]+)', text)
            if match:
                return match.group(1).replace(',', '')
        
        patterns = [
            r'(?:the\s+)?answer\s+is\s+(-?[\d,\.]+)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?answer\s+is\s+(-?[\d,\.]+)',
            r'=\s*(-?[\d,\.]+)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        
        numbers = re.findall(r'-?[\d,\.]+', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def _check_answer(self, extracted: str, ground_truth: str) -> bool:
        """
        Check if extracted answer matches ground truth.
        
        Args:
            extracted: Extracted answer.
            ground_truth: Ground truth answer.
        
        Returns:
            True if answers match.
        """
        if not extracted or not ground_truth:
            return False
        
        try:
            ext_val = float(extracted.replace(',', '').replace('$', '').strip())
            gt_val = float(ground_truth.replace(',', '').replace('$', '').strip())
            return abs(ext_val - gt_val) < 1e-6
        except (ValueError, AttributeError):
            return extracted.strip().lower() == ground_truth.strip().lower()
    
    def _check_format(self, text: str) -> bool:
        """
        Check if response has valid format.
        
        Args:
            text: Response text.
        
        Returns:
            True if format is valid.
        """
        if self.answer_start in text and self.answer_end in text:
            return True
        
        if "####" in text:
            return True
        
        numbers = re.findall(r'-?[\d,\.]+', text)
        return len(numbers) > 0
    
    def _get_response_end_mask(
        self,
        response: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get mask for valid response tokens.
        
        Args:
            response: Response tokens.
            attention_mask: Attention mask.
        
        Returns:
            Boolean mask tensor.
        """
        if attention_mask is not None:
            return (attention_mask == 1).bool()
        else:
            return (response != 0).bool()
    
    def _examine_samples(
        self,
        response_texts: list[str],
        extracted_answers: list[str],
        ground_truths: list[str],
        rewards: list[float],
    ):
        """
        Log sample details for debugging.
        
        Args:
            response_texts: Response texts.
            extracted_answers: Extracted answers.
            ground_truths: Ground truths.
            rewards: Rewards.
        """
        logger.info("=" * 50)
        logger.info("Curriculum-GRPO Reward Samples:")
        logger.info("=" * 50)
        
        for i, (resp, ext, gt, rew) in enumerate(
            zip(response_texts, extracted_answers, ground_truths, rewards)
        ):
            logger.info(f"\n--- Sample {i} ---")
            logger.info(f"Response (last 200 chars): ...{resp[-200:]}")
            logger.info(f"Extracted: {ext}")
            logger.info(f"Ground Truth: {gt}")
            logger.info(f"Reward: {rew}")
        
        logger.info("=" * 50)
