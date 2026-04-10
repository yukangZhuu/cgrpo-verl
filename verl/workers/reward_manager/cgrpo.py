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

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)

if not MATH_VERIFY_AVAILABLE:
    raise ImportError(
        "math_verify is required for CurriculumGRPORewardManager but is not installed. "
        "Without it, LaTeX and symbolic answer verification is disabled, which causes "
        "most correct answers to be scored as wrong — leading to training collapse. "
        "Install with:  pip install math-verify"
    )


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
        strict_format: bool = False,
        overlong_buffer_enable: bool = False,
        overlong_buffer_len: int = 1024,
        overlong_penalty_factor: float = 1.0,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.correct_score = correct_score
        self.strict_format = strict_format
        self.overlong_buffer_enable = overlong_buffer_enable
        self.overlong_buffer_len = overlong_buffer_len
        self.overlong_penalty_factor = overlong_penalty_factor

        if self.overlong_buffer_enable:
            logger.info(
                f"Overlong buffer enabled: buffer_len={overlong_buffer_len}, "
                f"penalty_factor={overlong_penalty_factor}"
            )
    
    def __call__(self, data: Any, return_dict: bool = False, **kwargs) -> Any:
        """
        Compute rewards for a batch of data.

        When ``return_dict=True`` (used by the parent trainer's validation path),
        returns a **dict** ``{"reward_tensor": ..., "reward_extra_info": ...}``
        to match the interface expected by ``RayPPOTrainer._compute_reward_legacy``.

        When ``return_dict=False``, returns a **tuple** ``(reward_tensor, extra_info)``.
        """
        if not hasattr(data, "batch"):
            raise ValueError(f"Unsupported data type: {type(data)}")

        reward_tensor, extra_info = self._compute_batch_reward(data)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": extra_info,
            }
        return reward_tensor, extra_info
    
    def _compute_batch_reward(
        self,
        data: Any,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute rewards for a batch.

        Returns:
            Tuple of (reward_tensor, extra_info_dict).
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
        last_valid_indices = []
        extra_info = {
            "extracted_answers": [],
            "ground_truths": [],
            "is_correct": [],
            "acc": [],
            "has_format": [],
            "failure_reasons": [],
            "is_truncated": [],
        }
        
        for i in range(batch_size):
            response_text = response_texts[i]
            ground_truth = ground_truths[i]
            
            extracted_answer = self._extract_answer(response_text)
            is_correct = self._check_answer(extracted_answer, ground_truth)
            has_format = self._check_format(response_text)
            
            response_mask = self._get_response_end_mask(
                responses[i],
                attention_mask[i] if attention_mask is not None else None,
                response_length,
            )
            last_valid_idx = response_mask.sum().item() - 1
            last_valid_indices.append(last_valid_idx)
            
            is_truncated = False
            if last_valid_idx == response_length - 1:
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                last_token = responses[i, last_valid_idx].item()
                if eos_id is None or last_token != eos_id:
                    is_truncated = True
            
            failure_reason = "none"
            if is_correct:
                reward = self.correct_score
            elif has_format and not self.strict_format:
                reward = self.format_score
                failure_reason = "wrong_answer"
            else:
                reward = 0.0
                if is_truncated:
                    failure_reason = "truncated"
                elif not has_format:
                    failure_reason = "format_error"
                else:
                    failure_reason = "wrong_answer"
            
            rewards.append(reward)
            extra_info["extracted_answers"].append(extracted_answer)
            extra_info["ground_truths"].append(ground_truth)
            extra_info["is_correct"].append(is_correct)
            extra_info["acc"].append(float(is_correct))
            extra_info["has_format"].append(has_format)
            extra_info["failure_reasons"].append(failure_reason)
            extra_info["is_truncated"].append(is_truncated)
        
        # --- Overlong reward shaping (DAPO-style) ---
        extra_info["overlong_penalty_applied"] = [False] * batch_size
        if self.overlong_buffer_enable:
            buffer_start = response_length - self.overlong_buffer_len
            for i in range(batch_size):
                actual_len = last_valid_indices[i] + 1
                if actual_len > buffer_start and not extra_info["is_correct"][i]:
                    fraction = min((actual_len - buffer_start) / max(self.overlong_buffer_len, 1), 1.0)
                    penalty = fraction * self.overlong_penalty_factor
                    rewards[i] = -penalty
                    extra_info["overlong_penalty_applied"][i] = True
                    if extra_info["is_truncated"][i]:
                        extra_info["failure_reasons"][i] = "truncated_penalized"

        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)
        for i, reward in enumerate(rewards):
            if last_valid_indices[i] >= 0:
                reward_tensor[i, last_valid_indices[i]] = reward
        
        if self.num_examine > 0:
            self._examine_samples(
                response_texts[:self.num_examine],
                extra_info["extracted_answers"][:self.num_examine],
                ground_truths[:self.num_examine],
                rewards[:self.num_examine],
                extra_info["failure_reasons"][:self.num_examine],
            )
        
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
        
        Only supports:
        1. \boxed{42}
        
        Args:
            text: Response text.
        
        Returns:
            Extracted answer string.
        """
        # Try extracting from \boxed{}
        boxed_start = r'\boxed{'
        start_idx = text.rfind(boxed_start)
        
        if start_idx != -1:
            content_start = start_idx + len(boxed_start)
            brace_count = 0
            content_end = -1
            
            for i in range(content_start, len(text)):
                char = text[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    if brace_count == 0:
                        content_end = i
                        break
                    else:
                        brace_count -= 1
            
            if content_end != -1:
                return text[content_start:content_end].strip()
        
        return ""
    
    def _check_answer(self, extracted: str, ground_truth: str) -> bool:
        """
        Check if extracted answer matches ground truth.

        Verification cascade (stops at first match):
        1. Exact string match (after whitespace normalization)
        2. Numeric comparison (float, tolerance 1e-6)
        3. math_verify with all extraction configs (LaTeX, Expr, String)
        4. math_verify with boxed-wrapped LaTeX parsing
        """
        if not extracted or not extracted.strip() or not ground_truth or not ground_truth.strip():
            return False

        gt_clean = re.sub(r'\s+', '', ground_truth.strip())
        ma_clean = re.sub(r'\s+', '', extracted.strip())
        if gt_clean == ma_clean:
            return True

        try:
            gt_f = float(ground_truth.strip())
            ma_f = float(extracted.strip())
            if abs(gt_f - ma_f) < 1e-6:
                return True
        except (ValueError, OverflowError):
            pass

        try:
            configs = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]
            parsed_gt = parse(ground_truth, extraction_config=configs)
            parsed_ma = parse(extracted, extraction_config=configs)
            if parsed_gt and parsed_ma and verify(parsed_gt, parsed_ma):
                return True
        except Exception:
            pass

        try:
            wrapped_gt = f'\\boxed{{{ground_truth.strip()}}}'
            wrapped_ma = f'\\boxed{{{extracted.strip()}}}'
            parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
            parsed_ma = parse(wrapped_ma, extraction_config=[LatexExtractionConfig()])
            if parsed_gt and parsed_ma and verify(parsed_gt, parsed_ma):
                return True
        except Exception:
            pass

        return False
    
    def _check_format(self, text: str) -> bool:
        """
        Check if response has valid format.
        
        Args:
            text: Response text.
        
        Returns:
            True if format is valid.
        """
        if r'\boxed{' in text:
            return True
        
        return False
    
    def _get_response_end_mask(
        self,
        response: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        response_length: int = None,
    ) -> torch.Tensor:
        """
        Get mask for valid response tokens.
        
        Args:
            response: Response tokens.
            attention_mask: Attention mask (full sequence).
            response_length: Length of response portion.
        
        Returns:
            Boolean mask tensor for response portion only.
        """
        if attention_mask is not None:
            if response_length is not None and len(attention_mask) > response_length:
                prompt_len = len(attention_mask) - response_length
                response_mask = attention_mask[prompt_len:]
            else:
                response_mask = attention_mask[-len(response):]
            return (response_mask == 1).bool()
        else:
            return (response != 0).bool()
    
    def _examine_samples(
        self,
        response_texts: list[str],
        extracted_answers: list[str],
        ground_truths: list[str],
        rewards: list[float],
        failure_reasons: list[str],
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
        
        for i, (resp, ext, gt, rew, reason) in enumerate(
            zip(response_texts, extracted_answers, ground_truths, rewards, failure_reasons)
        ):
            # Only log detailed info for failed samples or a few success ones
            # For now, log all passed samples (up to num_examine)
            
            logger.info(f"\n--- Sample {i} ---")
            logger.info(f"Response (last 200 chars): ...{resp[-200:]}")
            logger.info(f"Extracted: {ext}")
            logger.info(f"Ground Truth: {gt}")
            logger.info(f"Reward: {rew}")
            
            if rew == 0.0:
                logger.info("Status: FAILED")
                logger.info(f"Failure Reason: {reason}")
            else:
                logger.info("Status: SUCCESS")
        
        logger.info("=" * 50)
