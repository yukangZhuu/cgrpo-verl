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
        # Disable internal logging by default as we use Trainer's unified dump
        self.num_examine = 0 
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
            extra_info["has_format"].append(has_format)
            extra_info["failure_reasons"].append(failure_reason)
            extra_info["is_truncated"].append(is_truncated)
        
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)
        for i, reward in enumerate(rewards):
            response_mask = self._get_response_end_mask(
                responses[i],
                attention_mask[i] if attention_mask is not None else None,
                response_length,
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
                extra_info["failure_reasons"][:self.num_examine],
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
        Check if extracted answer matches ground truth using math-verify.
        
        Args:
            extracted: Extracted answer.
            ground_truth: Ground truth answer.
        
        Returns:
            True if answers match.
        """
        if MATH_VERIFY_AVAILABLE:
            return self._verify_answer_with_math_verify(ground_truth, extracted)
        else:
            return self._verify_answer_fallback(ground_truth, extracted)

    def _verify_answer_with_math_verify(self, ground_truth: str, model_answer: str) -> bool:
        if self._is_empty(ground_truth) or self._is_empty(model_answer):
            return False
        
        gt_clean = re.sub(r'\s+', '', ground_truth.strip())
        ma_clean = re.sub(r'\s+', '', model_answer.strip())
        if gt_clean == ma_clean:
            return True
        
        if self._is_number(ground_truth) and self._is_number(model_answer):
            if self._compare_numbers(ground_truth, model_answer):
                return True
        
        try:
            parsed_gt = parse(ground_truth, extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
                StringExtractionConfig()
            ])
            parsed_ma = parse(model_answer, extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
                StringExtractionConfig()
            ])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            mcq_answers = ['A', 'B', 'C', 'D', 'E']
            config = StringExtractionConfig(strings=tuple(mcq_answers))
            parsed_gt = parse(ground_truth, extraction_config=[config])
            parsed_ma = parse(model_answer, extraction_config=[config])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            parsed_gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
            parsed_ma = parse(model_answer, extraction_config=[ExprExtractionConfig()])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            wrapped_gt = self._wrap_latex(ground_truth)
            wrapped_ma = self._wrap_latex(model_answer)
            
            parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
            parsed_ma = parse(wrapped_ma, extraction_config=[LatexExtractionConfig()])
            
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            numbers_gt = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
            numbers_ma = re.findall(r'[-+]?\d*\.?\d+', model_answer)
            if numbers_gt and numbers_ma:
                if numbers_gt == numbers_ma:
                    return True
        except Exception as e:
            pass
        
        return False

    def _verify_answer_fallback(self, ground_truth: str, model_answer: str) -> bool:
        if self._is_empty(ground_truth) or self._is_empty(model_answer):
            return False
        
        gt_clean = re.sub(r'\s+', '', ground_truth.strip())
        ma_clean = re.sub(r'\s+', '', model_answer.strip())
        if gt_clean == ma_clean:
            return True
        
        if self._is_number(ground_truth) and self._is_number(model_answer):
            if self._compare_numbers(ground_truth, model_answer):
                return True
        
        try:
            numbers_gt = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
            numbers_ma = re.findall(r'[-+]?\d*\.?\d+', model_answer)
            if numbers_gt and numbers_ma:
                if numbers_gt == numbers_ma:
                    return True
        except Exception as e:
            pass
        
        return False

    def _is_empty(self, text: str) -> bool:
        return not text or text.strip() == ""

    def _is_number(self, text: str) -> bool:
        try:
            float(text.strip())
            return True
        except ValueError:
            return False

    def _compare_numbers(self, num1: str, num2: str) -> bool:
        try:
            n1 = float(num1.strip())
            n2 = float(num2.strip())
            return abs(n1 - n2) < 1e-6
        except ValueError:
            return False

    def _is_latex_wrapped(self, text: str) -> bool:
        text = text.strip()
        
        latex_patterns = [
            r'^\\\[.*\\\]$',          
            r'^\$\$.*\$\$$',          
            r'^\\boxed\{.*\}$',       
            r'^\$.*\$',                
            r'^\\\(.*\\\)$',          
            r'^\[.*\]$',              
        ]
        
        for pattern in latex_patterns:
            if re.match(pattern, text, re.DOTALL):
                return True
        
        return False

    def _wrap_latex(self, text: str) -> str:
        text = text.strip()
        
        if self._is_latex_wrapped(text):
            return text
        
        return f'\\boxed{{{text}}}'
    
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
