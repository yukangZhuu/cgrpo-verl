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
Curriculum Agent Loop for C-GRPO.
Implements backward chaining prompt construction.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("curriculum_agent")
class CurriculumAgentLoop(AgentLoopBase):
    """
    Agent loop for Curriculum-GRPO.
    
    Implements backward chaining by:
    1. Receiving current curriculum level k
    2. Building prompt with teacher prefix (first L-k steps)
    3. Student generates remaining k steps + answer
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        
        self.thinking_start = self.config.data.get("thinking_start", "<think>")
        self.thinking_end = self.config.data.get("thinking_end", "</think>")
        self.answer_start = self.config.data.get("answer_start", "<answer>")
        self.answer_end = self.config.data.get("answer_end", "</answer>")
        self.instruction_following = self.config.data.get(
            "instruction_following",
            None,
        )
        
        self.current_k = self.config.curriculum.get("initial_k", 1)
        
        logger.info(f"CurriculumAgentLoop initialized with k={self.current_k}")
    
    def set_curriculum_k(self, k: int):
        """Set current curriculum level."""
        self.current_k = k
        logger.info(f"CurriculumAgentLoop k updated to {k}")
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run curriculum-aware generation.
        
        Args:
            sampling_params: Sampling parameters.
            **kwargs: Additional arguments including:
                - raw_prompt: Original question
                - steps: Teacher's reasoning steps (optional, used if teacher_prefix not provided)
                - teacher_prefix: Pre-computed teacher prefix (optional)
                - current_k: Current curriculum level (optional, overrides self.current_k)
        
        Returns:
            AgentLoopOutput with generated response.
        """
        messages = list(kwargs["raw_prompt"])
        steps = kwargs.get("steps", [])
        teacher_prefix = kwargs.get("teacher_prefix", "")
        current_k = kwargs.get("current_k", self.current_k)
        
        # Unify steps and teacher_prefix handling
        # Priority: teacher_prefix > steps > nothing
        
        teacher_prefix = kwargs.get("teacher_prefix", "")
        steps = kwargs.get("steps", [])
        current_k = kwargs.get("current_k", self.current_k)
        
        # If teacher_prefix is not provided but steps are, derive prefix from steps
        if not teacher_prefix and steps:
            num_steps = len(steps)
            cut_index = max(0, num_steps - current_k)
            teacher_prefix_steps = steps[:cut_index]
            if teacher_prefix_steps:
                teacher_prefix = "\n".join(teacher_prefix_steps)
        
        # Build prompt
        prompt_ids = await self._build_prompt(
            messages=messages,
            teacher_prefix=teacher_prefix
        )
        
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
            )
        
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        
        response_mask = [1] * len(output.token_ids)
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=None,
            num_turns=2,
            metrics=metrics,
        )
        return output
    
    async def _build_prompt(
        self,
        messages: list[dict],
        teacher_prefix: str = "",
    ) -> list[int]:
        """
        Build curriculum-aware prompt with optional teacher prefix.
        
        Handles ChatML formatting and smart truncation.
        
        Args:
            messages: Original messages (user question).
            teacher_prefix: Pre-computed teacher prefix.
        
        Returns:
            Token IDs for the constructed prompt.
        """
        messages = self._add_instruction_following(messages)
        
        # If no prefix, just use standard chat template
        # BUT we still want to enforce thinking format if possible.
        # However, standard apply_chat_template usually ends with assistant header.
        # If we append thinking start, we force the model to think.
        
        # Construct prefix with thinking tags
        # Ensure there is a transition phrase before the teacher prefix
        # This helps the model align with the "thinking" mode
        thought_start_phrase = self.config.data.get("thought_start_phrase", "Okay, let's think step by step.")
        
        if teacher_prefix:
            assistant_prefix = f"{self.thinking_start}\n{thought_start_phrase}\n{teacher_prefix}"
        else:
            # Even without teacher prefix, we force thinking start
            assistant_prefix = f"{self.thinking_start}\n{thought_start_phrase}"
        
        # 1. Encode base messages (System + User)
        base_prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        base_ids = self.tokenizer.encode(base_prompt_text, add_special_tokens=False)
        
        # 2. Encode teacher prefix
        prefix_ids = self.tokenizer.encode(assistant_prefix, add_special_tokens=False)
        
        # 3. Combine and Truncate
        # Strategy: Head Truncation on Teacher Prefix
        # We prioritize keeping the base prompt (system instruction + question)
        # and the END of the teacher prefix (most recent reasoning steps).
        
        if len(base_ids) + len(prefix_ids) > self.prompt_length:
            remaining_len = self.prompt_length - len(base_ids)
            
            if remaining_len > 0:
                # Truncate prefix from head (keep last part of reasoning)
                prefix_ids = prefix_ids[-remaining_len:]
                prompt_ids = base_ids + prefix_ids
                logger.warning(
                    f"Truncated teacher prefix from {len(prefix_ids)+len(base_ids)} "
                    f"to {self.prompt_length} tokens (Head Truncation)"
                )
            else:
                # Base prompt itself is too long, fall back to standard tail truncation
                # This is a fallback and shouldn't happen with reasonable prompt_length
                prompt_ids = (base_ids + prefix_ids)[-self.prompt_length:]
                logger.warning(
                    f"Base prompt too long ({len(base_ids)} > {self.prompt_length}), "
                    f"performing standard tail truncation."
                )
        else:
            prompt_ids = base_ids + prefix_ids
        
        return prompt_ids
    
    def _add_instruction_following(self, messages: list[dict]) -> list[dict]:
        """
        Add instruction following to user message if not present.
        
        Args:
            messages: Original messages.
        
        Returns:
            Messages with instruction following added.
        """
        if not self.instruction_following:
            return messages
        
        messages = list(messages)
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if self.instruction_following and not content.rstrip().endswith(self.instruction_following):
                    messages[i] = {
                        "role": "user",
                        "content": content.rstrip() + " " + self.instruction_following,
                    }
                break
        
        return messages
