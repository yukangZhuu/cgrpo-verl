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
        
        self.thinking_start = self.config.data.get("thinking_start", "<think")
        self.thinking_end = self.config.data.get("thinking_end", "</think")
        self.answer_start = self.config.data.get("answer_start", "<answer>")
        self.answer_end = self.config.data.get("answer_end", "</answer>")
        
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
                - steps: Teacher's reasoning steps
                - current_k: Current curriculum level (optional, overrides self.current_k)
        
        Returns:
            AgentLoopOutput with generated response.
        """
        messages = list(kwargs["raw_prompt"])
        steps = kwargs.get("steps", [])
        current_k = kwargs.get("current_k", self.current_k)
        
        prompt_ids = await self._build_curriculum_prompt(
            messages=messages,
            steps=steps,
            current_k=current_k,
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
    
    async def _build_curriculum_prompt(
        self,
        messages: list[dict],
        steps: list[str],
        current_k: int,
    ) -> list[int]:
        """
        Build curriculum-aware prompt.
        
        Args:
            messages: Original messages (user question).
            steps: Teacher's reasoning steps.
            current_k: Current curriculum level.
        
        Returns:
            Token IDs for the constructed prompt.
        """
        if not steps:
            prompt_ids = await self.apply_chat_template(messages)
            return prompt_ids
        
        num_steps = len(steps)
        cut_index = max(0, num_steps - current_k)
        
        teacher_prefix_steps = steps[:cut_index]
        
        if teacher_prefix_steps:
            teacher_prefix = "\n".join(teacher_prefix_steps)
            assistant_prefix = f"{self.thinking_start}\n{teacher_prefix}"
            
            prompt_ids = await self._apply_curriculum_chat_template(
                messages=messages,
                assistant_prefix=assistant_prefix,
            )
        else:
            prompt_ids = await self.apply_chat_template(messages)
        
        return prompt_ids
    
    async def _apply_curriculum_chat_template(
        self,
        messages: list[dict],
        assistant_prefix: str,
    ) -> list[int]:
        """
        Apply chat template with assistant prefix.
        
        This creates a prompt where the assistant has already started
        responding with the teacher prefix.
        
        Args:
            messages: User messages.
            assistant_prefix: Pre-filled assistant content.
        
        Returns:
            Token IDs for the prompt.
        """
        full_messages = messages + [{"role": "assistant", "content": assistant_prefix}]
        
        prompt_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        if not prompt_text.endswith(assistant_prefix):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_text += assistant_prefix
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        if len(prompt_ids) > self.prompt_length:
            prompt_ids = prompt_ids[-self.prompt_length:]
            logger.warning(f"Prompt truncated to {self.prompt_length} tokens")
        
        return prompt_ids
