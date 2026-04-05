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
Agent Loop for CGRPO.
Supports three prompt construction modes: none, prefix, hint.

Final prompt layout (Qwen3 ChatML):

    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    <think>
    {teacher_prefix — only for prefix mode with non-empty guidance}
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("curriculum_agent")
class CurriculumAgentLoop(AgentLoopBase):
    """
    Agent loop for CGRPO / standard GRPO.

    Prompt construction depends on ``guidance_mode`` (from config):
      * ``none``   – standard prompt, no teacher guidance
      * ``prefix`` – teacher guidance_steps prepended inside <think> tag
      * ``hint``   – hints already embedded in user message by dataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.guidance_mode = self.config.data.get("guidance_mode", "none")

        logger.info(
            f"CurriculumAgentLoop initialized, guidance_mode={self.guidance_mode}"
        )

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        guidance_steps: list[str] = kwargs.get("guidance_steps", [])

        teacher_prefix = ""
        if self.guidance_mode == "prefix" and len(guidance_steps) > 0:
            teacher_prefix = "\n".join(guidance_steps)

        prompt_ids = await self._build_prompt(
            messages=messages,
            teacher_prefix=teacher_prefix,
        )

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
            )

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = (
                output.num_preempted
                if output.num_preempted is not None
                else -1
            )

        response_mask = [1] * len(output.token_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                output.log_probs[: self.response_length]
                if output.log_probs
                else None
            ),
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=None,
            num_turns=2,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Prompt construction — single-pass encoding
    # ------------------------------------------------------------------

    async def _build_prompt(
        self,
        messages: list[dict],
        teacher_prefix: str = "",
    ) -> list[int]:
        """
        Build prompt token ids via single-pass encoding.

        1. ``apply_chat_template`` → base text ending with
           ``<|im_start|>assistant\\n``
        2. Append ``<think>\\n`` (+ teacher_prefix for prefix mode)
        3. Encode the *complete* string once so that ``<think>`` is
           correctly recognised as a special token in context.
        """
        # --- assemble full text ---
        base_text: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if teacher_prefix:
            full_text = f"{base_text}<think>\n{teacher_prefix}\n"
        else:
            full_text = f"{base_text}<think>\n"

        # --- single-pass encode ---
        prompt_ids: list[int] = self.tokenizer.encode(
            full_text, add_special_tokens=False
        )

        # --- truncation (head-truncate teacher prefix part) ---
        if len(prompt_ids) > self.prompt_length:
            base_ids = self.tokenizer.encode(
                f"{base_text}<think>\n", add_special_tokens=False
            )
            if len(base_ids) < self.prompt_length:
                # Keep full base; truncate teacher prefix from the head
                # (preserve the END of the reasoning, closest to where the
                # student continues)
                prefix_budget = self.prompt_length - len(base_ids)
                teacher_ids = prompt_ids[len(base_ids) :]
                teacher_ids = teacher_ids[-prefix_budget:]
                prompt_ids = base_ids + teacher_ids
                logger.warning(
                    f"Head-truncated teacher prefix to fit prompt_length="
                    f"{self.prompt_length}"
                )
            else:
                # Even base is too long — tail-truncate everything
                prompt_ids = prompt_ids[-self.prompt_length :]
                logger.warning(
                    f"Base prompt too long ({len(base_ids)}), "
                    f"tail-truncating to {self.prompt_length}"
                )

        return prompt_ids
