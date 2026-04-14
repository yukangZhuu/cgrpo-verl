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
CGRPO Trainer.
Extends RayPPOTrainer for:
  - standard GRPO          (curriculum_method=none)
  - static-mixture GRPO    (curriculum_method=mixture)
  - per-sample adaptive    (curriculum_method=adaptive, AdaBack-style)
"""

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import uuid

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.utils.curriculum import TrainingMetricsTracker, PerSampleCurriculumState
from verl.utils.dataset.curriculum_dataset import CurriculumGRPODataset
from verl.workers.reward_manager.cgrpo import CurriculumGRPORewardManager

logger = logging.getLogger(__name__)


class CurriculumGRPOTrainer(RayPPOTrainer):
    """
    Trainer for CGRPO / standard GRPO / adaptive curriculum.

    Extends RayPPOTrainer with:
    1. Per-sample guidance via g_level / guidance_steps (from dataset or adaptive state)
    2. CurriculumGRPORewardManager for boxed-answer verification
    3. Optional AdaBack-style per-sample adaptive curriculum
    4. Debug sample dumping for analysis
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        from verl.utils.profiler import Profiler

        self.profiler = Profiler()
        self.metrics_tracker = TrainingMetricsTracker(
            ema_alpha=self.config.trainer.get("ema_alpha", 0.1),
        )
        self.guidance_mode = self.config.data.get("guidance_mode", "none")
        self.curriculum_method = self.config.data.get("curriculum_method", "none")
        self._training_start_time = None

        # --- Adaptive curriculum state (only when curriculum_method=adaptive) ---
        self.adaptive_state: Optional[PerSampleCurriculumState] = None
        if self.curriculum_method == "adaptive":
            ac = self.config.get("adaptive_curriculum", {})
            self.adaptive_state = PerSampleCurriculumState(
                tau=ac.get("tau", 0.4),
                p_zero=ac.get("p_zero", 0.1),
                default_rho=ac.get("default_rho", 0.5),
                min_step_delta=ac.get("min_step_delta", 1),
            )
            logger.info(
                f"Adaptive curriculum enabled: tau={self.adaptive_state.tau}, "
                f"p_zero={self.adaptive_state.p_zero}"
            )

        logger.info(
            f"CurriculumGRPOTrainer initialized: "
            f"guidance_mode={self.guidance_mode}, "
            f"curriculum_method={self.curriculum_method}"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        from verl.utils.tracking import Tracking

        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="CGRPO Training",
        )

        self._training_start_time = time.time()
        self.global_steps += 1

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = (
                    self.config.actor_rollout_ref.rollout.temperature
                )

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )

                # --- Adaptive guidance (before generation) ---
                if self.curriculum_method == "adaptive":
                    self._apply_adaptive_guidance(batch)

                # --- Generate ---
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                with self.profiler.context_manager("rollout"):
                    gen_batch_output = (
                        self.async_rollout_manager.generate_sequences(gen_batch_output)
                    )

                self.checkpoint_manager.sleep_replicas()

                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                batch = batch.union(gen_batch_output)

                if "response_mask" not in batch.batch.keys():
                    from verl.trainer.ppo.ray_trainer import compute_response_mask

                    batch.batch["response_mask"] = compute_response_mask(batch)

                # --- Reward ---
                with self.profiler.context_manager("reward_computation"):
                    reward_tensor, reward_extra_info = self._compute_reward(batch)

                batch.batch["token_level_scores"] = reward_tensor
                batch.batch["token_level_rewards"] = reward_tensor

                # --- Adaptive state update (after reward) ---
                if self.curriculum_method == "adaptive":
                    self._update_adaptive_state(batch, reward_extra_info)

                # --- Debug dump ---
                dump_freq = self.config.trainer.get("debug_dump_freq", 0)
                if dump_freq > 0 and self.global_steps % dump_freq == 0:
                    try:
                        self._dump_debug_samples(
                            batch=batch,
                            reward_tensor=reward_tensor,
                            reward_extra_info=reward_extra_info,
                            num_samples=self.config.trainer.get(
                                "debug_dump_num_samples", 10
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Debug dump failed: {e}")

                # --- Log probs & advantage ---
                old_log_prob, _ = self._compute_old_log_prob(batch)
                batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    ),
                    config=self.config.algorithm,
                )

                data_metrics = compute_data_metrics(
                    batch=batch, use_critic=self.use_critic
                )
                metrics.update(data_metrics)

                # --- Actor update ---
                with self.profiler.context_manager("actor_update"):
                    self._update_actor(batch)

                self.checkpoint_manager.update_weights()

                # --- Metrics ---
                acc_list = reward_extra_info.get("acc", [])
                batch_success_rate = (
                    sum(acc_list) / len(acc_list) if acc_list else 0.0
                )
                tracker_metrics = self.metrics_tracker.update(
                    batch_success_rate=batch_success_rate,
                    batch_size=len(batch),
                )
                metrics.update(tracker_metrics)

                reward_mean = reward_tensor.sum(dim=-1).mean().item()
                metrics["training/reward_mean"] = reward_mean

                truncated_list = reward_extra_info.get("is_truncated", [])
                overlong_rate = (
                    sum(truncated_list) / len(truncated_list)
                    if truncated_list
                    else 0.0
                )
                metrics["training/overlong_rate"] = overlong_rate

                if self.adaptive_state is not None:
                    metrics.update(self.adaptive_state.get_metrics())

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                timing_metrics = self.profiler.get_metrics()
                metrics.update(timing_metrics)

                tracking.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {"sr": f"{batch_success_rate:.3f}", "mode": self.guidance_mode}
                )

                self._save_progress(epoch=epoch, batch_sr=batch_success_rate)
                self.global_steps += 1

                if (
                    self.config.trainer.save_freq > 0
                    and self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                if (
                    self.config.trainer.test_freq > 0
                    and self.global_steps % self.config.trainer.test_freq == 0
                ):
                    val_metrics = self._validate()
                    tracking.log(data=val_metrics, step=self.global_steps)

        progress_bar.close()

    # ------------------------------------------------------------------
    # Adaptive curriculum hooks
    # ------------------------------------------------------------------

    def _apply_adaptive_guidance(self, batch: DataProto) -> None:
        """
        For each sample in the batch whose g_level == -1 (sentinel from dataset),
        sample a rho from the adaptive state, compute guidance_steps, and
        rebuild raw_prompt if needed (hint mode).

        Frozen anchors (frozen_g_level >= 0) are left untouched.
        """
        assert self.adaptive_state is not None
        batch_size = len(batch)

        new_g_levels = []
        new_guidance_steps_list = []
        new_raw_prompts = []

        for i in range(batch_size):
            item = batch[i]
            g_level = float(item.non_tensor_batch.get("g_level", 0.0))
            frozen = float(item.non_tensor_batch.get("frozen_g_level", -1.0))
            adaptive_id = str(item.non_tensor_batch.get("adaptive_id", str(i)))
            steps = item.non_tensor_batch.get("steps", [])
            if isinstance(steps, np.ndarray):
                steps = steps.tolist()

            if frozen >= 0:
                # Anchor sample — keep existing g_level and guidance
                new_g_levels.append(g_level)
                new_guidance_steps_list.append(
                    item.non_tensor_batch.get("guidance_steps", [])
                )
                new_raw_prompts.append(item.non_tensor_batch.get("raw_prompt", []))
                continue

            num_steps = len(steps) if isinstance(steps, list) else 0
            rho = self.adaptive_state.get_rho(adaptive_id, num_steps)
            guidance = PerSampleCurriculumState.compute_guidance_steps(steps, rho)

            new_g_levels.append(rho)
            question = item.non_tensor_batch.get(
                "question",
                self._extract_question_from_prompt(item.non_tensor_batch.get("raw_prompt", [])),
            )

            if self.guidance_mode == "hint":
                if guidance:
                    user_content = CurriculumGRPODataset._build_hint_user_content(
                        question, guidance
                    )
                else:
                    user_content = CurriculumGRPODataset._build_standard_user_content(
                        question
                    )
                from verl.utils.dataset.curriculum_dataset import SYSTEM_PROMPT
                raw_prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                new_raw_prompts.append(raw_prompt)
                new_guidance_steps_list.append([])
            elif self.guidance_mode == "prefix":
                new_raw_prompts.append(item.non_tensor_batch.get("raw_prompt", []))
                new_guidance_steps_list.append(guidance)
            else:
                new_raw_prompts.append(item.non_tensor_batch.get("raw_prompt", []))
                new_guidance_steps_list.append([])

        batch.non_tensor_batch["g_level"] = np.array(new_g_levels, dtype=object)
        batch.non_tensor_batch["guidance_steps"] = np.array(
            new_guidance_steps_list, dtype=object
        )
        batch.non_tensor_batch["raw_prompt"] = np.array(
            new_raw_prompts, dtype=object
        )

    def _update_adaptive_state(
        self, batch: DataProto, reward_extra_info: dict
    ) -> None:
        """
        After reward computation, update the per-sample rho intervals.
        The batch has been repeat()ed by rollout.n, so we aggregate per
        original sample using adaptive_id.
        """
        assert self.adaptive_state is not None
        n_rollouts = self.config.actor_rollout_ref.rollout.n

        acc_list = reward_extra_info.get("acc", [])
        if not acc_list:
            return

        batch_size = len(batch)
        num_originals = batch_size // n_rollouts

        for orig_idx in range(num_originals):
            start = orig_idx * n_rollouts
            end = start + n_rollouts

            item = batch[start]
            frozen = float(item.non_tensor_batch.get("frozen_g_level", -1.0))
            if frozen >= 0:
                continue

            adaptive_id = str(item.non_tensor_batch.get("adaptive_id", str(orig_idx)))
            steps = item.non_tensor_batch.get("steps", [])
            if isinstance(steps, np.ndarray):
                steps = steps.tolist()
            num_steps = len(steps) if isinstance(steps, list) else 0

            rollout_accs = acc_list[start:end]
            avg_reward = sum(rollout_accs) / len(rollout_accs) if rollout_accs else 0.0

            self.adaptive_state.update(adaptive_id, avg_reward, num_steps)

    @staticmethod
    def _extract_question_from_prompt(raw_prompt) -> str:
        """Best-effort extraction of the question text from raw_prompt messages."""
        if isinstance(raw_prompt, list):
            for msg in raw_prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    # The question is the first part before our instruction blocks
                    parts = content.split("\nPlease reason step by step")
                    if parts:
                        return parts[0].split("\nPlease solve the problem")[0].split("\nBelow are some")[0].strip()
                    return content
        return str(raw_prompt)

    # ------------------------------------------------------------------
    # Checkpoint — save/restore adaptive state
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if self.adaptive_state is not None:
            adaptive_path = os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.global_steps}",
                "adaptive_curriculum_state.json",
            )
            os.makedirs(os.path.dirname(adaptive_path), exist_ok=True)
            with open(adaptive_path, "w", encoding="utf-8") as f:
                json.dump(self.adaptive_state.state_dict(), f, indent=2)
            logger.info(f"Adaptive curriculum state saved to {adaptive_path}")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.adaptive_state is None:
            return
        if self.config.trainer.resume_mode == "disable":
            return

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if global_step_folder is None:
            return

        adaptive_path = os.path.join(
            global_step_folder, "adaptive_curriculum_state.json"
        )
        if os.path.exists(adaptive_path):
            with open(adaptive_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.adaptive_state.load_state_dict(state)
            logger.info(f"Adaptive curriculum state loaded from {adaptive_path}")

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self, batch: DataProto
    ) -> tuple[torch.Tensor, dict]:
        if self.reward_fn is not None and isinstance(
            self.reward_fn, CurriculumGRPORewardManager
        ):
            return self.reward_fn(batch, return_dict=False)
        else:
            from verl.trainer.ppo.reward import compute_reward

            return compute_reward(batch, self.reward_fn)

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _save_progress(self, epoch: int, batch_sr: float):
        progress_path = os.path.join(
            self.config.trainer.get("default_local_dir", "./checkpoints"),
            "progress.json",
        )
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)

        elapsed = time.time() - self._training_start_time if self._training_start_time else 0
        steps_per_epoch = len(self.train_dataloader)

        status = {
            "global_step": self.global_steps,
            "epoch": epoch,
            "total_epochs": self.config.trainer.total_epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": self.total_training_steps,
            "progress_pct": round(self.global_steps / max(self.total_training_steps, 1) * 100, 1),
            "batch_sr": round(batch_sr, 4),
            "sr_ema": round(self.metrics_tracker.sr_ema, 4),
            "guidance_mode": self.guidance_mode,
            "curriculum_method": self.curriculum_method,
            "elapsed_seconds": round(elapsed, 1),
            "avg_seconds_per_step": round(elapsed / max(self.global_steps, 1), 1),
            "experiment_name": self.config.trainer.experiment_name,
        }

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Debug dump
    # ------------------------------------------------------------------

    def _dump_debug_samples(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_info: dict,
        num_samples: int = 10,
    ):
        dump_dir = self.config.trainer.get("debug_dump_dir", "./debug_samples")
        os.makedirs(dump_dir, exist_ok=True)

        filename = os.path.join(dump_dir, f"step_{self.global_steps}.jsonl")

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]

        prompt_length = prompts.shape[1]
        n_samples = min(num_samples, len(batch))

        samples = []
        for i in range(n_samples):
            prompt_ids = prompts[i]
            response_ids = responses[i]

            prompt_text = self.tokenizer.decode(
                prompt_ids[prompt_ids != 0], skip_special_tokens=False
            )
            response_text = self.tokenizer.decode(
                response_ids[attention_mask[i, prompt_length:] == 1],
                skip_special_tokens=False,
            )

            item = batch[i]
            ground_truth = item.non_tensor_batch.get("reward_model", {}).get(
                "ground_truth", ""
            )
            g_level = item.non_tensor_batch.get("g_level", 0.0)
            guidance_steps = item.non_tensor_batch.get("guidance_steps", [])
            adaptive_id = item.non_tensor_batch.get("adaptive_id", "")

            extracted_answer = (
                reward_extra_info.get("extracted_answers", [""] * n_samples)[i]
                if reward_extra_info
                else ""
            )
            is_correct = (
                reward_extra_info.get("is_correct", [False] * n_samples)[i]
                if reward_extra_info
                else False
            )
            failure_reason = (
                reward_extra_info.get("failure_reasons", ["none"] * n_samples)[i]
                if reward_extra_info
                else "none"
            )

            reward_sum = reward_tensor[i].sum().item()

            sample = {
                "step": self.global_steps,
                "sample_idx": i,
                "adaptive_id": str(adaptive_id),
                "g_level": float(g_level) if not isinstance(g_level, float) else g_level,
                "guidance_mode": self.guidance_mode,
                "curriculum_method": self.curriculum_method,
                "guidance_steps_count": len(guidance_steps) if isinstance(guidance_steps, list) else 0,
                "ground_truth": str(ground_truth),
                "prompt_text": prompt_text[:1500] + "..." if len(prompt_text) > 1500 else prompt_text,
                "response_text": response_text[:1500] + "..." if len(response_text) > 1500 else response_text,
                "extracted_answer": extracted_answer,
                "is_correct": bool(is_correct),
                "failure_reason": failure_reason,
                "reward": float(reward_sum),
            }
            samples.append(sample)

        with open(filename, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Dumped {n_samples} debug samples to {filename}")
