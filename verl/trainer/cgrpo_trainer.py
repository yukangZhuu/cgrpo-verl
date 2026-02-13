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
Curriculum-GRPO Trainer.
Extends RayPPOTrainer with curriculum learning support.
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import uuid

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.curriculum import CurriculumConfig, CurriculumManager
from verl.utils.dataset.curriculum_dataset import CurriculumGRPODataset
from verl.workers.reward_manager.cgrpo import CurriculumGRPORewardManager

logger = logging.getLogger(__name__)


class CurriculumGRPOTrainer(RayPPOTrainer):
    """
    Trainer for Curriculum-GRPO.
    
    Extends RayPPOTrainer with:
    1. CurriculumManager for backward chaining
    2. Dynamic prompt construction based on curriculum level
    3. Curriculum-aware reward computation
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
        """
        Initialize Curriculum-GRPO Trainer.
        
        Args:
            config: Training configuration.
            tokenizer: Tokenizer.
            role_worker_mapping: Role to worker mapping.
            resource_pool_manager: Resource pool manager.
            ray_worker_group_cls: Ray worker group class.
            processor: Data processor.
            reward_fn: Reward function.
            val_reward_fn: Validation reward function.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            collate_fn: Collate function.
            train_sampler: Training sampler.
            device_name: Device name.
        """
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
        
        curriculum_config = CurriculumConfig(
            initial_k=self.config.curriculum.get("initial_k", 1),
            max_k=self.config.curriculum.get("max_k", 10),
            ema_alpha=self.config.curriculum.get("ema_alpha", 0.1),
            base_threshold=self.config.curriculum.get("base_threshold", 0.9),
            threshold_decay=self.config.curriculum.get("threshold_decay", 0.95),
            patience=self.config.curriculum.get("patience", 1000),
            min_steps_per_k=self.config.curriculum.get("min_steps_per_k", 100),
            warmup_steps=self.config.curriculum.get("warmup_steps", 0),
        )
        
        self.curriculum_manager = CurriculumManager(config=curriculum_config)
        
        logger.info(f"CurriculumGRPOTrainer initialized with curriculum config: {curriculum_config}")
    
    def _create_curriculum_batch(
        self,
        batch: DataProto,
        current_k: int,
    ) -> DataProto:
        """
        Create curriculum-aware batch with dynamic prompts.
        
        Args:
            batch: Original batch.
            current_k: Current curriculum level.
        
        Returns:
            Modified batch with curriculum prompts.
        """
        batch_size = len(batch)
        
        teacher_prefixes = []
        cut_indices = []
        
        for i in range(batch_size):
            item = batch[i]
            
            steps = item.non_tensor_batch.get("steps", [])
            question_messages = item.non_tensor_batch.get("raw_prompt", [])
            
            if isinstance(question_messages, list) and len(question_messages) > 0:
                if isinstance(question_messages[0], dict):
                    question = question_messages[0].get("content", "")
                else:
                    question = str(question_messages)
            else:
                question = str(question_messages)
            
            if steps:
                num_steps = len(steps)
                cut_index = max(0, num_steps - current_k)
                teacher_prefix_steps = steps[:cut_index]
                teacher_prefix = "\n".join(teacher_prefix_steps) if teacher_prefix_steps else ""
            else:
                cut_index = 0
                teacher_prefix = ""
            
            teacher_prefixes.append(teacher_prefix)
            cut_indices.append(cut_index)
        
        batch.non_tensor_batch["teacher_prefix"] = np.array(teacher_prefixes, dtype=object)
        batch.non_tensor_batch["cut_index"] = np.array(cut_indices, dtype=np.int32)
        
        return batch
    
    def fit(self):
        """
        Main training loop for Curriculum-GRPO.
        
        Extends the base fit() method with:
        1. Curriculum-aware prompt construction
        2. Curriculum level updates
        3. Curriculum metrics logging
        """
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
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
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Curriculum-GRPO Training",
        )
        
        self.global_steps += 1
        
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                
                current_k = self.curriculum_manager.get_current_k()
                batch = self._create_curriculum_batch(batch, current_k)
                
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["current_k"] = current_k
                
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                self.checkpoint_manager.sleep_replicas()
                
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                batch = batch.union(gen_batch_output)
                
                if "response_mask" not in batch.batch.keys():
                    from verl.trainer.ppo.ray_trainer import compute_response_mask
                    batch.batch["response_mask"] = compute_response_mask(batch)
                
                reward_tensor, reward_extra_info = self._compute_curriculum_reward(batch)
                batch.batch["token_level_scores"] = reward_tensor
                batch.batch["token_level_rewards"] = reward_tensor
                
                if self.config.trainer.get("debug_dump_samples", False):
                    self._dump_debug_samples(
                        batch=batch,
                        reward_tensor=reward_tensor,
                        reward_extra_info=reward_extra_info,
                        current_k=current_k,
                        num_samples=self.config.trainer.get("debug_num_samples", 5),
                    )
                
                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
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
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm,
                )
                
                actor_output = self._update_actor(batch)
                self.checkpoint_manager.update_weights()
                
                batch_success_rate = reward_tensor.sum(dim=-1).mean().item() / self.config.actor_rollout_ref.rollout.n
                curriculum_metrics = self.curriculum_manager.update(
                    batch_success_rate=batch_success_rate,
                    batch_size=len(batch),
                )
                metrics.update(curriculum_metrics)
                
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "k": current_k,
                    "sr": f"{batch_success_rate:.3f}",
                })
                
                self.global_steps += 1
                
                if self.global_steps % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    logger.log(data=val_metrics, step=self.global_steps)
                
                if self.curriculum_manager.should_stop():
                    logger.info(
                        f"Early stopping triggered at step {self.global_steps}, "
                        f"k={self.curriculum_manager.get_current_k()}, "
                        f"sr_ema={self.curriculum_manager.sr_ema:.4f}"
                    )
                    self._save_checkpoint()
                    progress_bar.close()
                    return
        
        progress_bar.close()
    
    def _compute_curriculum_reward(
        self,
        batch: DataProto,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute curriculum-aware rewards.
        
        Args:
            batch: Batch data.
        
        Returns:
            Tuple of (reward_tensor, extra_info_dict).
        """
        if self.reward_fn is not None and isinstance(
            self.reward_fn, CurriculumGRPORewardManager
        ):
            return self.reward_fn(batch, return_dict=True)
        else:
            from verl.trainer.ppo.reward import compute_reward
            return compute_reward(batch, self.reward_fn)
    
    def _dump_debug_samples(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_info: dict,
        current_k: int,
        num_samples: int = 5,
    ):
        """
        Dump debug samples to a JSONL file for analysis.
        
        Args:
            batch: Batch data.
            reward_tensor: Computed rewards.
            reward_extra_info: Extra info from reward computation.
            current_k: Current curriculum level.
            num_samples: Number of samples to dump.
        """
        dump_dir = self.config.trainer.get("debug_dump_dir", "./debug_samples")
        os.makedirs(dump_dir, exist_ok=True)
        
        filename = os.path.join(dump_dir, f"step_{self.global_steps}_k{current_k}.jsonl")
        
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        
        prompt_length = prompts.shape[1]
        response_length = responses.shape[1]
        
        n_samples = min(num_samples, len(batch))
        
        samples = []
        for i in range(n_samples):
            prompt_ids = prompts[i]
            response_ids = responses[i]
            
            prompt_text = self.tokenizer.decode(prompt_ids[prompt_ids != 0], skip_special_tokens=False)
            response_text = self.tokenizer.decode(response_ids[attention_mask[i, prompt_length:] == 1], skip_special_tokens=False)
            
            item = batch[i]
            teacher_prefix = item.non_tensor_batch.get("teacher_prefix", "")
            ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            cut_index = item.non_tensor_batch.get("cut_index", 0)
            
            extracted_answer = reward_extra_info.get("extracted_answers", [""] * n_samples)[i] if reward_extra_info else ""
            is_correct = reward_extra_info.get("is_correct", [False] * n_samples)[i] if reward_extra_info else False
            
            reward_sum = reward_tensor[i].sum().item()
            
            sample = {
                "step": self.global_steps,
                "k": current_k,
                "index": i,
                "cut_index": cut_index,
                "teacher_prefix": teacher_prefix[:500] + "..." if len(str(teacher_prefix)) > 500 else teacher_prefix,
                "ground_truth": str(ground_truth),
                "prompt_text": prompt_text,
                "response_text": response_text,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "reward": reward_sum,
            }
            samples.append(sample)
        
        with open(filename, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Dumped {n_samples} debug samples to {filename}")
        
        return samples
    
    def _save_checkpoint(self):
        """Save checkpoint including curriculum state."""
        super()._save_checkpoint()
        
        curriculum_state_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
            "curriculum_state.pt",
        )
        os.makedirs(os.path.dirname(curriculum_state_path), exist_ok=True)
        torch.save(self.curriculum_manager.state_dict(), curriculum_state_path)
        logger.info(f"Curriculum state saved to {curriculum_state_path}")
    
    def _load_checkpoint(self):
        """Load checkpoint including curriculum state."""
        super()._load_checkpoint()
        
        if self.config.trainer.resume_mode == "disable":
            return
        
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        
        if global_step_folder is None:
            return
        
        curriculum_state_path = os.path.join(global_step_folder, "curriculum_state.pt")
        
        if os.path.exists(curriculum_state_path):
            state_dict = torch.load(curriculum_state_path, weights_only=False)
            self.curriculum_manager.load_state_dict(state_dict)
            logger.info(f"Curriculum state loaded from {curriculum_state_path}")
        else:
            logger.info("No curriculum state found, starting fresh")
