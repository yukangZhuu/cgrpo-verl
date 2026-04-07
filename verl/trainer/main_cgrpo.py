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
Main entry point for CGRPO training.
Supports both standard GRPO and static-mixture curriculum GRPO.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.cgrpo_trainer import CurriculumGRPOTrainer
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.dataset.curriculum_dataset import CurriculumGRPODataset
from verl.workers.reward_manager.cgrpo import CurriculumGRPORewardManager


@hydra.main(config_path="config", config_name="cgrpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_cgrpo(config)


def run_cgrpo(config):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        print(f"Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner = CGRPOTaskRunner.remote()
    ray.get(task_runner.run.remote(config))


@ray.remote(num_cpus=1)
class CGRPOTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from omegaconf import open_dict

        print(
            f"CGRPOTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        with open_dict(config):
            config.trainer.use_legacy_worker_impl = "disable"

        self._setup_workers(config)

        from verl.trainer.ppo.utils import need_reference_policy

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=False,
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        reward_fn = CurriculumGRPORewardManager(
            tokenizer=tokenizer,
            num_examine=config.reward_model.get("num_examine", 0),
            format_score=config.reward_model.get("format_score", 0.0),
            correct_score=config.reward_model.get("correct_score", 1.0),
        )

        train_dataset = CurriculumGRPODataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=config.data.get("train_max_samples", -1),
        )

        val_dataset = None
        if config.data.get("val_files", None):
            val_dataset = CurriculumGRPODataset(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                config=config.data,
                max_samples=config.data.get("val_max_samples", -1),
            )

        from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
        from verl.trainer.ppo.ray_trainer import Role

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

        from verl.workers.engine_workers import ActorRolloutRefWorker

        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(
            ActorRolloutRefWorker
        )
        self.mapping[Role.ActorRolloutRef] = "global_pool"

        trainer = CurriculumGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.init_workers()
        trainer.fit()

    def _setup_workers(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping = {}
        self.mapping = {}


if __name__ == "__main__":
    main()
