# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
#
# Env fixes if you see errors:
#   - "Numba needs NumPy 2.2 or less": pip install "numpy>=1.0,<=2.2"
#   - "Failed to import Triton kernels" (triton_kernels vs Triton 3.4): in the SAME env where vLLM runs,
#     rename site-packages/triton_kernels to triton_kernels.disabled (Qwen3-0.6B does not need it).
# 若之后需要跑依赖 triton_kernels 的 MoE 模型，再考虑：
# 恢复：mv .../triton_kernels.disabled .../triton_kernels
# 并安装与 Triton 3.4 兼容的 triton_kernels 版本（若有官方更新）

# Run with: conda activate verl  (so Ray workers use verl's numpy and packages).
# If you see "Free memory ... is less than desired GPU memory utilization", lower
#   actor_rollout_ref.rollout.gpu_memory_utilization (e.g. 0.7 or 0.6) so vLLM fits alongside the actor on the same GPU.



set -x

# Ensure NumPy is compatible with Numba (required by vLLM). Numba supports NumPy <= 2.2.
python3 -c "
import numpy as np
v = np.__version__
major, minor = int(v.split('.')[0]), int(v.split('.')[1])
if major > 2 or (major == 2 and minor > 2):
    raise SystemExit(
        f'NumPy {v} is too new for Numba/vLLM. Install a compatible version:\n'
        '  pip install \"numpy>=1.0,<=2.2\"'
    )
"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/data/gsm8k/train.parquet \
    data.val_files=/root/autodl-tmp/data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_qwen3_0.6b_gsm8k' \
    trainer.experiment_name='verl_grpo_qwen3_0.6b_gsm8k_2.08' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@