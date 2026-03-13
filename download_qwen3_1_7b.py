import os
from huggingface_hub import snapshot_download

# 模型名称
model_name = "Qwen/Qwen3-1.7B"
# 本地保存路径
local_dir = "/root/autodl-tmp/models/Qwen3-1.7B"

print(f"Starting download of {model_name} to {local_dir}...")

# 确保目录存在
os.makedirs(local_dir, exist_ok=True)

# 下载模型
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 不使用软链接，直接下载文件
    resume_download=True,          # 支持断点续传
    max_workers=8                  # 使用多线程加速下载
)

print(f"Successfully downloaded {model_name} to {local_dir}")
