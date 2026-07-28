from huggingface_hub import snapshot_download
import os

# 设置镜像加速（国内网络推荐）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载整个模型仓库到指定文件夹
model_path = snapshot_download(
    repo_id="Shadow0482/iris-onnx",
    local_dir="D:/ai-eye/model",  # 指定下载到哪个文件夹
    local_dir_use_symlinks=False,  # 不使用软链接，直接复制文件
    resume_download=True,          # 支持断点续传
)

print(f"✅ 模型下载成功！保存在：{model_path}")