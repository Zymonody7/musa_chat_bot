"""使用 ModelScope 下载 Embedding 模型到本地（联网一次）
说明：统一改为使用 ModelScope 的 snapshot_download，不再进行 safetensors 强制处理。
若 ModelScope 不可用，则回退到 HuggingFace 下载，但同样不做 safetensors 特殊逻辑。
"""
import os
import shutil

# 基本配置（可通过环境变量覆盖）
OUTPUT_DIR = os.getenv("EMBEDDING_OUTPUT_DIR", "./models/embedding_model")
TEMP_DIR = os.getenv("EMBEDDING_TEMP_DIR", "./.temp_embedding_model")
MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-zh-v1.5")

print(f"开始下载 Embedding 模型: {MODEL_ID}")
print(f"输出目录: {OUTPUT_DIR}")

# 准备临时与输出目录
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

downloaded_dir = None

# 优先使用 ModelScope
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    print("使用 ModelScope 下载模型...")
    downloaded_dir = ms_snapshot_download(model_id=MODEL_ID, cache_dir=TEMP_DIR)
    print("✓ ModelScope 下载完成")
except Exception as ms_err:
    print(f"ModelScope 下载失败或不可用: {ms_err}")
    print("回退到 HuggingFace 下载（不做 safetensors 特殊处理）...")
    try:
        # 使用 HuggingFace（可选镜像）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download as hf_snapshot_download
        downloaded_dir = hf_snapshot_download(
            repo_id=MODEL_ID,
            local_dir=TEMP_DIR,
            resume_download=True,
        )
        print("✓ HuggingFace 下载完成")
    except Exception as hf_err:
        print(f"下载失败: {hf_err}")
        raise

# 将下载内容复制到 OUTPUT_DIR（覆盖）
print(f"复制文件到 {OUTPUT_DIR}...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(downloaded_dir):
    rel = os.path.relpath(root, downloaded_dir)
    dest_root = OUTPUT_DIR if rel == "." else os.path.join(OUTPUT_DIR, rel)
    os.makedirs(dest_root, exist_ok=True)
    for d in dirs:
        os.makedirs(os.path.join(dest_root, d), exist_ok=True)
    for f in files:
        src_f = os.path.join(root, f)
        dst_f = os.path.join(dest_root, f)
        shutil.copy2(src_f, dst_f)

# 清理临时目录
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"✓ 模型已保存到: {OUTPUT_DIR}")