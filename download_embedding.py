"""下载 Embedding 模型到本地（需要联网一次）
注意：由于 torch 版本限制，我们优先下载 safetensors 格式的模型
"""
import os
import shutil

# 配置 HuggingFace 镜像（必须在导入之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ.get("HUGGINGFACE_HUB_CACHE", "./.hf_cache")

# 配置
MODEL_NAME = "BAAI/bge-base-zh-v1.5"  # 中文 embedding 模型
OUTPUT_DIR = "./models/embedding_model"
TEMP_DIR = "./.temp_embedding_model"

print(f"开始下载 Embedding 模型: {MODEL_NAME}")
print(f"使用镜像: {os.environ['HF_ENDPOINT']}")
print(f"输出目录: {OUTPUT_DIR}")
print("注意：将优先使用 safetensors 格式以兼容旧版 torch")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查是否已经下载过
if os.path.exists(TEMP_DIR) and os.path.exists(os.path.join(TEMP_DIR, "config.json")):
    print(f"发现已下载的文件在 {TEMP_DIR}，直接使用...")
else:
    # 使用 huggingface_hub 下载
    from huggingface_hub import snapshot_download
    
    print("使用 huggingface_hub 下载（优先 safetensors 格式）...")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=TEMP_DIR,
            resume_download=True,
            ignore_patterns=["*.bin"] if not os.path.exists(TEMP_DIR) else None  # 忽略 .bin 文件
        )
        print("✓ 模型文件下载完成")
    except Exception as e:
        print(f"下载失败: {e}")
        raise

# 检查是否有 .bin 文件，如果有则尝试转换为 safetensors 或删除
bin_file = os.path.join(TEMP_DIR, "pytorch_model.bin")
if os.path.exists(bin_file):
    print(f"警告：发现 {bin_file}，但由于 torch 版本限制无法直接使用")
    print("尝试查找 safetensors 文件...")
    
    # 检查是否有 safetensors 文件
    import glob
    safetensors_files = glob.glob(os.path.join(TEMP_DIR, "*.safetensors"))
    if not safetensors_files:
        print("未找到 safetensors 文件，尝试下载 safetensors 版本...")
        # 删除 .bin 文件，重新下载（只下载 safetensors）
        os.remove(bin_file)
        from huggingface_hub import hf_hub_download
        try:
            # 尝试下载 safetensors 格式
            safetensors_path = hf_hub_download(
                repo_id=MODEL_NAME,
                filename="model.safetensors",
                local_dir=TEMP_DIR,
                resume_download=True
            )
            print(f"✓ 已下载 safetensors 文件")
        except Exception as e:
            print(f"无法下载 safetensors 文件: {e}")
            print("将保留 .bin 文件，但加载时可能失败（需要 torch >= 2.6）")
    else:
        print(f"✓ 找到 safetensors 文件: {safetensors_files}")

# 复制文件到目标目录（排除 .bin 文件）
print(f"复制文件到 {OUTPUT_DIR}...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 只复制非 .bin 文件
for item in os.listdir(TEMP_DIR):
    src = os.path.join(TEMP_DIR, item)
    dst = os.path.join(OUTPUT_DIR, item)
    if item.endswith('.bin'):
        print(f"跳过 .bin 文件: {item}（使用 safetensors 替代）")
        continue
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

# 清理临时目录
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"✓ 模型已保存到: {OUTPUT_DIR}")
print("现在可以离线使用了！")

