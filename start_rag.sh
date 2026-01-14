#!/bin/bash
# 离线 PDF RAG 推理系统启动脚本

# 配置路径（可根据实际情况修改）
LLM_MODEL_PATH="/root/autodl-tmp/local-model"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-./models/embedding_model}"
PDF_PATH="${PDF_PATH:-./rag/statics/PMPP-3rd-Edition.pdf}"
INDEX_DIR="${INDEX_DIR:-./rag_store}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "=========================================="
echo "离线 PDF RAG 推理系统启动脚本"
echo "=========================================="
echo "LLM 模型路径: $LLM_MODEL_PATH"
echo "Embedding 模型路径: $EMBEDDING_MODEL_PATH"
echo "PDF 文件路径: $PDF_PATH"
echo "索引目录: $INDEX_DIR"
echo "服务地址: $HOST:$PORT"
echo "=========================================="

# 检查 LLM 模型
if [ ! -d "$LLM_MODEL_PATH" ]; then
    echo "错误: LLM 模型目录不存在: $LLM_MODEL_PATH"
    exit 1
fi

# 检查 PDF 文件
if [ ! -f "$PDF_PATH" ]; then
    echo "错误: PDF 文件不存在: $PDF_PATH"
    exit 1
fi

# 检查索引是否存在
if [ ! -f "$INDEX_DIR/index.faiss" ] || [ ! -f "$INDEX_DIR/chunks.jsonl" ]; then
    echo ""
    echo "索引文件不存在，开始构建索引..."
    echo "=========================================="
    
    # 检查 embedding 模型
    if [ ! -d "$EMBEDDING_MODEL_PATH" ]; then
        echo "错误: Embedding 模型目录不存在: $EMBEDDING_MODEL_PATH"
        echo ""
        echo "请先准备 Embedding 模型，例如："
        echo "  - 使用 sentence-transformers 模型"
        echo "  - 或从 HuggingFace 下载到本地"
        echo ""
        echo "示例（需要联网下载一次）："
        echo "  from sentence_transformers import SentenceTransformer"
        echo "  model = SentenceTransformer('BAAI/bge-base-zh-v1.5')"
        echo "  model.save('./models/embedding_model')"
        exit 1
    fi
    
    python build_index.py \
        --pdf "$PDF_PATH" \
        --embedding-model "$EMBEDDING_MODEL_PATH" \
        --index-dir "$INDEX_DIR"
    
    if [ $? -ne 0 ]; then
        echo "索引构建失败！"
        exit 1
    fi
    
    echo ""
    echo "索引构建完成！"
    echo "=========================================="
else
    echo "索引文件已存在，跳过构建步骤"
fi

echo ""
echo "启动 FastAPI 服务..."
echo "=========================================="

# 设置环境变量并启动服务
export LLM_MODEL_PATH="$LLM_MODEL_PATH"
export EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH"
export INDEX_DIR="$INDEX_DIR"
export HOST="$HOST"
export PORT="$PORT"

python server.py

