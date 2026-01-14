"""构建 RAG 索引脚本"""
import argparse
import os
import logging

from rag.pdf_reader import PDFReader
from rag.chunker import Chunker
from rag.index import RAGIndex

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="构建 PDF RAG 索引（支持目录内多 PDF）")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="./rag/statics",
        help="包含多个 PDF 的目录（默认 rag/statics）"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="./models/embedding_model",
        help="Embedding 模型路径"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./rag_store",
        help="索引输出目录"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk 大小（字符数）"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=250,
        help="Chunk 重叠大小（字符数）"
    )
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=50,
        help="最小页面文本长度（小于此值跳过）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批量编码大小"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.pdf_dir):
        raise FileNotFoundError(f"PDF 目录不存在: {args.pdf_dir}")
    
    if not os.path.exists(args.embedding_model):
        raise FileNotFoundError(f"Embedding 模型不存在: {args.embedding_model}")
    
    logger.info("=" * 60)
    logger.info("开始构建 RAG 索引")
    logger.info("=" * 60)
    logger.info(f"PDF 目录: {args.pdf_dir}")
    logger.info(f"Embedding 模型: {args.embedding_model}")
    logger.info(f"索引输出目录: {args.index_dir}")
    logger.info(f"Chunk 大小: {args.chunk_size}, 重叠: {args.overlap}")
    
    # 1. 提取目录内所有 PDF 文本
    logger.info("\n[1/3] 提取 PDF 文本（遍历目录）...")
    pages_data = []
    pdf_files = [
        os.path.join(args.pdf_dir, f)
        for f in os.listdir(args.pdf_dir)
        if f.lower().endswith('.pdf')
    ]
    if not pdf_files:
        raise FileNotFoundError("目录中未找到 PDF 文件")
    for pdf_path in pdf_files:
        logger.info(f"读取: {pdf_path}")
        reader = PDFReader(pdf_path, min_page_chars=args.min_page_chars)
        pages = reader.extract_pages()
        pages_data.extend(pages)
    logger.info(f"共提取 {len(pages_data)} 页文本（来自 {len(pdf_files)} 个 PDF）")
    
    # 2. 切分 chunks
    logger.info("\n[2/3] 切分文本为 chunks...")
    chunker = Chunker(chunk_size=args.chunk_size, overlap=args.overlap)
    chunks = chunker.chunk_pages(pages_data)
    logger.info(f"生成了 {len(chunks)} 个 chunks")
    
    # 3. 构建索引
    logger.info("\n[3/3] 构建 FAISS 索引...")
    rag_index = RAGIndex(
        embedding_model_path=args.embedding_model,
        index_dir=args.index_dir
    )
    rag_index.build_index(chunks, batch_size=args.batch_size)
    
    logger.info("\n" + "=" * 60)
    logger.info("索引构建完成！")
    logger.info("=" * 60)
    logger.info(f"索引文件: {args.index_dir}/index.faiss")
    logger.info(f"Chunks 文件: {args.index_dir}/chunks.jsonl")


if __name__ == "__main__":
    main()


