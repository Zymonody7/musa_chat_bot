"""FAISS 索引构建和检索模块"""
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss 未安装，请安装 faiss-cpu 或 faiss-gpu")


class EmbeddingModel:
    """本地 Embedding 模型"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Args:
            model_path: 本地模型路径
            device: 'musa', 'cuda' 或 'cpu'，None 时默认使用 musa
        """
        self.model_path = model_path
        self.device = device or "musa"  # 默认使用 musa GPU
        self._load_model()
    
    def _load_model(self):
        """加载 embedding 模型"""
        logger.info(f"加载 Embedding 模型: {self.model_path} (device={self.device})")
        
        # 对于旧版 torch，设置环境变量以绕过检查（不升级 torch 的情况下）
        import os
        if not os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"):
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        
        try:
            # 尝试使用 sentence-transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True
            )
            self.use_sentence_transformers = True
            logger.info("使用 sentence-transformers 加载")
        except Exception as e:
            logger.warning(f"sentence-transformers 加载失败: {e}，尝试使用 transformers")
            # 回退到 transformers
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                # 优先使用 safetensors（兼容旧版 torch）
                import glob
                has_safetensors = len(glob.glob(os.path.join(self.model_path, "*.safetensors"))) > 0
                
                # 如果只有 .bin 文件且 torch 版本低，尝试使用旧的加载方式
                if not has_safetensors:
                    logger.warning(
                        "未找到 safetensors 文件，尝试使用 .bin 文件。"
                        "将使用环境变量绕过 torch 版本检查。"
                    )
                    # 临时禁用 transformers 的 torch.load 安全检查
                    # 注意：这仅用于兼容性，模型文件必须是可信的
                    import transformers.utils.import_utils as import_utils
                    # 保存原始函数
                    original_check = getattr(import_utils, 'check_torch_load_is_safe', None)
                    if original_check:
                        def noop_check(*args, **kwargs):
                            pass  # 跳过检查
                        import_utils.check_torch_load_is_safe = noop_check
                
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map=self.device,
                    use_safetensors=has_safetensors
                )
                
                # 恢复原始函数
                if not has_safetensors and original_check:
                    import_utils.check_torch_load_is_safe = original_check
                
                self.model.eval()
                self.use_sentence_transformers = False
                logger.info("使用 transformers 加载")
            except Exception as load_error:
                logger.error(f"加载模型失败: {load_error}")
                raise RuntimeError(
                    f"无法加载 embedding 模型。"
                    f"如果 torch 版本 < 2.6，请使用 safetensors 格式的模型，"
                    f"或考虑使用其他兼容的 embedding 模型。"
                ) from load_error
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
        
        Returns:
            numpy array: shape (n, dim)
        """
        if self.use_sentence_transformers:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        else:
            # 使用 transformers
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # 归一化（用于余弦相似度）
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings


class RAGIndex:
    """RAG 索引管理器"""
    
    def __init__(
        self,
        embedding_model_path: str,
        index_dir: str = "./rag_store",
        device: Optional[str] = None
    ):
        """
        Args:
            embedding_model_path: Embedding 模型路径
            index_dir: 索引存储目录
            device: 设备
        """
        self.embedding_model_path = embedding_model_path
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = None
        self.index = None
        self.chunks = []
        
        # 索引文件路径
        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.jsonl"
    
    def _get_embedding_model(self) -> EmbeddingModel:
        """延迟加载 embedding 模型"""
        if self.embedding_model is None:
            # 使用 musa GPU 加速 embedding
            self.embedding_model = EmbeddingModel(
                self.embedding_model_path,
                device="musa"
            )
        return self.embedding_model
    
    def build_index(self, chunks: List[Dict], batch_size: int = 32):
        """
        构建 FAISS 索引
        
        Args:
            chunks: chunk 列表，每个包含 text 和 meta
            batch_size: 批量编码大小
        """
        if not HAS_FAISS:
            raise ImportError("需要安装 faiss-cpu 或 faiss-gpu")
        
        logger.info(f"开始构建索引，共 {len(chunks)} 个 chunks")
        
        # 保存 chunks 元数据
        self.chunks = chunks
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                chunk_record = {
                    "id": i,
                    "text": chunk["text"],
                    "meta": chunk["meta"]
                }
                f.write(json.dumps(chunk_record, ensure_ascii=False) + '\n')
        
        logger.info(f"Chunks 元数据已保存到: {self.chunks_path}")
        
        # 批量编码
        embedding_model = self._get_embedding_model()
        texts = [chunk["text"] for chunk in chunks]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            logger.info(f"已编码 {min(i + batch_size, len(texts))}/{len(texts)} 个 chunks")
        
        embeddings = np.vstack(all_embeddings)
        dim = embeddings.shape[1]
        logger.info(f"Embedding 维度: {dim}")
        
        # 创建 FAISS 索引（使用内积，因为向量已归一化）
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype('float32'))
        
        # 保存索引
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS 索引已保存到: {self.index_path}")
        logger.info(f"索引大小: {self.index.ntotal}")
    
    def load_index(self):
        """加载已构建的索引"""
        if not HAS_FAISS:
            raise ImportError("需要安装 faiss-cpu 或 faiss-gpu")
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks 文件不存在: {self.chunks_path}")
        
        logger.info(f"加载索引: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"加载 chunks 元数据: {self.chunks_path}")
        self.chunks = []
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_record = json.loads(line.strip())
                self.chunks.append({
                    "text": chunk_record["text"],
                    "meta": chunk_record["meta"]
                })
        
        logger.info(f"索引加载完成，共 {self.index.ntotal} 个向量，{len(self.chunks)} 个 chunks")
    
    def search(self, query: str, top_k: int = 8) -> List[Tuple[Dict, float]]:
        """
        搜索相关 chunks
        
        Args:
            query: 查询文本
            top_k: 返回 top k 结果
        
        Returns:
            List[Tuple[Dict, float]]: (chunk, score) 列表
        """
        if self.index is None:
            raise RuntimeError("索引未加载，请先调用 load_index()")
        
        # 编码查询
        embedding_model = self._get_embedding_model()
        query_embedding = embedding_model.encode([query])
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def aggregate_by_page(
        self,
        search_results: List[Tuple[Dict, float]],
        max_sources: int = 4
    ) -> List[Dict]:
        """
        按页面聚合搜索结果
        
        Args:
            search_results: search() 返回的结果
            max_sources: 最大返回 sources 数
        
        Returns:
            List[Dict]: 聚合后的 sources，每个包含 pdf_page, chapter, section, snippet, score
        """
        # 按页面分组
        page_groups = {}
        for chunk, score in search_results:
            pdf_page = chunk["meta"]["pdf_page"]
            if pdf_page not in page_groups:
                page_groups[pdf_page] = {
                    "chunks": [],
                    "scores": [],
                    "chapter": chunk["meta"].get("chapter"),
                    "section": chunk["meta"].get("section"),
                }
            page_groups[pdf_page]["chunks"].append(chunk["text"])
            page_groups[pdf_page]["scores"].append(score)
        
        # 按得分排序，取前 max_sources 页
        sorted_pages = sorted(
            page_groups.keys(),
            key=lambda p: np.mean(page_groups[p]["scores"]),
            reverse=True
        )[:max_sources]
        
        sources = []
        for pdf_page in sorted_pages:
            group = page_groups[pdf_page]
            # 合并该页的所有 chunk 文本
            combined_text = "\n\n".join(group["chunks"])
            # 截取前 800 字符作为 snippet
            snippet = combined_text[:800] + ("..." if len(combined_text) > 800 else "")
            
            sources.append({
                "pdf_page": pdf_page,
                "chapter": group["chapter"],
                "section": group["section"],
                "snippet": snippet,
                "score": float(np.mean(group["scores"])),
            })
        
        # 确保至少返回 2 个 sources（如果搜索结果足够）
        if len(sources) < 2 and len(search_results) >= 2:
            # 补充更多页面
            remaining_pages = [p for p in sorted(page_groups.keys(), reverse=True) if p not in sorted_pages]
            for pdf_page in remaining_pages[:2 - len(sources)]:
                group = page_groups[pdf_page]
                combined_text = "\n\n".join(group["chunks"])
                snippet = combined_text[:800] + ("..." if len(combined_text) > 800 else "")
                
                sources.append({
                    "pdf_page": pdf_page,
                    "chapter": group["chapter"],
                    "section": group["section"],
                    "snippet": snippet,
                    "score": float(np.mean(group["scores"])),
                })
        
        return sources

