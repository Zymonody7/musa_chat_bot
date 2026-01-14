"""文本切分模块"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Chunker:
    """文本切分器，支持 overlap"""
    
    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 250
    ):
        """
        Args:
            chunk_size: chunk 大小（字符数）
            overlap: 重叠字符数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_page(self, page_data: Dict) -> List[Dict]:
        """
        切分单页文本为多个 chunk
        
        Args:
            page_data: 包含 text, pdf_page, chapter, section 的字典
        
        Returns:
            List[Dict]: 每个 chunk 包含 text, meta
        """
        text = page_data["text"]
        pdf_page = page_data["pdf_page"]
        chapter = page_data.get("chapter")
        section = page_data.get("section")
        doc_id = page_data.get("doc_id")
        
        chunks = []
        
        # 如果文本长度小于 chunk_size，直接作为一个 chunk
        if len(text) <= self.chunk_size:
            chunks.append({
                "text": text,
                "meta": {
                    "pdf_page": pdf_page,
                    "chapter": chapter,
                    "section": section,
                    "doc_id": doc_id,
                    "chunk_idx": 0,
                }
            })
            return chunks
        
        # 按 chunk_size 和 overlap 切分
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果剩余文本不足一个 chunk，取到末尾
            if end >= len(text):
                chunk_text = text[start:]
            else:
                # 尽量在换行符处截断
                chunk_text = text[start:end]
                # 向后查找最近的换行符（最多向后 200 字符）
                next_newline = chunk_text.rfind('\n', max(0, len(chunk_text) - 200))
                if next_newline > len(chunk_text) * 0.7:  # 如果换行符不太远
                    chunk_text = text[start:start + next_newline + 1]
                    end = start + next_newline + 1
            
            chunks.append({
                "text": chunk_text,
                "meta": {
                    "pdf_page": pdf_page,
                    "chapter": chapter,
                    "section": section,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                }
            })
            
            chunk_idx += 1
            # 下一个 chunk 的起始位置（考虑 overlap）
            start = end - self.overlap
            if start >= len(text):
                break
        
        logger.debug(
            f"页面 {pdf_page} 切分为 {len(chunks)} 个 chunks "
            f"(总长度={len(text)}, chunk_size={self.chunk_size})"
        )
        
        return chunks
    
    def chunk_pages(self, pages_data: List[Dict]) -> List[Dict]:
        """
        批量切分多页文本
        
        Args:
            pages_data: 页面数据列表
        
        Returns:
            List[Dict]: 所有 chunks 列表
        """
        all_chunks = []
        
        for page_data in pages_data:
            chunks = self.chunk_page(page_data)
            all_chunks.extend(chunks)
        
        logger.info(f"共生成 {len(all_chunks)} 个 chunks")
        
        return all_chunks


