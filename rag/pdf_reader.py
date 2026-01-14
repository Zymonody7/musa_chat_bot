"""PDF 文本提取模块（支持图片型 PDF 的 OCR 回退）"""
import re
import logging
from typing import List, Dict, Optional, Tuple
import os
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    try:
        import pdfplumber
        HAS_PDFPLUMBER = True
    except ImportError:
        HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)

# 可选 OCR 依赖
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False


class PDFReader:
    """PDF 文本提取器，支持章节识别"""
    
    def __init__(self, pdf_path: str, min_page_chars: int = 50):
        """
        Args:
            pdf_path: PDF 文件路径
            min_page_chars: 跳过文本长度小于此值的页面
        """
        self.pdf_path = pdf_path
        self.min_page_chars = min_page_chars
        self._check_dependencies()
        # 文档 ID（用于区分不同 PDF）
        self.doc_id = os.path.basename(self.pdf_path)
    
    def _check_dependencies(self):
        """检查 PDF 处理库"""
        if not HAS_PYMUPDF and not HAS_PDFPLUMBER:
            raise ImportError(
                "需要安装 PyMuPDF 或 pdfplumber："
                "pip install pymupdf 或 pip install pdfplumber"
            )
    
    def _extract_text_pymupdf(self, page) -> str:
        """使用 PyMuPDF 提取文本"""
        return page.get_text()

    def _extract_ocr_pymupdf(self, page) -> str:
        """基于 PyMuPDF 渲染并使用 Tesseract OCR 提取文本（若可用）"""
        if not HAS_TESSERACT:
            return ""
        try:
            # 渲染为像素图（提高分辨率以提升 OCR 效果）
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception:
            return ""
    
    def _extract_text_pdfplumber(self, page) -> str:
        """使用 pdfplumber 提取文本"""
        return page.extract_text() or ""
    
    def _detect_chapter_section(self, text_lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        从文本前 N 行检测章节/小节标题
        
        Returns:
            (chapter, section) 元组
        """
        patterns = [
            (r"^Chapter\s+\d+", "chapter"),
            (r"^CHAPTER\s+\d+", "chapter"),
            (r"^第[一二三四五六七八九十\d]+章", "chapter"),
            (r"^\d+(\.\d+){0,2}\s+\S+", "section"),
        ]
        
        chapter = None
        section = None
        
        # 检查前 5 行
        for line in text_lines[:5]:
            line = line.strip()
            if not line:
                continue
            
            for pattern, tag_type in patterns:
                if re.match(pattern, line):
                    if tag_type == "chapter":
                        chapter = line
                        section = None  # 新章节时清空小节
                    elif tag_type == "section":
                        section = line
                    break
        
        return chapter, section
    
    def extract_pages(self) -> List[Dict]:
        """
        提取 PDF 所有页面的文本
        
        Returns:
            List[Dict]: 每页包含 text, pdf_page, chapter, section
        """
        logger.info(f"开始读取 PDF: {self.pdf_path}")
        
        if HAS_PYMUPDF:
            doc = fitz.open(self.pdf_path)
            extract_func = self._extract_text_pymupdf
        else:
            doc = pdfplumber.open(self.pdf_path)
            extract_func = self._extract_text_pdfplumber
        
        pages_data = []
        current_chapter = None
        current_section = None
        
        total_pages = len(doc)
        logger.info(f"PDF 总页数: {total_pages}")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = extract_func(page).strip()
            # 若文本过短，尝试 OCR（仅在 PyMuPDF 可用时）
            if len(text) < self.min_page_chars and HAS_PYMUPDF:
                ocr_text = self._extract_ocr_pymupdf(page)
                if len(ocr_text) >= self.min_page_chars:
                    text = ocr_text
            
            # 跳过极短页
            if len(text) < self.min_page_chars:
                logger.debug(f"跳过第 {page_num + 1} 页（文本长度 {len(text)} < {self.min_page_chars}）")
                continue
            
            # 检测章节/小节（从前几行）
            text_lines = text.split('\n')
            chapter, section = self._detect_chapter_section(text_lines)
            
            # 更新当前章节/小节
            if chapter:
                current_chapter = chapter
                current_section = None
            if section:
                current_section = section
            
            pages_data.append({
                "text": text,
                "pdf_page": page_num + 1,  # 从 1 开始
                "chapter": current_chapter,
                "section": current_section,
                "doc_id": self.doc_id,
            })
            
            logger.debug(
                f"第 {page_num + 1} 页: "
                f"文本长度={len(text)}, "
                f"chapter={current_chapter}, "
                f"section={current_section}"
            )
        
        doc.close()
        logger.info(f"成功提取 {len(pages_data)} 页文本")
        
        return pages_data


