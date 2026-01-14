'use client';

import React, { useState, useEffect, useRef } from 'react';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCw, ExternalLink } from 'lucide-react';

interface PDFViewerProps {
  pdfUrl: string;
  initialPage?: number;
  onClose?: () => void;
  onPageChange?: (page: number) => void;
}

const PDFViewer: React.FC<PDFViewerProps> = ({ 
  pdfUrl, 
  initialPage = 1, 
  onClose,
  onPageChange 
}) => {
  const [currentPage, setCurrentPage] = useState<number>(initialPage);
  const [scale, setScale] = useState<number>(100);
  const [iframeSrc, setIframeSrc] = useState<string>(`${pdfUrl}#page=${initialPage}&toolbar=0&navpanes=0&scrollbar=1`);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const lastInitialPageRef = useRef<number>(initialPage);

  // 跳转到指定页面的辅助函数
  const jumpToPage = (page: number) => {
    const url = `${pdfUrl}#page=${page}&toolbar=0&navpanes=0&scrollbar=1`;
    setIframeSrc(url);
  };

  // 监听 initialPage 变化，跳转到指定页面
  useEffect(() => {
    if (initialPage && initialPage !== lastInitialPageRef.current) {
      lastInitialPageRef.current = initialPage;
      setCurrentPage(initialPage);
      jumpToPage(initialPage);
    }
  }, [initialPage, pdfUrl]);

  // 初始化时设置第一页
  useEffect(() => {
    if (pdfUrl) {
      jumpToPage(initialPage);
    }
  }, [pdfUrl]);

  const goToPrevPage = () => {
    if (currentPage > 1) {
      const newPage = currentPage - 1;
      setCurrentPage(newPage);
      lastInitialPageRef.current = newPage;
      onPageChange?.(newPage);
      jumpToPage(newPage);
    }
  };

  const goToNextPage = () => {
    const newPage = currentPage + 1;
    setCurrentPage(newPage);
    lastInitialPageRef.current = newPage;
    onPageChange?.(newPage);
    jumpToPage(newPage);
  };

  const handlePageInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 1) {
      setCurrentPage(value);
      lastInitialPageRef.current = value;
      onPageChange?.(value);
      jumpToPage(value);
    }
  };

  const zoomIn = () => {
    setScale(prev => Math.min(prev + 10, 200));
  };

  const zoomOut = () => {
    setScale(prev => Math.max(prev - 10, 50));
  };

  const resetZoom = () => {
    setScale(100);
  };

  const openInNewTab = () => {
    window.open(`${pdfUrl}#page=${currentPage}`, '_blank');
  };

  return (
    <div className="h-full w-full flex flex-col bg-background border-l border-slate-800">
      {/* 工具栏 - 统一风格 */}
      <div className="flex items-center justify-between p-3 bg-surface border-b border-slate-800">
        <div className="flex items-center gap-2">
          <button
            onClick={goToPrevPage}
            disabled={currentPage <= 1}
            className="p-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            title="上一页"
          >
            <ChevronLeft size={18} />
          </button>
          
          <div className="flex items-center gap-2 px-3">
            <input
              type="number"
              value={currentPage}
              onChange={handlePageInput}
              min={1}
              className="w-16 px-2 py-1.5 bg-background text-white text-sm rounded-lg border border-slate-700 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              placeholder="页码"
            />
            <span className="text-slate-400 text-xs">页</span>
          </div>
          
          <button
            onClick={goToNextPage}
            className="p-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-all"
            title="下一页"
          >
            <ChevronRight size={18} />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            className="p-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-all"
            title="缩小"
          >
            <ZoomOut size={18} />
          </button>
          <button
            onClick={resetZoom}
            className="px-3 py-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-all text-xs font-medium min-w-[3.5rem]"
            title="重置缩放"
          >
            {scale}%
          </button>
          <button
            onClick={zoomIn}
            className="p-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-all"
            title="放大"
          >
            <ZoomIn size={18} />
          </button>
          <button
            onClick={openInNewTab}
            className="p-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-all ml-1"
            title="在新标签页中打开"
          >
            <ExternalLink size={18} />
          </button>
        </div>
      </div>

      {/* PDF 内容区域 */}
      <div className="flex-1 overflow-auto bg-slate-900 p-4">
        <div className="w-full h-full flex items-center justify-center">
          <div 
            className="w-full h-full flex items-center justify-center"
            style={{
              transform: `scale(${scale / 100})`,
              transformOrigin: 'top center',
              transition: 'transform 0.2s ease',
            }}
          >
            <iframe
              ref={iframeRef}
              src={iframeSrc}
              key={iframeSrc}
              className="w-full border-0 rounded-lg shadow-2xl bg-white"
              style={{
                minHeight: '800px',
                height: '100%',
              }}
              title="PDF Viewer"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default PDFViewer;
