'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, MemoryItem, User } from '../types';
import { RagSource, buildQuestionWithMemory, streamAsk } from '../services/ragService';
import { Send, Bot, User as UserIcon, RefreshCw, Brain, ChevronDown, ChevronUp, FileText, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatInterfaceProps {
  user: User;
  memories: MemoryItem[];
  modelName: string;
  onSourceClick?: (page: number) => void;
  showPDF?: boolean;
  onTogglePDF?: () => void;
  sessionId?: string;
  onSessionIdChange?: (sessionId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ user, memories, modelName, onSourceClick, showPDF, onTogglePDF, sessionId, onSessionIdChange }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedThinks, setExpandedThinks] = useState<Set<string>>(new Set());
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>(sessionId);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const hasInitialized = useRef(false);
  const userHasScrolled = useRef(false);
  const isAutoScrolling = useRef(false);

  /**
   * 解析文本，提取think内容和正常回答
   */
  const parseThinkContent = (text: string): { thinkContent: string; answer: string } => {
    // 先移除 <|im_end|> 标记
    let cleanedText = text.replace(/<\|im_end\|>/gi, '').trim();
    
    // 匹配 <think>...</think> 或 <think>...</think> 格式
    const thinkRegex = /<(?:think|redacted_reasoning)>([\s\S]*?)<\/(?:think|redacted_reasoning)>/gi;
    let thinkContent = '';
    let answer = cleanedText;
    
    const matches = [...cleanedText.matchAll(thinkRegex)];
    if (matches.length > 0) {
      // 提取所有think内容
      thinkContent = matches.map(m => m[1].trim()).join('\n\n');
      // 移除think标记，保留正常回答
      answer = cleanedText.replace(thinkRegex, '').trim();
    }
    
    return { thinkContent, answer };
  };

  /**
   * 获取文本的前N行
   */
  const getFirstNLines = (text: string, n: number): string => {
    const lines = text.split('\n');
    return lines.slice(0, n).join('\n');
  };

  // 当sessionId变化时，加载对应的消息
  useEffect(() => {
    if (sessionId && sessionId !== currentSessionId) {
      setCurrentSessionId(sessionId);
      loadSessionMessages(sessionId);
    } else if (!sessionId && currentSessionId) {
      // 新会话，清空消息
      setCurrentSessionId(undefined);
      setMessages([
        {
          id: 'welcome',
          role: 'model',
          text: `Hello ${user.username}. 我是MUSA Chat，随时可以开始对话。`,
          timestamp: Date.now(),
        },
      ]);
      hasInitialized.current = true;
    }
  }, [sessionId]);

  // 加载session消息
  const loadSessionMessages = async (sessionId: string) => {
    try {
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        const loadedMessages: ChatMessage[] = data.messages.map((msg: any, idx: number) => ({
          id: `${sessionId}_${idx}`,
          role: msg.role === 'user' ? 'user' : 'model',
          text: msg.content,
          timestamp: msg.timestamp * 1000, // 后端返回的是秒，前端需要毫秒
          sources: msg.sources || [],
        }));
        setMessages(loadedMessages);
        hasInitialized.current = true;
      }
    } catch (error) {
      console.error('加载session消息失败:', error);
    }
  };

  // 初次欢迎信息
  useEffect(() => {
    if (!hasInitialized.current && !currentSessionId) {
      setMessages([
        {
          id: 'welcome',
          role: 'model',
          text: `Hello ${user.username}. 我是MUSA Chat，随时可以开始对话。`,
          timestamp: Date.now(),
        },
      ]);
      hasInitialized.current = true;
    }
  }, [user.username, currentSessionId]);

  const scrollToBottom = (force = false) => {
    if (!force && userHasScrolled.current) {
      return; // 用户手动滚动过，不自动滚动
    }
    isAutoScrolling.current = true;
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => {
      isAutoScrolling.current = false;
    }, 500);
  };

  // 监听滚动事件，检测用户是否手动滚动
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (isAutoScrolling.current) return;
      
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100; // 距离底部100px内
      
      if (isNearBottom) {
        userHasScrolled.current = false; // 接近底部，允许自动滚动
      } else {
        userHasScrolled.current = true; // 不在底部，用户手动滚动了
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // 只在消息更新时自动滚动（如果用户没有手动滚动）
  useEffect(() => {
    if (messages.length > 0) {
    scrollToBottom();
    }
  }, [messages.length]); // 只在消息数量变化时滚动，而不是每次消息内容变化

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: inputValue,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    const botMsgId = (Date.now() + 1).toString();
    const botMsg: ChatMessage = {
      id: botMsgId,
      role: 'model',
      text: '',
      timestamp: Date.now(),
      sources: [],
    };
    setMessages(prev => [...prev, botMsg]);

    try {
      let fullText = '';
      let currentSources: RagSource[] = [];
      const enrichedQuestion = buildQuestionWithMemory(userMsg.text, memories);

      for await (const event of streamAsk(enrichedQuestion, currentSessionId)) {
        if (event.type === 'retrieve') {
          // event.sources 已经在 ragService 中转换为 RagSource[] 格式
          currentSources = (event.sources as RagSource[]) || [];
          setMessages(prev => prev.map(msg => 
            msg.id === botMsgId ? { ...msg, sources: currentSources } : msg
          ));
          // 更新session_id
          if (event.session_id && event.session_id !== currentSessionId) {
            setCurrentSessionId(event.session_id);
            onSessionIdChange?.(event.session_id);
          }
        }
        if (event.type === 'token') {
          fullText = event.text || '';
          const { thinkContent, answer } = parseThinkContent(fullText);
          setMessages(prev => prev.map(msg => 
            msg.id === botMsgId ? { ...msg, text: answer, thinkContent, isThinking: !answer && !!thinkContent, sources: currentSources } : msg
          ));
          // 流式生成时，如果用户没有手动滚动，自动滚动到底部
          setTimeout(() => scrollToBottom(), 50);
        }
        if (event.type === 'done') {
          fullText = event.full_text || '';
          const { thinkContent, answer } = parseThinkContent(fullText);
          setMessages(prev => prev.map(msg => 
            msg.id === botMsgId ? { ...msg, text: answer, thinkContent, isThinking: false, sources: currentSources } : msg
          ));
          // 更新session_id
          if (event.session_id && event.session_id !== currentSessionId) {
            setCurrentSessionId(event.session_id);
            onSessionIdChange?.(event.session_id);
          }
        }
        if (event.type === 'error') {
          throw new Error(event.error);
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(msg => {
        if (msg.id === botMsgId) {
          return {
            ...msg,
            text: "后端服务暂不可用，请稍后重试。",
            isError: true
          };
        }
        return msg;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const resetChat = () => {
    setMessages([{
      id: Date.now().toString(),
      role: 'model',
      text: '对话已重置，记忆已同步，继续提问吧。',
      timestamp: Date.now(),
      sources: []
    }]);
    setExpandedThinks(new Set());
  };

  /**
   * 思考内容显示组件
   */
  const ThinkContentDisplay: React.FC<{
    thinkContent: string;
    isThinking: boolean;
    messageId: string;
    isExpanded: boolean;
    onToggle: () => void;
  }> = ({ thinkContent, isThinking, messageId, isExpanded, onToggle }) => {
    const lines = thinkContent.split('\n');
    const firstTwoLines = lines.slice(0, 2).join('\n');
    const hasMore = lines.length > 2 || thinkContent.length > firstTwoLines.length;

    return (
      <div className="mb-3 p-3 bg-slate-800/60 border border-slate-600/50 rounded-lg">
        <div className="text-xs text-slate-400 mb-2 font-semibold flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${isThinking ? 'bg-yellow-400 animate-pulse' : 'bg-slate-400'}`}></span>
            {isThinking ? '思考中...' : '思考过程'}
          </div>
          {!isThinking && hasMore && (
            <button
              onClick={onToggle}
              className="text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1"
            >
              {isExpanded ? (
                <>
                  <ChevronUp size={14} />
                  <span>收起</span>
                </>
              ) : (
                <>
                  <ChevronDown size={14} />
                  <span>展开</span>
                </>
              )}
            </button>
          )}
        </div>
        <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap font-mono">
          {isExpanded || isThinking ? thinkContent : firstTwoLines}
          {!isExpanded && !isThinking && hasMore && (
            <span className="text-slate-500 italic">...</span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-background/50 relative">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 p-4 bg-background/80 backdrop-blur-md border-b border-slate-800 z-10 flex justify-between items-center">
        <div>
           <h2 className="text-lg font-semibold text-white">Live Session</h2>
           <p className="text-xs text-primary flex items-center gap-1">
             <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
             Connected: {modelName}
           </p>
        </div>
        <div className="flex items-center gap-2">
          {onTogglePDF && (
            <button 
              onClick={onTogglePDF}
              className={`p-2 rounded-lg transition-colors ${
                showPDF 
                  ? 'text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20' 
                  : 'text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700'
              }`}
              title={showPDF ? "隐藏PDF" : "显示PDF"}
            >
              {showPDF ? <X size={18} /> : <FileText size={18} />}
            </button>
          )}
        <button 
            onClick={resetChat}
            className="p-2 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
            title="Reset Session"
        >
            <RefreshCw size={18} />
        </button>
        </div>
      </div>

      {/* Messages Area */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 pt-20 pb-24 space-y-6"
      >
        {messages.map((msg) => {
          const isUser = msg.role === 'user';
          return (
            <div key={msg.id} className="space-y-2">
              {/* Sources - 显示在回答上方 */}
              {!isUser && msg.sources && msg.sources.length > 0 && (
                <div className="bg-surface border border-slate-700 rounded-xl p-3 text-xs text-slate-300 max-w-[80%] ml-12">
                  <p className="text-slate-400 mb-2 text-[10px]">检索到 {msg.sources.length} 个来源：</p>
            <div className="grid gap-2 md:grid-cols-2">
                    {msg.sources.map((src, idx) => (
                      <div 
                        key={idx} 
                        className={`bg-background/40 border border-slate-700 rounded-lg p-2 transition-all ${
                          src.metadata?.page !== undefined && onSourceClick
                            ? 'cursor-pointer hover:border-primary hover:bg-background/60'
                            : ''
                        }`}
                        onClick={() => {
                          if (src.metadata?.page !== undefined && onSourceClick) {
                            onSourceClick(src.metadata.page);
                          }
                        }}
                      >
                  <div className="text-[11px] text-primary font-mono mb-1">
                    {src.metadata?.chapter ? `${src.metadata.chapter}` : ''}
                    {src.metadata?.section ? ` • ${src.metadata.section}` : ''}
                          {src.metadata?.page !== undefined ? (
                            <span className={onSourceClick ? 'underline hover:text-primaryHover' : ''}>
                              {' • Page '}{src.metadata.page}
                            </span>
                          ) : ''}
                    {!src.metadata?.chapter && !src.metadata?.page && '未知来源'}
                  </div>
                  {src.page_content && (
                    <p className="text-[11px] text-slate-300 leading-relaxed">
                            {src.page_content.slice(0, 150)}
                            {src.page_content.length > 150 ? '...' : ''}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
              
            <div 
              className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in-up`}
            >
              <div className={`max-w-[80%] flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                  isUser ? 'bg-primary' : 'bg-orange-700'
                }`}>
                  {isUser ? <UserIcon size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
                </div>

                {/* Bubble */}
                  <div className={`p-4 rounded-2xl max-w-full ${
                  isUser 
                    ? 'bg-surface border border-slate-700 text-white rounded-tr-none' 
                    : 'bg-primary/20 border border-primary/30 text-slate-100 rounded-tl-none'
                } ${msg.isError ? 'border-red-500/50 bg-red-900/10' : ''}`}>
                  {/* Think内容显示 */}
                  {msg.thinkContent && (
                    <ThinkContentDisplay
                      thinkContent={msg.thinkContent}
                      isThinking={msg.isThinking || false}
                      messageId={msg.id}
                      isExpanded={expandedThinks.has(msg.id)}
                      onToggle={() => {
                        setExpandedThinks(prev => {
                          const next = new Set(prev);
                          if (next.has(msg.id)) {
                            next.delete(msg.id);
                          } else {
                            next.add(msg.id);
                          }
                          return next;
                        });
                      }}
                    />
                  )}
                  {/* 正常回答内容 - 使用 Markdown */}
                  <div className="prose prose-invert prose-sm max-w-none leading-relaxed text-sm min-h-[1.5em] break-words overflow-wrap-anywhere">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        code: ({ children, className, ...props }: any) => {
                          const isInline = !className?.includes('language-');
                          if (isInline) {
                            return (
                              <code className="bg-slate-800/50 px-1.5 py-0.5 rounded text-xs font-mono text-orange-300 break-words" {...props}>
                                {children}
                              </code>
                            );
                          }
                          return (
                            <code className="block bg-slate-900/50 p-3 rounded-lg text-xs font-mono text-slate-300 overflow-x-auto max-w-full" {...props}>
                              {children}
                            </code>
                          );
                        },
                        pre: ({ children }) => (
                          <pre className="mb-2 overflow-x-auto max-w-full">{children}</pre>
                        ),
                        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                        li: ({ children }) => <li className="ml-2">{children}</li>,
                        blockquote: ({ children }) => (
                          <blockquote className="border-l-4 border-slate-600 pl-4 italic text-slate-400 mb-2">
                            {children}
                          </blockquote>
                        ),
                        h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-base font-bold mb-2">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-sm font-bold mb-1">{children}</h3>,
                        strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                        em: ({ children }) => <em className="italic">{children}</em>,
                        a: ({ children, href }) => (
                          <a href={href} className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
                            {children}
                          </a>
                        ),
                      }}
                    >
                      {msg.text}
                    </ReactMarkdown>
                    {msg.role === 'model' && isLoading && msg.id === messages[messages.length - 1].id && (
                        <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-primary/50 animate-pulse" />
                    )}
                  </div>
                  <div className="mt-1 text-[10px] opacity-40 text-right">
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-background border-t border-slate-800">
        <div className="max-w-4xl mx-auto relative flex items-end gap-2 bg-surface p-2 rounded-xl border border-slate-700 focus-within:border-primary transition-colors">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message to Musa..."
            className="w-full bg-transparent text-white placeholder-slate-500 text-sm resize-none focus:outline-none max-h-32 min-h-[44px] py-3 px-2"
            rows={1}
            style={{ height: 'auto', minHeight: '44px' }}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="p-3 bg-primary hover:bg-primaryHover disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg transition-all"
          >
            {isLoading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Send size={20} />}
          </button>
        </div>
        <p className="text-center text-[10px] text-slate-600 mt-2">
            AI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;