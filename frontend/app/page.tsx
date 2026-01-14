'use client';

import React, { useState, useEffect } from 'react';
import { User, MemoryItem, ViewState, ChatSession } from '../types';
import { INITIAL_MEMORIES, DEFAULT_MODEL, DEFAULT_USER } from '../constants';
import AuthModal from '../components/AuthModal';
import ChatInterface from '../components/ChatInterface';
import MemoryPanel from '../components/MemoryPanel';
import PDFViewer from '../components/PDFViewer';
import { MessageSquare, Brain, LogOut, Plus, FileText, Trash2 } from 'lucide-react';
import Image from 'next/image';

// PDF URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const PDF_URL = `${API_BASE_URL}/pdf`;

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [currentView, setCurrentView] = useState<ViewState>('chat');
  const [memories, setMemories] = useState<MemoryItem[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('musachat_memories');
      return saved ? JSON.parse(saved) : INITIAL_MEMORIES;
    }
    return INITIAL_MEMORIES;
  });
  const [modelName, setModelName] = useState(DEFAULT_MODEL);
  const [chatSessionId, setChatSessionId] = useState(0);
  const [showPDF, setShowPDF] = useState<boolean>(true);
  const [pdfPage, setPdfPage] = useState<number>(1);
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>(undefined);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showSessionList, setShowSessionList] = useState<boolean>(false);

  // Persist memories
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('musachat_memories', JSON.stringify(memories));
    }
  }, [memories]);

  // Check for session - 默认自动登录
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedUser = localStorage.getItem('musachat_user');
      if (storedUser) {
        setUser(JSON.parse(storedUser));
      } else {
        // 默认自动登录
        setUser(DEFAULT_USER);
        localStorage.setItem('musachat_user', JSON.stringify(DEFAULT_USER));
      }
    }
  }, []);

  const handleLogin = (newUser: User) => {
    setUser(newUser);
    if (typeof window !== 'undefined') {
      localStorage.setItem('musachat_user', JSON.stringify(newUser));
    }
    setIsAuthOpen(false);
  };

  const handleLogout = () => {
    setUser(null);
    if (typeof window !== 'undefined') {
      localStorage.removeItem('musachat_user');
    }
    setIsAuthOpen(true);
  };

  const handleNewChat = () => {
    setChatSessionId(prev => prev + 1);
    setCurrentSessionId(undefined);
    setCurrentView('chat');
  };

  // 加载session列表
  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`);
      if (response.ok) {
        const data = await response.json();
        setSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('加载session列表失败:', error);
    }
  };

  // 删除session
  const handleDeleteSession = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        await loadSessions();
        if (currentSessionId === sessionId) {
          setCurrentSessionId(undefined);
          setChatSessionId(prev => prev + 1);
        }
      }
    } catch (error) {
      console.error('删除session失败:', error);
    }
  };

  // 切换session
  const handleSwitchSession = (sessionId: string) => {
    setCurrentSessionId(sessionId);
    setChatSessionId(prev => prev + 1);
    setShowSessionList(false);
  };

  // 初始化时加载session列表
  useEffect(() => {
    if (user) {
      loadSessions();
    }
  }, [user]);

  // 当sessionId变化时，重新加载session列表
  useEffect(() => {
    if (currentSessionId) {
      loadSessions();
    }
  }, [currentSessionId]);

  const handleSourceClick = (page: number) => {
    // 确保PDF显示，然后跳转到指定页面
    if (!showPDF) {
      setShowPDF(true);
      // 延迟设置页码，确保PDF已经加载
      setTimeout(() => {
        setPdfPage(page);
      }, 300);
    } else {
      // 直接设置页码，PDFViewer 会响应变化
      setPdfPage(page);
    }
  };

  return (
    <div className="h-screen w-full bg-background text-textMain flex overflow-hidden font-sans">
      
      {/* Sidebar */}
      <aside className="w-20 md:w-64 bg-surface border-r border-slate-800 flex flex-col shrink-0 transition-all duration-300">
        <div className="p-6 flex items-center gap-3 border-b border-slate-800">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 overflow-hidden">
            <Image 
              src="/musa.png" 
              alt="MUSA Logo" 
              width={32} 
              height={32} 
              className="object-contain"
            />
          </div>
          <h1 className="font-bold text-xl hidden md:block bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
            MusaChat
          </h1>
        </div>

        <div className="p-4 pb-0">
          <button
             onClick={handleNewChat}
             className="w-full flex items-center justify-center gap-2 p-3 rounded-xl bg-primary hover:bg-primaryHover text-white transition-all shadow-lg shadow-primary/20 group"
           >
             <Plus size={20} className="group-hover:rotate-90 transition-transform duration-300"/>
             <span className="hidden md:block font-bold">New Chat</span>
           </button>
        </div>

        {/* Session List */}
        {currentView === 'chat' && (
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            <div className="text-xs text-slate-500 mb-2 hidden md:block">对话历史</div>
            {sessions.length === 0 ? (
              <div className="text-xs text-slate-500 text-center py-4 hidden md:block">暂无对话记录</div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                    currentSessionId === session.session_id
                      ? 'bg-slate-800 border border-slate-700'
                      : 'hover:bg-slate-800/50 border border-transparent'
                  }`}
                  onClick={() => handleSwitchSession(session.session_id)}
                >
                  <div className="text-sm text-white truncate hidden md:block">{session.title}</div>
                  <div className="text-xs text-slate-500 mt-1 hidden md:block">
                    {new Date(session.updated_at * 1000).toLocaleDateString('zh-CN', {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteSession(session.session_id);
                    }}
                    className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-slate-700 transition-opacity"
                  >
                    <Trash2 size={14} className="text-slate-400 hover:text-red-400" />
                  </button>
                </div>
              ))
            )}
          </div>
        )}

        <nav className="p-4 space-y-2 border-t border-slate-800">
          <button
            onClick={() => setCurrentView('chat')}
            className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all ${
              currentView === 'chat' 
                ? 'bg-slate-800 text-white border border-slate-700' 
                : 'text-textMuted hover:bg-slate-800 hover:text-white'
            }`}
          >
            <MessageSquare size={20} />
            <span className="hidden md:block font-medium">Chat</span>
          </button>

          <button
            onClick={() => setCurrentView('memory')}
            className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all ${
              currentView === 'memory' 
                ? 'bg-primary/10 text-primary border border-primary/20' 
                : 'text-textMuted hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Brain size={20} />
            <span className="hidden md:block font-medium">Shared Memory</span>
          </button>
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3 mb-4 px-2">
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold text-white shrink-0">
              {user?.username?.charAt(0).toUpperCase() || '?'}
            </div>
            <div className="hidden md:block overflow-hidden">
              <p className="text-sm font-medium text-white truncate">{user?.username}</p>
              <p className="text-xs text-textMuted truncate">{user?.email}</p>
            </div>
          </div>
          
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 p-3 rounded-xl text-slate-400 hover:bg-red-500/10 hover:text-red-400 transition-colors"
          >
            <LogOut size={20} />
            <span className="hidden md:block font-medium">Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 relative overflow-hidden flex">
        {currentView === 'chat' ? (
          user ? (
            <div className="flex-1 flex overflow-hidden">
              {/* 聊天界面 */}
              <div className={`flex-1 flex flex-col transition-all ${showPDF ? 'w-1/2 lg:w-1/2' : 'w-full'}`}>
            <ChatInterface 
              key={chatSessionId} 
              user={user} 
              memories={memories} 
              modelName={modelName} 
              onSourceClick={handleSourceClick}
              showPDF={showPDF}
              onTogglePDF={() => setShowPDF(!showPDF)}
              sessionId={currentSessionId}
              onSessionIdChange={(sessionId) => {
                setCurrentSessionId(sessionId);
                loadSessions();
              }}
            />
              </div>
              {/* PDF查看器 */}
              {showPDF && (
                <div className="w-1/2 lg:w-1/2 flex-shrink-0 border-l border-slate-800 hidden md:flex">
                  <PDFViewer
                    pdfUrl={PDF_URL}
                    initialPage={pdfPage}
                    onPageChange={setPdfPage}
                    onClose={() => setShowPDF(false)}
                  />
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-textMuted">
              Please login to access chat.
            </div>
          )
        ) : (
          <MemoryPanel 
            memories={memories} 
            setMemories={setMemories} 
            modelName={modelName}
            setModelName={setModelName}
          />
        )}
      </main>

      {/* Auth Modal */}
      <AuthModal 
        isOpen={isAuthOpen} 
        onLogin={handleLogin} 
        onClose={() => { if(user) setIsAuthOpen(false); }} 
      />
    </div>
  );
}

