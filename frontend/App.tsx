import React, { useState, useEffect } from 'react';
import { User, MemoryItem, ViewState } from './types';
import { INITIAL_MEMORIES, DEFAULT_MODEL } from './constants';
import AuthModal from './components/AuthModal';
import ChatInterface from './components/ChatInterface';
import MemoryPanel from './components/MemoryPanel';
import { MessageSquare, Brain, LogOut, Github, Globe, Plus } from 'lucide-react';

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [currentView, setCurrentView] = useState<ViewState>('chat');
  const [memories, setMemories] = useState<MemoryItem[]>(() => {
    const saved = localStorage.getItem('musachat_memories');
    return saved ? JSON.parse(saved) : INITIAL_MEMORIES;
  });
  const [modelName, setModelName] = useState(DEFAULT_MODEL);
  const [chatSessionId, setChatSessionId] = useState(0);

  // Persist memories
  useEffect(() => {
    localStorage.setItem('musachat_memories', JSON.stringify(memories));
  }, [memories]);

  // Check for session
  useEffect(() => {
    const storedUser = localStorage.getItem('musachat_user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    } else {
      setIsAuthOpen(true);
    }
  }, []);

  const handleLogin = (newUser: User) => {
    setUser(newUser);
    localStorage.setItem('musachat_user', JSON.stringify(newUser));
    setIsAuthOpen(false);
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('musachat_user');
    setIsAuthOpen(true);
  };

  const handleNewChat = () => {
    setChatSessionId(prev => prev + 1);
    setCurrentView('chat');
  };

  return (
    <div className="h-screen w-full bg-background text-textMain flex overflow-hidden font-sans">
      
      {/* Sidebar */}
      <aside className="w-20 md:w-64 bg-surface border-r border-slate-800 flex flex-col shrink-0 transition-all duration-300">
        <div className="p-6 flex items-center gap-3 border-b border-slate-800">
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-orange-700 rounded-lg flex items-center justify-center shrink-0 shadow-lg shadow-primary/20">
            <Globe size={18} className="text-white" />
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

        <nav className="flex-1 p-4 space-y-2">
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
      <main className="flex-1 relative overflow-hidden flex flex-col">
        {currentView === 'chat' ? (
          user ? (
            <ChatInterface 
              key={chatSessionId} 
              user={user} 
              memories={memories} 
              modelName={modelName} 
            />
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

export default App;