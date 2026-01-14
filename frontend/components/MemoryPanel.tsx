'use client';

import React, { useState } from 'react';
import { MemoryItem, MemoryType } from '../types';
import { Plus, Trash2, Power, Brain, Cpu, Save, User, Book, Terminal } from 'lucide-react';

interface MemoryPanelProps {
  memories: MemoryItem[];
  setMemories: React.Dispatch<React.SetStateAction<MemoryItem[]>>;
  modelName: string;
  setModelName: (name: string) => void;
}

const MemoryPanel: React.FC<MemoryPanelProps> = ({ memories, setMemories, modelName, setModelName }) => {
  const [newMemory, setNewMemory] = useState('');
  const [memoryType, setMemoryType] = useState<MemoryType>('instruction');

  const addMemory = () => {
    if (!newMemory.trim()) return;
    const item: MemoryItem = {
      id: Date.now().toString(),
      content: newMemory,
      type: memoryType,
      createdAt: Date.now(),
      isActive: true,
    };
    setMemories([...memories, item]);
    setNewMemory('');
  };

  const toggleMemory = (id: string) => {
    setMemories(memories.map(m => 
      m.id === id ? { ...m, isActive: !m.isActive } : m
    ));
  };

  const deleteMemory = (id: string) => {
    setMemories(memories.filter(m => m.id !== id));
  };

  const getTypeIcon = (type: MemoryType) => {
    switch (type) {
      case 'personality': return <User size={14} />;
      case 'knowledge': return <Book size={14} />;
      case 'instruction': return <Terminal size={14} />;
      default: return <Brain size={14} />;
    }
  };

  const getTypeColor = (type: MemoryType) => {
    switch (type) {
      case 'personality': return 'text-purple-400 bg-purple-400/10 border-purple-400/20';
      case 'knowledge': return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
      case 'instruction': return 'text-orange-400 bg-orange-400/10 border-orange-400/20';
      default: return 'text-slate-400 bg-slate-400/10 border-slate-400/20';
    }
  };

  return (
    <div className="h-full flex flex-col p-6 overflow-hidden max-w-4xl mx-auto w-full animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-primary/20 rounded-xl">
          <Brain className="text-primary" size={28} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-white">Shared Memory System</h2>
          <p className="text-textMuted text-sm">Configure the neural context and personality of the model.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 flex-1 min-h-0">
        
        {/* Left Col: Configuration */}
        <div className="md:col-span-1 space-y-6 overflow-y-auto pr-2">
           {/* Model Config Card */}
           <div className="bg-surface rounded-xl p-5 border border-slate-700">
            <div className="flex items-center gap-2 mb-4 text-primary">
              <Cpu size={20} />
              <h3 className="font-semibold text-white">Model Configuration</h3>
            </div>
            <div className="space-y-3">
              <label className="text-xs font-medium text-textMuted uppercase tracking-wider">Target Model ID</label>
              <input 
                type="text" 
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full bg-background border border-slate-600 rounded-lg p-2 text-sm text-white focus:border-primary focus:outline-none"
                placeholder="e.g. QWen3-14B"
              />
              <p className="text-xs text-slate-500">
                Specify the deployed model ID. The default is optimized for speed and reasoning.
              </p>
            </div>
          </div>

          {/* New Memory Input */}
          <div className="bg-surface rounded-xl p-5 border border-slate-700">
             <div className="flex items-center gap-2 mb-4 text-primary">
              <Save size={20} />
              <h3 className="font-semibold text-white">Inject Memory</h3>
            </div>
            
            <div className="flex gap-2 mb-3">
              {(['personality', 'knowledge', 'instruction'] as MemoryType[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setMemoryType(t)}
                  className={`flex-1 py-2 rounded-lg text-xs font-medium border transition-all flex items-center justify-center gap-1 ${
                    memoryType === t 
                      ? getTypeColor(t) + ' border-current'
                      : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
                  }`}
                  title={t.charAt(0).toUpperCase() + t.slice(1)}
                >
                  {getTypeIcon(t)}
                </button>
              ))}
            </div>

            <textarea
              value={newMemory}
              onChange={(e) => setNewMemory(e.target.value)}
              placeholder={`Define ${memoryType}...`}
              className="w-full h-32 bg-background border border-slate-600 rounded-lg p-3 text-sm text-white resize-none focus:border-primary focus:outline-none mb-3"
            />
            <button
              onClick={addMemory}
              disabled={!newMemory.trim()}
              className="w-full bg-primary hover:bg-primaryHover disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <Plus size={16} /> Add to Core
            </button>
          </div>
        </div>

        {/* Right Col: Memory List */}
        <div className="md:col-span-2 bg-surface rounded-xl border border-slate-700 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
            <h3 className="font-semibold text-white">Active Neural Pathways</h3>
            <span className="text-xs font-mono bg-primary/20 text-primary px-2 py-1 rounded">
              {memories.filter(m => m.isActive).length} Active / {memories.length} Total
            </span>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {memories.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-textMuted opacity-50">
                <Brain size={48} className="mb-4" />
                <p>No shared memories initialized.</p>
              </div>
            ) : (
              memories.map((memory) => (
                <div 
                  key={memory.id} 
                  className={`group relative p-4 rounded-xl border transition-all duration-200 ${
                    memory.isActive 
                      ? 'bg-slate-800/50 border-primary/30 shadow-sm' 
                      : 'bg-slate-900/30 border-slate-800 opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-2 w-full">
                      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium border w-fit ${getTypeColor(memory.type || 'instruction')}`}>
                         {getTypeIcon(memory.type || 'instruction')}
                         <span className="uppercase tracking-wider">{memory.type || 'instruction'}</span>
                      </div>
                      <p className="text-sm text-slate-200 leading-relaxed font-mono">
                        {memory.content}
                      </p>
                    </div>
                    
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => toggleMemory(memory.id)}
                        className={`p-1.5 rounded-lg transition-colors ${
                          memory.isActive 
                            ? 'text-emerald-400 bg-emerald-400/10 hover:bg-emerald-400/20' 
                            : 'text-slate-500 bg-slate-800 hover:bg-slate-700'
                        }`}
                        title={memory.isActive ? "Deactivate" : "Activate"}
                      >
                        <Power size={16} />
                      </button>
                      <button
                        onClick={() => deleteMemory(memory.id)}
                        className="p-1.5 rounded-lg text-red-400 bg-red-400/10 hover:bg-red-400/20 opacity-0 group-hover:opacity-100 transition-all"
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-[10px] text-slate-500">
                    <span className="font-mono">ID: {memory.id.slice(-6)}</span>
                    <span>•</span>
                    <span>{new Date(memory.createdAt).toLocaleDateString()}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MemoryPanel;