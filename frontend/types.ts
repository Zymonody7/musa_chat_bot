export interface User {
  id: string;
  username: string;
  email: string;
}

export type MemoryType = 'personality' | 'knowledge' | 'instruction';

export interface MemoryItem {
  id: string;
  content: string;
  type: MemoryType;
  createdAt: number;
  isActive: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  thinkContent?: string;  // 思考过程内容
  isThinking?: boolean;   // 是否正在思考中
  timestamp: number;
  isError?: boolean;
  sources?: Array<{
    page_content: string;
    metadata?: {
      source?: string;
      file?: string;
      page?: number;
      chapter?: string;
      section?: string;
    };
  }>;  // 该消息的来源
}

export type ViewState = 'chat' | 'memory';

export interface ChatConfig {
  modelName: string;
}

export interface ChatSession {
  session_id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}