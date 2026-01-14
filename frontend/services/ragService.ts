import { MemoryItem } from '../types';

// 后端 API 地址
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export interface PageContent {
  page: number;
  text: string;
  chapter?: string;
  section?: string;
  chunks: Array<{
    text: string;
    meta: any;
    chunk_idx: number;
  }>;
}

/**
 * 获取指定页面的内容
 */
export async function getPageContent(pageNum: number): Promise<PageContent> {
  const response = await fetch(`${API_BASE_URL}/page/${pageNum}`);
  if (!response.ok) {
    throw new Error(`获取页面内容失败: ${response.statusText}`);
  }
  return response.json();
}

export interface RagSource {
  page_content: string;
  metadata?: {
    source?: string;
    file?: string;
    page?: number;
    chapter?: string;
    section?: string;
  };
}

// 后端返回的原始 source 格式
interface BackendSource {
  pdf_page?: number;
  chapter?: string;
  section?: string;
  snippet?: string;
  score?: number;
}

export interface StreamEvent {
  type: 'retrieve' | 'token' | 'done' | 'error';
  sources?: BackendSource[] | RagSource[];
  token?: string;
  text?: string;
  full_text?: string;
  error?: string;
  session_id?: string;
}

/**
 * 将后端返回的 source 格式转换为前端期望的格式
 */
function transformSource(backendSource: BackendSource): RagSource {
  return {
    page_content: backendSource.snippet || '',
    metadata: {
      page: backendSource.pdf_page,
      chapter: backendSource.chapter,
      section: backendSource.section,
    },
  };
}

/**
 * 构建包含记忆的问题
 */
export function buildQuestionWithMemory(question: string, memories: MemoryItem[]): string {
  const activeMemories = memories.filter(m => m.isActive);
  if (activeMemories.length === 0) {
    return question;
  }

  const memoryContext = activeMemories
    .map(m => `[${m.type.toUpperCase()}] ${m.content}`)
    .join('\n\n');

  return `${memoryContext}\n\n用户问题: ${question}`;
}

/**
 * 流式调用后端 RAG 服务
 */
export async function* streamAsk(question: string, sessionId?: string): AsyncGenerator<StreamEvent, void, unknown> {
  try {
    const response = await fetch(`${API_BASE_URL}/ask/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        question,
        session_id: sessionId || null
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      yield {
        type: 'error',
        error: `HTTP ${response.status}: ${errorText || '请求失败'}`,
      };
      return;
    }

    if (!response.body) {
      yield {
        type: 'error',
        error: '响应体为空',
      };
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue; // 跳过空行
          
          if (trimmedLine.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmedLine.slice(6));
              // 转换 retrieve 事件中的 sources 格式
              if (data.type === 'retrieve' && data.sources && Array.isArray(data.sources)) {
                data.sources = data.sources.map(transformSource);
              }
              yield data as StreamEvent;
            } catch (e) {
              console.warn('解析 SSE 数据失败:', trimmedLine, e);
            }
          }
        }
      }

      // 处理剩余的 buffer
      if (buffer.trim()) {
        const lines = buffer.split('\n');
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue; // 跳过空行
          
          if (trimmedLine.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmedLine.slice(6));
              // 转换 retrieve 事件中的 sources 格式
              if (data.type === 'retrieve' && data.sources && Array.isArray(data.sources)) {
                data.sources = data.sources.map(transformSource);
              }
              yield data as StreamEvent;
            } catch (e) {
              console.warn('解析 SSE 数据失败:', trimmedLine, e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('流式请求错误:', error);
    yield {
      type: 'error',
      error: error instanceof Error ? error.message : '网络错误或后端服务不可用',
    };
  }
}
