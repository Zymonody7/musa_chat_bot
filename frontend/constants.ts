import { MemoryItem } from './types';

export const INITIAL_MEMORIES: MemoryItem[] = [
  {
    id: 'mem_1',
    content: 'You are MusaChat, a helpful and knowledgeable AI assistant.',
    type: 'personality',
    createdAt: Date.now(),
    isActive: true,
  },
  {
    id: 'mem_2',
    content: 'The user prefers concise answers with code examples where applicable.',
    type: 'instruction',
    createdAt: Date.now(),
    isActive: true,
  },
];

export const DEFAULT_MODEL = 'QWen3-14B';

export const MOCK_USER = {
  id: 'usr_123',
  username: 'Zymonody',
  email: 'user@example.com'
};

export const DEFAULT_USER = {
  id: 'usr_default',
  username: 'zymonody',
  email: 'zymonody@gmail.com'
};