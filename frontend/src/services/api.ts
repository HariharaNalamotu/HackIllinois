import { getStoredApiKey, getStoredBackendUrl } from '../components/SettingsModal';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface WorkflowConfig {
  nodes: any[];
  edges: any[];
}

export interface FeedbackPayload {
  messageId: string;
  rating: 'up' | 'down';
  feedback?: string;
}

export interface RLAIFScore {
  helpfulness: number;
  accuracy: number;
  safety: number;
  overall: number;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onToolCall?: (toolCall: { name: string; arguments: string }) => void;
  onSubAgent?: (step: { agent: string; content: string }) => void;
  onRlaifScore?: (score: RLAIFScore) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

function getHeaders(): Record<string, string> {
  const apiKey = getStoredApiKey();
  return {
    'Content-Type': 'application/json',
    ...(apiKey ? { 'X-API-Key': apiKey } : {}),
  };
}

function getBaseUrl(): string {
  return getStoredBackendUrl();
}

export async function chatStream(
  messages: ChatMessage[],
  workflowConfig: WorkflowConfig,
  callbacks: StreamCallbacks
): Promise<void> {
  const baseUrl = getBaseUrl();

  try {
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ messages, workflowConfig }),
    });

    if (!res.ok) {
      const errText = await res.text();
      callbacks.onError(errText || `HTTP ${res.status}`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      callbacks.onError('No response body');
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          callbacks.onDone();
          return;
        }

        try {
          const parsed = JSON.parse(data);
          switch (parsed.type) {
            case 'token':
              callbacks.onToken(parsed.content);
              break;
            case 'tool_call':
              callbacks.onToolCall?.(parsed);
              break;
            case 'subagent_step':
              callbacks.onSubAgent?.(parsed);
              break;
            case 'rlaif_score':
              callbacks.onRlaifScore?.(parsed.score);
              break;
            case 'error':
              callbacks.onError(parsed.content);
              return;
          }
        } catch {
          // Skip malformed JSON lines
        }
      }
    }

    callbacks.onDone();
  } catch (err: any) {
    callbacks.onError(err.message || 'Network error');
  }
}

export async function submitFeedback(payload: FeedbackPayload): Promise<boolean> {
  const baseUrl = getBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/api/feedback`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(payload),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function healthCheck(): Promise<boolean> {
  const baseUrl = getBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
