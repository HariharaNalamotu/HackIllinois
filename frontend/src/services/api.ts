import { getStoredApiKey, getStoredBackendUrl } from '../components/SettingsModal';
import type { PipelineSpec } from '../utils/pipelineBuilder';

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

// ── Training pipeline API ─────────────────────────────────────────────────────

const WORKER_BASE = 'https://hackillinois-api.harihara-nalamotu.workers.dev';

/** Submit a training job. Returns the job_id. */
export async function submitTrainingJob(
  workflowId: string,
  pipelineSpec: PipelineSpec,
  files: Array<{ nodeId: string; file: File }>
): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.set('pipeline_spec', JSON.stringify(pipelineSpec));
  for (const { nodeId, file } of files) {
    fd.append('files[]', file, `${nodeId}__${file.name}`);
  }

  const res = await fetch(`${WORKER_BASE}/api/workflow/${workflowId}/train`, {
    method: 'POST',
    body: fd,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Training submission failed: ${err.slice(0, 200)}`);
  }
  return res.json();
}

/** Poll training job status. */
export async function pollJobStatus(jobId: string): Promise<{
  status: 'running' | 'complete' | 'error' | 'not_found';
  result?: unknown;
  error?: string;
}> {
  const res = await fetch(`${WORKER_BASE}/api/pipeline/${jobId}/status`);
  if (!res.ok) return { status: 'not_found' };
  return res.json();
}

/**
 * Stream training logs via SSE.
 * Returns a cleanup function to cancel the stream.
 */
export function streamJobLogs(
  jobId: string,
  onLog: (line: string) => void,
  onDone: (result?: unknown) => void
): () => void {
  const url = `${WORKER_BASE}/api/pipeline/${jobId}/logs`;
  const es  = new EventSource(url);

  es.onmessage = (e) => {
    if (e.data === '[DONE]') {
      es.close();
      onDone();
      return;
    }
    try {
      const parsed = JSON.parse(e.data);
      if (parsed.type === 'log')    onLog(parsed.message || e.data);
      if (parsed.type === 'result') { es.close(); onDone(parsed.result); }
    } catch {
      onLog(e.data);
    }
  };

  es.onerror = () => {
    es.close();
    onDone();
  };

  return () => es.close();
}

/** Download a trained model as a tar archive (triggers browser download). */
export async function downloadModel(modelName: string): Promise<void> {
  const res = await fetch(`${WORKER_BASE}/api/models/${encodeURIComponent(modelName)}/download`);
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);

  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `${modelName}.tar`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Run inference via the deployment endpoint. */
export async function runInference(
  workflowId: string,
  inputData: string | File | File[]
): Promise<{ result: unknown; llmResponse?: string }> {
  let body: FormData | string;
  let contentType: string | undefined;

  if (typeof inputData === 'string') {
    body        = inputData;
    contentType = 'text/plain';
  } else if (inputData instanceof File) {
    const fd = new FormData();
    fd.append('file', inputData, inputData.name);
    body = fd;
  } else {
    const fd = new FormData();
    for (const f of inputData) fd.append('files[]', f, f.name);
    body = fd;
  }

  const headers: Record<string, string> = {};
  if (contentType) headers['Content-Type'] = contentType;

  const res = await fetch(`${WORKER_BASE}/api/deploy/${workflowId}`, {
    method: 'POST',
    headers,
    body,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Inference failed: ${err.slice(0, 200)}`);
  }
  return res.json();
}

/** Send a chat message to the LLM proxy. */
export async function sendLLMChat(
  provider: string,
  model: string,
  messages: Array<{ role: string; content: string }>,
  ollamaUrl?: string,
  systemPrompt?: string
): Promise<{ content: string }> {
  const res = await fetch(`${WORKER_BASE}/api/llm/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model, messages, ollamaUrl, systemPrompt }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`LLM chat failed: ${err.slice(0, 200)}`);
  }
  return res.json();
}
