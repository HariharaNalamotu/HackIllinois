// Core agentic orchestration service
// Takes workflow config + user messages, orchestrates OpenAI calls with tools, sub-agents, RLAIF

import type { Context } from 'hono';
import type { Env } from '../index';
import { buildToolDefinitions, executeTool, ToolDefinition } from './tool_service';
import { getSubAgentConfigs, buildSubAgentTools, executeSubAgent, SubAgentConfig } from './subagent_service';
import { getRLAIFConfig, evaluateResponse, getGoodResponseExamples } from './rlaif_service';
import { getPositiveFeedbackContext } from './rlhf_service';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface WorkflowConfig {
  nodes: any[];
  edges: any[];
}

interface ChatRequest {
  messages: ChatMessage[];
  workflowConfig: WorkflowConfig;
}

// Build the system prompt from workflow configuration
function buildSystemPrompt(nodes: any[]): string {
  const llmNode = nodes.find((n: any) => n.data?.type === 'agenticLLM');
  let basePrompt = llmNode?.data?.parameters?.subAgentPrompt || 'You are a helpful AI assistant.';

  // Add RLHF context (positive feedback examples)
  basePrompt += getPositiveFeedbackContext();

  // Add RLAIF context (high-scoring response patterns)
  basePrompt += getGoodResponseExamples();

  return basePrompt;
}

// Map frontend model names to OpenAI model IDs
function resolveModel(nodes: any[]): string {
  const llmNode = nodes.find((n: any) => n.data?.type === 'agenticLLM');
  const model = llmNode?.data?.parameters?.subAgentModel || 'gpt-5.2';
  const modelMap: Record<string, string> = {
    'gpt-5.2': 'gpt-4o',
    'gpt-5-mini': 'gpt-4o-mini',
    'gpt-5-nano': 'gpt-4o-mini',
  };
  return modelMap[model] || 'gpt-4o-mini';
}

// SSE helper: write a server-sent event
function sseEvent(type: string, data: any): string {
  return `data: ${JSON.stringify({ type, ...data })}\n\n`;
}

export async function chatHandler(c: Context<{ Bindings: Env }>) {
  // Accept API key from either X-API-Key header or the worker's env
  const apiKey = c.req.header('X-API-Key') || c.env.OPENAI_API_KEY;
  if (!apiKey) {
    return c.json({ error: 'No OpenAI API key available. Set it in Settings or configure OPENAI_API_KEY on the worker.' }, 401);
  }

  let body: ChatRequest;
  try {
    body = await c.req.json<ChatRequest>();
  } catch {
    return c.json({ error: 'Invalid request body' }, 400);
  }

  const { messages, workflowConfig } = body;
  if (!messages || !Array.isArray(messages)) {
    return c.json({ error: 'messages array is required' }, 400);
  }

  const nodes = workflowConfig?.nodes || [];

  // Build system prompt
  const systemPrompt = buildSystemPrompt(nodes);
  const model = resolveModel(nodes);

  // Build tools from agentTool nodes + sub-agent nodes
  const toolDefs = buildToolDefinitions(nodes);
  const subAgentConfigs = getSubAgentConfigs(nodes);
  const subAgentTools = buildSubAgentTools(subAgentConfigs);
  const allTools: ToolDefinition[] = [...toolDefs, ...subAgentTools];

  // Build message list with system prompt
  const openaiMessages: any[] = [
    { role: 'system', content: systemPrompt },
    ...messages.map((m) => ({ role: m.role, content: m.content })),
  ];

  // RLAIF config
  const rlaifConfig = getRLAIFConfig(nodes);

  // Create a streaming response
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  const write = async (text: string) => {
    await writer.write(encoder.encode(text));
  };

  // Process in background
  const processChat = async () => {
    let fullResponse = '';
    const lastUserMessage = messages[messages.length - 1]?.content || '';

    try {
      // Initial OpenAI call
      fullResponse = await streamOpenAIResponse(
        openaiMessages,
        model,
        allTools.length > 0 ? allTools : undefined,
        apiKey,
        write,
        subAgentConfigs,
      );

      // RLAIF evaluation (if configured)
      if (rlaifConfig && fullResponse) {
        try {
          const score = await evaluateResponse(
            lastUserMessage,
            fullResponse,
            rlaifConfig.model,
            apiKey
          );
          await write(sseEvent('rlaif_score', { score }));
        } catch (err) {
          console.error('RLAIF evaluation failed:', err);
        }
      }

      await write('data: [DONE]\n\n');
    } catch (err: any) {
      await write(sseEvent('error', { content: err.message || 'Unknown error' }));
    } finally {
      await writer.close();
    }
  };

  // Kick off processing without awaiting (streams to response)
  processChat();

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

// Stream an OpenAI chat completion, handling tool calls iteratively
async function streamOpenAIResponse(
  messages: any[],
  model: string,
  tools: ToolDefinition[] | undefined,
  apiKey: string,
  write: (text: string) => Promise<void>,
  subAgentConfigs: SubAgentConfig[],
  depth: number = 0
): Promise<string> {
  // Prevent infinite tool call loops
  if (depth > 10) {
    await write(sseEvent('token', { content: '\n[Max tool call depth reached]' }));
    return '[Max depth reached]';
  }

  const requestBody: any = {
    model,
    messages,
    stream: true,
    temperature: 0.7,
  };

  if (tools && tools.length > 0) {
    requestBody.tools = tools;
  }

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OpenAI API error ${response.status}: ${errText}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullContent = '';

  // Accumulate tool calls across chunks
  const toolCallAccumulator: Map<number, { id: string; name: string; arguments: string }> = new Map();
  let hasToolCalls = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') continue;

      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta;

        // Content tokens
        if (delta?.content) {
          fullContent += delta.content;
          await write(sseEvent('token', { content: delta.content }));
        }

        // Tool calls
        if (delta?.tool_calls) {
          hasToolCalls = true;
          for (const tc of delta.tool_calls) {
            const idx = tc.index;
            if (!toolCallAccumulator.has(idx)) {
              toolCallAccumulator.set(idx, { id: tc.id || '', name: '', arguments: '' });
            }
            const acc = toolCallAccumulator.get(idx)!;
            if (tc.id) acc.id = tc.id;
            if (tc.function?.name) acc.name += tc.function.name;
            if (tc.function?.arguments) acc.arguments += tc.function.arguments;
          }
        }
      } catch {
        // Skip malformed chunks
      }
    }
  }

  // If the model made tool calls, execute them and continue
  if (hasToolCalls && toolCallAccumulator.size > 0) {
    const assistantMsg: any = {
      role: 'assistant',
      content: fullContent || null,
      tool_calls: Array.from(toolCallAccumulator.values()).map((tc) => ({
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: tc.arguments },
      })),
    };
    messages.push(assistantMsg);

    // Execute each tool call
    for (const tc of toolCallAccumulator.values()) {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.arguments);
      } catch {}

      // Check if this is a sub-agent call
      const subAgent = subAgentConfigs.find((sa) => sa.name === tc.name);
      let result: string;

      if (subAgent) {
        await write(sseEvent('subagent_step', { agent: tc.name, content: `Delegating: "${args.task}"` }));
        try {
          result = await executeSubAgent(subAgent, args.task as string, apiKey);
          await write(sseEvent('subagent_step', { agent: tc.name, content: `Response: ${result.slice(0, 200)}...` }));
        } catch (err: any) {
          result = `Sub-agent error: ${err.message}`;
          await write(sseEvent('subagent_step', { agent: tc.name, content: result }));
        }
      } else {
        // Regular tool call
        await write(sseEvent('tool_call', { name: tc.name, arguments: tc.arguments }));
        result = executeTool(tc.name, args);
      }

      // Add tool result to conversation
      messages.push({
        role: 'tool',
        tool_call_id: tc.id,
        content: result,
      });
    }

    // Continue the conversation with tool results
    return await streamOpenAIResponse(messages, model, tools, apiKey, write, subAgentConfigs, depth + 1);
  }

  return fullContent;
}
