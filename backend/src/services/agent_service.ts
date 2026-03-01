// Core agentic orchestration service
// Takes workflow config + user messages, orchestrates OpenAI calls with tools, sub-agents, RLAIF

import type { Context } from 'hono';
import type { AppContext } from '../index';
import { buildToolDefinitions, executeTool, registerToolApiKeys, ToolDefinition } from './tool_service';
import { getSubAgentConfigs, buildSubAgentTools, executeSubAgent, resetSubAgentSessions, SubAgentConfig } from './subagent_service';
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

  // Add tool usage instructions if tools are present
  const toolNodes = nodes.filter((n: any) => n.data?.type === 'agentTool' && n.data.parameters?.functionName);
  if (toolNodes.length > 0) {
    basePrompt += '\n\nYou have access to tools that can perform real actions. When a user asks you to do something that a tool can handle, ALWAYS call the tool — do not say you cannot do it. Your tools are fully functional and their results are delivered directly to the user.';
    basePrompt += ' Tools that produce files (PDFs, images, documents, etc.) will automatically deliver the file to the user as a download — you do not need to worry about file delivery, just call the tool.';
    basePrompt += ' Never tell the user you have "tool limitations" or suggest they use external services for tasks your tools can handle.';
    basePrompt += ' IMPORTANT: If one tool call fails (e.g. web search returns an error), do NOT give up on the remaining tools. Continue with the other tools using your existing knowledge. For example, if search fails but you have a PDF generation tool, still generate the PDF with whatever information you have.';
  }

  // Add RLHF context (positive feedback examples)
  basePrompt += getPositiveFeedbackContext();

  // Add RLAIF context (high-scoring response patterns)
  basePrompt += getGoodResponseExamples();

  return basePrompt;
}

// Resolve the model from the workflow's LLM node
function resolveModel(nodes: any[]): string {
  const llmNode = nodes.find((n: any) => n.data?.type === 'agenticLLM');
  return llmNode?.data?.parameters?.subAgentModel || 'gpt-5.2';
}

// SSE helper: write a server-sent event
function sseEvent(type: string, data: any): string {
  return `data: ${JSON.stringify({ type, ...data })}\n\n`;
}

export async function chatHandler(c: Context<AppContext>) {
  const apiKey = c.req.header('X-API-Key');
  if (!apiKey) {
    return c.json({ error: 'X-API-Key header is required. Set your OpenAI API key in Settings.' }, 401);
  }

  if (!apiKey.startsWith('sk-')) {
    return c.json({ error: 'Invalid API key format. OpenAI keys start with "sk-".' }, 401);
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

  if (!workflowConfig?.nodes || !Array.isArray(workflowConfig.nodes)) {
    return c.json({ error: 'workflowConfig with nodes array is required' }, 400);
  }

  const nodes = workflowConfig.nodes;

  // Build system prompt
  const systemPrompt = buildSystemPrompt(nodes);
  const model = resolveModel(nodes);

  const edges = workflowConfig?.edges || [];

  // Register per-tool API keys from node config
  registerToolApiKeys(nodes);

  // Build tools from agentTool nodes + sub-agent nodes
  const toolDefs = buildToolDefinitions(nodes);
  const subAgentConfigs = getSubAgentConfigs(nodes, edges);
  const subAgentTools = buildSubAgentTools(subAgentConfigs);
  // Only give the main agent tools that aren't exclusively connected to a sub-agent
  const subAgentToolNodeIds = new Set(
    edges
      .filter((e: any) => {
        const targetNode = nodes.find((n: any) => n.id === e.target);
        const sourceNode = nodes.find((n: any) => n.id === e.source);
        return targetNode?.data?.type === 'subAgent' || sourceNode?.data?.type === 'subAgent';
      })
      .flatMap((e: any) => [e.source, e.target])
      .filter((id: string) => nodes.find((n: any) => n.id === id)?.data?.type === 'agentTool')
  );
  const mainAgentTools = toolDefs.filter(
    (td) => !subAgentToolNodeIds.has(
      nodes.find((n: any) => n.data?.type === 'agentTool' && n.data.parameters.functionName === td.function.name)?.id
    )
  );
  const allTools: ToolDefinition[] = [...mainAgentTools, ...subAgentTools];

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
    let lastUserMessage = messages[messages.length - 1]?.content || '';

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

  // Kick off processing — use waitUntil so Cloudflare Workers doesn't kill
  // the promise before it finishes writing to the stream.
  const promise = processChat();
  try {
    c.executionCtx.waitUntil(promise);
  } catch {
    // Fallback: if executionCtx isn't available (e.g. local dev), the
    // TransformStream backpressure will keep the promise alive anyway.
  }

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      'Access-Control-Allow-Origin': '*',
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
    if (response.status === 401) {
      throw new Error('Invalid OpenAI API key. Please check your key in Settings.');
    }
    if (response.status === 429) {
      throw new Error('OpenAI rate limit exceeded. Please wait a moment and try again.');
    }
    if (response.status === 402) {
      throw new Error('OpenAI billing issue. Check your account has sufficient credits.');
    }
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
    // Add the assistant message with tool calls to the conversation
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
        const subMessage = (args.message || args.task || '') as string;
        await write(sseEvent('subagent_step', { agent: tc.name, content: `Parent → Sub-agent: "${subMessage.slice(0, 100)}${subMessage.length > 100 ? '...' : ''}"` }));
        try {
          result = await executeSubAgent(
            subAgent,
            subMessage,
            apiKey,
            async (stepContent) => {
              await write(sseEvent('subagent_step', { agent: tc.name, content: stepContent }));
            }
          );
          await write(sseEvent('subagent_step', { agent: tc.name, content: `Sub-agent → Parent: "${result.slice(0, 200)}${result.length > 200 ? '...' : ''}"` }));
        } catch (err: any) {
          result = `Sub-agent error: ${err.message}`;
          await write(sseEvent('subagent_step', { agent: tc.name, content: result }));
        }
      } else {
        // Regular tool call
        await write(sseEvent('tool_call', { name: tc.name, arguments: tc.arguments }));
        result = await executeTool(tc.name, args);
      }

      // Check for file outputs in tool result and emit file_output events
      let cleanResult = result;
      try {
        const parsed = JSON.parse(result);
        // Log tool results for debugging
        if (parsed?.error) {
          console.log(`[tool-error] ${tc.name}: ${parsed.error}`);
        }
        if (parsed && typeof parsed === 'object') {
          // Primary: explicit __files__ array
          if (Array.isArray(parsed.__files__)) {
            for (const file of parsed.__files__) {
              await write(sseEvent('file_output', {
                file: {
                  filename: file.filename || 'untitled',
                  mimeType: file.mimeType || 'application/octet-stream',
                  data: file.data || '',
                  displayType: file.displayType || (file.mimeType?.startsWith('image/') ? 'image' : 'download'),
                },
              }));
            }
            const { __files__, ...rest } = parsed;
            cleanResult = JSON.stringify(rest);
          }
          // Fallback: detect single file-like result (has data + mimeType or filename)
          else if (parsed.data && typeof parsed.data === 'string' && (parsed.mimeType || parsed.filename)) {
            const mimeType = parsed.mimeType || 'application/octet-stream';
            await write(sseEvent('file_output', {
              file: {
                filename: parsed.filename || `output.${mimeType.split('/')[1] || 'bin'}`,
                mimeType,
                data: parsed.data,
                displayType: parsed.displayType || (mimeType.startsWith('image/') ? 'image' : 'download'),
              },
            }));
            // Pass a summary to the model instead of the raw data
            const { data: _fileData, ...rest } = parsed;
            cleanResult = JSON.stringify({ ...rest, fileDelivered: true });
          }
        }
      } catch {
        // Result is not JSON — use as-is
      }

      // Add tool result to conversation
      messages.push({
        role: 'tool',
        tool_call_id: tc.id,
        content: cleanResult,
      });
    }

    // Continue the conversation with tool results
    return await streamOpenAIResponse(messages, model, tools, apiKey, write, subAgentConfigs, depth + 1);
  }

  return fullContent;
}
