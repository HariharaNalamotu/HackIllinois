// Sub-agent orchestration service
// Each subAgent node becomes a callable tool for the main agent
// Sub-agents maintain persistent conversation state across invocations,
// enabling cyclical parent ↔ sub-agent dialogue until the parent is satisfied

import type { Context } from 'hono';
import type { AppContext } from '../index';
import { ToolDefinition, executeTool } from './tool_service';

interface SubAgentNode {
  id: string;
  data: {
    type: 'subAgent';
    label: string;
    parameters: {
      subAgentModel: string;
      subAgentPrompt: string;
    };
  };
}

export interface SubAgentConfig {
  id: string;
  name: string;
  model: string;
  systemPrompt: string;
  tools: ToolDefinition[];
}

interface SubAgentSession {
  config: SubAgentConfig;
  // Persistent conversation history — survives across parent invocations
  conversationHistory: any[];
  invocationCount: number;
  status: 'registered' | 'active' | 'error';
}

function resolveModel(model: string): string {
  return model || 'gpt-4o-mini';
}

// Persistent sub-agent sessions keyed by agent name
const agentSessions: Map<string, SubAgentSession> = new Map();

// Extract sub-agent configs from workflow nodes, including their connected tools
export function getSubAgentConfigs(nodes: any[], edges: any[]): SubAgentConfig[] {
  const subAgentNodes = nodes.filter((n: any) => n.data?.type === 'subAgent') as SubAgentNode[];

  return subAgentNodes.map((node, index) => {
    const connectedToolNodeIds = edges
      .filter((e: any) => e.target === node.id || e.source === node.id)
      .map((e: any) => e.target === node.id ? e.source : e.target);

    const connectedToolNodes = nodes.filter(
      (n: any) => n.data?.type === 'agentTool' && connectedToolNodeIds.includes(n.id)
    );

    const tools: ToolDefinition[] = connectedToolNodes
      .filter((n: any) => n.data.parameters.functionName)
      .map((n: any) => {
        const params = n.data.parameters;
        const properties: Record<string, unknown> = {};
        const required: string[] = [];

        for (const param of params.parameters || []) {
          const prop: Record<string, unknown> = {
            type: param.type,
            description: param.description,
          };
          if (param.type === 'object' && param.objectProperties) {
            prop.properties = {};
            for (const op of param.objectProperties) {
              (prop.properties as Record<string, unknown>)[op.key] = { type: 'string', description: op.value };
            }
          }
          if (param.type === 'array' && param.arrayValues) {
            prop.items = { type: 'string', enum: param.arrayValues };
          }
          properties[param.name] = prop;
          if (param.required) required.push(param.name);
        }

        return {
          type: 'function' as const,
          function: {
            name: params.functionName,
            description: params.functionDescription || `Execute ${params.functionName}`,
            parameters: { type: 'object' as const, properties, required },
          },
        };
      });

    return {
      id: node.id,
      name: `sub_agent_${index + 1}`,
      model: node.data.parameters.subAgentModel || 'gpt-4o-mini',
      systemPrompt: node.data.parameters.subAgentPrompt || 'You are a helpful assistant.',
      tools,
    };
  });
}

// Convert sub-agents into tool definitions so the main agent can "call" them
// The parent can call the same sub-agent multiple times — it retains conversation context
export function buildSubAgentTools(configs: SubAgentConfig[]): ToolDefinition[] {
  return configs.map((config) => {
    const toolList = config.tools.length > 0
      ? ` Available tools: ${config.tools.map((t) => t.function.name).join(', ')}.`
      : '';
    return {
      type: 'function' as const,
      function: {
        name: config.name,
        description: `Delegate a task to sub-agent "${config.name}". This agent retains conversation context — you can call it multiple times for follow-up instructions until you are satisfied with its output. System: ${config.systemPrompt.slice(0, 80)}...${toolList}`,
        parameters: {
          type: 'object' as const,
          properties: {
            message: {
              type: 'string',
              description: 'The instruction or follow-up message to send to this sub-agent. On first call, describe the task. On subsequent calls, provide feedback or further instructions.',
            },
          },
          required: ['message'],
        },
      },
    };
  });
}

// Get or create a session for a sub-agent
function getOrCreateSession(config: SubAgentConfig): SubAgentSession {
  let session = agentSessions.get(config.name);
  if (!session) {
    session = {
      config,
      conversationHistory: [
        { role: 'system', content: config.systemPrompt },
      ],
      invocationCount: 0,
      status: 'active',
    };
    agentSessions.set(config.name, session);
  }
  return session;
}

// Reset all sub-agent sessions (called when a new chat session or training starts)
export function resetSubAgentSessions(): void {
  agentSessions.clear();
}

// Execute a sub-agent call with persistent conversation state
// Each call adds to the sub-agent's conversation history, enabling
// the parent agent to have a back-and-forth dialogue with the sub-agent
export async function executeSubAgent(
  config: SubAgentConfig,
  message: string,
  apiKey: string,
  onStep?: (content: string) => Promise<void>
): Promise<string> {
  const model = resolveModel(config.model);
  const session = getOrCreateSession(config);

  session.invocationCount++;

  // Add the parent's message to the persistent conversation
  session.conversationHistory.push({ role: 'user', content: message });

  if (onStep) {
    await onStep(`Invocation #${session.invocationCount}: "${message.slice(0, 100)}${message.length > 100 ? '...' : ''}"`);
  }

  const maxToolIterations = 5;

  // Inner loop: let the sub-agent use its tools until it produces a final text response
  for (let i = 0; i < maxToolIterations; i++) {
    const requestBody: any = {
      model,
      messages: session.conversationHistory,
      temperature: 0.7,
    };

    if (config.tools.length > 0) {
      requestBody.tools = config.tools;
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
      session.status = 'error';
      if (response.status === 401) throw new Error('Invalid OpenAI API key.');
      if (response.status === 429) throw new Error('Rate limited. Try again shortly.');
      const err = await response.text();
      throw new Error(`Sub-agent error (${response.status}): ${err}`);
    }

    const data = await response.json() as any;
    const choice = data.choices?.[0];
    const msg = choice?.message;

    if (!msg) {
      session.status = 'error';
      return 'No response from sub-agent.';
    }

    // If the sub-agent made tool calls, execute them and loop
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      session.conversationHistory.push(msg);

      for (const tc of msg.tool_calls) {
        const fnName = tc.function.name;
        let args: Record<string, unknown> = {};
        try { args = JSON.parse(tc.function.arguments); } catch {}

        if (onStep) {
          await onStep(`Tool: ${fnName}(${JSON.stringify(args)})`);
        }

        const result = await executeTool(fnName, args);
        session.conversationHistory.push({
          role: 'tool',
          tool_call_id: tc.id,
          content: result,
        });
      }

      continue;
    }

    // Final text response — add to persistent history and return
    const responseContent = msg.content || 'No response from sub-agent.';
    session.conversationHistory.push({ role: 'assistant', content: responseContent });

    if (onStep) {
      await onStep(`Responded (invocation #${session.invocationCount}, ${session.conversationHistory.length} messages in context)`);
    }

    return responseContent;
  }

  return 'Sub-agent reached max tool call iterations.';
}

// POST /api/train/subagents — register and validate sub-agent configurations
export async function subagentTrainHandler(c: Context<AppContext>) {
  const apiKey = c.req.header('X-API-Key');
  if (!apiKey) {
    return c.json({ error: 'X-API-Key header is required.' }, 401);
  }

  if (!apiKey.startsWith('sk-')) {
    return c.json({ error: 'Invalid API key format.' }, 401);
  }

  let body: { workflowConfig: { nodes: any[]; edges: any[] } };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: 'Invalid request body' }, 400);
  }

  const nodes = body.workflowConfig?.nodes || [];
  const edges = body.workflowConfig?.edges || [];
  const configs = getSubAgentConfigs(nodes, edges);

  if (configs.length === 0) {
    return c.json({ error: 'No sub-agent nodes found in workflow.' }, 400);
  }

  // Clear previous sessions
  resetSubAgentSessions();

  const results: { name: string; model: string; tools: number; status: string; error?: string }[] = [];

  for (const config of configs) {
    const resolvedModel = resolveModel(config.model);

    // Validate: send a short test message to verify model + API key work
    try {
      const testResponse = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: resolvedModel,
          messages: [
            { role: 'system', content: config.systemPrompt },
            { role: 'user', content: 'Respond with "ready" to confirm you are operational.' },
          ],
          ...( resolvedModel.startsWith('gpt-5') ? { max_completion_tokens: 10 } : { max_tokens: 10 }),
          temperature: 0,
        }),
      });

      if (!testResponse.ok) {
        const errText = await testResponse.text();
        results.push({
          name: config.name,
          model: resolvedModel,
          tools: config.tools.length,
          status: 'error',
          error: `Validation failed (${testResponse.status}): ${errText.slice(0, 100)}`,
        });
        continue;
      }

      // Pre-create the session so it's ready for use
      const session = getOrCreateSession(config);
      session.status = 'registered';

      results.push({
        name: config.name,
        model: resolvedModel,
        tools: config.tools.length,
        status: 'registered',
      });
    } catch (err: any) {
      results.push({
        name: config.name,
        model: resolvedModel,
        tools: config.tools.length,
        status: 'error',
        error: err.message,
      });
    }
  }

  const allRegistered = results.every((r) => r.status === 'registered');

  return c.json({
    success: allRegistered,
    subagents: results,
    message: allRegistered
      ? `${results.length} sub-agent(s) registered and validated. Each retains conversation context across calls for cyclical parent ↔ sub-agent dialogue.`
      : `Some sub-agents failed validation. Check errors.`,
  });
}
