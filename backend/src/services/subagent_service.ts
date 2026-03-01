// Sub-agent orchestration service
// Each subAgent node becomes a callable tool for the main agent

import { ToolDefinition } from './tool_service';

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
}

// Extract sub-agent configs from workflow nodes
export function getSubAgentConfigs(nodes: any[]): SubAgentConfig[] {
  const subAgentNodes = nodes.filter((n: any) => n.data?.type === 'subAgent') as SubAgentNode[];

  return subAgentNodes.map((node, index) => ({
    id: node.id,
    name: `sub_agent_${index + 1}`,
    model: node.data.parameters.subAgentModel || 'gpt-4o-mini',
    systemPrompt: node.data.parameters.subAgentPrompt || 'You are a helpful assistant.',
  }));
}

// Convert sub-agents into tool definitions so the main agent can "call" them
export function buildSubAgentTools(configs: SubAgentConfig[]): ToolDefinition[] {
  return configs.map((config) => ({
    type: 'function' as const,
    function: {
      name: config.name,
      description: `Delegate a task to sub-agent "${config.name}". System: ${config.systemPrompt.slice(0, 100)}...`,
      parameters: {
        type: 'object' as const,
        properties: {
          task: {
            type: 'string',
            description: 'The task or question to delegate to this sub-agent',
          },
        },
        required: ['task'],
      },
    },
  }));
}

// Execute a sub-agent call via OpenAI
export async function executeSubAgent(
  config: SubAgentConfig,
  task: string,
  apiKey: string
): Promise<string> {
  const modelMap: Record<string, string> = {
    'gpt-5.2': 'gpt-4o',
    'gpt-5-mini': 'gpt-4o-mini',
    'gpt-5-nano': 'gpt-4o-mini',
  };
  const model = modelMap[config.model] || 'gpt-4o-mini';

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: config.systemPrompt },
        { role: 'user', content: task },
      ],
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Sub-agent API error: ${err}`);
  }

  const data = await response.json() as any;
  return data.choices?.[0]?.message?.content || 'No response from sub-agent.';
}
