// Tool execution service
// Converts agentTool nodes into OpenAI function calling format and handles execution

export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, unknown>;
      required: string[];
    };
  };
}

interface ToolParameter {
  id: string;
  name: string;
  type: string;
  description: string;
  required: boolean;
  objectProperties?: { key: string; value: string }[];
  arrayValues?: string[];
}

interface AgentToolNode {
  data: {
    type: 'agentTool';
    parameters: {
      functionName: string;
      functionDescription: string;
      parameters: ToolParameter[];
    };
  };
}

// Convert agentTool nodes from the workflow into OpenAI function calling schema
export function buildToolDefinitions(nodes: any[]): ToolDefinition[] {
  const toolNodes = nodes.filter((n: any) => n.data?.type === 'agentTool') as AgentToolNode[];

  return toolNodes
    .filter((n) => n.data.parameters.functionName)
    .map((node) => {
      const params = node.data.parameters;
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

        if (param.required) {
          required.push(param.name);
        }
      }

      return {
        type: 'function' as const,
        function: {
          name: params.functionName,
          description: params.functionDescription || `Execute ${params.functionName}`,
          parameters: {
            type: 'object' as const,
            properties,
            required,
          },
        },
      };
    });
}

// Execute a tool call (returns mock results for demo)
export function executeTool(name: string, args: Record<string, unknown>): string {
  return JSON.stringify({
    tool: name,
    arguments: args,
    result: `[Mock result for ${name}] — executed successfully with provided arguments.`,
    timestamp: new Date().toISOString(),
  });
}
