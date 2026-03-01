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

// In-memory store of per-tool API keys (keyed by function name)
const toolApiKeys: Map<string, string> = new Map();

export function setToolApiKey(toolName: string, apiKey: string) {
  if (apiKey) {
    toolApiKeys.set(toolName, apiKey);
  }
}

// Extract API keys from workflow nodes for runtime injection
export function registerToolApiKeys(nodes: any[]) {
  const toolNodes = nodes.filter((n: any) => n.data?.type === 'agentTool' && n.data.parameters?.functionName);
  for (const node of toolNodes) {
    const { functionName, apiKey } = node.data.parameters;
    if (functionName && apiKey) {
      toolApiKeys.set(functionName, apiKey);
    }
  }
}

// Strip TypeScript syntax from generated code so it runs as plain JS
function stripTypeScript(code: string): string {
  return code
    .replace(/^export\s+(default\s+)?/gm, '')
    .replace(/^module\.exports\s*=.*/gm, '')
    .replace(/\((\w+)\s*:\s*\{[^}]*\}\s*\)/g, '($1)')
    .replace(/(\w+)\s*:\s*(string|number|boolean|any|void|object|unknown|never)\s*(?=[,)=])/g, '$1')
    .replace(/\)\s*:\s*Promise<[^>]*>\s*\{/g, ') {')
    .replace(/\)\s*:\s*\w[\w<>,\s|[\]{}]*\s*\{/g, ') {')
    .replace(/^(interface|type)\s+\w+[\s\S]*?^}/gm, '')
    .replace(/\s+as\s+\w+(\[\])?/g, '')
    .replace(/(const|let|var)\s+(\w+)\s*:\s*[^=]+=\s*/g, '$1 $2 = ');
}

// Execute a tool call — runs codex-generated code via new Function()
// Requires Node.js runtime (not Cloudflare Workers which blocks eval)
export async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  const { getGeneratedCode } = await import('./codegen_service');
  const code = getGeneratedCode(name);

  if (!code) {
    return JSON.stringify({
      error: `No implementation for tool "${name}". Click "Train Model" first to generate code.`,
    });
  }

  // Inject the per-tool API key if one is configured
  const apiKey = toolApiKeys.get(name);
  if (apiKey) {
    args.__apiKey = apiKey;
  }

  const cleanCode = stripTypeScript(code);

  try {
    const fn = new Function('args', `
      return (async () => {
        ${cleanCode}
        return typeof ${name} === 'function' ? ${name}(args) : { error: 'Function not found in generated code' };
      })();
    `);
    const result = await fn(args);
    return JSON.stringify(result);
  } catch (err: any) {
    return JSON.stringify({
      tool: name,
      error: `Execution failed: ${err.message}`,
      codePreview: cleanCode.slice(0, 300),
    });
  }
}
