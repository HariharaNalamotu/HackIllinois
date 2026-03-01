// Code generation service
// Uses GPT-5.3-codex with Tavily web search to generate
// executable implementations for agent tools

import type { Context } from 'hono';
import type { AppContext } from '../index';

interface ToolNodeData {
  functionName: string;
  functionDescription: string;
  hasApiKey: boolean;
  parameters: {
    name: string;
    type: string;
    description: string;
    required: boolean;
  }[];
}

interface GeneratedTool {
  name: string;
  code: string;
}

// In-memory store of generated tool code (keyed by function name)
const generatedToolCode: Map<string, string> = new Map();

export function getGeneratedCode(toolName: string): string | undefined {
  return generatedToolCode.get(toolName);
}

// Search for relevant API documentation using Tavily
async function searchForAPIs(
  toolDescription: string,
  tavilyApiKey: string
): Promise<string> {
  try {
    const response = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tavilyApiKey}`,
      },
      body: JSON.stringify({
        query: `best free public API for: ${toolDescription}. Show endpoint URL, request format, and example response.`,
        search_depth: 'advanced',
        include_answer: true,
        max_results: 5,
      }),
    });

    if (!response.ok) {
      console.error(`Tavily search failed (${response.status})`);
      return '';
    }

    const data = await response.json() as any;

    // Build context from search results
    let context = '';
    if (data.answer) {
      context += `Web search summary:\n${data.answer}\n\n`;
    }
    if (data.results && Array.isArray(data.results)) {
      for (const result of data.results) {
        context += `Source: ${result.title} (${result.url})\n${result.content}\n\n`;
      }
    }

    return context;
  } catch (err) {
    console.error('Tavily search error:', err);
    return '';
  }
}

// Generate implementation code for a single tool using codex with Tavily search context
async function generateToolCode(
  tool: ToolNodeData,
  apiKey: string,
  tavilyApiKey?: string
): Promise<GeneratedTool> {
  const paramSignature = (tool.parameters || [])
    .map((p) => `${p.name}: ${p.type}${p.required ? '' : '?'} — ${p.description}`)
    .join('\n  ');

  // Search for relevant APIs using Tavily if key is available
  let searchContext = '';
  if (tavilyApiKey) {
    searchContext = await searchForAPIs(tool.functionDescription, tavilyApiKey);
  }

  const prompt = `Generate a complete, working JavaScript function for a Cloudflare Workers environment (no Node.js APIs, only fetch and Web APIs). Do NOT use TypeScript syntax — no type annotations, no interfaces, no generics, no "as" casts.

Tool name: ${tool.functionName}
Description: ${tool.functionDescription}
Parameters:
  ${paramSignature || 'None'}

${searchContext ? `Relevant API documentation from web search:\n${searchContext}\n` : ''}Requirements:
- Standalone async function, single object argument with the parameters above
- Return a JSON-serializable result
- Use fetch() for any HTTP calls
- Include proper error handling with try/catch
${tool.hasApiKey
    ? `- An API key IS provided for this tool. Access it via args.__apiKey (injected at runtime). You may use APIs that require authentication.
- IMPORTANT for web search tools: If the tool is for web searching, use the Tavily Search API (POST https://api.tavily.com/search with Authorization: Bearer header). The args.__apiKey will be a Tavily API key. Do NOT send it to OpenAI or any other provider.`
    : `- No API key is provided for this tool. You MUST NOT use any API that requires a private/secret API key. Only use completely free, public, no-auth APIs, or implement the functionality locally using pure JavaScript (e.g. generate PDFs/files programmatically without calling an external service). If the task is web search, use a free search API like DuckDuckGo HTML (fetch https://html.duckduckgo.com/html/?q=... and parse the results) or Wikipedia API. If the task is PDF generation, build the PDF in-memory using raw PDF format strings.`
}
- If the function produces files, images, PDFs, or binary data, return them in a __files__ array alongside the result. Each file object must have: { filename: string, mimeType: string, data: string (base64-encoded), displayType: 'image' | 'download' }. Example: return { result: "Generated report", __files__: [{ filename: "report.pdf", mimeType: "application/pdf", data: base64String, displayType: "download" }] }
- Define a single async function named "${tool.functionName}" (do NOT use export keyword, just declare the function)

Respond with ONLY the JavaScript code, no markdown fences, no explanation.`;

  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-5.3-codex',
      input: prompt,
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    if (response.status === 401) throw new Error('Invalid OpenAI API key.');
    if (response.status === 429) throw new Error('Rate limited. Try again shortly.');
    throw new Error(`Code generation failed (${response.status}): ${errText}`);
  }

  const data = await response.json() as any;
  let code = data.output?.[0]?.content?.[0]?.text || '';

  // Clean up any markdown fences that might have slipped through
  code = code.replace(/^```(?:typescript|ts|javascript|js)?\n?/gm, '').replace(/```$/gm, '').trim();

  return { name: tool.functionName, code };
}

// POST /api/train handler
export async function trainHandler(c: Context<AppContext>) {
  const apiKey = c.req.header('X-API-Key');
  if (!apiKey) {
    return c.json({ error: 'X-API-Key header is required.' }, 401);
  }

  if (!apiKey.startsWith('sk-')) {
    return c.json({ error: 'Invalid API key format.' }, 401);
  }

  const tavilyApiKey = c.req.header('X-Tavily-Key') || '';

  let body: { workflowConfig: { nodes: any[]; edges: any[] } };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: 'Invalid request body' }, 400);
  }

  const nodes = body.workflowConfig?.nodes || [];
  const toolNodes = nodes.filter((n: any) => n.data?.type === 'agentTool' && n.data.parameters?.functionName);

  if (toolNodes.length === 0) {
    return c.json({ error: 'No agent tool nodes found in workflow.' }, 400);
  }

  const results: GeneratedTool[] = [];

  for (const node of toolNodes) {
    try {
      const tool: ToolNodeData = {
        functionName: node.data.parameters.functionName,
        functionDescription: node.data.parameters.functionDescription || '',
        hasApiKey: !!(node.data.parameters.apiKey),
        parameters: (node.data.parameters.parameters || []).map((p: any) => ({
          name: p.name,
          type: p.type,
          description: p.description,
          required: p.required,
        })),
      };

      const generated = await generateToolCode(tool, apiKey, tavilyApiKey || undefined);
      results.push(generated);

      // Store the generated code for use in tool execution
      generatedToolCode.set(tool.functionName, generated.code);
    } catch (err: any) {
      results.push({
        name: node.data.parameters.functionName,
        code: `// Error generating code: ${err.message}`,
      });
    }
  }

  return c.json({ tools: results });
}
