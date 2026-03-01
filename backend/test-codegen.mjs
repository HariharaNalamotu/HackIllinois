#!/usr/bin/env node
// Test script for tool code generation
// Usage: OPENAI_API_KEY=sk-... node test-codegen.mjs

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error('Set OPENAI_API_KEY env var first: OPENAI_API_KEY=sk-... node test-codegen.mjs');
  process.exit(1);
}

const tools = [
  {
    functionName: 'search_web',
    functionDescription: 'Search the web for information',
    parameters: [
      { name: 'query', type: 'string', description: 'Search query', required: true },
    ],
  },
  {
    functionName: 'write_pdf',
    functionDescription: 'Generate a PDF document from text content',
    parameters: [
      { name: 'content', type: 'string', description: 'Text content for the PDF', required: true },
      { name: 'title', type: 'string', description: 'Document title', required: true },
    ],
  },
];

function buildPrompt(tool) {
  const paramSignature = (tool.parameters || [])
    .map((p) => `${p.name}: ${p.type}${p.required ? '' : '?'} — ${p.description}`)
    .join('\n  ');

  return `Generate a complete, working JavaScript function for a Cloudflare Workers environment (no Node.js APIs, only fetch and Web APIs). Do NOT use TypeScript syntax — no type annotations, no interfaces, no generics, no "as" casts.

Tool name: ${tool.functionName}
Description: ${tool.functionDescription}
Parameters:
  ${paramSignature || 'None'}

Requirements:
- Standalone async function, single object argument with the parameters above
- Return a JSON-serializable result
- Use fetch() for any HTTP calls
- Include proper error handling with try/catch
- Use real public APIs where possible
- If the tool needs an API key for authentication, access it via args.__apiKey (it will be injected at runtime)
- If the function produces files, images, PDFs, or binary data, return them in a __files__ array alongside the result. Each file object must have: { filename: string, mimeType: string, data: string (base64-encoded), displayType: 'image' | 'download' }. Example: return { result: "Generated report", __files__: [{ filename: "report.pdf", mimeType: "application/pdf", data: base64String, displayType: "download" }] }
- Define a single async function named "${tool.functionName}" (do NOT use export keyword, just declare the function)

Respond with ONLY the TypeScript code, no markdown fences, no explanation.`;
}

function cleanCode(raw) {
  return raw
    .replace(/^```(?:typescript|ts|javascript|js)?\n?/gm, '')
    .replace(/```$/gm, '')
    .replace(/^export\s+(default\s+)?/gm, '')
    .replace(/^module\.exports\s*=.*/gm, '')
    .trim();
}

// Strip TypeScript annotations for JS runtime
function stripTypeAnnotations(code) {
  return code
    // Strip TS function parameter type annotations: (args: { foo: string }) -> (args)
    .replace(/\((\w+)\s*:\s*\{[^}]*\}\s*\)/g, '($1)')
    // Strip simple param types: (x: string, y: number) -> (x, y)
    .replace(/(\w+)\s*:\s*(string|number|boolean|any|void|object|unknown|never)\s*(?=[,)=])/g, '$1')
    // Strip return type annotations: ): Promise<...> { -> ) {
    .replace(/\)\s*:\s*Promise<[^>]*>\s*\{/g, ') {')
    .replace(/\)\s*:\s*\w[\w<>,\s|[\]{}]*\s*\{/g, ') {')
    // Strip interface/type declarations
    .replace(/^(interface|type)\s+\w+[\s\S]*?^}/gm, '')
    // Strip "as Type" casts
    .replace(/\s+as\s+\w+(\[\])?/g, '')
    // Strip generic type params on variables
    .replace(/(const|let|var)\s+(\w+)\s*:\s*[^=]+=\s*/g, '$1 $2 = ')
    .trim();
}

function validateCode(name, code) {
  const errors = [];

  if (!code.includes(`function ${name}`) && !code.includes(`${name} =`)) {
    errors.push(`Function "${name}" not found in generated code`);
  }

  if (code.includes(': string') || code.includes(': number') || code.includes(': boolean')) {
    errors.push('Contains TypeScript type annotations (needs stripping for runtime)');
  }
  if (code.match(/\binterface\s+\w+/)) {
    errors.push('Contains TypeScript interface declarations');
  }

  try {
    new Function('args', 'fetch', `
      return (async () => {
        ${code}
        return typeof ${name} === 'function' ? ${name}(args) : { error: 'not found' };
      })();
    `);
  } catch (err) {
    errors.push(`Syntax error: ${err.message}`);
  }

  return errors;
}

async function generateToolCode(tool, maxAttempts = 3) {
  const prompt = buildPrompt(tool);

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    console.log(`\n--- ${tool.functionName}: Attempt ${attempt}/${maxAttempts} ---`);

    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-5.3-codex',
        input: prompt,
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error(`API error (${response.status}):`, errText.slice(0, 500));
      if (response.status === 401) {
        console.error('Invalid API key. Exiting.');
        process.exit(1);
      }
      continue;
    }

    const data = await response.json();
    const rawCode = data.output?.[0]?.content?.[0]?.text || '';

    console.log('Raw output length:', rawCode.length);
    console.log('First 300 chars:', rawCode.slice(0, 300));

    let code = cleanCode(rawCode);
    console.log('\nAfter cleaning, length:', code.length);

    let errors = validateCode(tool.functionName, code);

    if (errors.some((e) => e.includes('TypeScript'))) {
      console.log('Stripping TypeScript annotations...');
      code = stripTypeAnnotations(code);
      errors = validateCode(tool.functionName, code);
    }

    if (errors.length === 0) {
      console.log('VALID code generated!');
      console.log('\n--- Generated code ---');
      console.log(code);
      console.log('--- End ---\n');
      return { name: tool.functionName, code, attempts: attempt };
    }

    console.log('Validation errors:', errors);

    if (attempt < maxAttempts) {
      console.log('Retrying...');
    }
  }

  console.error(`FAILED after ${maxAttempts} attempts for ${tool.functionName}`);
  return { name: tool.functionName, code: null, attempts: maxAttempts };
}

async function main() {
  console.log('Testing tool code generation...\n');
  console.log('Model: gpt-5.3-codex (Responses API)');
  console.log('Tools to generate:', tools.map((t) => t.functionName).join(', '));

  const results = [];
  for (const tool of tools) {
    const result = await generateToolCode(tool);
    results.push(result);
  }

  console.log('\n========== SUMMARY ==========');
  for (const r of results) {
    console.log(`${r.name}: ${r.code ? `OK (${r.attempts} attempt(s))` : 'FAILED'}`);
  }
}

main().catch(console.error);
