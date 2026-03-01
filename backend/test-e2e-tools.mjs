#!/usr/bin/env node
// End-to-end test: reproduces the "web search → write markdown → convert to PDF" workflow
//
// Usage:
//   OPENAI_API_KEY=sk-... node test-e2e-tools.mjs
//   OPENAI_API_KEY=sk-... TAVILY_API_KEY=tvly-... node test-e2e-tools.mjs
//   OPENAI_API_KEY=sk-... BACKEND_URL=http://localhost:8787 node test-e2e-tools.mjs
//
// What it tests:
//   1. Code generation for search_web and write_pdf tools
//   2. That generated code actually executes (no "Code generation from strings disallowed")
//   3. That the /api/chat SSE stream delivers tokens incrementally
//   4. That tool_call and file_output events are emitted properly

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8787';

if (!OPENAI_API_KEY) {
  console.error('Usage: OPENAI_API_KEY=sk-... node test-e2e-tools.mjs');
  process.exit(1);
}

// ─── Colors ──────────────────────────────────────────────────────────
const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';

function pass(msg) { console.log(`  ${GREEN}✓${RESET} ${msg}`); }
function fail(msg) { console.log(`  ${RED}✗${RESET} ${msg}`); }
function info(msg) { console.log(`  ${CYAN}ℹ${RESET} ${msg}`); }
function warn(msg) { console.log(`  ${YELLOW}⚠${RESET} ${msg}`); }
function header(msg) { console.log(`\n${BOLD}${msg}${RESET}`); }

// ─── Workflow config that mirrors what the frontend sends ────────────
const WORKFLOW_CONFIG = {
  nodes: [
    {
      id: 'llm-1',
      type: 'custom',
      position: { x: 300, y: 200 },
      data: {
        type: 'agenticLLM',
        label: 'Agentic LLM',
        parameters: {
          subAgentModel: 'gpt-4o',
          subAgentPrompt: 'You are a helpful research assistant. When asked to create documents, first search the web for information, then generate the document using available tools. Always use your tools — never say you cannot do something that a tool can handle.',
        },
      },
    },
    {
      id: 'tool-search',
      type: 'custom',
      position: { x: 100, y: 400 },
      data: {
        type: 'agentTool',
        label: 'Web Search',
        parameters: {
          functionName: 'search_web',
          functionDescription: 'Search the web for current information on any topic. Returns relevant search results with titles, URLs, and content snippets.',
          parameters: [
            { id: 'p1', name: 'query', type: 'string', description: 'The search query', required: true },
          ],
          apiKey: TAVILY_API_KEY || '',
        },
      },
    },
    {
      id: 'tool-pdf',
      type: 'custom',
      position: { x: 500, y: 400 },
      data: {
        type: 'agentTool',
        label: 'PDF Generator',
        parameters: {
          functionName: 'generate_pdf',
          functionDescription: 'Generate a PDF document from markdown text content. Returns the PDF as a downloadable file.',
          parameters: [
            { id: 'p2', name: 'markdown_content', type: 'string', description: 'Markdown formatted content for the PDF', required: true },
            { id: 'p3', name: 'title', type: 'string', description: 'Document title', required: true },
          ],
          apiKey: '',
        },
      },
    },
    {
      id: 'output-1',
      type: 'custom',
      position: { x: 300, y: 600 },
      data: { type: 'output', label: 'Output' },
    },
  ],
  edges: [
    { id: 'e1', source: 'llm-1', target: 'tool-search', sourceHandle: 'bottom', targetHandle: 'top' },
    { id: 'e2', source: 'llm-1', target: 'tool-pdf', sourceHandle: 'bottom', targetHandle: 'top' },
    { id: 'e3', source: 'llm-1', target: 'output-1', sourceHandle: 'bottom', targetHandle: 'top' },
  ],
};

let totalPassed = 0;
let totalFailed = 0;

function assert(condition, passMsg, failMsg) {
  if (condition) {
    pass(passMsg);
    totalPassed++;
  } else {
    fail(failMsg || passMsg);
    totalFailed++;
  }
}

// ─── Test 1: Backend health check ───────────────────────────────────
async function testHealth() {
  header('Test 1: Backend health check');
  try {
    const res = await fetch(`${BACKEND_URL}/api/health`);
    assert(res.ok, `Backend is running at ${BACKEND_URL}`, `Backend not reachable at ${BACKEND_URL} (status ${res.status})`);
    if (!res.ok) {
      console.error(`\n${RED}Backend is not running. Start it with: cd backend && npm run dev${RESET}\n`);
      process.exit(1);
    }
  } catch (err) {
    fail(`Cannot connect to ${BACKEND_URL}: ${err.message}`);
    console.error(`\n${RED}Backend is not running. Start it with: cd backend && npm run dev${RESET}\n`);
    process.exit(1);
  }
}

// ─── Test 2: Code generation (POST /api/train) ─────────────────────
async function testCodeGeneration() {
  header('Test 2: Code generation via /api/train');

  const res = await fetch(`${BACKEND_URL}/api/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': OPENAI_API_KEY,
      ...(TAVILY_API_KEY ? { 'X-Tavily-Key': TAVILY_API_KEY } : {}),
    },
    body: JSON.stringify({ workflowConfig: WORKFLOW_CONFIG }),
  });

  assert(res.ok, `POST /api/train returned ${res.status}`, `POST /api/train failed with ${res.status}`);

  const data = await res.json();
  assert(Array.isArray(data.tools), `Got ${data.tools?.length || 0} tool(s) generated`, 'No tools array in response');

  for (const tool of data.tools || []) {
    const hasCode = tool.code && !tool.code.startsWith('// Error');
    assert(hasCode, `${tool.name}: code generated (${tool.code?.length || 0} chars)`, `${tool.name}: code generation FAILED — ${tool.code?.slice(0, 100)}`);

    if (hasCode) {
      // Check the code doesn't have obvious issues
      const hasFunctionDef = tool.code.includes(`function ${tool.name}`) || tool.code.includes(`${tool.name} =`);
      assert(hasFunctionDef, `${tool.name}: contains function definition`, `${tool.name}: missing function definition for "${tool.name}"`);
    }
  }

  return data.tools || [];
}

// ─── Test 3: Tool execution locally (simulates what executeTool does) ─
async function testLocalToolExecution(tools) {
  header('Test 3: Local tool execution (new Function)');
  info('This tests whether generated code can run via new Function() — the exact error path');

  for (const tool of tools) {
    if (!tool.code || tool.code.startsWith('// Error')) {
      warn(`Skipping ${tool.name} — no valid code`);
      continue;
    }

    // Apply the same stripping as tool_service.ts
    const cleanCode = tool.code
      .replace(/^export\s+(default\s+)?/gm, '')
      .replace(/^module\.exports\s*=.*/gm, '')
      .replace(/\((\w+)\s*:\s*\{[^}]*\}\s*\)/g, '($1)')
      .replace(/(\w+)\s*:\s*(string|number|boolean|any|void|object|unknown|never)\s*(?=[,)=])/g, '$1')
      .replace(/\)\s*:\s*Promise<[^>]*>\s*\{/g, ') {')
      .replace(/\)\s*:\s*\w[\w<>,\s|[\]{}]*\s*\{/g, ') {')
      .replace(/^(interface|type)\s+\w+[\s\S]*?^}/gm, '')
      .replace(/\s+as\s+\w+(\[\])?/g, '')
      .replace(/(const|let|var)\s+(\w+)\s*:\s*[^=]+=\s*/g, '$1 $2 = ');

    try {
      // This is exactly what tool_service.ts does:
      const fn = new Function('args', 'fetch', `
        return (async () => {
          ${cleanCode}
          return typeof ${tool.name} === 'function' ? ${tool.name}(args) : { error: 'Function not found' };
        })();
      `);
      pass(`${tool.name}: new Function() creation succeeded`);

      // Try executing with test args
      const testArgs = tool.name === 'search_web'
        ? { query: 'test query', __apiKey: TAVILY_API_KEY || '' }
        : { markdown_content: '# Test\nHello world', title: 'Test Doc' };

      info(`${tool.name}: executing with test args...`);
      const result = await Promise.race([
        fn(testArgs, globalThis.fetch),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout after 15s')), 15000)),
      ]);

      const resultStr = JSON.stringify(result);
      assert(
        result && !result.error,
        `${tool.name}: executed successfully (${resultStr.slice(0, 150)}${resultStr.length > 150 ? '...' : ''})`,
        `${tool.name}: execution returned error — ${resultStr.slice(0, 200)}`
      );
    } catch (err) {
      fail(`${tool.name}: new Function() FAILED — ${err.message}`);
      if (err.message.includes('disallowed')) {
        info('This is the "Code generation from strings disallowed" error');
        info('In Cloudflare Workers this is fixed by [[unsafe.bindings]] type = "unsafe_eval" in wrangler.toml');
      }
    }
  }
}

// ─── Test 4: Full SSE streaming chat ────────────────────────────────
async function testStreamingChat() {
  header('Test 4: Full SSE streaming chat (/api/chat)');
  info('Sending: "Generate a PDF Document about Jeffrey Epstein."');

  const startTime = Date.now();
  const events = { token: [], tool_call: [], file_output: [], error: [], subagent_step: [], rlaif_score: [] };
  let firstTokenTime = null;
  let done = false;

  try {
    const res = await fetch(`${BACKEND_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': OPENAI_API_KEY,
      },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'Generate a PDF Document about Jeffrey Epstein.' }],
        workflowConfig: WORKFLOW_CONFIG,
      }),
    });

    assert(res.ok, `POST /api/chat returned ${res.status}`, `POST /api/chat failed with ${res.status}`);
    assert(
      res.headers.get('content-type')?.includes('text/event-stream'),
      'Response Content-Type is text/event-stream',
      `Unexpected Content-Type: ${res.headers.get('content-type')}`
    );

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let chunkCount = 0;

    while (true) {
      const { done: streamDone, value } = await reader.read();
      if (streamDone) break;

      chunkCount++;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          done = true;
          continue;
        }

        try {
          const parsed = JSON.parse(data);
          const type = parsed.type;
          if (events[type]) {
            events[type].push(parsed);
          }

          if (type === 'token' && !firstTokenTime) {
            firstTokenTime = Date.now();
          }

          // Log key events as they arrive
          if (type === 'tool_call') {
            info(`Tool call: ${parsed.name}(${parsed.arguments?.slice(0, 80)}...)`);
          } else if (type === 'file_output') {
            info(`File output: ${parsed.file?.filename} (${parsed.file?.mimeType}, ${parsed.file?.data?.length || 0} chars base64)`);
          } else if (type === 'error') {
            warn(`Error event: ${parsed.content}`);
          }
        } catch {}
      }
    }

    const elapsed = Date.now() - startTime;
    const ttft = firstTokenTime ? firstTokenTime - startTime : null;

    // Validate results
    info(`Total chunks received: ${chunkCount}`);
    info(`Total time: ${(elapsed / 1000).toFixed(1)}s`);
    if (ttft) info(`Time to first token: ${(ttft / 1000).toFixed(1)}s`);

    assert(done, 'Received [DONE] sentinel', 'Stream ended without [DONE]');
    assert(events.token.length > 0, `Got ${events.token.length} token events`, 'No token events — streaming is broken');
    assert(chunkCount > 1, `Stream delivered ${chunkCount} chunks (not all-at-once)`, 'Only 1 chunk — stream is NOT incremental (all-at-once)');

    // Check streaming quality
    if (events.token.length > 0 && chunkCount > 1) {
      const tokensPerChunk = events.token.length / chunkCount;
      if (tokensPerChunk < 5) {
        pass(`Good streaming granularity (~${tokensPerChunk.toFixed(1)} tokens/chunk)`);
      } else {
        warn(`Coarse streaming (~${tokensPerChunk.toFixed(1)} tokens/chunk) — may appear laggy`);
      }
    }

    // Tool call validation
    assert(events.tool_call.length > 0, `Got ${events.tool_call.length} tool_call events`, 'No tool_call events — model did not use tools');

    if (events.tool_call.length > 0) {
      const toolNames = events.tool_call.map(tc => tc.name);
      const hasSearch = toolNames.includes('search_web');
      const hasPdf = toolNames.includes('generate_pdf');
      assert(hasSearch, 'Model called search_web tool', 'Model did NOT call search_web');
      assert(hasPdf, 'Model called generate_pdf tool', 'Model did NOT call generate_pdf');
    }

    // Error check
    if (events.error.length > 0) {
      for (const err of events.error) {
        fail(`Error from backend: ${err.content}`);
        if (err.content.includes('disallowed')) {
          info('ROOT CAUSE: "Code generation from strings disallowed" — check wrangler.toml [[unsafe.bindings]] type = "unsafe_eval"');
        }
      }
    } else {
      pass('No error events in stream');
    }

    // File output validation
    if (events.file_output.length > 0) {
      pass(`Got ${events.file_output.length} file_output events`);
      for (const fo of events.file_output) {
        const file = fo.file;
        assert(file.filename, `File has filename: ${file.filename}`, 'File missing filename');
        assert(file.mimeType, `File has mimeType: ${file.mimeType}`, 'File missing mimeType');
        assert(file.data && file.data.length > 0, `File has data (${file.data.length} chars)`, 'File has no data');
      }
    } else {
      warn('No file_output events — PDF generation may have failed or model skipped it');
    }

    // Print the full assistant response
    const fullText = events.token.map(t => t.content).join('');
    header('Assistant response:');
    console.log(DIM + fullText.slice(0, 500) + (fullText.length > 500 ? '...' : '') + RESET);

  } catch (err) {
    fail(`Chat stream failed: ${err.message}`);
  }
}

// ─── Test 5: Verify wrangler.toml config ────────────────────────────
async function testWranglerConfig() {
  header('Test 5: Verify wrangler.toml configuration');

  try {
    const fs = await import('fs');
    const toml = fs.readFileSync(new URL('./wrangler.toml', import.meta.url), 'utf-8');

    const hasUnsafeBinding = toml.includes('[[unsafe.bindings]]');
    const hasUnsafeEvalType = toml.includes('type = "unsafe_eval"');
    const hasOldTypo = toml.includes('[unsafe.capabilties]');
    const hasOldCapabilities = toml.includes('[unsafe.capabilities]') && !toml.includes('[[unsafe.bindings]]');

    assert(hasUnsafeBinding, 'wrangler.toml has [[unsafe.bindings]]', 'wrangler.toml missing [[unsafe.bindings]]');
    assert(hasUnsafeEvalType, 'Has type = "unsafe_eval" binding', 'Missing type = "unsafe_eval" — new Function() will fail');
    assert(!hasOldTypo, 'No misspelled [unsafe.capabilties]', 'wrangler.toml has MISSPELLED [unsafe.capabilties] — this was the original bug!');
    assert(!hasOldCapabilities, 'Not using deprecated [unsafe.capabilities]', 'Using old [unsafe.capabilities] which wrangler 3.x ignores');
  } catch (err) {
    warn(`Could not read wrangler.toml: ${err.message}`);
  }
}

// ─── Run all tests ──────────────────────────────────────────────────
async function main() {
  console.log(`${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}`);
  console.log(`${BOLD}║  E2E Test: Web Search → PDF Generation Workflow             ║${RESET}`);
  console.log(`${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}`);
  console.log(`Backend: ${BACKEND_URL}`);
  console.log(`API Key: ${OPENAI_API_KEY.slice(0, 8)}...`);
  console.log(`Tavily:  ${TAVILY_API_KEY ? TAVILY_API_KEY.slice(0, 8) + '...' : '(not set)'}`);

  await testWranglerConfig();
  await testHealth();
  const tools = await testCodeGeneration();
  await testLocalToolExecution(tools);
  await testStreamingChat();

  // Summary
  header('═══════════════════════ SUMMARY ═══════════════════════');
  console.log(`  ${GREEN}Passed: ${totalPassed}${RESET}`);
  console.log(`  ${totalFailed > 0 ? RED : DIM}Failed: ${totalFailed}${RESET}`);
  console.log();

  if (totalFailed > 0) {
    console.log(`${YELLOW}Common fixes:${RESET}`);
    console.log(`  1. wrangler.toml: ensure [[unsafe.bindings]] with type = "unsafe_eval" and name = "UNSAFE_EVAL"`);
    console.log(`  2. Restart backend after wrangler.toml changes: cd backend && npm run dev`);
    console.log(`  3. Set TAVILY_API_KEY for web search to work`);
    console.log(`  4. Ensure you ran "Train Model" in the frontend before testing`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(`\n${RED}Fatal error: ${err.message}${RESET}`);
  process.exit(1);
});
