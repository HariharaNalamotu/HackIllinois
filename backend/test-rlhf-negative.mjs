#!/usr/bin/env node
// Test: RLHF negative feedback actually changes model behavior
//
// Usage:
//   OPENAI_API_KEY=sk-... node test-rlhf-negative.mjs
//   OPENAI_API_KEY=sk-... BACKEND_URL=http://localhost:8787 node test-rlhf-negative.mjs
//
// What it tests:
//   1. Submit a chat message and capture the response
//   2. Submit negative feedback with criticism about that response
//   3. Ask the same question again
//   4. Verify the new response avoids the criticized behavior

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8787';

if (!OPENAI_API_KEY) {
  console.error('Usage: OPENAI_API_KEY=sk-... node test-rlhf-negative.mjs');
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

let totalPassed = 0;
let totalFailed = 0;

function assert(condition, passMsg, failMsg) {
  if (condition) {
    pass(passMsg);
    totalPassed++;
    return true;
  } else {
    fail(failMsg || passMsg);
    totalFailed++;
    return false;
  }
}

// ─── Minimal workflow config (just an LLM node, no tools) ────────────
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
          subAgentModel: 'gpt-4o-mini',
          subAgentPrompt: 'You are a helpful assistant. Keep responses concise (2-4 sentences max).',
        },
      },
    },
    {
      id: 'rlhf-1',
      type: 'custom',
      position: { x: 300, y: 400 },
      data: {
        type: 'rlhf',
        label: 'RLHF',
        parameters: { iterations: 3 },
      },
    },
  ],
  edges: [
    { id: 'e1', source: 'llm-1', target: 'rlhf-1', sourceHandle: 'bottom', targetHandle: 'top' },
  ],
};

// ─── Helper: read full response from SSE stream ──────────────────────
async function chatAndCollect(messages) {
  const res = await fetch(`${BACKEND_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': OPENAI_API_KEY,
    },
    body: JSON.stringify({ messages, workflowConfig: WORKFLOW_CONFIG }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat failed (${res.status}): ${text}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullContent = '';

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
        if (parsed.type === 'token') {
          fullContent += parsed.content;
        }
        if (parsed.type === 'error') {
          throw new Error(`Stream error: ${parsed.content}`);
        }
      } catch (e) {
        if (e.message.startsWith('Stream error')) throw e;
      }
    }
  }

  return fullContent;
}

// ─── Test 1: Health check ────────────────────────────────────────────
async function testHealth() {
  header('Test 1: Backend health check');
  try {
    const res = await fetch(`${BACKEND_URL}/api/health`);
    assert(res.ok, `Backend running at ${BACKEND_URL}`, `Backend not reachable at ${BACKEND_URL}`);
    if (!res.ok) process.exit(1);
  } catch (err) {
    fail(`Cannot connect to ${BACKEND_URL}: ${err.message}`);
    console.error(`\n${RED}Start the backend: cd backend && npm run dev${RESET}\n`);
    process.exit(1);
  }
}

// ─── Test 2: Start RLHF session ─────────────────────────────────────
async function testStartSession() {
  header('Test 2: Start RLHF training session');

  const res = await fetch(`${BACKEND_URL}/api/train/rlhf`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': OPENAI_API_KEY,
    },
    body: JSON.stringify({ workflowConfig: WORKFLOW_CONFIG }),
  });

  const data = await res.json();
  assert(res.ok && data.success, 'RLHF session started', `Failed to start session: ${JSON.stringify(data)}`);
  assert(data.session?.status === 'active', 'Session is active', `Session status: ${data.session?.status}`);
}

// ─── Test 3: The core test — negative feedback changes behavior ──────
async function testNegativeFeedbackEffect() {
  header('Test 3: Negative feedback changes model behavior');

  const TEST_QUESTION = 'Explain what a neural network is.';

  // Step 1: Get initial response
  info('Step 1: Asking the model the question...');
  const response1 = await chatAndCollect([{ role: 'user', content: TEST_QUESTION }]);
  info(`Response 1 (${response1.length} chars): "${response1.slice(0, 150)}..."`);
  assert(response1.length > 0, 'Got initial response', 'Empty initial response');

  // Step 2: Submit negative feedback — criticize a specific aspect
  // We'll tell the model its response was too technical and used jargon
  const CRITICISM = 'Too technical and uses jargon. Explain it like I am five years old. Use a simple analogy instead of technical terms.';

  info(`Step 2: Submitting negative feedback: "${CRITICISM}"`);

  const snippet = `User: ${TEST_QUESTION}\nAssistant: ${response1}`;

  // First submit the thumbs-down
  const fb1 = await fetch(`${BACKEND_URL}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messageId: 'test-msg-1',
      rating: 'down',
      conversationSnippet: snippet,
    }),
  });
  assert(fb1.ok, 'Thumbs-down feedback submitted', 'Failed to submit thumbs-down');

  // Then submit the text criticism
  const fb2 = await fetch(`${BACKEND_URL}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messageId: 'test-msg-1',
      rating: 'down',
      feedback: CRITICISM,
      conversationSnippet: snippet,
    }),
  });
  assert(fb2.ok, 'Text criticism submitted', 'Failed to submit text criticism');

  // Step 3: Verify feedback is stored
  info('Step 3: Verifying feedback is stored...');
  const feedbackRes = await fetch(`${BACKEND_URL}/api/feedback`);
  const feedbackData = await feedbackRes.json();
  const negativeEntries = feedbackData.feedback.filter(e => e.rating === 'down');
  assert(negativeEntries.length >= 1, `${negativeEntries.length} negative feedback entries stored`, 'No negative feedback stored');

  const hasSnippet = negativeEntries.some(e => e.conversationSnippet && e.conversationSnippet.includes(TEST_QUESTION));
  assert(hasSnippet, 'Negative feedback includes the conversation snippet', 'Conversation snippet not stored with feedback');

  const hasCriticism = negativeEntries.some(e => e.feedback === CRITICISM);
  assert(hasCriticism, 'Criticism text is stored', 'Criticism text not stored');

  // Step 4: Ask the same question again — model should adapt
  info('Step 4: Asking the SAME question again after negative feedback...');
  const response2 = await chatAndCollect([{ role: 'user', content: TEST_QUESTION }]);
  info(`Response 2 (${response2.length} chars): "${response2.slice(0, 150)}..."`);
  assert(response2.length > 0, 'Got second response', 'Empty second response');

  // Step 5: Analyze whether the model changed behavior
  header('Test 4: Analyzing behavior change');

  const r1Lower = response1.toLowerCase();
  const r2Lower = response2.toLowerCase();

  // The responses should be meaningfully different
  assert(
    response1 !== response2,
    'Second response is different from first',
    'Responses are IDENTICAL — feedback had no effect'
  );

  // Check if the second response uses simpler language / analogies
  // (since we asked for "explain like I'm five" and "simple analogy")
  const simpleWords = ['like', 'imagine', 'think of', 'pretend', 'simple', 'easy', 'basically', 'analogy', 'example', 'picture'];
  const jargonWords = ['neurons', 'weights', 'backpropagation', 'gradient', 'activation function', 'perceptron', 'epoch', 'optimization'];

  const r2SimpleCount = simpleWords.filter(w => r2Lower.includes(w)).length;
  const r1SimpleCount = simpleWords.filter(w => r1Lower.includes(w)).length;
  const r2JargonCount = jargonWords.filter(w => r2Lower.includes(w)).length;
  const r1JargonCount = jargonWords.filter(w => r1Lower.includes(w)).length;

  info(`Response 1 — simple words: ${r1SimpleCount}, jargon words: ${r1JargonCount}`);
  info(`Response 2 — simple words: ${r2SimpleCount}, jargon words: ${r2JargonCount}`);

  // The second response should use more simple language OR less jargon
  const improved = r2SimpleCount > r1SimpleCount || r2JargonCount < r1JargonCount || r2SimpleCount >= 2;
  assert(
    improved,
    'Second response uses simpler language (more analogies or less jargon)',
    'Second response did NOT become simpler — feedback may not be injected into prompt correctly'
  );

  // Check the session advanced
  const sessionRes = await fetch(`${BACKEND_URL}/api/feedback`);
  const sessionData = await sessionRes.json();
  if (sessionData.session) {
    info(`Session iteration: ${sessionData.session.currentIteration}/${sessionData.session.iterations}`);
    assert(
      sessionData.session.currentIteration >= 2,
      'Session iteration advanced from feedback',
      `Session iteration unexpectedly at ${sessionData.session.currentIteration}`
    );
  }

  // Print side-by-side comparison
  header('═══ Response Comparison ═══');
  console.log(`\n${YELLOW}BEFORE feedback:${RESET}`);
  console.log(DIM + response1.slice(0, 400) + (response1.length > 400 ? '...' : '') + RESET);
  console.log(`\n${YELLOW}AFTER feedback ("${CRITICISM}"):${RESET}`);
  console.log(DIM + response2.slice(0, 400) + (response2.length > 400 ? '...' : '') + RESET);
}

// ─── Run all tests ──────────────────────────────────────────────────
async function main() {
  console.log(`${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}`);
  console.log(`${BOLD}║  RLHF Negative Feedback Test                                ║${RESET}`);
  console.log(`${BOLD}║  Verifies the model avoids patterns from rejected responses  ║${RESET}`);
  console.log(`${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}`);
  console.log(`Backend: ${BACKEND_URL}`);
  console.log(`API Key: ${OPENAI_API_KEY.slice(0, 8)}...`);

  await testHealth();
  await testStartSession();
  await testNegativeFeedbackEffect();

  // Summary
  header('═══════════════════════ SUMMARY ═══════════════════════');
  console.log(`  ${GREEN}Passed: ${totalPassed}${RESET}`);
  console.log(`  ${totalFailed > 0 ? RED : DIM}Failed: ${totalFailed}${RESET}`);
  console.log();

  if (totalFailed > 0) {
    console.log(`${YELLOW}Possible issues:${RESET}`);
    console.log(`  1. Ensure backend is running: cd backend && npm run dev`);
    console.log(`  2. Check that getPositiveFeedbackContext() includes negative feedback`);
    console.log(`  3. Check that buildNegativeFeedbackPrompt() includes conversationSnippet`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(`\n${RED}Fatal error: ${err.message}${RESET}`);
  process.exit(1);
});
