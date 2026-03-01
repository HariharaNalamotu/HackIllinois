const key = process.env.OPENAI_API_KEY;

// Test 1: gpt-5.3-codex via chat completions
console.log("--- Test 1: gpt-5.3-codex via /v1/chat/completions ---");
let res = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ model: 'gpt-5.3-codex', messages: [{ role: 'user', content: 'Say hi' }], max_tokens: 10 }),
});
console.log(`Status: ${res.status}`);
console.log(await res.text());

// Test 2: gpt-5.3-codex via responses API
console.log("\n--- Test 2: gpt-5.3-codex via /v1/responses ---");
res = await fetch('https://api.openai.com/v1/responses', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ model: 'gpt-5.3-codex', input: 'Say hi' }),
});
console.log(`Status: ${res.status}`);
const text2 = await res.text();
console.log(text2.slice(0, 500));

// Test 3: gpt-5.2 via chat completions
console.log("\n--- Test 3: gpt-5.2 via /v1/chat/completions ---");
res = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ model: 'gpt-5.2', messages: [{ role: 'user', content: 'Say hi' }], max_tokens: 10 }),
});
console.log(`Status: ${res.status}`);
console.log(await res.text());
