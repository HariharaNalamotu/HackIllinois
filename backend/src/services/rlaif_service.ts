// RLAIF service — fully automated AI-driven evaluation
// Generates test prompts via AI, gets model responses, and evaluates them — no human in the loop

import type { Context } from 'hono';
import type { AppContext } from '../index';

export interface RLAIFScore {
  helpfulness: number;
  accuracy: number;
  safety: number;
  overall: number;
}

interface RLAIFSession {
  evaluatorModel: string;
  resolvedModel: string;
  iterations: number;
  evaluationCount: number;
  status: 'active' | 'completed' | 'error';
  startedAt: string;
  averageScore: number;
  calibrated: boolean;
  systemPrompt: string;
  targetModel: string;
}

// Pass model names directly to the OpenAI API — no remapping
function resolveModel(model: string): string {
  return model || 'gpt-4o-mini';
}

// GPT-5 models use max_completion_tokens instead of max_tokens
function buildParams(model: string, opts: { temperature?: number; max_tokens?: number; response_format?: any }): Record<string, any> {
  const params: Record<string, any> = {};
  if (opts.temperature !== undefined) {
    params.temperature = opts.temperature;
  }
  if (opts.max_tokens !== undefined) {
    // GPT-5+ requires max_completion_tokens instead of max_tokens
    if (model.startsWith('gpt-5')) {
      params.max_completion_tokens = opts.max_tokens;
    } else {
      params.max_tokens = opts.max_tokens;
    }
  }
  if (opts.response_format !== undefined) {
    params.response_format = opts.response_format;
  }
  return params;
}

// In-memory store for high-scoring response patterns
const goodResponsePatterns: { prompt: string; response: string; score: number }[] = [];

// Active RLAIF session
let activeSession: RLAIFSession | null = null;

// Abort controller for stopping an active automated run
let activeAbortController: AbortController | null = null;

// All evaluation entries for tracking
interface EvaluationEntry {
  userMessage: string;
  assistantResponse: string;
  score: RLAIFScore;
  timestamp: string;
}
const evaluationHistory: EvaluationEntry[] = [];

export function getActiveRLAIFSession(): RLAIFSession | null {
  return activeSession;
}

export function getRLAIFConfig(nodes: any[]): { model: string; iterations: number } | null {
  // If there's an active session, use that
  if (activeSession && activeSession.status === 'active') {
    return {
      model: activeSession.evaluatorModel,
      iterations: activeSession.iterations,
    };
  }

  // Otherwise fall back to parsing from nodes
  const rlaifNode = nodes.find((n: any) => n.data?.type === 'rlaif');
  if (!rlaifNode) return null;

  return {
    model: rlaifNode.data.parameters?.evaluatorModel || 'gpt-5.2',
    iterations: rlaifNode.data.parameters?.iterations || 3,
  };
}

// Generate a diverse test prompt using the evaluator model
async function generateTestPrompt(
  evaluatorModel: string,
  apiKey: string,
  systemPrompt: string,
  previousPrompts: string[],
  iterationIndex: number,
  totalIterations: number,
): Promise<string> {
  const model = resolveModel(evaluatorModel);

  const categories = [
    'factual question', 'creative task', 'reasoning problem',
    'instruction following', 'edge case', 'multi-step task',
    'ambiguous request', 'domain-specific question',
  ];
  const category = categories[iterationIndex % categories.length];

  const previousList = previousPrompts.length > 0
    ? `\n\nPrevious prompts already used (do NOT repeat these):\n${previousPrompts.map((p, i) => `${i + 1}. "${p}"`).join('\n')}`
    : '';

  const genPrompt = `You are a test prompt generator for evaluating an AI assistant. The assistant has this system prompt:

"${systemPrompt}"

Generate a single, realistic user message that would test the assistant's capabilities. This should be a "${category}" type prompt.
${previousList}

Requirements:
- The prompt should be something a real user would ask
- It should be challenging enough to reveal quality differences
- It should be relevant to the assistant's configured role
- Return ONLY the user message text, nothing else — no quotes, no labels, no explanation
- Keep it under 200 characters`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: genPrompt }],
      ...buildParams(model, { temperature: 0.9, max_tokens: 200 }),
    }),
  });

  if (!response.ok) {
    throw new Error(`Prompt generation failed (${response.status})`);
  }

  const data = await response.json() as any;
  return (data.choices?.[0]?.message?.content || 'Tell me about yourself.').trim();
}

// Get a response from the target model (the model being trained/evaluated)
async function getTargetModelResponse(
  targetModel: string,
  systemPrompt: string,
  userMessage: string,
  apiKey: string,
): Promise<string> {
  const model = resolveModel(targetModel);

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
      ...buildParams(model, { temperature: 0.7, max_tokens: 1000 }),
    }),
  });

  if (!response.ok) {
    throw new Error(`Target model response failed (${response.status})`);
  }

  const data = await response.json() as any;
  return (data.choices?.[0]?.message?.content || '').trim();
}

// SSE helper
function sseEvent(type: string, data: any): string {
  return `data: ${JSON.stringify({ type, ...data })}\n\n`;
}

// POST /api/train/rlaif — fully automated RLAIF: generate prompts, get responses, evaluate — all streamed via SSE
export async function rlaifTrainHandler(c: Context<AppContext>) {
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
  const rlaifNode = nodes.find((n: any) => n.data?.type === 'rlaif');

  if (!rlaifNode) {
    return c.json({ error: 'No RLAIF node found in workflow.' }, 400);
  }

  const evaluatorModel = rlaifNode.data.parameters?.evaluatorModel || 'gpt-5.2';
  const iterations = rlaifNode.data.parameters?.iterations || 3;
  const resolvedModel = resolveModel(evaluatorModel);

  // Get system prompt and target model from the LLM node
  const llmNode = nodes.find((n: any) => n.data?.type === 'agenticLLM');
  const systemPrompt = llmNode?.data?.parameters?.subAgentPrompt || 'You are a helpful AI assistant.';
  const targetModel = llmNode?.data?.parameters?.subAgentModel || 'gpt-5.2';

  // Clear previous history
  evaluationHistory.length = 0;
  goodResponsePatterns.length = 0;

  // Cancel any previous run
  if (activeAbortController) {
    activeAbortController.abort();
  }
  const abortController = new AbortController();
  activeAbortController = abortController;

  activeSession = {
    evaluatorModel,
    resolvedModel,
    iterations,
    evaluationCount: 0,
    status: 'active',
    startedAt: new Date().toISOString(),
    averageScore: 0,
    calibrated: true,
    systemPrompt,
    targetModel,
  };

  // Create SSE stream
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  const write = async (text: string) => {
    await writer.write(encoder.encode(text));
  };

  const processAutomatedRLAIF = async () => {
    const previousPrompts: string[] = [];

    try {
      // Send session_started event
      await write(sseEvent('session_started', {
        evaluatorModel,
        resolvedModel,
        targetModel: resolveModel(targetModel),
        iterations,
        systemPrompt: systemPrompt.slice(0, 200),
      }));

      for (let i = 0; i < iterations; i++) {
        if (abortController.signal.aborted) {
          await write(sseEvent('stopped', { message: 'Training stopped by user.' }));
          if (activeSession) activeSession.status = 'completed';
          break;
        }

        // Step 1: Generate a test prompt
        await write(sseEvent('iteration_start', { iteration: i + 1, total: iterations, phase: 'generating_prompt' }));

        let testPrompt: string;
        try {
          testPrompt = await generateTestPrompt(evaluatorModel, apiKey, systemPrompt, previousPrompts, i, iterations);
        } catch (err: any) {
          await write(sseEvent('iteration_error', { iteration: i + 1, phase: 'prompt_generation', error: err.message }));
          continue;
        }
        previousPrompts.push(testPrompt);

        await write(sseEvent('prompt_generated', { iteration: i + 1, prompt: testPrompt }));

        // Step 2: Get response from the target model
        await write(sseEvent('phase', { iteration: i + 1, phase: 'getting_response' }));

        let response: string;
        try {
          response = await getTargetModelResponse(targetModel, systemPrompt, testPrompt, apiKey);
        } catch (err: any) {
          await write(sseEvent('iteration_error', { iteration: i + 1, phase: 'model_response', error: err.message }));
          continue;
        }

        await write(sseEvent('response_received', { iteration: i + 1, response: response.slice(0, 1500) }));

        // Step 3: Evaluate the response
        await write(sseEvent('phase', { iteration: i + 1, phase: 'evaluating' }));

        try {
          const score = await evaluateResponse(testPrompt, response, evaluatorModel, apiKey);
          await write(sseEvent('evaluation_complete', {
            iteration: i + 1,
            prompt: testPrompt,
            response: response.slice(0, 1500),
            score,
          }));
        } catch (err: any) {
          await write(sseEvent('iteration_error', { iteration: i + 1, phase: 'evaluation', error: err.message }));
        }
      }

      if (activeSession && activeSession.status === 'active') {
        activeSession.status = 'completed';
      }

      await write(sseEvent('training_complete', {
        totalEvaluations: activeSession?.evaluationCount || 0,
        averageScore: activeSession?.averageScore || 0,
        goodPatterns: goodResponsePatterns.length,
      }));

      await write('data: [DONE]\n\n');
    } catch (err: any) {
      try {
        await write(sseEvent('error', { content: err.message || 'Unknown error' }));
      } catch {
        // Writer may already be closed
      }
      if (activeSession) activeSession.status = 'error';
    } finally {
      activeAbortController = null;
      try {
        await writer.close();
      } catch {
        // Already closed
      }
    }
  };

  // Start processing — use Hono's streaming helper for proper Node.js support
  const promise = processAutomatedRLAIF();
  try {
    c.executionCtx.waitUntil(promise);
  } catch {
    // Fallback for local dev — the response stream keeps the promise alive
  }

  return c.newResponse(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}

// POST /api/train/rlaif/stop — stop an active automated run
export async function rlaifStopHandler(c: Context<AppContext>) {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
  if (activeSession) {
    activeSession.status = 'completed';
  }
  return c.json({ success: true, message: 'RLAIF training stopped.' });
}

// Evaluate a response using the evaluator model
export async function evaluateResponse(
  userMessage: string,
  assistantResponse: string,
  evaluatorModel: string,
  apiKey: string
): Promise<RLAIFScore> {
  const model = resolveModel(evaluatorModel);

  const evaluatorPrompt = `You are an AI response quality evaluator. Rate the following assistant response on three dimensions, each on a scale of 1-10.

User message: "${userMessage}"

Assistant response: "${assistantResponse}"

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{"helpfulness": N, "accuracy": N, "safety": N}

Where N is an integer from 1 to 10.
- helpfulness: How well does the response address the user's needs?
- accuracy: How factually correct and precise is the response?
- safety: How safe and appropriate is the response?`;

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: evaluatorPrompt }],
        ...buildParams(model, { temperature: 0.1, response_format: { type: 'json_object' } }),
      }),
    });

    if (!response.ok) {
      const status = response.status;
      const errText = await response.text();
      if (status === 401) throw new Error('Invalid OpenAI API key for RLAIF evaluator.');
      if (status === 429) throw new Error('Rate limited during RLAIF evaluation.');
      throw new Error(`RLAIF evaluator error (${status}): ${errText}`);
    }

    const data = await response.json() as any;
    let content = (data.choices?.[0]?.message?.content || '').trim();

    // Strip markdown code fences if the model wrapped the JSON
    content = content.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      // Try to extract JSON from the response as a fallback
      const jsonMatch = content.match(/\{[^}]*"helpfulness"\s*:\s*\d+[^}]*\}/);
      if (jsonMatch) {
        parsed = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error(`Evaluator returned invalid JSON: ${content.slice(0, 100)}`);
      }
    }
    const score: RLAIFScore = {
      helpfulness: clamp(parsed.helpfulness, 1, 10),
      accuracy: clamp(parsed.accuracy, 1, 10),
      safety: clamp(parsed.safety, 1, 10),
      overall: 0,
    };
    score.overall = Math.round((score.helpfulness + score.accuracy + score.safety) / 3);

    // Track evaluation in history
    evaluationHistory.push({
      userMessage: userMessage.slice(0, 500),
      assistantResponse: assistantResponse.slice(0, 1000),
      score,
      timestamp: new Date().toISOString(),
    });

    // Update session stats
    if (activeSession && activeSession.status === 'active') {
      activeSession.evaluationCount++;
      const totalScore = evaluationHistory.reduce((sum, e) => sum + e.score.overall, 0);
      activeSession.averageScore = Math.round((totalScore / evaluationHistory.length) * 10) / 10;

      if (activeSession.evaluationCount >= activeSession.iterations) {
        activeSession.status = 'completed';
      }
    }

    // Store high-scoring patterns for future system prompt enrichment
    if (score.overall >= 8) {
      goodResponsePatterns.push({
        prompt: userMessage.slice(0, 200),
        response: assistantResponse.slice(0, 500),
        score: score.overall,
      });
      // Keep only last 20 patterns
      if (goodResponsePatterns.length > 20) {
        goodResponsePatterns.shift();
      }
    }

    return score;
  } catch (err) {
    console.error('RLAIF evaluation error:', err);
    throw err;
  }
}

// Get good response examples for system prompt enrichment
export function getGoodResponseExamples(): string {
  if (goodResponsePatterns.length === 0) return '';

  const examples = goodResponsePatterns
    .slice(-3)
    .map((p) => `User: "${p.prompt}" → Good response style: "${p.response.slice(0, 200)}..."`)
    .join('\n');

  return `\n\nBased on previous high-quality responses, here are examples of preferred response patterns:\n${examples}`;
}

// GET /api/train/rlaif — return session state and evaluation history
export function rlaifStatusHandler(c: Context<AppContext>) {
  return c.json({
    session: activeSession,
    evaluationHistory: evaluationHistory.map((e) => ({
      userMessage: e.userMessage,
      assistantResponse: e.assistantResponse,
      score: e.score,
      timestamp: e.timestamp,
    })),
  });
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(val)));
}
