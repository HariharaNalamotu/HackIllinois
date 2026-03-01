// RLHF feedback service — stores human feedback and enriches future prompts
// Uses in-memory storage (resets on worker restart; use KV for persistence)

import type { Context } from 'hono';
import type { AppContext } from '../index';

interface FeedbackEntry {
  messageId: string;
  rating: 'up' | 'down';
  feedback?: string;
  timestamp: string;
  conversationSnippet?: string;
}

interface RLHFSession {
  workflowId: string;
  iterations: number;
  currentIteration: number;
  status: 'active' | 'completed';
  startedAt: string;
  systemPromptOverrides: string;
}

// In-memory feedback storage
const feedbackStore: FeedbackEntry[] = [];

// Active RLHF session
let activeSession: RLHFSession | null = null;

// POST /api/feedback
export async function feedbackHandler(c: Context<AppContext>) {
  try {
    const body = await c.req.json<{
      messageId: string;
      rating: 'up' | 'down';
      feedback?: string;
      conversationSnippet?: string;
    }>();

    if (!body.messageId || !body.rating) {
      return c.json({ error: 'messageId and rating are required' }, 400);
    }

    const entry: FeedbackEntry = {
      messageId: body.messageId,
      rating: body.rating,
      feedback: body.feedback,
      timestamp: new Date().toISOString(),
      conversationSnippet: body.conversationSnippet,
    };

    feedbackStore.push(entry);

    // Keep only last 100 entries
    if (feedbackStore.length > 100) {
      feedbackStore.splice(0, feedbackStore.length - 100);
    }

    // If an RLHF session is active, advance iteration on each piece of feedback
    if (activeSession && activeSession.status === 'active') {
      activeSession.currentIteration++;
      // Rebuild prompt overrides from accumulated feedback
      activeSession.systemPromptOverrides = buildPromptFromFeedback();
      if (activeSession.currentIteration >= activeSession.iterations) {
        activeSession.status = 'completed';
      }
    }

    return c.json({
      success: true,
      session: activeSession ? {
        iteration: activeSession.currentIteration,
        total: activeSession.iterations,
        status: activeSession.status,
      } : null,
    });
  } catch (err) {
    return c.json({ error: 'Invalid request body' }, 400);
  }
}

// GET /api/feedback
export async function getFeedbackHandler(c: Context<AppContext>) {
  return c.json({
    feedback: feedbackStore.slice(-50),
    session: activeSession,
  });
}

// POST /api/train/rlhf — initialize an RLHF training session
export async function rlhfTrainHandler(c: Context<AppContext>) {
  const apiKey = c.req.header('X-API-Key');
  if (!apiKey) {
    return c.json({ error: 'X-API-Key header is required.' }, 401);
  }

  let body: { workflowConfig: { nodes: any[]; edges: any[] } };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: 'Invalid request body' }, 400);
  }

  const nodes = body.workflowConfig?.nodes || [];
  const rlhfNode = nodes.find((n: any) => n.data?.type === 'rlhf');

  if (!rlhfNode) {
    return c.json({ error: 'No RLHF node found in workflow.' }, 400);
  }

  const iterations = rlhfNode.data.parameters?.iterations || 3;

  // Clear previous feedback if starting fresh
  feedbackStore.length = 0;

  activeSession = {
    workflowId: 'current',
    iterations,
    currentIteration: 0,
    status: 'active',
    startedAt: new Date().toISOString(),
    systemPromptOverrides: '',
  };

  return c.json({
    success: true,
    session: {
      iterations,
      status: 'active',
      message: `RLHF session started. Provide ${iterations} rounds of feedback (thumbs up/down) on responses in the chat to train the model.`,
    },
  });
}

// Build system prompt additions from accumulated feedback
function buildPromptFromFeedback(): string {
  const positive = feedbackStore.filter((e) => e.rating === 'up');
  const negative = feedbackStore.filter((e) => e.rating === 'down');

  let override = '';

  if (positive.length > 0) {
    const examples = positive
      .slice(-5)
      .filter((e) => e.conversationSnippet)
      .map((e) => e.conversationSnippet)
      .join('\n---\n');
    if (examples) {
      override += `\nResponses the user liked:\n${examples}\nEmulate this style.`;
    }
  }

  if (negative.length > 0) {
    override += buildNegativeFeedbackPrompt(negative);
  }

  return override;
}

// Build a detailed negative feedback prompt that includes both the bad response
// and the user's criticism so the model knows exactly what to avoid
function buildNegativeFeedbackPrompt(negative: FeedbackEntry[]): string {
  const recentNegative = negative.slice(-5);
  let prompt = '\n\nCRITICAL — The user has rejected the following responses. You MUST NOT repeat these patterns:';

  for (const entry of recentNegative) {
    if (entry.conversationSnippet) {
      prompt += `\n\n--- BAD RESPONSE (DO NOT REPEAT) ---\n${entry.conversationSnippet}`;
      if (entry.feedback) {
        prompt += `\nUser's criticism: "${entry.feedback}"`;
      }
      prompt += '\n--- END BAD RESPONSE ---';
    } else if (entry.feedback) {
      prompt += `\nUser complaint: "${entry.feedback}"`;
    }
  }

  prompt += '\n\nWhen answering similar questions, take a completely different approach from the rejected responses above. Address the user\'s criticisms directly.';
  return prompt;
}

// Get all feedback context (positive + negative) to include in system prompt
export function getPositiveFeedbackContext(): string {
  // If there's an active RLHF session, use its accumulated overrides
  if (activeSession && activeSession.systemPromptOverrides) {
    return activeSession.systemPromptOverrides;
  }

  let context = '';

  // Positive feedback
  const positiveEntries = feedbackStore
    .filter((e) => e.rating === 'up' && e.conversationSnippet)
    .slice(-5);

  if (positiveEntries.length > 0) {
    const examples = positiveEntries
      .map((e) => e.conversationSnippet)
      .join('\n---\n');
    context += `\n\nThe user has previously given positive feedback on responses similar to these:\n${examples}\nTry to maintain this quality and style.`;
  }

  // Negative feedback — tell the model what NOT to do
  const negativeEntries = feedbackStore.filter((e) => e.rating === 'down');
  if (negativeEntries.length > 0) {
    context += buildNegativeFeedbackPrompt(negativeEntries);
  }

  return context;
}

// Get the active session (for status display)
export function getActiveSession(): RLHFSession | null {
  return activeSession;
}
