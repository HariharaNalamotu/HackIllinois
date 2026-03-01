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

// In-memory feedback storage
const feedbackStore: FeedbackEntry[] = [];

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

    return c.json({ success: true });
  } catch (err) {
    return c.json({ error: 'Invalid request body' }, 400);
  }
}

// GET /api/feedback
export async function getFeedbackHandler(c: Context<AppContext>) {
  return c.json({ feedback: feedbackStore.slice(-50) });
}

// Get positive feedback examples to include in system prompt
export function getPositiveFeedbackContext(): string {
  const positiveEntries = feedbackStore
    .filter((e) => e.rating === 'up' && e.conversationSnippet)
    .slice(-5);

  if (positiveEntries.length === 0) return '';

  const examples = positiveEntries
    .map((e) => e.conversationSnippet)
    .join('\n---\n');

  return `\n\nThe user has previously given positive feedback on responses similar to these:\n${examples}\nTry to maintain this quality and style.`;
}
