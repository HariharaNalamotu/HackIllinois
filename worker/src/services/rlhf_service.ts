// RLHF feedback service — stores human feedback and enriches future prompts
// Uses in-memory storage (resets on worker restart)

interface FeedbackEntry {
  messageId: string;
  rating: 'up' | 'down';
  feedback?: string;
  timestamp: string;
  conversationSnippet?: string;
}

// In-memory feedback storage
const feedbackStore: FeedbackEntry[] = [];

export function storeFeedback(entry: {
  messageId: string;
  rating: 'up' | 'down';
  feedback?: string;
  conversationSnippet?: string;
}): void {
  feedbackStore.push({
    ...entry,
    timestamp: new Date().toISOString(),
  });
  // Keep only last 100 entries
  if (feedbackStore.length > 100) {
    feedbackStore.splice(0, feedbackStore.length - 100);
  }
}

export function getRecentFeedback(): FeedbackEntry[] {
  return feedbackStore.slice(-50);
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
