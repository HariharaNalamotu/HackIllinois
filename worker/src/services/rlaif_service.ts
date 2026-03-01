// RLAIF service — evaluates agent responses using a separate evaluator model

export interface RLAIFScore {
  helpfulness: number;
  accuracy: number;
  safety: number;
  overall: number;
}

interface RLAIFNode {
  data: {
    type: 'rlaif';
    parameters: {
      evaluatorModel: string;
      iterations: number;
    };
  };
}

// In-memory store for high-scoring response patterns
const goodResponsePatterns: { prompt: string; response: string; score: number }[] = [];

export function getRLAIFConfig(nodes: any[]): { model: string; iterations: number } | null {
  const rlaifNode = nodes.find((n: any) => n.data?.type === 'rlaif') as RLAIFNode | undefined;
  if (!rlaifNode) return null;

  return {
    model: rlaifNode.data.parameters.evaluatorModel || 'gpt-5.2',
    iterations: rlaifNode.data.parameters.iterations || 3,
  };
}

// Evaluate a response using the evaluator model
export async function evaluateResponse(
  userMessage: string,
  assistantResponse: string,
  evaluatorModel: string,
  apiKey: string
): Promise<RLAIFScore> {
  const modelMap: Record<string, string> = {
    'gpt-5.2': 'gpt-4o',
    'gpt-5-mini': 'gpt-4o-mini',
    'gpt-5-nano': 'gpt-4o-mini',
  };
  const model = modelMap[evaluatorModel] || 'gpt-4o-mini';

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
        temperature: 0.1,
      }),
    });

    if (!response.ok) {
      console.error('RLAIF evaluation failed:', await response.text());
      return { helpfulness: 5, accuracy: 5, safety: 5, overall: 5 };
    }

    const data = await response.json() as any;
    const content = data.choices?.[0]?.message?.content || '';

    const parsed = JSON.parse(content);
    const score: RLAIFScore = {
      helpfulness: clamp(parsed.helpfulness, 1, 10),
      accuracy: clamp(parsed.accuracy, 1, 10),
      safety: clamp(parsed.safety, 1, 10),
      overall: 0,
    };
    score.overall = Math.round((score.helpfulness + score.accuracy + score.safety) / 3);

    // Store high-scoring patterns for future system prompt enrichment
    if (score.overall >= 8) {
      goodResponsePatterns.push({
        prompt: userMessage.slice(0, 200),
        response: assistantResponse.slice(0, 500),
        score: score.overall,
      });
      if (goodResponsePatterns.length > 20) {
        goodResponsePatterns.shift();
      }
    }

    return score;
  } catch (err) {
    console.error('RLAIF evaluation error:', err);
    return { helpfulness: 5, accuracy: 5, safety: 5, overall: 5 };
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

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(val)));
}
