import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { chatHandler } from './services/agent_service';
import { feedbackHandler, getFeedbackHandler, rlhfTrainHandler } from './services/rlhf_service';
import { rlaifTrainHandler, rlaifStatusHandler, rlaifStopHandler } from './services/rlaif_service';
import { subagentTrainHandler } from './services/subagent_service';
import { trainHandler } from './services/codegen_service';

export type Bindings = {
  ALLOWED_ORIGINS: string;
  ENVIRONMENT: string;
};

export type AppContext = {
  Bindings: Bindings;
};

const app = new Hono<AppContext>();

// CORS
app.use('/api/*', async (c, next) => {
  const origins = (c.env?.ALLOWED_ORIGINS || 'http://localhost:5173').split(',');
  const corsMiddleware = cors({
    origin: origins,
    allowMethods: ['GET', 'POST', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'X-API-Key', 'X-Tavily-Key'],
    maxAge: 86400,
  });
  return corsMiddleware(c, next);
});

// Health check
app.get('/api/health', (c) => {
  return c.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Chat endpoint (streaming SSE)
app.post('/api/chat', chatHandler);

// Train — generate tool code via codex
app.post('/api/train', trainHandler);

// Training endpoints
app.post('/api/train/rlhf', rlhfTrainHandler);
app.post('/api/train/rlaif', rlaifTrainHandler);
app.post('/api/train/subagents', subagentTrainHandler);

// RLHF feedback
app.post('/api/feedback', feedbackHandler);
app.get('/api/feedback', getFeedbackHandler);

// RLAIF status/history + stop
app.get('/api/train/rlaif', rlaifStatusHandler);
app.post('/api/train/rlaif/stop', rlaifStopHandler);

// 404
app.notFound((c) => c.json({ error: 'Not found' }, 404));

app.onError((err, c) => {
  console.error('Unhandled error:', err);
  return c.json({ error: 'Internal server error' }, 500);
});

export default app;
