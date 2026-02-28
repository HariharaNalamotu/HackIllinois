import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { chatHandler } from './services/agent_service';
import { feedbackHandler, getFeedbackHandler } from './services/rlhf_service';

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
  const origins = (c.env.ALLOWED_ORIGINS || 'http://localhost:5173').split(',');
  const corsMiddleware = cors({
    origin: origins,
    allowMethods: ['GET', 'POST', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'X-API-Key'],
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

// RLHF feedback
app.post('/api/feedback', feedbackHandler);
app.get('/api/feedback', getFeedbackHandler);

// 404
app.notFound((c) => c.json({ error: 'Not found' }, 404));

app.onError((err, c) => {
  console.error('Unhandled error:', err);
  return c.json({ error: 'Internal server error' }, 500);
});

export default app;
