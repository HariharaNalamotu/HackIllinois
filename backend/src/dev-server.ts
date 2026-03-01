// Node.js dev server — runs the same Hono app on Node.js instead of Cloudflare Workers.
// This allows new Function() / eval() for dynamic tool code execution.
// Usage: npm run dev

import { serve } from '@hono/node-server';
import app from './index';

// Inject environment variables that wrangler.toml [vars] would normally provide
const env = {
  ENVIRONMENT: 'development',
  ALLOWED_ORIGINS: 'http://localhost:5173,http://localhost:3000',
};

// Patch env into every request context (Hono expects c.env.*)
app.use('*', async (c, next) => {
  // @ts-ignore — inject env bindings for Hono
  c.env = { ...env, ...c.env };
  await next();
});

const port = Number(process.env.PORT) || 8787;

console.log(`ML Workflow API running on http://localhost:${port}`);
console.log(`Environment: ${env.ENVIRONMENT}`);
console.log(`Allowed origins: ${env.ALLOWED_ORIGINS}`);
console.log(`Runtime: Node.js ${process.version} (eval/new Function supported)`);

serve({ fetch: app.fetch, port });
