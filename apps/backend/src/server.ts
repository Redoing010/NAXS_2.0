import Fastify, { type FastifyInstance } from 'fastify';
import tracePlugin from './plugins/trace';
import { registerErrorHandler } from './middleware/errorHandler';

export async function buildServer(): Promise<FastifyInstance> {
  const app = Fastify({
    logger: true,
  });

  await app.register(tracePlugin);
  registerErrorHandler(app);

  app.get('/healthz', async (request) => ({ status: 'ok', traceId: request.traceId }));

  return app;
}

async function start(): Promise<void> {
  const app = await buildServer();
  const port = Number(process.env.PORT ?? 8080);
  const host = process.env.HOST ?? '0.0.0.0';

  try {
    await app.listen({ port, host });
    app.log.info({ port, host }, 'Server started');
  } catch (error) {
    app.log.error({ err: error }, 'Failed to start server');
    process.exit(1);
  }
}

if (require.main === module) {
  void start();
}
