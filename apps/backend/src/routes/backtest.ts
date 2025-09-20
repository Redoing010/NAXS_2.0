import fp from 'fastify-plugin';
import type { FastifyInstance } from 'fastify';
import { AppError } from '../utils/errors';
import { createJob, publishBacktestCommand } from '../mq/index';
import type { BacktestCommandMessage } from '../mq/index';

interface BacktestRequestBody {
  strategy: string;
  params?: Record<string, unknown>;
  profileId: string;
  evidenceRefs?: string[];
}

function normalizeHeader(value: string | string[] | undefined): string | undefined {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }

  if (Array.isArray(value)) {
    return normalizeHeader(value[0]);
  }

  return undefined;
}

async function registerBacktestRoutes(app: FastifyInstance): Promise<void> {
  app.post<{ Body: BacktestRequestBody }>('/backtest/run', async (request, reply) => {
    const idempotencyKey = normalizeHeader(request.headers['idempotency-key']);

    if (!idempotencyKey) {
      throw new AppError('IDEMPOTENCY_KEY_REQUIRED', 'Idempotency-Key header is required', 400);
    }

    const { body } = request;

    if (!body || typeof body.strategy !== 'string' || body.strategy.trim() === '') {
      throw new AppError('BACKTEST_STRATEGY_REQUIRED', 'strategy is required', 400);
    }

    if (typeof body.profileId !== 'string' || body.profileId.trim() === '') {
      throw new AppError('BACKTEST_PROFILE_REQUIRED', 'profileId is required', 400);
    }

    const job = createJob('backtest', request.traceId);

    const command: BacktestCommandMessage = {
      jobId: job.id,
      strategy: body.strategy,
      params: body.params ?? {},
      profileId: body.profileId,
      evidenceRefs: body.evidenceRefs ?? [],
      traceId: request.traceId,
    };

    await publishBacktestCommand(command);

    reply.status(202).send({ id: job.id, status: job.state });
  });
}

export default fp(registerBacktestRoutes, {
  fastify: '4.x',
  name: 'backtest-routes',
});
