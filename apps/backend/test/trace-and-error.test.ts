import { describe, expect, it } from 'vitest';
import type { FastifyInstance } from 'fastify';
import { buildServer } from '../src/server';
import { AppError } from '../src/utils/errors';

async function createServer(): Promise<FastifyInstance> {
  return buildServer();
}

describe('trace plugin and error handler', () => {
  it('generates a trace id when header is missing', async () => {
    const server = await createServer();

    const response = await server.inject({
      method: 'GET',
      url: '/healthz',
    });

    const traceIdHeader = response.headers['x-trace-id'];

    expect(typeof traceIdHeader).toBe('string');
    expect((traceIdHeader as string).length).toBeGreaterThan(0);

    await server.close();
  });

  it('reuses the first value when header is an array', async () => {
    const server = await createServer();

    const traceId = 'trace-array-1';
    const response = await server.inject({
      method: 'GET',
      url: '/healthz',
      headers: {
        'x-trace-id': [traceId, 'ignored'],
      },
    });

    expect(response.headers['x-trace-id']).toBe(traceId);

    await server.close();
  });

  it('returns standard error body when route throws', async () => {
    const server = await createServer();

    server.get('/boom', () => {
      throw new AppError('BAD_REQUEST', 'Request failed', 400, { reason: 'boom' });
    });

    await server.ready();

    const traceId = 'trace-error-1';
    const response = await server.inject({
      method: 'GET',
      url: '/boom',
      headers: {
        'x-trace-id': traceId,
      },
    });

    expect(response.statusCode).toBe(400);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const payload = response.json();

    expect(payload).toMatchObject({
      code: 'BAD_REQUEST',
      message: 'Request failed',
      traceId,
      details: { reason: 'boom' },
    });

    expect(typeof payload.timestamp).toBe('string');
    expect(Number.isNaN(Date.parse(payload.timestamp))).toBe(false);

    await server.close();
  });
});
