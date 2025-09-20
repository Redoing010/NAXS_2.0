import fp from 'fastify-plugin';
import { randomUUID } from 'node:crypto';
import type { FastifyInstance } from 'fastify';

const TRACE_HEADER = 'x-trace-id';

function extractTraceId(rawHeader: string | string[] | undefined): string | undefined {
  if (typeof rawHeader === 'string') {
    return rawHeader.trim() === '' ? undefined : rawHeader.trim();
  }

  if (Array.isArray(rawHeader) && rawHeader.length > 0) {
    const first = rawHeader[0]?.trim();
    return first ? first : undefined;
  }

  return undefined;
}

async function tracePlugin(fastify: FastifyInstance): Promise<void> {
  fastify.decorateRequest('traceId', '');

  fastify.addHook('onRequest', async (request, reply) => {
    const providedTraceId = extractTraceId(request.headers[TRACE_HEADER]);
    const traceId = providedTraceId ?? randomUUID();

    request.traceId = traceId;
    reply.header(TRACE_HEADER, traceId);
  });
}

export default fp(tracePlugin, {
  fastify: '4.x',
  name: 'trace-plugin',
});

export { TRACE_HEADER };
