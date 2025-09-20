import type { FastifyError, FastifyInstance } from 'fastify';
import { AppError, err } from '../utils/errors';

const DEFAULT_ERROR_CODE = 'INTERNAL_ERROR';

export function registerErrorHandler(app: FastifyInstance): void {
  app.setErrorHandler((error, request, reply) => {
    const traceId = request.traceId ?? '';
    request.log.error({ err: error, traceId }, 'request failed');

    if (error instanceof AppError) {
      reply
        .status(error.statusCode)
        .send(
          err({
            code: error.code,
            message: error.message,
            traceId,
            details: error.details,
          }),
        );
      return;
    }

    const fastifyError = error as FastifyError & {
      cause?: unknown;
      validation?: unknown;
    };

    const statusCode = typeof fastifyError.statusCode === 'number' ? fastifyError.statusCode : 500;
    const code = typeof fastifyError.code === 'string' ? fastifyError.code : DEFAULT_ERROR_CODE;
    const message = statusCode >= 500 ? 'Internal server error' : fastifyError.message;
    const details = statusCode >= 500 ? undefined : fastifyError.validation ?? fastifyError.cause;

    reply.status(statusCode).send(
      err({
        code,
        message,
        traceId,
        details,
      }),
    );
  });
}
