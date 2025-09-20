export interface ErrorResponse {
  code: string;
  message: string;
  details?: unknown;
  traceId: string;
  timestamp: string;
}

export class AppError extends Error {
  public readonly code: string;

  public readonly statusCode: number;

  public readonly details?: unknown;

  constructor(code: string, message: string, statusCode = 500, details?: unknown) {
    super(message);
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
  }
}

interface ErrorParams {
  code: string;
  message: string;
  traceId: string;
  details?: unknown;
}

export function err({ code, message, traceId, details }: ErrorParams): ErrorResponse {
  return {
    code,
    message,
    ...(details === undefined ? {} : { details }),
    traceId,
    timestamp: new Date().toISOString(),
  };
}
