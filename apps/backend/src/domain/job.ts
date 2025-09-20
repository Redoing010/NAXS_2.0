export enum JobState {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCEEDED = 'SUCCEEDED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
}

export interface TransitionResult {
  allowed: boolean;
  statusCode?: number;
  reason?: string;
}

const terminalStates: ReadonlySet<JobState> = new Set([
  JobState.SUCCEEDED,
  JobState.FAILED,
  JobState.CANCELLED,
]);

export function canTransition(from: JobState, to: JobState): TransitionResult {
  if (from === to) {
    return {
      allowed: false,
      statusCode: 409,
      reason: `Job already in state ${to}`,
    };
  }

  if (terminalStates.has(from)) {
    return {
      allowed: false,
      statusCode: 409,
      reason: `Job is already finished with state ${from}`,
    };
  }

  if (to === JobState.RUNNING) {
    return from === JobState.PENDING
      ? { allowed: true }
      : {
          allowed: false,
          statusCode: 422,
          reason: 'Jobs can enter RUNNING only from PENDING',
        };
  }

  if (to === JobState.SUCCEEDED || to === JobState.FAILED) {
    return from === JobState.RUNNING
      ? { allowed: true }
      : {
          allowed: false,
          statusCode: 422,
          reason: `Jobs can enter ${to} only from RUNNING`,
        };
  }

  if (to === JobState.CANCELLED) {
    return from === JobState.PENDING || from === JobState.RUNNING
      ? { allowed: true }
      : {
          allowed: false,
          statusCode: 409,
          reason: 'Jobs can be cancelled only before completion',
        };
  }

  return {
    allowed: false,
    statusCode: 422,
    reason: `Transition to ${to} is not supported`,
  };
}
