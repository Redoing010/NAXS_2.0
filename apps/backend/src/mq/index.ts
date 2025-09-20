import { randomUUID } from 'node:crypto';
import { setImmediate as scheduleImmediate } from 'node:timers';
import type { JobRepository, JobRow } from '../services/jobService';
import { JobState } from '../domain/job';

export interface BacktestCommandMessage {
  jobId: string;
  strategy: string;
  params: Record<string, unknown>;
  profileId: string;
  evidenceRefs: string[];
  traceId: string;
}

export type BacktestConsumer = (message: BacktestCommandMessage) => Promise<void>;

let currentBacktestHandler: BacktestConsumer | null = null;
const jobStore = new Map<string, JobRow>();

export const jobRepository: JobRepository = {
  async findById(id: string): Promise<JobRow | null> {
    return jobStore.get(id) ?? null;
  },
  async update(id: string, patch: Partial<JobRow>): Promise<JobRow> {
    const current = jobStore.get(id);

    if (!current) {
      throw new Error(`Job ${id} not found`);
    }

    const updated: JobRow = {
      ...current,
      ...patch,
      state: patch.state ?? current.state,
      progress: patch.progress ?? current.progress,
      result: patch.result !== undefined ? patch.result : current.result,
      error: patch.error !== undefined ? patch.error : current.error,
      traceId: patch.traceId !== undefined ? patch.traceId : current.traceId,
      updatedAt: patch.updatedAt ?? new Date().toISOString(),
    };

    jobStore.set(id, updated);

    return updated;
  },
};

export function createJob(kind: string, traceId: string): JobRow {
  const now = new Date().toISOString();
  const job: JobRow = {
    id: randomUUID(),
    kind,
    state: JobState.PENDING,
    progress: 0,
    result: null,
    error: null,
    traceId,
    createdAt: now,
    updatedAt: now,
  };

  jobStore.set(job.id, job);

  return job;
}

export function getJob(jobId: string): JobRow | undefined {
  return jobStore.get(jobId);
}

export function registerBacktestConsumer(handler: BacktestConsumer): void {
  currentBacktestHandler = handler;
}

export async function publishBacktestCommand(message: BacktestCommandMessage): Promise<void> {
  if (process.env.MQ_DISABLED === 'true') {
    if (!currentBacktestHandler) {
      throw new Error('No backtest consumer registered');
    }

    await currentBacktestHandler(message);
    return;
  }

  scheduleImmediate(() => {
    if (!currentBacktestHandler) {
      return;
    }

    void currentBacktestHandler(message).catch((error) => {
      // eslint-disable-next-line no-console
      console.error('Backtest consumer failed', error);
    });
  });
}

export function resetJobStore(): void {
  jobStore.clear();
}
