import { describe, expect, it } from 'vitest';
import type { FastifyInstance } from 'fastify';
import { buildServer } from '../src/server';
import { JobService, type JobRepository, type JobRow } from '../src/services/jobService';
import { JobState } from '../src/domain/job';

class InMemoryJobRepository implements JobRepository {
  private readonly store = new Map<string, JobRow>();

  constructor(initialJobs: JobRow[]) {
    initialJobs.forEach((job) => {
      this.store.set(job.id, job);
    });
  }

  async findById(id: string): Promise<JobRow | null> {
    return this.store.get(id) ?? null;
  }

  async update(id: string, patch: Partial<JobRow>): Promise<JobRow> {
    const current = this.store.get(id);

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

    this.store.set(id, updated);

    return updated;
  }
}

interface UpdateJobBody {
  state: JobState;
  progress?: number;
  result?: Record<string, unknown> | null;
  error?: Record<string, unknown> | null;
}

function createJobRow(id: string, state: JobState): JobRow {
  const now = new Date().toISOString();

  return {
    id,
    kind: 'backtest',
    state,
    progress: 0,
    result: null,
    error: null,
    traceId: null,
    createdAt: now,
    updatedAt: now,
  };
}

async function createJobServer(initialJobs: JobRow[]): Promise<FastifyInstance> {
  const server = await buildServer();
  const repository = new InMemoryJobRepository(initialJobs);
  const service = new JobService(repository);

  server.post<{ Params: { id: string }; Body: UpdateJobBody }>('/jobs/:id/state', async (request) => {
    const { id } = request.params;
    const { state, progress, result, error } = request.body;

    const patch: Partial<JobRow> = {
      state,
      ...(progress === undefined ? {} : { progress }),
      ...(result === undefined ? {} : { result }),
      ...(error === undefined ? {} : { error }),
    };

    const updated = await service.updateState(id, patch);

    return updated;
  });

  await server.ready();

  return server;
}

describe('job state transitions', () => {
  it('allows transition from PENDING to RUNNING', async () => {
    const server = await createJobServer([createJobRow('job-1', JobState.PENDING)]);

    const traceId = 'trace-job-allowed';
    const response = await server.inject({
      method: 'POST',
      url: '/jobs/job-1/state',
      headers: {
        'content-type': 'application/json',
        'x-trace-id': traceId,
      },
      payload: {
        state: JobState.RUNNING,
      },
    });

    expect(response.statusCode).toBe(200);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const body = response.json() as JobRow;

    expect(body.state).toBe(JobState.RUNNING);
    expect(typeof body.updatedAt).toBe('string');

    await server.close();
  });

  it('rejects SUCCEEDED to RUNNING transition with 409', async () => {
    const server = await createJobServer([createJobRow('job-2', JobState.SUCCEEDED)]);

    const traceId = 'trace-job-409';
    const response = await server.inject({
      method: 'POST',
      url: '/jobs/job-2/state',
      headers: {
        'content-type': 'application/json',
        'x-trace-id': traceId,
      },
      payload: {
        state: JobState.RUNNING,
      },
    });

    expect(response.statusCode).toBe(409);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const payload = response.json() as Record<string, unknown>;

    expect(payload).toMatchObject({
      code: 'JOB_INVALID_TRANSITION',
      message: 'Job is already finished with state SUCCEEDED',
      traceId,
    });
    expect(typeof payload.timestamp).toBe('string');

    await server.close();
  });

  it('rejects FAILED to SUCCEEDED transition with 409', async () => {
    const server = await createJobServer([createJobRow('job-3', JobState.FAILED)]);

    const traceId = 'trace-job-409-b';
    const response = await server.inject({
      method: 'POST',
      url: '/jobs/job-3/state',
      headers: {
        'content-type': 'application/json',
        'x-trace-id': traceId,
      },
      payload: {
        state: JobState.SUCCEEDED,
      },
    });

    expect(response.statusCode).toBe(409);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const payload = response.json() as Record<string, unknown>;

    expect(payload).toMatchObject({
      code: 'JOB_INVALID_TRANSITION',
      message: 'Job is already finished with state FAILED',
      traceId,
    });
    expect(typeof payload.timestamp).toBe('string');

    await server.close();
  });

  it('rejects PENDING to SUCCEEDED transition with 422', async () => {
    const server = await createJobServer([createJobRow('job-4', JobState.PENDING)]);

    const traceId = 'trace-job-422';
    const response = await server.inject({
      method: 'POST',
      url: '/jobs/job-4/state',
      headers: {
        'content-type': 'application/json',
        'x-trace-id': traceId,
      },
      payload: {
        state: JobState.SUCCEEDED,
      },
    });

    expect(response.statusCode).toBe(422);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const payload = response.json() as Record<string, unknown>;

    expect(payload).toMatchObject({
      code: 'JOB_INVALID_TRANSITION',
      message: 'Jobs can enter SUCCEEDED only from RUNNING',
      traceId,
    });
    expect(typeof payload.timestamp).toBe('string');

    await server.close();
  });
});
