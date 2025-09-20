import { describe, expect, it, beforeEach } from 'vitest';
import { JobService } from '../src/services/jobService';
import { JobState } from '../src/domain/job';
import { createJob, getJob, jobRepository, resetJobStore } from '../src/mq/index';
import type { BacktestsRepo, BacktestRecord, InsertBacktestResultInput } from '../src/repos/backtestsRepo';
import { handleBacktestMessage } from '../../backtest_worker/src/worker';
import type { BacktestCompletedEvent } from '../src/events/publish';

class TestBacktestsRepo implements BacktestsRepo {
  public records: BacktestRecord[] = [];

  async insertResult(input: InsertBacktestResultInput): Promise<BacktestRecord> {
    const record: BacktestRecord = {
      id: `rec-${this.records.length + 1}`,
      createdAt: new Date().toISOString(),
      ...input,
    };

    this.records.push(record);

    return record;
  }

  async findByEvidenceId(evidenceId: string): Promise<BacktestRecord | null> {
    return this.records.find((record) => record.evidenceId === evidenceId) ?? null;
  }

  reset(): void {
    this.records = [];
  }
}

describe('backtest worker', () => {
  beforeEach(() => {
    resetJobStore();
  });

  it('marks job as succeeded and stores result when adapter succeeds', async () => {
    const repo = new TestBacktestsRepo();
    const events: BacktestCompletedEvent[] = [];
    const job = createJob('backtest', 'trace-success');
    const jobService = new JobService(jobRepository);

    await handleBacktestMessage(
      {
        jobId: job.id,
        strategy: 'demo',
        params: { window: 10 },
        profileId: 'default',
        evidenceRefs: [],
        traceId: 'trace-success',
      },
      {
        repo,
        publishEvent: async (event) => {
          events.push(event);
        },
        runAdapter: async () => ({
          metrics: { annRet: 0.12 },
          evidenceId: 'evidence-1',
        }),
        jobService,
      },
    );

    const storedJob = getJob(job.id);

    expect(storedJob).not.toBeUndefined();
    expect(storedJob?.state).toBe(JobState.SUCCEEDED);
    expect(storedJob?.result).toMatchObject({
      metrics: { annRet: 0.12 },
      evidenceIds: ['evidence-1'],
    });
    expect(repo.records).toHaveLength(1);
    expect(repo.records[0].evidenceId).toBe('evidence-1');
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      jobId: job.id,
      evidenceId: 'evidence-1',
      traceId: 'trace-success',
    });
  });

  it('marks job as failed when adapter throws', async () => {
    const repo = new TestBacktestsRepo();
    const job = createJob('backtest', 'trace-failure');
    const jobService = new JobService(jobRepository);

    await handleBacktestMessage(
      {
        jobId: job.id,
        strategy: 'demo',
        params: { window: 10 },
        profileId: 'default',
        evidenceRefs: [],
        traceId: 'trace-failure',
      },
      {
        repo,
        publishEvent: async () => {
          throw new Error('should not publish on failure');
        },
        runAdapter: async () => {
          throw new Error('adapter failed');
        },
        jobService,
      },
    );

    const storedJob = getJob(job.id);

    expect(storedJob?.state).toBe(JobState.FAILED);
    expect(storedJob?.error).toMatchObject({
      code: 'DEPENDENCY_FAILED',
    });
    expect(repo.records).toHaveLength(0);
  });
});
