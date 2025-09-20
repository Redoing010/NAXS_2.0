import { AppError } from '../utils/errors';
import { JobState, canTransition } from '../domain/job';

export interface JobRow {
  id: string;
  kind: string;
  state: JobState;
  progress: number;
  result: Record<string, unknown> | null;
  error: Record<string, unknown> | null;
  traceId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface JobRepository {
  findById(id: string): Promise<JobRow | null>;
  update(id: string, patch: Partial<JobRow>): Promise<JobRow>;
}

export class JobService {
  constructor(private readonly repository: JobRepository) {}

  async updateState(jobId: string, patch: Partial<JobRow>): Promise<JobRow> {
    if (!patch.state) {
      throw new AppError('JOB_STATE_REQUIRED', 'State is required to update a job', 422);
    }

    const existingJob = await this.repository.findById(jobId);

    if (!existingJob) {
      throw new AppError('JOB_NOT_FOUND', `Job ${jobId} not found`, 404);
    }

    const evaluation = canTransition(existingJob.state, patch.state);

    if (!evaluation.allowed) {
      const statusCode = evaluation.statusCode ?? 409;
      const message =
        evaluation.reason ?? `Cannot transition job from ${existingJob.state} to ${patch.state}`;
      throw new AppError('JOB_INVALID_TRANSITION', message, statusCode);
    }

    const updatePayload: Partial<JobRow> = {
      ...patch,
      state: patch.state,
      updatedAt: patch.updatedAt ?? new Date().toISOString(),
    };

    const updatedJob = await this.repository.update(jobId, updatePayload);

    return updatedJob;
  }
}
