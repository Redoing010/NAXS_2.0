import { spawn } from 'node:child_process';
import path from 'node:path';
import type { JobRow } from '../../backend/src/services/jobService';
import { JobService } from '../../backend/src/services/jobService';
import { JobState } from '../../backend/src/domain/job';
import type { BacktestCommandMessage } from '../../backend/src/mq/index';
import { jobRepository, registerBacktestConsumer } from '../../backend/src/mq/index';
import type { BacktestsRepo, BacktestRecord } from '../../backend/src/repos/backtestsRepo';
import { backtestsRepo } from '../../backend/src/repos/backtestsRepo';
import type { BacktestCompletedEvent } from '../../backend/src/events/publish';
import { publishBacktestCompleted } from '../../backend/src/events/publish';

export interface AdapterResult {
  metrics: Record<string, unknown>;
  evidenceId: string;
}

export interface BacktestWorkerDeps {
  repo?: BacktestsRepo;
  publishEvent?: (event: BacktestCompletedEvent) => Promise<void>;
  runAdapter?: (params: Record<string, unknown>, profileId: string) => Promise<AdapterResult>;
  jobService?: JobService;
}

const defaultJobService = new JobService(jobRepository);

function resolveAdapterPath(): string {
  return path.resolve(__dirname, '../../backtest_py/qlib_adapter.py');
}

async function runPythonAdapter(
  params: Record<string, unknown>,
  profileId: string,
): Promise<AdapterResult> {
  const pythonExecutable = process.env.PYTHON_EXECUTABLE ?? 'python3';
  const scriptPath = resolveAdapterPath();

  return new Promise<AdapterResult>((resolve, reject) => {
    const child = spawn(pythonExecutable, [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const input = JSON.stringify({ params, profileId });
    child.stdin.write(input);
    child.stdin.end();

    let stdout = '';
    let stderr = '';

    child.stdout.setEncoding('utf-8');
    child.stdout.on('data', (chunk: string) => {
      stdout += chunk;
    });

    child.stderr.setEncoding('utf-8');
    child.stderr.on('data', (chunk: string) => {
      stderr += chunk;
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`qlib adapter exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        const parsed = JSON.parse(stdout) as { metrics: Record<string, unknown>; evidenceId: string };
        resolve({ metrics: parsed.metrics, evidenceId: parsed.evidenceId });
      } catch (error) {
        reject(new Error(`Failed to parse qlib adapter response: ${(error as Error).message}`));
      }
    });

    child.on('error', (error) => {
      reject(error);
    });
  });
}

export async function handleBacktestMessage(
  message: BacktestCommandMessage,
  deps: BacktestWorkerDeps = {},
): Promise<void> {
  const service = deps.jobService ?? defaultJobService;
  const repo = deps.repo ?? backtestsRepo;
  const publishEvent = deps.publishEvent ?? publishBacktestCompleted;
  const runAdapter = deps.runAdapter ?? runPythonAdapter;

  await service.updateState(message.jobId, {
    state: JobState.RUNNING,
    progress: 0,
  });

  try {
    const adapterResult = await runAdapter(message.params, message.profileId);

    const record: BacktestRecord = await repo.insertResult({
      jobId: message.jobId,
      profileId: message.profileId,
      metrics: adapterResult.metrics,
      evidenceId: adapterResult.evidenceId,
      traceId: message.traceId,
    });

    const updatedJob: JobRow = await service.updateState(message.jobId, {
      state: JobState.SUCCEEDED,
      progress: 1,
      result: {
        metrics: adapterResult.metrics,
        evidenceIds: [record.evidenceId],
      },
      error: null,
    });

    await publishEvent({
      jobId: updatedJob.id,
      profileId: message.profileId,
      metrics: adapterResult.metrics,
      evidenceId: record.evidenceId,
      traceId: message.traceId,
    });
  } catch (error) {
    const failureMessage = error instanceof Error ? error.message : 'Backtest execution failed';

    await service.updateState(message.jobId, {
      state: JobState.FAILED,
      result: null,
      error: {
        code: 'DEPENDENCY_FAILED',
        message: failureMessage,
      },
    });
  }
}

export function startBacktestWorker(deps: BacktestWorkerDeps = {}): void {
  registerBacktestConsumer(async (message) => {
    await handleBacktestMessage(message, deps);
  });
}
