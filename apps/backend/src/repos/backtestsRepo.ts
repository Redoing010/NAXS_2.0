import { randomUUID } from 'node:crypto';

export interface BacktestMetrics {
  annRet?: number;
  sharpe?: number;
  maxDD?: number;
  turnover?: number;
  [key: string]: unknown;
}

export interface InsertBacktestResultInput {
  jobId: string;
  profileId: string;
  metrics: BacktestMetrics;
  evidenceId: string;
  traceId: string;
}

export interface BacktestRecord extends InsertBacktestResultInput {
  id: string;
  createdAt: string;
}

export interface BacktestsRepo {
  insertResult(input: InsertBacktestResultInput): Promise<BacktestRecord>;
  findByEvidenceId(evidenceId: string): Promise<BacktestRecord | null>;
  reset(): void;
}

class InMemoryBacktestsRepo implements BacktestsRepo {
  private readonly store = new Map<string, BacktestRecord>();

  async insertResult(input: InsertBacktestResultInput): Promise<BacktestRecord> {
    const createdAt = new Date().toISOString();
    const record: BacktestRecord = {
      id: randomUUID(),
      ...input,
      createdAt,
    };

    this.store.set(record.evidenceId, record);

    return record;
  }

  async findByEvidenceId(evidenceId: string): Promise<BacktestRecord | null> {
    return this.store.get(evidenceId) ?? null;
  }

  reset(): void {
    this.store.clear();
  }
}

export const backtestsRepo: BacktestsRepo = new InMemoryBacktestsRepo();
