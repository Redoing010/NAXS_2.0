import { randomUUID } from 'node:crypto';

export interface InsertEvidenceInput {
  source: string;
  uri: string;
  checksum: string;
}

export interface EvidenceRecord {
  id: string;
  source: string;
  uri: string;
  checksum: string;
  ts: string;
}

export interface EvidenceRepo {
  insert(input: InsertEvidenceInput): Promise<EvidenceRecord>;
  findById(id: string): Promise<EvidenceRecord | null>;
  reset(): void;
}

export class InMemoryEvidenceRepo implements EvidenceRepo {
  private readonly store = new Map<string, EvidenceRecord>();

  async insert(input: InsertEvidenceInput): Promise<EvidenceRecord> {
    const record: EvidenceRecord = {
      id: randomUUID(),
      source: input.source,
      uri: input.uri,
      checksum: input.checksum,
      ts: new Date().toISOString(),
    };

    this.store.set(record.id, record);

    return record;
  }

  async findById(id: string): Promise<EvidenceRecord | null> {
    return this.store.get(id) ?? null;
  }

  reset(): void {
    this.store.clear();
  }
}

export const evidenceRepo: EvidenceRepo = new InMemoryEvidenceRepo();
