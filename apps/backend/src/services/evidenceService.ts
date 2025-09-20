import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';
import { AppError } from '../utils/errors';
import {
  evidenceRepo,
  type EvidenceRepo,
  type EvidenceRecord,
  type InsertEvidenceInput,
} from '../repos/evidenceRepo';

export interface EvidenceStorage {
  putObject(key: string, body: Buffer): Promise<string>;
  getObject(uri: string): Promise<Buffer | null>;
  reset(): void;
}

export class InMemoryEvidenceStorage implements EvidenceStorage {
  private readonly store = new Map<string, Buffer>();

  constructor(private readonly baseUri = 'oss://mock-bucket') {}

  async putObject(key: string, body: Buffer): Promise<string> {
    const uri = `${this.baseUri}/${key}`;
    this.store.set(uri, Buffer.from(body));
    return uri;
  }

  async getObject(uri: string): Promise<Buffer | null> {
    const stored = this.store.get(uri);
    return stored ? Buffer.from(stored) : null;
  }

  reset(): void {
    this.store.clear();
  }
}

export interface CreateEvidenceFromBuffer {
  source: string;
  buffer: Buffer;
  filename?: string;
}

export interface CreateEvidenceFromUri {
  source: string;
  uri: string;
}

export type CreateEvidenceInput = CreateEvidenceFromBuffer | CreateEvidenceFromUri;

function isBufferInput(input: CreateEvidenceInput): input is CreateEvidenceFromBuffer {
  return (input as CreateEvidenceFromBuffer).buffer !== undefined;
}

function buildObjectKey(filename?: string): string {
  const id = randomUUID();

  if (!filename) {
    return id;
  }

  const safeName = filename.trim();

  if (safeName === '') {
    return id;
  }

  const extension = path.extname(safeName);

  return extension ? `${id}${extension}` : `${id}-${safeName.replace(/\s+/g, '-')}`;
}

function computeChecksum(content: Buffer | string): string {
  const hash = createHash('sha256');
  hash.update(content);
  return hash.digest('hex');
}

export class EvidenceService {
  constructor(private readonly repo: EvidenceRepo, private readonly storage: EvidenceStorage) {}

  async create(input: CreateEvidenceInput): Promise<EvidenceRecord> {
    if (isBufferInput(input)) {
      const key = buildObjectKey(input.filename);
      const checksum = computeChecksum(input.buffer);
      const uri = await this.storage.putObject(key, input.buffer);

      return this.persist({
        source: input.source,
        uri,
        checksum,
      });
    }

    const checksum = computeChecksum(input.uri);

    return this.persist({
      source: input.source,
      uri: input.uri,
      checksum,
    });
  }

  async getById(id: string): Promise<EvidenceRecord> {
    const record = await this.repo.findById(id);

    if (!record) {
      throw new AppError('EVIDENCE_NOT_FOUND', `Evidence ${id} not found`, 404);
    }

    return record;
  }

  private async persist(input: InsertEvidenceInput): Promise<EvidenceRecord> {
    return this.repo.insert(input);
  }
}

export interface CreateEvidenceServiceDeps {
  repo?: EvidenceRepo;
  storage?: EvidenceStorage;
}

export function createEvidenceService(deps: CreateEvidenceServiceDeps = {}): EvidenceService {
  const repo = deps.repo ?? evidenceRepo;
  const storage = deps.storage ?? new InMemoryEvidenceStorage();

  return new EvidenceService(repo, storage);
}

export const evidenceService = createEvidenceService();
