import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import type { FastifyInstance } from 'fastify';
import { buildServer } from '../src/server';
import evidenceRoutes from '../src/routes/evidence';
import {
  createEvidenceService,
  InMemoryEvidenceStorage,
  type EvidenceService,
} from '../src/services/evidenceService';
import { InMemoryEvidenceRepo } from '../src/repos/evidenceRepo';

async function setupServer(service: EvidenceService): Promise<FastifyInstance> {
  const app = await buildServer();
  await app.register(evidenceRoutes, { service });
  await app.ready();
  return app;
}

describe('evidence service and routes', () => {
  it('stores buffer payloads, uploads to OSS, and exposes via route', async () => {
    const repo = new InMemoryEvidenceRepo();
    const storage = new InMemoryEvidenceStorage('oss://test-bucket');
    const service = createEvidenceService({ repo, storage });

    const payload = Buffer.from('mock-backtest-chart');
    const record = await service.create({ source: 'backtest', buffer: payload, filename: 'chart.png' });

    expect(record.source).toBe('backtest');
    expect(record.uri.startsWith('oss://test-bucket/')).toBe(true);

    const stored = await storage.getObject(record.uri);
    expect(stored?.equals(payload)).toBe(true);

    const checksum = createHash('sha256').update(payload).digest('hex');
    expect(record.checksum).toBe(checksum);

    const server = await setupServer(service);

    const traceId = 'trace-evidence-buffer';
    const response = await server.inject({
      method: 'GET',
      url: `/evidence/${record.id}`,
      headers: {
        'x-trace-id': traceId,
      },
    });

    expect(response.statusCode).toBe(200);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const body = response.json();
    expect(body).toMatchObject({
      id: record.id,
      source: 'backtest',
      uri: record.uri,
      checksum,
    });

    await server.close();
  });

  it('records metadata for existing URIs without uploading content', async () => {
    const repo = new InMemoryEvidenceRepo();
    const storage = new InMemoryEvidenceStorage('oss://noop');
    const service = createEvidenceService({ repo, storage });

    const uri = 'oss://existing/bucket/report.json';
    const record = await service.create({ source: 'report', uri });

    expect(record).toMatchObject({
      source: 'report',
      uri,
    });

    const checksum = createHash('sha256').update(uri).digest('hex');
    expect(record.checksum).toBe(checksum);

    const stored = await storage.getObject(uri);
    expect(stored).toBeNull();

    const server = await setupServer(service);
    const traceId = 'trace-evidence-uri';
    const response = await server.inject({
      method: 'GET',
      url: `/evidence/${record.id}`,
      headers: {
        'x-trace-id': traceId,
      },
    });

    expect(response.statusCode).toBe(200);
    expect(response.headers['x-trace-id']).toBe(traceId);

    const body = response.json();
    expect(body).toMatchObject({
      id: record.id,
      source: 'report',
      uri,
      checksum,
    });

    await server.close();
  });
});
