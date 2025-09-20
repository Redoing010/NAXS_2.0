import fp from 'fastify-plugin';
import type { FastifyInstance } from 'fastify';
import { evidenceService, type EvidenceService } from '../services/evidenceService';

interface EvidenceRouteOptions {
  service?: EvidenceService;
}

async function evidenceRoutes(app: FastifyInstance, options: EvidenceRouteOptions = {}): Promise<void> {
  const service = options.service ?? evidenceService;

  app.get<{ Params: { id: string } }>('/evidence/:id', async (request) => {
    const { id } = request.params;
    const record = await service.getById(id);
    return record;
  });
}

export default fp(evidenceRoutes, {
  fastify: '4.x',
  name: 'evidence-routes',
});
