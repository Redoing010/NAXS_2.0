import { useCallback } from 'react';
import apiService from '../services/api';
import {
  useKnowledgeGraph,
  useKnowledgeGraphActions,
  type KnowledgeGraphContextPayload,
  type KnowledgeGraphEntity,
  type KnowledgeGraphRelation,
} from '../store';

const normalizeEntity = (item: unknown): KnowledgeGraphEntity => {
  if (!item || typeof item !== 'object') {
    return {
      id: String(item ?? `entity_${Date.now()}`),
      name: String(item ?? '未知实体'),
    };
  }

  const record = item as Record<string, unknown>;

  return {
    id:
      (record.id as string | undefined) ??
      (record.entity_id as string | undefined) ??
      (record.identifier as string | undefined) ??
      (record.name as string | undefined) ??
      `entity_${Date.now()}`,
    name:
      (record.name as string | undefined) ??
      (record.label as string | undefined) ??
      (record.title as string | undefined) ??
      '未知实体',
    type: (record.type as string | undefined) ?? (record.category as string | undefined) ?? (record.kind as string | undefined),
    description:
      (record.description as string | undefined) ??
      (record.summary as string | undefined) ??
      (record.brief as string | undefined),
    attributes:
      (record.attributes as Record<string, unknown> | undefined) ??
      (record.metadata as Record<string, unknown> | undefined) ??
      (record.properties as Record<string, unknown> | undefined),
    score:
      (record.score as number | undefined) ??
      (record.weight as number | undefined) ??
      (record.confidence as number | undefined),
  };
};

const normalizeRelation = (item: unknown): KnowledgeGraphRelation => {
  if (!item || typeof item !== 'object') {
    const id = String(item ?? `relation_${Date.now()}`);
    return {
      id,
      source: id,
      target: id,
      type: 'related_to',
    };
  }

  const record = item as Record<string, unknown>;
  const source =
    (record.source as string | undefined) ??
    (record.from as string | undefined) ??
    (record.head as string | undefined) ??
    (record.entity_a as string | undefined) ??
    '';
  const target =
    (record.target as string | undefined) ??
    (record.to as string | undefined) ??
    (record.tail as string | undefined) ??
    (record.entity_b as string | undefined) ??
    '';
  const type =
    (record.type as string | undefined) ??
    (record.relation as string | undefined) ??
    (record.edge_type as string | undefined) ??
    'related_to';

  return {
    id: (record.id as string | undefined) ?? `${source}-${type}-${target}`,
    source,
    target,
    type,
    weight:
      (record.weight as number | undefined) ??
      (record.score as number | undefined) ??
      (record.probability as number | undefined),
    metadata:
      (record.metadata as Record<string, unknown> | undefined) ??
      (record.attributes as Record<string, unknown> | undefined) ??
      (record.details as Record<string, unknown> | undefined),
  };
};

export const useKnowledgeGraphService = () => {
  const knowledgeGraph = useKnowledgeGraph();
  const {
    setKnowledgeGraphEnabled,
    setKnowledgeGraphLoading,
    setKnowledgeGraphError,
    setKnowledgeGraphResults,
    setKnowledgeGraphSelectedEntity,
    setKnowledgeGraphRelations,
    addKnowledgeGraphHistory,
    clearKnowledgeGraph,
  } = useKnowledgeGraphActions();

  const toggleKnowledgeGraph = useCallback(
    (enabled: boolean) => {
      setKnowledgeGraphEnabled(enabled);
    },
    [setKnowledgeGraphEnabled]
  );

  const searchKnowledgeGraph = useCallback(
    async (query: { entity?: string; relation?: string; limit?: number }) => {
      const searchTerm = query.entity?.trim();
      if (!searchTerm) {
        setKnowledgeGraphError('请输入要检索的实体名称');
        return;
      }

      setKnowledgeGraphLoading(true);
      setKnowledgeGraphError(null);

      try {
        const response = await apiService.searchKnowledgeGraph({
          entity: searchTerm,
          relation: query.relation,
          limit: query.limit ?? 20,
        });

        if (response.status === 'ok' && Array.isArray(response.data)) {
          const results = response.data.map(normalizeEntity);
          setKnowledgeGraphResults(results);
          addKnowledgeGraphHistory({
            id: `kg_query_${Date.now()}`,
            query: searchTerm,
            relation: query.relation,
            timestamp: new Date().toISOString(),
            results: results.length,
          });
          if (results.length > 0) {
            setKnowledgeGraphSelectedEntity(results[0]);
          }
        } else {
          setKnowledgeGraphError(response.error || '知识图谱查询失败');
          setKnowledgeGraphResults([]);
        }
      } catch (error: unknown) {
        console.error('Knowledge graph search failed:', error);
        setKnowledgeGraphError(error instanceof Error ? error.message : '知识图谱服务不可用');
        setKnowledgeGraphResults([]);
      } finally {
        setKnowledgeGraphLoading(false);
      }
    },
    [addKnowledgeGraphHistory, setKnowledgeGraphError, setKnowledgeGraphLoading, setKnowledgeGraphResults, setKnowledgeGraphSelectedEntity]
  );

  const selectEntity = useCallback(
    async (entity: KnowledgeGraphEntity) => {
      if (!entity?.id) {
        setKnowledgeGraphSelectedEntity(undefined);
        setKnowledgeGraphRelations([]);
        return;
      }

      setKnowledgeGraphLoading(true);
      setKnowledgeGraphError(null);

      try {
        const [detailResponse, relationsResponse] = await Promise.all([
          apiService.getEntityDetails(entity.id),
          apiService.getRelatedEntities(entity.id),
        ]);

        const entityDetails =
          detailResponse.status === 'ok' ? { ...entity, ...normalizeEntity(detailResponse.data) } : entity;
        const relations =
          relationsResponse.status === 'ok' && Array.isArray(relationsResponse.data)
            ? relationsResponse.data.map(normalizeRelation)
            : [];

        setKnowledgeGraphSelectedEntity(entityDetails);
        setKnowledgeGraphRelations(relations);
      } catch (error: unknown) {
        console.error('Failed to load entity details:', error);
        setKnowledgeGraphError(error instanceof Error ? error.message : '获取实体详情失败');
        setKnowledgeGraphSelectedEntity(entity);
      } finally {
        setKnowledgeGraphLoading(false);
      }
    },
    [setKnowledgeGraphError, setKnowledgeGraphLoading, setKnowledgeGraphRelations, setKnowledgeGraphSelectedEntity]
  );

  const resetKnowledgeGraph = useCallback(() => {
    clearKnowledgeGraph();
  }, [clearKnowledgeGraph]);

  const getKnowledgeGraphContext = useCallback((): KnowledgeGraphContextPayload => {
    if (!knowledgeGraph.enabled) {
      return { enabled: false };
    }

    const focusEntity = knowledgeGraph.selectedEntity;
    const entities = focusEntity ? [focusEntity] : knowledgeGraph.searchResults.slice(0, 5);
    const relations = knowledgeGraph.relations.slice(0, 20);

    return {
      enabled: true,
      focus: focusEntity?.name ?? knowledgeGraph.searchResults[0]?.name,
      entities,
      relations,
    };
  }, [knowledgeGraph.enabled, knowledgeGraph.relations, knowledgeGraph.searchResults, knowledgeGraph.selectedEntity]);

  return {
    knowledgeGraph,
    toggleKnowledgeGraph,
    searchKnowledgeGraph,
    selectEntity,
    resetKnowledgeGraph,
    getKnowledgeGraphContext,
  };
};

export type { KnowledgeGraphEntity, KnowledgeGraphRelation, KnowledgeGraphContextPayload };
