import React, { useState } from 'react';
import { clsx } from 'clsx';
import {
  Network,
  Search,
  Loader2,
  ToggleLeft,
  ToggleRight,
  Layers,
  Link,
  Sparkles,
  History,
  Info,
  Eraser,
  AlertCircle,
} from 'lucide-react';
import { useKnowledgeGraphService } from '../../hooks/useKnowledgeGraph';

interface KnowledgeGraphPanelProps {
  className?: string;
}

const KnowledgeGraphPanel: React.FC<KnowledgeGraphPanelProps> = ({ className }) => {
  const [query, setQuery] = useState('');
  const [relation, setRelation] = useState('');
  const { knowledgeGraph, toggleKnowledgeGraph, searchKnowledgeGraph, selectEntity, resetKnowledgeGraph } =
    useKnowledgeGraphService();

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    searchKnowledgeGraph({ entity: query, relation });
  };

  const handleSelectEntity = (entityId: string) => {
    const entity = knowledgeGraph.searchResults.find((item) => item.id === entityId);
    if (entity) {
      selectEntity(entity);
    }
  };

  return (
    <div
      className={clsx(
        'flex flex-col h-full bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden',
        className
      )}
    >
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-9 h-9 bg-primary-100 rounded-lg flex items-center justify-center">
            <Network className="w-4 h-4 text-primary-600" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-900">知识图谱联动</h3>
            <p className="text-xs text-gray-500">检索结构化知识，辅助 QWEN 推理</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => toggleKnowledgeGraph(!knowledgeGraph.enabled)}
          className={clsx(
            'inline-flex items-center px-3 py-1 text-xs font-medium rounded-full border transition-colors',
            knowledgeGraph.enabled
              ? 'border-primary-200 bg-primary-50 text-primary-700'
              : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50'
          )}
        >
          {knowledgeGraph.enabled ? (
            <ToggleRight className="w-4 h-4 mr-1" />
          ) : (
            <ToggleLeft className="w-4 h-4 mr-1" />
          )}
          {knowledgeGraph.enabled ? '已开启' : '未开启'}
        </button>
      </div>

      <div className="p-4 space-y-4 flex-1 overflow-y-auto">
        <form onSubmit={handleSubmit} className="space-y-2">
          <label className="text-xs font-medium text-gray-600">实体检索</label>
          <div className="flex space-x-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="输入公司/行业/事件等关键词"
                className="w-full pl-9 pr-3 py-2 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-100 focus:border-primary-300"
              />
            </div>
            <input
              value={relation}
              onChange={(event) => setRelation(event.target.value)}
              placeholder="关系类型 (可选)"
              className="w-32 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-100 focus:border-primary-300"
            />
            <button
              type="submit"
              className="inline-flex items-center px-3 py-2 text-sm font-medium text-white bg-primary-600 rounded-lg hover:bg-primary-700"
              disabled={knowledgeGraph.loading}
            >
              {knowledgeGraph.loading ? <Loader2 className="w-4 h-4 animate-spin" /> : '检索'}
            </button>
          </div>
        </form>

        {knowledgeGraph.error && (
          <div className="flex items-start space-x-2 p-3 bg-red-50 border border-red-100 rounded-lg text-xs text-red-600">
            <AlertCircle className="w-4 h-4 mt-0.5" />
            <div>
              <p className="font-medium">知识图谱异常</p>
              <p>{knowledgeGraph.error}</p>
            </div>
          </div>
        )}

        {knowledgeGraph.history.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2 text-xs text-gray-500">
                <History className="w-3 h-3" />
                <span>最近检索</span>
              </div>
              <button
                type="button"
                onClick={resetKnowledgeGraph}
                className="inline-flex items-center text-xs text-gray-400 hover:text-gray-600"
              >
                <Eraser className="w-3 h-3 mr-1" /> 清空
              </button>
            </div>
            <div className="space-y-1">
              {knowledgeGraph.history.slice(0, 5).map((item) => (
                <div key={item.id} className="text-xs text-gray-500 flex items-center justify-between">
                  <span className="truncate">{item.query}</span>
                  <span>{new Date(item.timestamp).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="flex items-center space-x-2 text-xs text-gray-500 mb-2">
            <Layers className="w-3 h-3" />
            <span>实体候选</span>
          </div>
          {knowledgeGraph.searchResults.length === 0 ? (
            <p className="text-xs text-gray-400">暂无结果，尝试新的关键词检索。</p>
          ) : (
            <div className="space-y-2">
              {knowledgeGraph.searchResults.slice(0, 10).map((entity) => {
                const isSelected = knowledgeGraph.selectedEntity?.id === entity.id;
                return (
                  <button
                    type="button"
                    key={entity.id}
                    onClick={() => handleSelectEntity(entity.id)}
                    className={clsx(
                      'w-full text-left border rounded-lg px-3 py-2',
                      'transition-colors duration-150',
                      isSelected
                        ? 'border-primary-300 bg-primary-50 text-primary-700'
                        : 'border-gray-200 hover:border-primary-200 hover:bg-primary-50/50'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{entity.name}</span>
                      {typeof entity.score === 'number' && (
                        <span className="text-xs text-gray-500">置信度 {Math.round(entity.score * 100)}%</span>
                      )}
                    </div>
                    {entity.type && <p className="text-xs text-gray-500">类型：{entity.type}</p>}
                    {entity.description && (
                      <p className="text-xs text-gray-500 truncate">{entity.description}</p>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {knowledgeGraph.selectedEntity && (
          <div className="border border-primary-100 rounded-lg p-3 bg-primary-50/40">
            <div className="flex items-start space-x-2">
              <Sparkles className="w-4 h-4 text-primary-500 mt-0.5" />
              <div>
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-primary-700">已选实体</h4>
                  <span className="text-xs text-primary-600">
                    {knowledgeGraph.relations.length} 条关系已加载
                  </span>
                </div>
                <p className="text-sm text-primary-800 mt-1">{knowledgeGraph.selectedEntity.name}</p>
                {knowledgeGraph.selectedEntity.description && (
                  <p className="text-xs text-primary-700 leading-snug mt-1">
                    {knowledgeGraph.selectedEntity.description}
                  </p>
                )}
              </div>
            </div>

            {knowledgeGraph.relations.length > 0 ? (
              <div className="mt-3 space-y-1 max-h-40 overflow-y-auto pr-1">
                {knowledgeGraph.relations.slice(0, 20).map((relation) => (
                  <div
                    key={relation.id}
                    className="flex items-center justify-between text-xs text-primary-700 bg-white/70 rounded px-2 py-1"
                  >
                    <span className="truncate">{relation.source}</span>
                    <Link className="w-3 h-3 text-primary-500 mx-1" />
                    <span className="truncate">{relation.target}</span>
                    <span className="ml-2 text-[11px] text-primary-500">{relation.type}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-primary-600 mt-2">未检索到关系，可尝试指定关系类型。</p>
            )}
          </div>
        )}

        <div className="flex items-start space-x-2 text-xs text-gray-400 bg-gray-50 border border-gray-100 rounded-lg p-3">
          <Info className="w-4 h-4" />
          <p>
            开启知识图谱后，系统会在调用 QWEN 进行推理时注入结构化实体与关系上下文，帮助模型生成更可靠的投资分析。
          </p>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeGraphPanel;
