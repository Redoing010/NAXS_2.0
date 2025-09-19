# 知识检索器 - 提供图谱查询和知识检索功能

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    np = None

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_SEARCH = "entity_search"          # 实体搜索
    RELATION_SEARCH = "relation_search"      # 关系搜索
    PATH_SEARCH = "path_search"              # 路径搜索
    NEIGHBOR_SEARCH = "neighbor_search"      # 邻居搜索
    SIMILARITY_SEARCH = "similarity_search"  # 相似性搜索
    SUBGRAPH_SEARCH = "subgraph_search"      # 子图搜索
    SEMANTIC_SEARCH = "semantic_search"      # 语义搜索

@dataclass
class QueryResult:
    """查询结果"""
    query_type: QueryType
    results: List[Dict[str, Any]]
    total_count: int
    execution_time: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class EntityMatch:
    """实体匹配结果"""
    entity_id: str
    entity_name: str
    entity_type: str
    similarity_score: float
    properties: Dict[str, Any]

@dataclass
class RelationMatch:
    """关系匹配结果"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    confidence: float
    properties: Dict[str, Any]

class KnowledgeRetriever:
    """知识检索器
    
    提供多种知识图谱查询和检索功能：
    1. 实体搜索和匹配
    2. 关系查询和分析
    3. 路径发现和推理
    4. 语义相似性搜索
    5. 子图提取和分析
    """
    
    def __init__(self, graph_builder, config: Dict[str, Any] = None):
        self.graph_builder = graph_builder
        self.config = config or {}
        
        # 查询缓存
        self.query_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 1800)  # 30分钟
        
        # 初始化文本向量化器（用于语义搜索）
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,  # 中文没有内置停用词
                ngram_range=(1, 2)
            )
            self.entity_vectors = None
            self.entity_texts = {}
        else:
            self.vectorizer = None
        
        logger.info("知识检索器初始化完成")
    
    def _get_cache_key(self, query_type: str, **kwargs) -> str:
        """生成缓存键"""
        params_str = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{query_type}_{params_str}"
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """检查缓存是否有效"""
        if 'timestamp' not in cache_entry:
            return False
        
        elapsed = (datetime.now() - cache_entry['timestamp']).total_seconds()
        return elapsed < self.cache_ttl
    
    async def search_entities(self, query: str, entity_types: List[str] = None,
                            limit: int = 10, similarity_threshold: float = 0.3) -> QueryResult:
        """搜索实体
        
        Args:
            query: 搜索查询
            entity_types: 限制的实体类型
            limit: 返回结果数量限制
            similarity_threshold: 相似度阈值
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(
                'entity_search', 
                query=query, 
                types=entity_types, 
                limit=limit,
                threshold=similarity_threshold
            )
            
            if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
                logger.debug("使用实体搜索缓存结果")
                return self.query_cache[cache_key]['result']
            
            matches = []
            
            # 获取图谱
            if self.graph_builder.backend == 'networkx':
                graph = self.graph_builder.graph
                
                for node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    node_name = node_data.get('name', '')
                    node_type = node_data.get('type', '')
                    
                    # 类型过滤
                    if entity_types and node_type not in entity_types:
                        continue
                    
                    # 计算相似度
                    similarity = self._calculate_text_similarity(query, node_name)
                    
                    if similarity >= similarity_threshold:
                        match = EntityMatch(
                            entity_id=node_id,
                            entity_name=node_name,
                            entity_type=node_type,
                            similarity_score=similarity,
                            properties=node_data
                        )
                        matches.append(match)
            
            # 按相似度排序
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # 限制结果数量
            matches = matches[:limit]
            
            # 转换为字典格式
            results = [
                {
                    'entity_id': match.entity_id,
                    'entity_name': match.entity_name,
                    'entity_type': match.entity_type,
                    'similarity_score': match.similarity_score,
                    'properties': match.properties
                }
                for match in matches
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = QueryResult(
                query_type=QueryType.ENTITY_SEARCH,
                results=results,
                total_count=len(results),
                execution_time=execution_time,
                confidence=0.8,
                metadata={'query': query, 'entity_types': entity_types}
            )
            
            # 缓存结果
            self.query_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            logger.info(f"实体搜索完成，找到{len(results)}个匹配实体")
            return result
            
        except Exception as e:
            logger.error(f"实体搜索失败: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query_type=QueryType.ENTITY_SEARCH,
                results=[],
                total_count=0,
                execution_time=execution_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def search_relations(self, source_entity: str = None, target_entity: str = None,
                             relation_types: List[str] = None, limit: int = 10) -> QueryResult:
        """搜索关系
        
        Args:
            source_entity: 源实体ID
            target_entity: 目标实体ID
            relation_types: 关系类型列表
            limit: 返回结果数量限制
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            matches = []
            
            if self.graph_builder.backend == 'networkx':
                graph = self.graph_builder.graph
                
                for u, v, edge_data in graph.edges(data=True):
                    # 源实体过滤
                    if source_entity and u != source_entity:
                        continue
                    
                    # 目标实体过滤
                    if target_entity and v != target_entity:
                        continue
                    
                    # 关系类型过滤
                    edge_type = edge_data.get('type', '')
                    if relation_types and edge_type not in relation_types:
                        continue
                    
                    match = RelationMatch(
                        source_id=u,
                        target_id=v,
                        relation_type=edge_type,
                        weight=edge_data.get('weight', 1.0),
                        confidence=edge_data.get('confidence', 1.0),
                        properties=edge_data
                    )
                    matches.append(match)
            
            # 按权重和置信度排序
            matches.sort(key=lambda x: (x.weight * x.confidence), reverse=True)
            
            # 限制结果数量
            matches = matches[:limit]
            
            # 转换为字典格式
            results = [
                {
                    'source_id': match.source_id,
                    'target_id': match.target_id,
                    'relation_type': match.relation_type,
                    'weight': match.weight,
                    'confidence': match.confidence,
                    'properties': match.properties
                }
                for match in matches
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = QueryResult(
                query_type=QueryType.RELATION_SEARCH,
                results=results,
                total_count=len(results),
                execution_time=execution_time,
                confidence=0.9,
                metadata={
                    'source_entity': source_entity,
                    'target_entity': target_entity,
                    'relation_types': relation_types
                }
            )
            
            logger.info(f"关系搜索完成，找到{len(results)}个匹配关系")
            return result
            
        except Exception as e:
            logger.error(f"关系搜索失败: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query_type=QueryType.RELATION_SEARCH,
                results=[],
                total_count=0,
                execution_time=execution_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def find_neighbors(self, entity_id: str, max_depth: int = 2, 
                           relation_types: List[str] = None, limit: int = 20) -> QueryResult:
        """查找邻居节点
        
        Args:
            entity_id: 实体ID
            max_depth: 最大搜索深度
            relation_types: 关系类型过滤
            limit: 返回结果数量限制
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            if self.graph_builder.backend == 'networkx':
                graph = self.graph_builder.graph
                
                if entity_id not in graph:
                    return QueryResult(
                        query_type=QueryType.NEIGHBOR_SEARCH,
                        results=[],
                        total_count=0,
                        execution_time=0.0,
                        confidence=0.0,
                        metadata={'error': f'实体不存在: {entity_id}'}
                    )
                
                neighbors = []
                visited = set()
                queue = [(entity_id, 0)]  # (节点ID, 深度)
                
                while queue and len(neighbors) < limit:
                    current_id, depth = queue.pop(0)
                    
                    if current_id in visited or depth > max_depth:
                        continue
                    
                    visited.add(current_id)
                    
                    # 获取邻居
                    for neighbor_id in graph.neighbors(current_id):
                        if neighbor_id not in visited:
                            # 检查关系类型
                            edge_data = graph.get_edge_data(current_id, neighbor_id)
                            if edge_data:
                                for edge_key, edge_attrs in edge_data.items():
                                    edge_type = edge_attrs.get('type')
                                    if not relation_types or edge_type in relation_types:
                                        neighbor_data = graph.nodes[neighbor_id]
                                        
                                        neighbor_info = {
                                            'entity_id': neighbor_id,
                                            'entity_name': neighbor_data.get('name', ''),
                                            'entity_type': neighbor_data.get('type', ''),
                                            'relation_type': edge_type,
                                            'depth': depth + 1,
                                            'weight': edge_attrs.get('weight', 1.0),
                                            'confidence': edge_attrs.get('confidence', 1.0),
                                            'properties': neighbor_data
                                        }
                                        neighbors.append(neighbor_info)
                                        
                                        if depth + 1 < max_depth:
                                            queue.append((neighbor_id, depth + 1))
                                        break
                
                # 按深度和权重排序
                neighbors.sort(key=lambda x: (x['depth'], -x['weight']))
                neighbors = neighbors[:limit]
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = QueryResult(
                    query_type=QueryType.NEIGHBOR_SEARCH,
                    results=neighbors,
                    total_count=len(neighbors),
                    execution_time=execution_time,
                    confidence=0.9,
                    metadata={
                        'entity_id': entity_id,
                        'max_depth': max_depth,
                        'relation_types': relation_types
                    }
                )
                
                logger.info(f"邻居搜索完成，找到{len(neighbors)}个邻居节点")
                return result
            
        except Exception as e:
            logger.error(f"邻居搜索失败: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query_type=QueryType.NEIGHBOR_SEARCH,
                results=[],
                total_count=0,
                execution_time=execution_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def find_paths(self, source_id: str, target_id: str, 
                        max_paths: int = 5, max_length: int = 4) -> QueryResult:
        """查找路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            max_paths: 最大路径数
            max_length: 最大路径长度
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            if self.graph_builder.backend == 'networkx':
                graph = self.graph_builder.graph
                
                if source_id not in graph or target_id not in graph:
                    return QueryResult(
                        query_type=QueryType.PATH_SEARCH,
                        results=[],
                        total_count=0,
                        execution_time=0.0,
                        confidence=0.0,
                        metadata={'error': '源节点或目标节点不存在'}
                    )
                
                paths = []
                
                try:
                    # 查找所有简单路径
                    all_paths = nx.all_simple_paths(
                        graph, source_id, target_id, cutoff=max_length
                    )
                    
                    for path in all_paths:
                        if len(paths) >= max_paths:
                            break
                        
                        # 计算路径权重
                        path_weight = 0.0
                        path_relations = []
                        
                        for i in range(len(path) - 1):
                            edge_data = graph.get_edge_data(path[i], path[i+1])
                            if edge_data:
                                for edge_attrs in edge_data.values():
                                    weight = edge_attrs.get('weight', 1.0)
                                    path_weight += weight
                                    path_relations.append({
                                        'source': path[i],
                                        'target': path[i+1],
                                        'type': edge_attrs.get('type', ''),
                                        'weight': weight
                                    })
                                    break
                        
                        path_info = {
                            'path': path,
                            'length': len(path) - 1,
                            'weight': path_weight,
                            'relations': path_relations,
                            'nodes': [
                                {
                                    'id': node_id,
                                    'name': graph.nodes[node_id].get('name', ''),
                                    'type': graph.nodes[node_id].get('type', '')
                                }
                                for node_id in path
                            ]
                        }
                        paths.append(path_info)
                
                except nx.NetworkXNoPath:
                    pass  # 没有路径
                
                # 按路径权重排序
                paths.sort(key=lambda x: x['weight'], reverse=True)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = QueryResult(
                    query_type=QueryType.PATH_SEARCH,
                    results=paths,
                    total_count=len(paths),
                    execution_time=execution_time,
                    confidence=0.8,
                    metadata={
                        'source_id': source_id,
                        'target_id': target_id,
                        'max_paths': max_paths,
                        'max_length': max_length
                    }
                )
                
                logger.info(f"路径搜索完成，找到{len(paths)}条路径")
                return result
            
        except Exception as e:
            logger.error(f"路径搜索失败: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query_type=QueryType.PATH_SEARCH,
                results=[],
                total_count=0,
                execution_time=execution_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def semantic_search(self, query: str, limit: int = 10) -> QueryResult:
        """语义搜索
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            if self.vectorizer is None or TfidfVectorizer is None:
                logger.warning("语义搜索需要scikit-learn库")
                return QueryResult(
                    query_type=QueryType.SEMANTIC_SEARCH,
                    results=[],
                    total_count=0,
                    execution_time=0.0,
                    confidence=0.0,
                    metadata={'error': 'scikit-learn未安装'}
                )
            
            # 构建实体文本索引（如果还没有）
            if self.entity_vectors is None:
                await self._build_text_index()
            
            if not self.entity_texts:
                return QueryResult(
                    query_type=QueryType.SEMANTIC_SEARCH,
                    results=[],
                    total_count=0,
                    execution_time=0.0,
                    confidence=0.0,
                    metadata={'error': '没有可搜索的文本内容'}
                )
            
            # 向量化查询
            query_vector = self.vectorizer.transform([query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, self.entity_vectors)[0]
            
            # 获取最相似的实体
            entity_ids = list(self.entity_texts.keys())
            scored_entities = list(zip(entity_ids, similarities))
            scored_entities.sort(key=lambda x: x[1], reverse=True)
            
            # 构建结果
            results = []
            for entity_id, similarity in scored_entities[:limit]:
                if self.graph_builder.backend == 'networkx':
                    graph = self.graph_builder.graph
                    if entity_id in graph:
                        node_data = graph.nodes[entity_id]
                        result_item = {
                            'entity_id': entity_id,
                            'entity_name': node_data.get('name', ''),
                            'entity_type': node_data.get('type', ''),
                            'similarity_score': float(similarity),
                            'text_content': self.entity_texts[entity_id],
                            'properties': node_data
                        }
                        results.append(result_item)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = QueryResult(
                query_type=QueryType.SEMANTIC_SEARCH,
                results=results,
                total_count=len(results),
                execution_time=execution_time,
                confidence=0.7,
                metadata={'query': query}
            )
            
            logger.info(f"语义搜索完成，找到{len(results)}个相关实体")
            return result
            
        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query_type=QueryType.SEMANTIC_SEARCH,
                results=[],
                total_count=0,
                execution_time=execution_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _build_text_index(self):
        """构建文本索引"""
        try:
            if self.graph_builder.backend == 'networkx':
                graph = self.graph_builder.graph
                
                # 收集实体文本
                texts = []
                entity_ids = []
                
                for node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    
                    # 构建实体文本（名称 + 属性）
                    text_parts = [node_data.get('name', '')]
                    
                    # 添加相关属性文本
                    for key, value in node_data.items():
                        if key not in ['name', 'type', 'created_at', 'updated_at'] and isinstance(value, str):
                            text_parts.append(value)
                    
                    entity_text = ' '.join(text_parts)
                    if entity_text.strip():
                        texts.append(entity_text)
                        entity_ids.append(node_id)
                        self.entity_texts[node_id] = entity_text
                
                # 训练向量化器
                if texts:
                    self.entity_vectors = self.vectorizer.fit_transform(texts)
                    logger.info(f"文本索引构建完成，索引了{len(texts)}个实体")
                
        except Exception as e:
            logger.error(f"文本索引构建失败: {str(e)}")
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的字符串相似度计算
        if not text1 or not text2:
            return 0.0
        
        # 完全匹配
        if text1 == text2:
            return 1.0
        
        # 包含关系
        if text1 in text2 or text2 in text1:
            return 0.8
        
        # 计算编辑距离相似度
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)
    
    async def get_subgraph(self, entity_ids: List[str], include_neighbors: bool = True,
                          max_neighbors: int = 5) -> QueryResult:
        """获取子图
        
        Args:
            entity_ids: 实体ID列表
            include_neighbors: 是否包含邻居节点
            max_neighbors: 每个节点的最大邻居数
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            subgraph_data = await self.graph_builder.export_subgraph(
                entity_ids, include_neighbors
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = QueryResult(
                query_type=QueryType.SUBGRAPH_SEARCH,
                results=[subgraph_data],
                total_count=1,
                execution_time=execution_time,
                confidence=1.0,
                metadata={
                    'entity_ids': entity_ids,
                    'include_neighbors': include_neighbors,
                    'max_neighbors': max_neighbors
                }
            )
            
            logger.info(f"子图提取完成，包含{subgraph_data['stats']['node_count']}个节点")
            return result
            
        except Exception as e:
            logger.error(f"子图提取失败: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query_type=QueryType.SUBGRAPH_SEARCH,
                results=[],
                total_count=0,
                execution_time=execution_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def clear_cache(self):
        """清空查询缓存"""
        self.query_cache.clear()
        logger.info("查询缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        valid_entries = sum(
            1 for entry in self.query_cache.values() 
            if self._is_cache_valid(entry)
        )
        
        return {
            'total_entries': len(self.query_cache),
            'valid_entries': valid_entries,
            'invalid_entries': len(self.query_cache) - valid_entries,
            'cache_ttl': self.cache_ttl
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            'backend': self.graph_builder.backend,
            'cache_enabled': True,
            'semantic_search_available': self.vectorizer is not None,
            'indexed_entities': len(self.entity_texts) if hasattr(self, 'entity_texts') else 0,
            'supported_query_types': [qt.value for qt in QueryType]
        }