# 图谱构建模块
# 实现知识图谱的结构定义、模式管理和构建功能

import logging
import json
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """节点类型"""
    # 实体类型
    COMPANY = "company"            # 公司
    PERSON = "person"              # 人物
    INDUSTRY = "industry"          # 行业
    CONCEPT = "concept"            # 概念
    EVENT = "event"                # 事件
    NEWS = "news"                  # 新闻
    REPORT = "report"              # 报告
    
    # 金融类型
    STOCK = "stock"                # 股票
    FUND = "fund"                  # 基金
    INDEX = "index"                # 指数
    BOND = "bond"                  # 债券
    COMMODITY = "commodity"        # 商品
    
    # 指标类型
    FINANCIAL_METRIC = "financial_metric"  # 财务指标
    MARKET_METRIC = "market_metric"        # 市场指标
    ECONOMIC_INDICATOR = "economic_indicator"  # 经济指标
    
    # 其他
    LOCATION = "location"          # 地理位置
    TIME_PERIOD = "time_period"    # 时间周期
    CUSTOM = "custom"              # 自定义

class EdgeType(Enum):
    """边类型"""
    # 关系类型
    OWNS = "owns"                  # 拥有
    CONTROLS = "controls"          # 控制
    INVESTS_IN = "invests_in"      # 投资
    COMPETES_WITH = "competes_with"  # 竞争
    COOPERATES_WITH = "cooperates_with"  # 合作
    SUPPLIES_TO = "supplies_to"    # 供应
    
    # 影响关系
    INFLUENCES = "influences"      # 影响
    CAUSES = "causes"              # 导致
    CORRELATES_WITH = "correlates_with"  # 相关
    DEPENDS_ON = "depends_on"      # 依赖
    
    # 分类关系
    BELONGS_TO = "belongs_to"      # 属于
    CONTAINS = "contains"          # 包含
    IS_TYPE_OF = "is_type_of"      # 是...的类型
    
    # 时间关系
    OCCURS_BEFORE = "occurs_before"  # 发生在...之前
    OCCURS_AFTER = "occurs_after"    # 发生在...之后
    OCCURS_DURING = "occurs_during"  # 发生在...期间
    
    # 其他
    MENTIONS = "mentions"          # 提及
    REFERENCES = "references"      # 引用
    SIMILAR_TO = "similar_to"      # 相似
    CUSTOM = "custom"              # 自定义

@dataclass
class GraphNode:
    """图节点"""
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    labels: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'properties': self.properties,
            'labels': list(self.labels),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """从字典创建节点"""
        return cls(
            id=data['id'],
            type=NodeType(data['type']),
            name=data['name'],
            properties=data.get('properties', {}),
            labels=set(data.get('labels', [])),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        )

@dataclass
class GraphEdge:
    """图边"""
    id: str
    type: EdgeType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type.value,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'properties': self.properties,
            'weight': self.weight,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """从字典创建边"""
        return cls(
            id=data['id'],
            type=EdgeType(data['type']),
            source_id=data['source_id'],
            target_id=data['target_id'],
            properties=data.get('properties', {}),
            weight=data.get('weight', 1.0),
            confidence=data.get('confidence', 1.0),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        )

@dataclass
class GraphSchema:
    """图谱模式定义"""
    name: str
    description: str = ""
    node_types: Set[NodeType] = field(default_factory=set)
    edge_types: Set[EdgeType] = field(default_factory=set)
    
    # 节点属性定义
    node_properties: Dict[NodeType, Dict[str, type]] = field(default_factory=dict)
    
    # 边属性定义
    edge_properties: Dict[EdgeType, Dict[str, type]] = field(default_factory=dict)
    
    # 约束规则
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_node_type(self, node_type: NodeType, properties: Dict[str, type] = None):
        """添加节点类型
        
        Args:
            node_type: 节点类型
            properties: 属性定义
        """
        self.node_types.add(node_type)
        if properties:
            self.node_properties[node_type] = properties
    
    def add_edge_type(self, edge_type: EdgeType, properties: Dict[str, type] = None):
        """添加边类型
        
        Args:
            edge_type: 边类型
            properties: 属性定义
        """
        self.edge_types.add(edge_type)
        if properties:
            self.edge_properties[edge_type] = properties
    
    def add_constraint(self, constraint_type: str, **kwargs):
        """添加约束
        
        Args:
            constraint_type: 约束类型
            **kwargs: 约束参数
        """
        constraint = {
            'type': constraint_type,
            'created_at': datetime.now().isoformat(),
            **kwargs
        }
        self.constraints.append(constraint)
    
    def validate_node(self, node: GraphNode) -> List[str]:
        """验证节点
        
        Args:
            node: 图节点
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 检查节点类型
        if node.type not in self.node_types:
            errors.append(f"不支持的节点类型: {node.type.value}")
        
        # 检查属性
        if node.type in self.node_properties:
            required_props = self.node_properties[node.type]
            for prop_name, prop_type in required_props.items():
                if prop_name not in node.properties:
                    errors.append(f"缺少必需属性: {prop_name}")
                elif not isinstance(node.properties[prop_name], prop_type):
                    errors.append(f"属性类型错误: {prop_name} 应为 {prop_type.__name__}")
        
        return errors
    
    def validate_edge(self, edge: GraphEdge) -> List[str]:
        """验证边
        
        Args:
            edge: 图边
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 检查边类型
        if edge.type not in self.edge_types:
            errors.append(f"不支持的边类型: {edge.type.value}")
        
        # 检查属性
        if edge.type in self.edge_properties:
            required_props = self.edge_properties[edge.type]
            for prop_name, prop_type in required_props.items():
                if prop_name not in edge.properties:
                    errors.append(f"缺少必需属性: {prop_name}")
                elif not isinstance(edge.properties[prop_name], prop_type):
                    errors.append(f"属性类型错误: {prop_name} 应为 {prop_type.__name__}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'node_types': [nt.value for nt in self.node_types],
            'edge_types': [et.value for et in self.edge_types],
            'node_properties': {
                nt.value: {prop: pt.__name__ for prop, pt in props.items()}
                for nt, props in self.node_properties.items()
            },
            'edge_properties': {
                et.value: {prop: pt.__name__ for prop, pt in props.items()}
                for et, props in self.edge_properties.items()
            },
            'constraints': self.constraints
        }

@dataclass
class GraphConfig:
    """图谱配置"""
    name: str
    schema: GraphSchema
    
    # 存储配置
    storage_type: str = "memory"  # memory, neo4j, etc.
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # 构建配置
    auto_create_nodes: bool = True
    auto_merge_duplicates: bool = True
    duplicate_threshold: float = 0.8
    
    # 性能配置
    batch_size: int = 1000
    max_nodes: int = 100000
    max_edges: int = 500000
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'schema': self.schema.to_dict(),
            'storage_type': self.storage_type,
            'storage_config': self.storage_config,
            'auto_create_nodes': self.auto_create_nodes,
            'auto_merge_duplicates': self.auto_merge_duplicates,
            'duplicate_threshold': self.duplicate_threshold,
            'batch_size': self.batch_size,
            'max_nodes': self.max_nodes,
            'max_edges': self.max_edges
        }

class GraphBuilder:
    """图谱构建器
    
    负责知识图谱的构建、管理和维护
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.schema = config.schema
        
        # 图数据存储
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        
        # 索引
        self.node_type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        self.edge_type_index: Dict[EdgeType, Set[str]] = defaultdict(set)
        self.node_name_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 邻接表
        self.adjacency_list: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # 统计信息
        self.stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'nodes_updated': 0,
            'edges_updated': 0,
            'validation_errors': 0
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"图谱构建器初始化完成: {config.name}")
    
    def add_node(self, node: GraphNode, validate: bool = True) -> bool:
        """添加节点
        
        Args:
            node: 图节点
            validate: 是否验证
            
        Returns:
            是否添加成功
        """
        with self.lock:
            # 验证节点
            if validate:
                errors = self.schema.validate_node(node)
                if errors:
                    self.stats['validation_errors'] += 1
                    logger.warning(f"节点验证失败: {node.id} - {errors}")
                    return False
            
            # 检查是否已存在
            if node.id in self.nodes:
                # 更新现有节点
                existing_node = self.nodes[node.id]
                existing_node.properties.update(node.properties)
                existing_node.labels.update(node.labels)
                existing_node.updated_at = datetime.now()
                
                self.stats['nodes_updated'] += 1
                logger.debug(f"节点已更新: {node.id}")
            else:
                # 添加新节点
                self.nodes[node.id] = node
                
                # 更新索引
                self.node_type_index[node.type].add(node.id)
                self.node_name_index[node.name.lower()].add(node.id)
                
                self.stats['nodes_created'] += 1
                logger.debug(f"节点已添加: {node.id}")
            
            return True
    
    def add_edge(self, edge: GraphEdge, validate: bool = True) -> bool:
        """添加边
        
        Args:
            edge: 图边
            validate: 是否验证
            
        Returns:
            是否添加成功
        """
        with self.lock:
            # 验证边
            if validate:
                errors = self.schema.validate_edge(edge)
                if errors:
                    self.stats['validation_errors'] += 1
                    logger.warning(f"边验证失败: {edge.id} - {errors}")
                    return False
            
            # 检查源节点和目标节点是否存在
            if edge.source_id not in self.nodes:
                if self.config.auto_create_nodes:
                    # 自动创建源节点
                    source_node = GraphNode(
                        id=edge.source_id,
                        type=NodeType.CUSTOM,
                        name=edge.source_id
                    )
                    self.add_node(source_node, validate=False)
                else:
                    logger.warning(f"源节点不存在: {edge.source_id}")
                    return False
            
            if edge.target_id not in self.nodes:
                if self.config.auto_create_nodes:
                    # 自动创建目标节点
                    target_node = GraphNode(
                        id=edge.target_id,
                        type=NodeType.CUSTOM,
                        name=edge.target_id
                    )
                    self.add_node(target_node, validate=False)
                else:
                    logger.warning(f"目标节点不存在: {edge.target_id}")
                    return False
            
            # 检查是否已存在
            if edge.id in self.edges:
                # 更新现有边
                existing_edge = self.edges[edge.id]
                existing_edge.properties.update(edge.properties)
                existing_edge.weight = edge.weight
                existing_edge.confidence = edge.confidence
                existing_edge.updated_at = datetime.now()
                
                self.stats['edges_updated'] += 1
                logger.debug(f"边已更新: {edge.id}")
            else:
                # 添加新边
                self.edges[edge.id] = edge
                
                # 更新索引
                self.edge_type_index[edge.type].add(edge.id)
                
                # 更新邻接表
                self.adjacency_list[edge.source_id]['out'].add(edge.target_id)
                self.adjacency_list[edge.target_id]['in'].add(edge.source_id)
                
                self.stats['edges_created'] += 1
                logger.debug(f"边已添加: {edge.id}")
            
            return True
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            图节点
        """
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """获取边
        
        Args:
            edge_id: 边ID
            
        Returns:
            图边
        """
        return self.edges.get(edge_id)
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """按类型查找节点
        
        Args:
            node_type: 节点类型
            
        Returns:
            节点列表
        """
        node_ids = self.node_type_index.get(node_type, set())
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def find_nodes_by_name(self, name: str, fuzzy: bool = False) -> List[GraphNode]:
        """按名称查找节点
        
        Args:
            name: 节点名称
            fuzzy: 是否模糊匹配
            
        Returns:
            节点列表
        """
        name_lower = name.lower()
        
        if fuzzy:
            # 模糊匹配
            matching_nodes = []
            for node_name, node_ids in self.node_name_index.items():
                if name_lower in node_name or node_name in name_lower:
                    matching_nodes.extend([self.nodes[node_id] for node_id in node_ids if node_id in self.nodes])
            return matching_nodes
        else:
            # 精确匹配
            node_ids = self.node_name_index.get(name_lower, set())
            return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def find_edges_by_type(self, edge_type: EdgeType) -> List[GraphEdge]:
        """按类型查找边
        
        Args:
            edge_type: 边类型
            
        Returns:
            边列表
        """
        edge_ids = self.edge_type_index.get(edge_type, set())
        return [self.edges[edge_id] for edge_id in edge_ids if edge_id in self.edges]
    
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """获取邻居节点
        
        Args:
            node_id: 节点ID
            direction: 方向 (in, out, both)
            
        Returns:
            邻居节点ID列表
        """
        neighbors = set()
        
        if node_id in self.adjacency_list:
            if direction in ["out", "both"]:
                neighbors.update(self.adjacency_list[node_id].get('out', set()))
            if direction in ["in", "both"]:
                neighbors.update(self.adjacency_list[node_id].get('in', set()))
        
        return list(neighbors)
    
    def get_edges_between(self, source_id: str, target_id: str) -> List[GraphEdge]:
        """获取两个节点之间的边
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            
        Returns:
            边列表
        """
        edges = []
        for edge in self.edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                edges.append(edge)
        return edges
    
    def remove_node(self, node_id: str) -> bool:
        """删除节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否删除成功
        """
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # 删除相关的边
            edges_to_remove = []
            for edge in self.edges.values():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_remove.append(edge.id)
            
            for edge_id in edges_to_remove:
                self.remove_edge(edge_id)
            
            # 删除节点
            del self.nodes[node_id]
            
            # 更新索引
            self.node_type_index[node.type].discard(node_id)
            self.node_name_index[node.name.lower()].discard(node_id)
            
            # 清理邻接表
            if node_id in self.adjacency_list:
                del self.adjacency_list[node_id]
            
            logger.debug(f"节点已删除: {node_id}")
            return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """删除边
        
        Args:
            edge_id: 边ID
            
        Returns:
            是否删除成功
        """
        with self.lock:
            if edge_id not in self.edges:
                return False
            
            edge = self.edges[edge_id]
            
            # 删除边
            del self.edges[edge_id]
            
            # 更新索引
            self.edge_type_index[edge.type].discard(edge_id)
            
            # 更新邻接表
            if edge.source_id in self.adjacency_list:
                self.adjacency_list[edge.source_id]['out'].discard(edge.target_id)
            if edge.target_id in self.adjacency_list:
                self.adjacency_list[edge.target_id]['in'].discard(edge.source_id)
            
            logger.debug(f"边已删除: {edge_id}")
            return True
    
    def clear(self):
        """清空图谱"""
        with self.lock:
            self.nodes.clear()
            self.edges.clear()
            self.node_type_index.clear()
            self.edge_type_index.clear()
            self.node_name_index.clear()
            self.adjacency_list.clear()
            
            # 重置统计
            self.stats = {
                'nodes_created': 0,
                'edges_created': 0,
                'nodes_updated': 0,
                'edges_updated': 0,
                'validation_errors': 0
            }
            
            logger.info(f"图谱已清空: {self.config.name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            node_type_counts = {nt.value: len(nodes) for nt, nodes in self.node_type_index.items()}
            edge_type_counts = {et.value: len(edges) for et, edges in self.edge_type_index.items()}
            
            return {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'node_type_counts': node_type_counts,
                'edge_type_counts': edge_type_counts,
                'creation_stats': self.stats.copy(),
                'avg_degree': (2 * len(self.edges)) / len(self.nodes) if len(self.nodes) > 0 else 0
            }
    
    def export_graph(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """导出图谱
        
        Args:
            format: 导出格式 (json, dict)
            
        Returns:
            导出的图谱数据
        """
        graph_data = {
            'config': self.config.to_dict(),
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'statistics': self.get_statistics(),
            'exported_at': datetime.now().isoformat()
        }
        
        if format == "json":
            return json.dumps(graph_data, ensure_ascii=False, indent=2)
        else:
            return graph_data
    
    def import_graph(self, data: Union[str, Dict[str, Any]], merge: bool = False):
        """导入图谱
        
        Args:
            data: 图谱数据
            merge: 是否合并到现有图谱
        """
        if isinstance(data, str):
            graph_data = json.loads(data)
        else:
            graph_data = data
        
        if not merge:
            self.clear()
        
        # 导入节点
        for node_data in graph_data.get('nodes', []):
            node = GraphNode.from_dict(node_data)
            self.add_node(node, validate=False)
        
        # 导入边
        for edge_data in graph_data.get('edges', []):
            edge = GraphEdge.from_dict(edge_data)
            self.add_edge(edge, validate=False)
        
        logger.info(f"图谱导入完成: {len(graph_data.get('nodes', []))} 个节点, {len(graph_data.get('edges', []))} 条边")

# 便捷函数
def create_graph_builder(config: GraphConfig) -> GraphBuilder:
    """创建图谱构建器
    
    Args:
        config: 图谱配置
        
    Returns:
        图谱构建器实例
    """
    return GraphBuilder(config)

def create_graph_config(name: str, schema: GraphSchema, **kwargs) -> GraphConfig:
    """创建图谱配置
    
    Args:
        name: 图谱名称
        schema: 图谱模式
        **kwargs: 其他配置参数
        
    Returns:
        图谱配置实例
    """
    return GraphConfig(name=name, schema=schema, **kwargs)

def create_graph_schema(name: str, description: str = "") -> GraphSchema:
    """创建图谱模式
    
    Args:
        name: 模式名称
        description: 模式描述
        
    Returns:
        图谱模式实例
    """
    return GraphSchema(name=name, description=description)

def create_graph_node(id: str, type: NodeType, name: str, **kwargs) -> GraphNode:
    """创建图节点
    
    Args:
        id: 节点ID
        type: 节点类型
        name: 节点名称
        **kwargs: 其他参数
        
    Returns:
        图节点实例
    """
    return GraphNode(id=id, type=type, name=name, **kwargs)

def create_graph_edge(id: str, type: EdgeType, source_id: str, target_id: str, **kwargs) -> GraphEdge:
    """创建图边
    
    Args:
        id: 边ID
        type: 边类型
        source_id: 源节点ID
        target_id: 目标节点ID
        **kwargs: 其他参数
        
    Returns:
        图边实例
    """
    return GraphEdge(id=id, type=type, source_id=source_id, target_id=target_id, **kwargs)

def create_financial_schema() -> GraphSchema:
    """创建金融领域图谱模式
    
    Returns:
        金融图谱模式
    """
    schema = GraphSchema(
        name="financial_knowledge_graph",
        description="金融投资知识图谱模式"
    )
    
    # 添加节点类型
    schema.add_node_type(NodeType.COMPANY, {'code': str, 'industry': str, 'market_cap': float})
    schema.add_node_type(NodeType.STOCK, {'symbol': str, 'exchange': str, 'sector': str})
    schema.add_node_type(NodeType.PERSON, {'role': str, 'company': str})
    schema.add_node_type(NodeType.EVENT, {'date': str, 'impact': str})
    schema.add_node_type(NodeType.INDUSTRY, {'sector': str, 'growth_rate': float})
    
    # 添加边类型
    schema.add_edge_type(EdgeType.OWNS, {'percentage': float})
    schema.add_edge_type(EdgeType.COMPETES_WITH, {'intensity': str})
    schema.add_edge_type(EdgeType.INFLUENCES, {'impact_score': float})
    schema.add_edge_type(EdgeType.BELONGS_TO, {})
    
    return schema