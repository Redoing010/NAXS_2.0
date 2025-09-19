# 知识图谱模块
# 实现知识图谱的构建、管理和查询功能

from .graph_builder import (
    NodeType,
    EdgeType,
    GraphNode,
    GraphEdge,
    GraphSchema,
    GraphConfig,
    GraphBuilder,
    create_graph_builder
)
from .entity_extractor import (
    EntityType,
    Entity,
    EntityExtractionResult,
    TextProcessor,
    EntityExtractor,
    create_entity_extractor
)
from .relation_modeler import (
    RelationType,
    Relation,
    RelationExtractionResult,
    RelationModeler,
    create_relation_modeler
)
from .graph_algorithms import (
    CentralityType,
    CommunityAlgorithm,
    CentralityMetrics,
    Community,
    CommunityDetection,
    PathFinding,
    GraphMetrics,
    GraphAlgorithms,
    create_graph_algorithms,
    calculate_centrality,
    detect_communities,
    find_shortest_path
)
from .knowledge_base import (
    KnowledgeType,
    ConfidenceLevel,
    QueryType,
    Knowledge,
    QueryResult,
    InferenceRule,
    InferenceResult,
    KnowledgeBase,
    create_knowledge_base,
    create_knowledge,
    create_inference_rule
)
from .graph_storage import (
    StorageType,
    CompressionType,
    StorageConfig,
    StorageStats,
    GraphStorage,
    create_storage_config,
    create_graph_storage
)

__version__ = "1.0.0"

__all__ = [
    # 图构建
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "GraphSchema",
    "GraphConfig",
    "GraphBuilder",
    "create_graph_builder",
    
    # 实体抽取
    "EntityType",
    "Entity",
    "EntityExtractionResult",
    "TextProcessor",
    "EntityExtractor",
    "create_entity_extractor",
    
    # 关系建模
    "RelationType",
    "Relation",
    "RelationExtractionResult",
    "RelationModeler",
    "create_relation_modeler",
    
    # 图算法
    "CentralityType",
    "CommunityAlgorithm",
    "CentralityMetrics",
    "Community",
    "CommunityDetection",
    "PathFinding",
    "GraphMetrics",
    "GraphAlgorithms",
    "create_graph_algorithms",
    "calculate_centrality",
    "detect_communities",
    "find_shortest_path",
    
    # 知识库
    "KnowledgeType",
    "ConfidenceLevel",
    "QueryType",
    "Knowledge",
    "QueryResult",
    "InferenceRule",
    "InferenceResult",
    "KnowledgeBase",
    "create_knowledge_base",
    "create_knowledge",
    "create_inference_rule",
    
    # 图存储
    "StorageType",
    "CompressionType",
    "StorageConfig",
    "StorageStats",
    "GraphStorage",
    "create_storage_config",
    "create_graph_storage",
]

# 模块描述
__doc__ = """
知识图谱系统

这个模块实现了智能投研系统的知识图谱功能，提供以下核心能力：

1. **图谱构建**：
   - 定义图谱模式和结构
   - 支持多种节点和边类型
   - 灵活的图谱配置管理

2. **实体抽取**：
   - 从文本中识别和抽取实体
   - 支持公司、人物、事件、指标等实体类型
   - 基于规则和机器学习的抽取方法

3. **关系建模**：
   - 识别实体间的关系
   - 支持因果关系、关联关系等
   - 关系强度和置信度计算

4. **图算法**：
   - 中心性分析（PageRank、度中心性等）
   - 社区发现和聚类
   - 路径查找和图遍历
   - 图结构分析和度量

5. **知识库管理**：
   - 知识查询和检索
   - 推理引擎和规则系统
   - 知识更新和维护

6. **存储后端**：
   - 支持Neo4j图数据库
   - 内存存储用于快速原型
   - 可扩展的存储接口

主要应用场景：
- 公司关系网络分析
- 事件影响链追踪
- 市场情绪传播分析
- 投资决策支持
- 风险关联发现
"""