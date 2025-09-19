# 关系建模模块
# 实现实体间关系的识别、抽取和建模功能

import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from collections import defaultdict
import math

from .entity_extractor import Entity, EntityType

logger = logging.getLogger(__name__)

class RelationType(Enum):
    """关系类型"""
    # 所有权关系
    OWNS = "owns"                  # 拥有
    CONTROLS = "controls"          # 控制
    HOLDS = "holds"                # 持有
    INVESTS_IN = "invests_in"      # 投资
    
    # 业务关系
    COMPETES_WITH = "competes_with"  # 竞争
    COOPERATES_WITH = "cooperates_with"  # 合作
    SUPPLIES_TO = "supplies_to"    # 供应
    PURCHASES_FROM = "purchases_from"  # 采购
    PARTNERS_WITH = "partners_with"  # 合作伙伴
    
    # 影响关系
    INFLUENCES = "influences"      # 影响
    AFFECTS = "affects"            # 影响
    CAUSES = "causes"              # 导致
    LEADS_TO = "leads_to"          # 导致
    CORRELATES_WITH = "correlates_with"  # 相关
    DEPENDS_ON = "depends_on"      # 依赖
    
    # 分类关系
    BELONGS_TO = "belongs_to"      # 属于
    CONTAINS = "contains"          # 包含
    IS_PART_OF = "is_part_of"      # 是...的一部分
    IS_TYPE_OF = "is_type_of"      # 是...的类型
    INCLUDES = "includes"          # 包括
    
    # 时间关系
    OCCURS_BEFORE = "occurs_before"  # 发生在...之前
    OCCURS_AFTER = "occurs_after"    # 发生在...之后
    OCCURS_DURING = "occurs_during"  # 发生在...期间
    CONCURRENT_WITH = "concurrent_with"  # 同时发生
    
    # 地理关系
    LOCATED_IN = "located_in"      # 位于
    NEAR = "near"                  # 靠近
    ADJACENT_TO = "adjacent_to"    # 邻近
    
    # 人物关系
    WORKS_FOR = "works_for"        # 为...工作
    MANAGES = "manages"            # 管理
    REPORTS_TO = "reports_to"      # 汇报给
    FOUNDED_BY = "founded_by"      # 由...创立
    
    # 信息关系
    MENTIONS = "mentions"          # 提及
    REFERENCES = "references"      # 引用
    DISCUSSES = "discusses"        # 讨论
    REPORTS_ON = "reports_on"      # 报道
    
    # 相似关系
    SIMILAR_TO = "similar_to"      # 相似
    EQUIVALENT_TO = "equivalent_to"  # 等价
    DIFFERENT_FROM = "different_from"  # 不同
    
    # 其他
    RELATED_TO = "related_to"      # 相关
    ASSOCIATED_WITH = "associated_with"  # 关联
    CUSTOM = "custom"              # 自定义

@dataclass
class Relation:
    """关系对象"""
    source_entity: Entity          # 源实体
    target_entity: Entity          # 目标实体
    relation_type: RelationType    # 关系类型
    confidence: float = 1.0        # 置信度
    strength: float = 1.0          # 关系强度
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    evidence: List[str] = field(default_factory=list)  # 证据文本
    context: str = ""              # 上下文
    
    @property
    def id(self) -> str:
        """关系ID"""
        return f"{self.source_entity.text}_{self.relation_type.value}_{self.target_entity.text}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'source_entity': self.source_entity.to_dict(),
            'target_entity': self.target_entity.to_dict(),
            'relation_type': self.relation_type.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'properties': self.properties,
            'evidence': self.evidence,
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """从字典创建关系"""
        return cls(
            source_entity=Entity.from_dict(data['source_entity']),
            target_entity=Entity.from_dict(data['target_entity']),
            relation_type=RelationType(data['relation_type']),
            confidence=data.get('confidence', 1.0),
            strength=data.get('strength', 1.0),
            properties=data.get('properties', {}),
            evidence=data.get('evidence', []),
            context=data.get('context', '')
        )

@dataclass
class RelationPattern:
    """关系模式"""
    relation_type: RelationType
    pattern: str                   # 正则表达式模式
    source_types: Set[EntityType] = field(default_factory=set)  # 源实体类型
    target_types: Set[EntityType] = field(default_factory=set)  # 目标实体类型
    confidence: float = 0.8        # 模式置信度
    bidirectional: bool = False    # 是否双向
    
    def matches(self, text: str, source_entity: Entity, target_entity: Entity) -> bool:
        """检查模式是否匹配
        
        Args:
            text: 文本
            source_entity: 源实体
            target_entity: 目标实体
            
        Returns:
            是否匹配
        """
        # 检查实体类型
        if self.source_types and source_entity.type not in self.source_types:
            return False
        if self.target_types and target_entity.type not in self.target_types:
            return False
        
        # 检查文本模式
        return bool(re.search(self.pattern, text, re.IGNORECASE))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'relation_type': self.relation_type.value,
            'pattern': self.pattern,
            'source_types': [t.value for t in self.source_types],
            'target_types': [t.value for t in self.target_types],
            'confidence': self.confidence,
            'bidirectional': self.bidirectional
        }

@dataclass
class RelationExtractionResult:
    """关系抽取结果"""
    text: str                      # 原始文本
    entities: List[Entity]         # 输入实体
    relations: List[Relation]      # 抽取的关系
    processing_time: float = 0.0   # 处理时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[Relation]:
        """按类型获取关系
        
        Args:
            relation_type: 关系类型
            
        Returns:
            关系列表
        """
        return [rel for rel in self.relations if rel.relation_type == relation_type]
    
    def get_relations_for_entity(self, entity: Entity) -> List[Relation]:
        """获取实体相关的关系
        
        Args:
            entity: 实体
            
        Returns:
            关系列表
        """
        return [rel for rel in self.relations 
                if rel.source_entity.text == entity.text or rel.target_entity.text == entity.text]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'text': self.text,
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'relation_count': len(self.relations)
        }

class RelationModeler:
    """关系建模器
    
    负责从文本和实体中识别和抽取关系
    """
    
    def __init__(self):
        # 关系模式
        self.patterns = self._build_relation_patterns()
        
        # 关系词典
        self.relation_keywords = self._build_relation_keywords()
        
        # 统计信息
        self.stats = {
            'texts_processed': 0,
            'relations_extracted': 0,
            'processing_time': 0.0
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("关系建模器初始化完成")
    
    def _build_relation_patterns(self) -> List[RelationPattern]:
        """构建关系模式
        
        Returns:
            关系模式列表
        """
        patterns = [
            # 所有权关系
            RelationPattern(
                relation_type=RelationType.OWNS,
                pattern=r'(.+?)拥有(.+?)',
                confidence=0.9
            ),
            RelationPattern(
                relation_type=RelationType.CONTROLS,
                pattern=r'(.+?)控制(.+?)',
                confidence=0.9
            ),
            RelationPattern(
                relation_type=RelationType.HOLDS,
                pattern=r'(.+?)持有(.+?)',
                confidence=0.8
            ),
            RelationPattern(
                relation_type=RelationType.INVESTS_IN,
                pattern=r'(.+?)投资(.+?)',
                confidence=0.8
            ),
            
            # 业务关系
            RelationPattern(
                relation_type=RelationType.COMPETES_WITH,
                pattern=r'(.+?)与(.+?)竞争',
                confidence=0.8,
                bidirectional=True
            ),
            RelationPattern(
                relation_type=RelationType.COOPERATES_WITH,
                pattern=r'(.+?)与(.+?)合作',
                confidence=0.8,
                bidirectional=True
            ),
            RelationPattern(
                relation_type=RelationType.SUPPLIES_TO,
                pattern=r'(.+?)向(.+?)供应',
                confidence=0.8
            ),
            RelationPattern(
                relation_type=RelationType.PARTNERS_WITH,
                pattern=r'(.+?)与(.+?)建立伙伴关系',
                confidence=0.9,
                bidirectional=True
            ),
            
            # 影响关系
            RelationPattern(
                relation_type=RelationType.INFLUENCES,
                pattern=r'(.+?)影响(.+?)',
                confidence=0.7
            ),
            RelationPattern(
                relation_type=RelationType.CAUSES,
                pattern=r'(.+?)导致(.+?)',
                confidence=0.8
            ),
            RelationPattern(
                relation_type=RelationType.LEADS_TO,
                pattern=r'(.+?)引起(.+?)',
                confidence=0.8
            ),
            
            # 分类关系
            RelationPattern(
                relation_type=RelationType.BELONGS_TO,
                pattern=r'(.+?)属于(.+?)',
                confidence=0.9
            ),
            RelationPattern(
                relation_type=RelationType.IS_PART_OF,
                pattern=r'(.+?)是(.+?)的一部分',
                confidence=0.9
            ),
            RelationPattern(
                relation_type=RelationType.INCLUDES,
                pattern=r'(.+?)包括(.+?)',
                confidence=0.8
            ),
            
            # 人物关系
            RelationPattern(
                relation_type=RelationType.WORKS_FOR,
                pattern=r'(.+?)在(.+?)工作',
                source_types={EntityType.PERSON},
                target_types={EntityType.COMPANY, EntityType.ORGANIZATION},
                confidence=0.9
            ),
            RelationPattern(
                relation_type=RelationType.MANAGES,
                pattern=r'(.+?)管理(.+?)',
                source_types={EntityType.PERSON, EntityType.EXECUTIVE},
                confidence=0.8
            ),
            RelationPattern(
                relation_type=RelationType.FOUNDED_BY,
                pattern=r'(.+?)由(.+?)创立',
                target_types={EntityType.PERSON},
                confidence=0.9
            ),
            
            # 地理关系
            RelationPattern(
                relation_type=RelationType.LOCATED_IN,
                pattern=r'(.+?)位于(.+?)',
                target_types={EntityType.REGION, EntityType.CITY, EntityType.COUNTRY},
                confidence=0.9
            ),
            
            # 信息关系
            RelationPattern(
                relation_type=RelationType.MENTIONS,
                pattern=r'(.+?)提到(.+?)',
                confidence=0.6
            ),
            RelationPattern(
                relation_type=RelationType.REPORTS_ON,
                pattern=r'(.+?)报道(.+?)',
                source_types={EntityType.NEWS, EntityType.REPORT},
                confidence=0.8
            )
        ]
        
        return patterns
    
    def _build_relation_keywords(self) -> Dict[RelationType, List[str]]:
        """构建关系关键词
        
        Returns:
            关系类型到关键词的映射
        """
        keywords = {
            RelationType.OWNS: ['拥有', '持有', '所有', '占有'],
            RelationType.CONTROLS: ['控制', '掌控', '主导', '支配'],
            RelationType.INVESTS_IN: ['投资', '入股', '参股', '注资'],
            RelationType.COMPETES_WITH: ['竞争', '对手', '竞争对手', '角逐'],
            RelationType.COOPERATES_WITH: ['合作', '协作', '联合', '携手'],
            RelationType.INFLUENCES: ['影响', '作用', '冲击', '波及'],
            RelationType.CAUSES: ['导致', '引起', '造成', '产生'],
            RelationType.BELONGS_TO: ['属于', '隶属', '归属'],
            RelationType.LOCATED_IN: ['位于', '坐落', '设在', '在'],
            RelationType.WORKS_FOR: ['工作', '任职', '就职', '供职'],
            RelationType.MANAGES: ['管理', '领导', '主管', '负责'],
            RelationType.MENTIONS: ['提及', '提到', '涉及', '谈到'],
            RelationType.SIMILAR_TO: ['相似', '类似', '相像', '近似']
        }
        
        return keywords
    
    def extract_relations(self, text: str, entities: List[Entity]) -> RelationExtractionResult:
        """抽取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系抽取结果
        """
        start_time = datetime.now()
        
        with self.lock:
            self.stats['texts_processed'] += 1
        
        relations = []
        
        # 1. 基于模式的关系抽取
        pattern_relations = self._extract_by_patterns(text, entities)
        relations.extend(pattern_relations)
        
        # 2. 基于关键词的关系抽取
        keyword_relations = self._extract_by_keywords(text, entities)
        relations.extend(keyword_relations)
        
        # 3. 基于距离的关系抽取
        distance_relations = self._extract_by_distance(text, entities)
        relations.extend(distance_relations)
        
        # 去重和合并
        relations = self._merge_relations(relations)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        with self.lock:
            self.stats['relations_extracted'] += len(relations)
            self.stats['processing_time'] += processing_time
        
        result = RelationExtractionResult(
            text=text,
            entities=entities,
            relations=relations,
            processing_time=processing_time,
            metadata={
                'method': 'rule_based',
                'patterns_used': len(self.patterns),
                'entity_pairs_checked': len(entities) * (len(entities) - 1) // 2
            }
        )
        
        logger.debug(f"关系抽取完成: {len(relations)} 个关系, 耗时 {processing_time:.3f}s")
        
        return result
    
    def _extract_by_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于模式抽取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系列表
        """
        relations = []
        
        # 检查每对实体
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i == j:
                    continue
                
                # 获取实体间的文本片段
                start_pos = min(source_entity.start_pos, target_entity.start_pos)
                end_pos = max(source_entity.end_pos, target_entity.end_pos)
                context = text[start_pos:end_pos]
                
                # 检查每个模式
                for pattern in self.patterns:
                    if pattern.matches(context, source_entity, target_entity):
                        relation = Relation(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relation_type=pattern.relation_type,
                            confidence=pattern.confidence,
                            context=context,
                            evidence=[context]
                        )
                        relations.append(relation)
                        
                        # 如果是双向关系，添加反向关系
                        if pattern.bidirectional:
                            reverse_relation = Relation(
                                source_entity=target_entity,
                                target_entity=source_entity,
                                relation_type=pattern.relation_type,
                                confidence=pattern.confidence * 0.9,  # 反向关系置信度稍低
                                context=context,
                                evidence=[context]
                            )
                            relations.append(reverse_relation)
        
        return relations
    
    def _extract_by_keywords(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于关键词抽取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系列表
        """
        relations = []
        
        # 检查每对实体
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i == j:
                    continue
                
                # 获取实体间的文本片段
                start_pos = min(source_entity.start_pos, target_entity.start_pos)
                end_pos = max(source_entity.end_pos, target_entity.end_pos)
                context = text[start_pos:end_pos]
                
                # 检查关系关键词
                for relation_type, keywords in self.relation_keywords.items():
                    for keyword in keywords:
                        if keyword in context:
                            relation = Relation(
                                source_entity=source_entity,
                                target_entity=target_entity,
                                relation_type=relation_type,
                                confidence=0.6,  # 关键词匹配的置信度较低
                                context=context,
                                evidence=[keyword]
                            )
                            relations.append(relation)
                            break
        
        return relations
    
    def _extract_by_distance(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于距离抽取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系列表
        """
        relations = []
        max_distance = 100  # 最大距离阈值
        
        # 检查每对实体
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i == j:
                    continue
                
                # 计算实体间距离
                distance = abs(source_entity.start_pos - target_entity.start_pos)
                
                if distance <= max_distance:
                    # 距离越近，关系强度越高
                    strength = 1.0 - (distance / max_distance)
                    
                    # 根据实体类型推断可能的关系
                    relation_type = self._infer_relation_type(source_entity, target_entity)
                    
                    if relation_type:
                        context = text[min(source_entity.start_pos, target_entity.start_pos):
                                     max(source_entity.end_pos, target_entity.end_pos)]
                        
                        relation = Relation(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relation_type=relation_type,
                            confidence=0.4,  # 基于距离的置信度较低
                            strength=strength,
                            context=context,
                            properties={'distance': distance}
                        )
                        relations.append(relation)
        
        return relations
    
    def _infer_relation_type(self, source_entity: Entity, target_entity: Entity) -> Optional[RelationType]:
        """根据实体类型推断关系类型
        
        Args:
            source_entity: 源实体
            target_entity: 目标实体
            
        Returns:
            推断的关系类型
        """
        # 基于实体类型的关系推断规则
        type_rules = {
            (EntityType.COMPANY, EntityType.COMPANY): RelationType.COMPETES_WITH,
            (EntityType.COMPANY, EntityType.INDUSTRY): RelationType.BELONGS_TO,
            (EntityType.COMPANY, EntityType.REGION): RelationType.LOCATED_IN,
            (EntityType.PERSON, EntityType.COMPANY): RelationType.WORKS_FOR,
            (EntityType.EXECUTIVE, EntityType.COMPANY): RelationType.MANAGES,
            (EntityType.STOCK, EntityType.COMPANY): RelationType.BELONGS_TO,
            (EntityType.EVENT, EntityType.COMPANY): RelationType.AFFECTS,
            (EntityType.NEWS, EntityType.COMPANY): RelationType.MENTIONS,
            (EntityType.CONCEPT, EntityType.COMPANY): RelationType.RELATED_TO
        }
        
        key = (source_entity.type, target_entity.type)
        return type_rules.get(key)
    
    def _merge_relations(self, relations: List[Relation]) -> List[Relation]:
        """合并重复的关系
        
        Args:
            relations: 关系列表
            
        Returns:
            合并后的关系列表
        """
        if not relations:
            return relations
        
        # 按关系ID分组
        relation_groups = defaultdict(list)
        for relation in relations:
            relation_groups[relation.id].append(relation)
        
        merged_relations = []
        for group in relation_groups.values():
            if len(group) == 1:
                merged_relations.append(group[0])
            else:
                # 合并同一关系的多个实例
                merged = group[0]
                
                # 取最高置信度
                merged.confidence = max(rel.confidence for rel in group)
                
                # 取最高强度
                merged.strength = max(rel.strength for rel in group)
                
                # 合并证据
                all_evidence = set()
                for rel in group:
                    all_evidence.update(rel.evidence)
                merged.evidence = list(all_evidence)
                
                # 合并属性
                for rel in group:
                    merged.properties.update(rel.properties)
                
                merged_relations.append(merged)
        
        return merged_relations
    
    def add_relation_pattern(self, pattern: RelationPattern):
        """添加关系模式
        
        Args:
            pattern: 关系模式
        """
        self.patterns.append(pattern)
        logger.debug(f"添加关系模式: {pattern.relation_type.value} - {pattern.pattern}")
    
    def add_relation_keywords(self, relation_type: RelationType, keywords: List[str]):
        """添加关系关键词
        
        Args:
            relation_type: 关系类型
            keywords: 关键词列表
        """
        if relation_type not in self.relation_keywords:
            self.relation_keywords[relation_type] = []
        
        self.relation_keywords[relation_type].extend(keywords)
        logger.debug(f"添加关系关键词: {relation_type.value} - {len(keywords)} 个词")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            stats = self.stats.copy()
            
            if stats['texts_processed'] > 0:
                stats['avg_relations_per_text'] = stats['relations_extracted'] / stats['texts_processed']
                stats['avg_processing_time'] = stats['processing_time'] / stats['texts_processed']
            else:
                stats['avg_relations_per_text'] = 0
                stats['avg_processing_time'] = 0
            
            stats['pattern_count'] = len(self.patterns)
            stats['keyword_count'] = sum(len(keywords) for keywords in self.relation_keywords.values())
            
            return stats

# 便捷函数
def create_relation_modeler() -> RelationModeler:
    """创建关系建模器
    
    Returns:
        关系建模器实例
    """
    return RelationModeler()

def create_relation_pattern(
    relation_type: RelationType,
    pattern: str,
    source_types: Set[EntityType] = None,
    target_types: Set[EntityType] = None,
    **kwargs
) -> RelationPattern:
    """创建关系模式
    
    Args:
        relation_type: 关系类型
        pattern: 正则表达式模式
        source_types: 源实体类型集合
        target_types: 目标实体类型集合
        **kwargs: 其他参数
        
    Returns:
        关系模式实例
    """
    return RelationPattern(
        relation_type=relation_type,
        pattern=pattern,
        source_types=source_types or set(),
        target_types=target_types or set(),
        **kwargs
    )

def create_relation(
    source_entity: Entity,
    target_entity: Entity,
    relation_type: RelationType,
    **kwargs
) -> Relation:
    """创建关系
    
    Args:
        source_entity: 源实体
        target_entity: 目标实体
        relation_type: 关系类型
        **kwargs: 其他参数
        
    Returns:
        关系实例
    """
    return Relation(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=relation_type,
        **kwargs
    )

def extract_relations_from_text(text: str, entities: List[Entity]) -> RelationExtractionResult:
    """从文本中抽取关系（便捷函数）
    
    Args:
        text: 输入文本
        entities: 实体列表
        
    Returns:
        关系抽取结果
    """
    modeler = create_relation_modeler()
    return modeler.extract_relations(text, entities)

def create_financial_relation_modeler() -> RelationModeler:
    """创建金融领域专用的关系建模器
    
    Returns:
        金融关系建模器
    """
    modeler = RelationModeler()
    
    # 添加金融特定的关系模式
    financial_patterns = [
        RelationPattern(
            relation_type=RelationType.OWNS,
            pattern=r'(.+?)持股(.+?)',
            source_types={EntityType.COMPANY, EntityType.PERSON},
            target_types={EntityType.COMPANY, EntityType.STOCK},
            confidence=0.9
        ),
        RelationPattern(
            relation_type=RelationType.INVESTS_IN,
            pattern=r'(.+?)投资(.+?)',
            source_types={EntityType.COMPANY, EntityType.FUND, EntityType.PERSON},
            target_types={EntityType.COMPANY, EntityType.STOCK, EntityType.FUND},
            confidence=0.8
        ),
        RelationPattern(
            relation_type=RelationType.AFFECTS,
            pattern=r'(.+?)对(.+?)产生影响',
            confidence=0.7
        )
    ]
    
    for pattern in financial_patterns:
        modeler.add_relation_pattern(pattern)
    
    # 添加金融关系关键词
    financial_keywords = {
        RelationType.OWNS: ['持股', '控股', '参股', '股权'],
        RelationType.INVESTS_IN: ['投资', '入股', '注资', '融资'],
        RelationType.COMPETES_WITH: ['竞争', '对手', '竞品'],
        RelationType.AFFECTS: ['影响', '冲击', '利好', '利空']
    }
    
    for relation_type, keywords in financial_keywords.items():
        modeler.add_relation_keywords(relation_type, keywords)
    
    return modeler