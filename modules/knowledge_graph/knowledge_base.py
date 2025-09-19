# 知识库模块
# 实现知识的存储、检索、管理和推理功能

import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import re
import hashlib

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """知识类型"""
    FACT = "fact"                  # 事实
    RULE = "rule"                  # 规则
    CONCEPT = "concept"            # 概念
    RELATION = "relation"          # 关系
    EVENT = "event"                # 事件
    PATTERN = "pattern"            # 模式

class ConfidenceLevel(Enum):
    """置信度级别"""
    VERY_LOW = "very_low"          # 很低 (0.0-0.2)
    LOW = "low"                    # 低 (0.2-0.4)
    MEDIUM = "medium"              # 中等 (0.4-0.6)
    HIGH = "high"                  # 高 (0.6-0.8)
    VERY_HIGH = "very_high"        # 很高 (0.8-1.0)

class QueryType(Enum):
    """查询类型"""
    EXACT = "exact"                # 精确匹配
    FUZZY = "fuzzy"                # 模糊匹配
    SEMANTIC = "semantic"          # 语义匹配
    PATTERN = "pattern"            # 模式匹配
    GRAPH = "graph"                # 图查询

@dataclass
class Knowledge:
    """知识条目"""
    id: str
    type: KnowledgeType
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.id:
            # 生成唯一ID
            content = f"{self.subject}_{self.predicate}_{self.object}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:16]
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """获取置信度级别"""
        if self.confidence <= 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence <= 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence <= 0.6:
            return ConfidenceLevel.MEDIUM
        elif self.confidence <= 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def to_triple(self) -> Tuple[str, str, str]:
        """转换为三元组"""
        return (self.subject, self.predicate, self.object)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type.value,
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Knowledge':
        """从字典创建知识条目"""
        return cls(
            id=data['id'],
            type=KnowledgeType(data['type']),
            subject=data['subject'],
            predicate=data['predicate'],
            object=data['object'],
            confidence=data.get('confidence', 1.0),
            source=data.get('source', ''),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            metadata=data.get('metadata', {}),
            tags=set(data.get('tags', []))
        )

@dataclass
class QueryResult:
    """查询结果"""
    knowledge_items: List[Knowledge] = field(default_factory=list)
    total_count: int = 0
    query_time: float = 0.0
    query_type: str = ""
    
    @property
    def is_empty(self) -> bool:
        """是否为空结果"""
        return len(self.knowledge_items) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'knowledge_items': [item.to_dict() for item in self.knowledge_items],
            'total_count': self.total_count,
            'query_time': self.query_time,
            'query_type': self.query_type,
            'is_empty': self.is_empty
        }

@dataclass
class InferenceRule:
    """推理规则"""
    id: str
    name: str
    conditions: List[str]  # 条件模式
    conclusion: str        # 结论模式
    confidence: float = 1.0
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'conditions': self.conditions,
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'enabled': self.enabled
        }

@dataclass
class InferenceResult:
    """推理结果"""
    new_knowledge: List[Knowledge] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'new_knowledge': [k.to_dict() for k in self.new_knowledge],
            'applied_rules': self.applied_rules,
            'inference_time': self.inference_time,
            'knowledge_count': len(self.new_knowledge)
        }

class KnowledgeIndex:
    """知识索引
    
    提供多种索引方式以支持快速查询
    """
    
    def __init__(self):
        # 主索引：按ID
        self.id_index: Dict[str, Knowledge] = {}
        
        # 三元组索引
        self.subject_index: Dict[str, Set[str]] = defaultdict(set)
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)
        self.object_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 类型索引
        self.type_index: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        
        # 标签索引
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 时间索引
        self.time_index: Dict[str, Set[str]] = defaultdict(set)  # 按日期分组
        
        # 置信度索引
        self.confidence_index: Dict[ConfidenceLevel, Set[str]] = defaultdict(set)
        
        # 全文索引（简单实现）
        self.text_index: Dict[str, Set[str]] = defaultdict(set)
        
        self.lock = threading.RLock()
    
    def add_knowledge(self, knowledge: Knowledge):
        """添加知识到索引"""
        with self.lock:
            kid = knowledge.id
            
            # 主索引
            self.id_index[kid] = knowledge
            
            # 三元组索引
            self.subject_index[knowledge.subject].add(kid)
            self.predicate_index[knowledge.predicate].add(kid)
            self.object_index[knowledge.object].add(kid)
            
            # 类型索引
            self.type_index[knowledge.type].add(kid)
            
            # 标签索引
            for tag in knowledge.tags:
                self.tag_index[tag].add(kid)
            
            # 时间索引
            date_key = knowledge.timestamp.strftime('%Y-%m-%d')
            self.time_index[date_key].add(kid)
            
            # 置信度索引
            self.confidence_index[knowledge.confidence_level].add(kid)
            
            # 全文索引
            text_content = f"{knowledge.subject} {knowledge.predicate} {knowledge.object}"
            words = re.findall(r'\w+', text_content.lower())
            for word in words:
                self.text_index[word].add(kid)
    
    def remove_knowledge(self, knowledge_id: str):
        """从索引中移除知识"""
        with self.lock:
            if knowledge_id not in self.id_index:
                return
            
            knowledge = self.id_index[knowledge_id]
            
            # 从各个索引中移除
            self.subject_index[knowledge.subject].discard(knowledge_id)
            self.predicate_index[knowledge.predicate].discard(knowledge_id)
            self.object_index[knowledge.object].discard(knowledge_id)
            self.type_index[knowledge.type].discard(knowledge_id)
            
            for tag in knowledge.tags:
                self.tag_index[tag].discard(knowledge_id)
            
            date_key = knowledge.timestamp.strftime('%Y-%m-%d')
            self.time_index[date_key].discard(knowledge_id)
            
            self.confidence_index[knowledge.confidence_level].discard(knowledge_id)
            
            # 从全文索引中移除
            text_content = f"{knowledge.subject} {knowledge.predicate} {knowledge.object}"
            words = re.findall(r'\w+', text_content.lower())
            for word in words:
                self.text_index[word].discard(knowledge_id)
            
            # 从主索引中移除
            del self.id_index[knowledge_id]
    
    def query_by_subject(self, subject: str) -> Set[str]:
        """按主语查询"""
        return self.subject_index.get(subject, set())
    
    def query_by_predicate(self, predicate: str) -> Set[str]:
        """按谓语查询"""
        return self.predicate_index.get(predicate, set())
    
    def query_by_object(self, object_val: str) -> Set[str]:
        """按宾语查询"""
        return self.object_index.get(object_val, set())
    
    def query_by_type(self, knowledge_type: KnowledgeType) -> Set[str]:
        """按类型查询"""
        return self.type_index.get(knowledge_type, set())
    
    def query_by_tag(self, tag: str) -> Set[str]:
        """按标签查询"""
        return self.tag_index.get(tag, set())
    
    def query_by_text(self, text: str) -> Set[str]:
        """全文查询"""
        words = re.findall(r'\w+', text.lower())
        if not words:
            return set()
        
        result = self.text_index.get(words[0], set())
        for word in words[1:]:
            result = result.intersection(self.text_index.get(word, set()))
        
        return result
    
    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """获取知识条目"""
        return self.id_index.get(knowledge_id)
    
    def get_all_ids(self) -> Set[str]:
        """获取所有知识ID"""
        return set(self.id_index.keys())

class KnowledgeBase:
    """知识库
    
    提供知识的存储、检索、管理和推理功能
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.index = KnowledgeIndex()
        self.inference_rules: Dict[str, InferenceRule] = {}
        
        # 统计信息
        self.stats = {
            'total_knowledge': 0,
            'total_queries': 0,
            'total_inferences': 0,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        
        self.lock = threading.RLock()
        
        logger.info(f"知识库 '{name}' 初始化完成")
    
    def add_knowledge(self, knowledge: Knowledge) -> bool:
        """添加知识
        
        Args:
            knowledge: 知识条目
            
        Returns:
            是否添加成功
        """
        try:
            with self.lock:
                self.index.add_knowledge(knowledge)
                self.stats['total_knowledge'] += 1
                self.stats['last_updated'] = datetime.now()
                
                logger.debug(f"添加知识: {knowledge.subject} -> {knowledge.predicate} -> {knowledge.object}")
                return True
        except Exception as e:
            logger.error(f"添加知识失败: {e}")
            return False
    
    def add_knowledge_batch(self, knowledge_list: List[Knowledge]) -> int:
        """批量添加知识
        
        Args:
            knowledge_list: 知识列表
            
        Returns:
            成功添加的数量
        """
        success_count = 0
        for knowledge in knowledge_list:
            if self.add_knowledge(knowledge):
                success_count += 1
        
        logger.info(f"批量添加知识完成: {success_count}/{len(knowledge_list)}")
        return success_count
    
    def remove_knowledge(self, knowledge_id: str) -> bool:
        """移除知识
        
        Args:
            knowledge_id: 知识ID
            
        Returns:
            是否移除成功
        """
        try:
            with self.lock:
                if knowledge_id in self.index.id_index:
                    self.index.remove_knowledge(knowledge_id)
                    self.stats['total_knowledge'] -= 1
                    self.stats['last_updated'] = datetime.now()
                    
                    logger.debug(f"移除知识: {knowledge_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"移除知识失败: {e}")
            return False
    
    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """获取知识
        
        Args:
            knowledge_id: 知识ID
            
        Returns:
            知识条目
        """
        return self.index.get_knowledge(knowledge_id)
    
    def query_knowledge(self, query_type: QueryType = QueryType.EXACT, **kwargs) -> QueryResult:
        """查询知识
        
        Args:
            query_type: 查询类型
            **kwargs: 查询参数
            
        Returns:
            查询结果
        """
        start_time = datetime.now()
        
        try:
            with self.lock:
                self.stats['total_queries'] += 1
                
                if query_type == QueryType.EXACT:
                    result_ids = self._exact_query(**kwargs)
                elif query_type == QueryType.FUZZY:
                    result_ids = self._fuzzy_query(**kwargs)
                elif query_type == QueryType.SEMANTIC:
                    result_ids = self._semantic_query(**kwargs)
                elif query_type == QueryType.PATTERN:
                    result_ids = self._pattern_query(**kwargs)
                else:
                    result_ids = set()
                
                # 获取知识条目
                knowledge_items = []
                for kid in result_ids:
                    knowledge = self.index.get_knowledge(kid)
                    if knowledge:
                        knowledge_items.append(knowledge)
                
                # 排序（按置信度和时间）
                knowledge_items.sort(key=lambda k: (-k.confidence, -k.timestamp.timestamp()))
                
                query_time = (datetime.now() - start_time).total_seconds()
                
                result = QueryResult(
                    knowledge_items=knowledge_items,
                    total_count=len(knowledge_items),
                    query_time=query_time,
                    query_type=query_type.value
                )
                
                logger.debug(f"查询完成: {len(knowledge_items)} 条结果, 耗时 {query_time:.3f}s")
                return result
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return QueryResult(query_type=query_type.value)
    
    def _exact_query(self, **kwargs) -> Set[str]:
        """精确查询"""
        result_ids = None
        
        if 'subject' in kwargs:
            ids = self.index.query_by_subject(kwargs['subject'])
            result_ids = ids if result_ids is None else result_ids.intersection(ids)
        
        if 'predicate' in kwargs:
            ids = self.index.query_by_predicate(kwargs['predicate'])
            result_ids = ids if result_ids is None else result_ids.intersection(ids)
        
        if 'object' in kwargs:
            ids = self.index.query_by_object(kwargs['object'])
            result_ids = ids if result_ids is None else result_ids.intersection(ids)
        
        if 'type' in kwargs:
            knowledge_type = KnowledgeType(kwargs['type']) if isinstance(kwargs['type'], str) else kwargs['type']
            ids = self.index.query_by_type(knowledge_type)
            result_ids = ids if result_ids is None else result_ids.intersection(ids)
        
        if 'tag' in kwargs:
            ids = self.index.query_by_tag(kwargs['tag'])
            result_ids = ids if result_ids is None else result_ids.intersection(ids)
        
        return result_ids or set()
    
    def _fuzzy_query(self, **kwargs) -> Set[str]:
        """模糊查询"""
        if 'text' in kwargs:
            return self.index.query_by_text(kwargs['text'])
        
        # 其他模糊查询逻辑
        return set()
    
    def _semantic_query(self, **kwargs) -> Set[str]:
        """语义查询（简化实现）"""
        # 这里可以集成词向量或语言模型进行语义匹配
        # 目前使用简单的文本匹配
        if 'text' in kwargs:
            return self.index.query_by_text(kwargs['text'])
        
        return set()
    
    def _pattern_query(self, **kwargs) -> Set[str]:
        """模式查询"""
        # 支持通配符和正则表达式
        result_ids = set()
        
        if 'pattern' in kwargs:
            pattern = kwargs['pattern']
            for kid, knowledge in self.index.id_index.items():
                text = f"{knowledge.subject} {knowledge.predicate} {knowledge.object}"
                if re.search(pattern, text, re.IGNORECASE):
                    result_ids.add(kid)
        
        return result_ids
    
    def add_inference_rule(self, rule: InferenceRule) -> bool:
        """添加推理规则
        
        Args:
            rule: 推理规则
            
        Returns:
            是否添加成功
        """
        try:
            with self.lock:
                self.inference_rules[rule.id] = rule
                logger.debug(f"添加推理规则: {rule.name}")
                return True
        except Exception as e:
            logger.error(f"添加推理规则失败: {e}")
            return False
    
    def remove_inference_rule(self, rule_id: str) -> bool:
        """移除推理规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否移除成功
        """
        try:
            with self.lock:
                if rule_id in self.inference_rules:
                    del self.inference_rules[rule_id]
                    logger.debug(f"移除推理规则: {rule_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"移除推理规则失败: {e}")
            return False
    
    def run_inference(self, max_iterations: int = 10) -> InferenceResult:
        """运行推理
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            推理结果
        """
        start_time = datetime.now()
        new_knowledge = []
        applied_rules = []
        
        try:
            with self.lock:
                self.stats['total_inferences'] += 1
                
                for iteration in range(max_iterations):
                    iteration_new = []
                    
                    for rule in self.inference_rules.values():
                        if not rule.enabled:
                            continue
                        
                        # 简化的规则匹配和应用
                        rule_results = self._apply_inference_rule(rule)
                        if rule_results:
                            iteration_new.extend(rule_results)
                            applied_rules.append(rule.id)
                    
                    if not iteration_new:
                        break  # 没有新知识生成，停止推理
                    
                    # 添加新知识到知识库
                    for knowledge in iteration_new:
                        if self.add_knowledge(knowledge):
                            new_knowledge.append(knowledge)
                    
                    logger.debug(f"推理第{iteration + 1}轮: 生成 {len(iteration_new)} 条新知识")
                
                inference_time = (datetime.now() - start_time).total_seconds()
                
                result = InferenceResult(
                    new_knowledge=new_knowledge,
                    applied_rules=applied_rules,
                    inference_time=inference_time
                )
                
                logger.info(f"推理完成: 生成 {len(new_knowledge)} 条新知识, 耗时 {inference_time:.3f}s")
                return result
                
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return InferenceResult()
    
    def _apply_inference_rule(self, rule: InferenceRule) -> List[Knowledge]:
        """应用推理规则
        
        Args:
            rule: 推理规则
            
        Returns:
            新生成的知识列表
        """
        # 简化的规则应用实现
        # 实际应用中需要更复杂的模式匹配和变量绑定
        new_knowledge = []
        
        # 这里只是一个示例实现
        # 实际需要根据具体的规则语法来实现
        
        return new_knowledge
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            stats = self.stats.copy()
            stats['knowledge_by_type'] = {}
            
            for knowledge_type, ids in self.index.type_index.items():
                stats['knowledge_by_type'][knowledge_type.value] = len(ids)
            
            stats['inference_rules_count'] = len(self.inference_rules)
            
            return stats
    
    def export_knowledge(self, format_type: str = "json") -> str:
        """导出知识
        
        Args:
            format_type: 导出格式
            
        Returns:
            导出的数据
        """
        with self.lock:
            if format_type == "json":
                data = {
                    'name': self.name,
                    'knowledge': [k.to_dict() for k in self.index.id_index.values()],
                    'inference_rules': [r.to_dict() for r in self.inference_rules.values()],
                    'statistics': self.get_statistics()
                }
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
    
    def import_knowledge(self, data: str, format_type: str = "json") -> bool:
        """导入知识
        
        Args:
            data: 导入的数据
            format_type: 数据格式
            
        Returns:
            是否导入成功
        """
        try:
            with self.lock:
                if format_type == "json":
                    parsed_data = json.loads(data)
                    
                    # 导入知识
                    if 'knowledge' in parsed_data:
                        knowledge_list = [Knowledge.from_dict(k) for k in parsed_data['knowledge']]
                        self.add_knowledge_batch(knowledge_list)
                    
                    # 导入推理规则
                    if 'inference_rules' in parsed_data:
                        for rule_data in parsed_data['inference_rules']:
                            rule = InferenceRule(**rule_data)
                            self.add_inference_rule(rule)
                    
                    logger.info(f"知识导入完成")
                    return True
                else:
                    raise ValueError(f"不支持的导入格式: {format_type}")
        except Exception as e:
            logger.error(f"知识导入失败: {e}")
            return False

# 便捷函数
def create_knowledge_base(name: str = "default") -> KnowledgeBase:
    """创建知识库
    
    Args:
        name: 知识库名称
        
    Returns:
        知识库实例
    """
    return KnowledgeBase(name)

def create_knowledge(subject: str, predicate: str, object_val: str, 
                   knowledge_type: KnowledgeType = KnowledgeType.FACT,
                   confidence: float = 1.0, source: str = "",
                   tags: Set[str] = None, metadata: Dict[str, Any] = None) -> Knowledge:
    """创建知识条目
    
    Args:
        subject: 主语
        predicate: 谓语
        object_val: 宾语
        knowledge_type: 知识类型
        confidence: 置信度
        source: 来源
        tags: 标签
        metadata: 元数据
        
    Returns:
        知识条目
    """
    return Knowledge(
        id="",  # 自动生成
        type=knowledge_type,
        subject=subject,
        predicate=predicate,
        object=object_val,
        confidence=confidence,
        source=source,
        tags=tags or set(),
        metadata=metadata or {}
    )

def create_inference_rule(rule_id: str, name: str, conditions: List[str], 
                        conclusion: str, confidence: float = 1.0) -> InferenceRule:
    """创建推理规则
    
    Args:
        rule_id: 规则ID
        name: 规则名称
        conditions: 条件列表
        conclusion: 结论
        confidence: 置信度
        
    Returns:
        推理规则
    """
    return InferenceRule(
        id=rule_id,
        name=name,
        conditions=conditions,
        conclusion=conclusion,
        confidence=confidence
    )