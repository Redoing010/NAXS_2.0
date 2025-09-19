# 实体抽取模块
# 实现从文本中识别和抽取实体的功能

import logging
import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from collections import defaultdict, Counter
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """实体类型"""
    # 组织机构
    COMPANY = "company"            # 公司
    ORGANIZATION = "organization"  # 组织
    INSTITUTION = "institution"    # 机构
    
    # 人物
    PERSON = "person"              # 人物
    EXECUTIVE = "executive"        # 高管
    ANALYST = "analyst"            # 分析师
    
    # 金融工具
    STOCK = "stock"                # 股票
    FUND = "fund"                  # 基金
    BOND = "bond"                  # 债券
    INDEX = "index"                # 指数
    COMMODITY = "commodity"        # 商品
    CURRENCY = "currency"          # 货币
    
    # 行业概念
    INDUSTRY = "industry"          # 行业
    SECTOR = "sector"              # 板块
    CONCEPT = "concept"            # 概念
    THEME = "theme"                # 主题
    
    # 地理位置
    COUNTRY = "country"            # 国家
    REGION = "region"              # 地区
    CITY = "city"                  # 城市
    
    # 时间
    DATE = "date"                  # 日期
    TIME_PERIOD = "time_period"    # 时间周期
    
    # 数值
    MONEY = "money"                # 金额
    PERCENTAGE = "percentage"      # 百分比
    NUMBER = "number"              # 数字
    
    # 事件
    EVENT = "event"                # 事件
    NEWS = "news"                  # 新闻
    ANNOUNCEMENT = "announcement"  # 公告
    
    # 指标
    FINANCIAL_METRIC = "financial_metric"  # 财务指标
    MARKET_METRIC = "market_metric"        # 市场指标
    ECONOMIC_INDICATOR = "economic_indicator"  # 经济指标
    
    # 其他
    PRODUCT = "product"            # 产品
    TECHNOLOGY = "technology"      # 技术
    POLICY = "policy"              # 政策
    REGULATION = "regulation"      # 法规

@dataclass
class Entity:
    """实体对象"""
    text: str                      # 实体文本
    type: EntityType               # 实体类型
    start_pos: int                 # 开始位置
    end_pos: int                   # 结束位置
    confidence: float = 1.0        # 置信度
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    aliases: Set[str] = field(default_factory=set)  # 别名
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'text': self.text,
            'type': self.type.value,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'confidence': self.confidence,
            'properties': self.properties,
            'aliases': list(self.aliases)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """从字典创建实体"""
        return cls(
            text=data['text'],
            type=EntityType(data['type']),
            start_pos=data['start_pos'],
            end_pos=data['end_pos'],
            confidence=data.get('confidence', 1.0),
            properties=data.get('properties', {}),
            aliases=set(data.get('aliases', []))
        )

@dataclass
class EntityExtractionResult:
    """实体抽取结果"""
    text: str                      # 原始文本
    entities: List[Entity]         # 抽取的实体
    processing_time: float = 0.0   # 处理时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """按类型获取实体
        
        Args:
            entity_type: 实体类型
            
        Returns:
            实体列表
        """
        return [entity for entity in self.entities if entity.type == entity_type]
    
    def get_unique_entities(self) -> List[Entity]:
        """获取去重后的实体
        
        Returns:
            去重实体列表
        """
        seen = set()
        unique_entities = []
        
        for entity in self.entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'text': self.text,
            'entities': [entity.to_dict() for entity in self.entities],
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'entity_count': len(self.entities),
            'unique_entity_count': len(self.get_unique_entities())
        }

class TextProcessor:
    """文本预处理器"""
    
    def __init__(self):
        # 初始化jieba
        jieba.initialize()
        
        # 加载自定义词典
        self._load_custom_dict()
        
        logger.debug("文本处理器初始化完成")
    
    def _load_custom_dict(self):
        """加载自定义词典"""
        # 金融相关词汇
        financial_terms = [
            "股票", "基金", "债券", "期货", "期权", "外汇",
            "上证指数", "深证成指", "创业板指", "科创板",
            "市盈率", "市净率", "净资产收益率", "毛利率",
            "营业收入", "净利润", "现金流", "资产负债率",
            "涨停", "跌停", "停牌", "复牌", "分红", "配股"
        ]
        
        for term in financial_terms:
            jieba.add_word(term)
    
    def clean_text(self, text: str) -> str:
        """清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?()\[\]{}"\'-]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """分词和词性标注
        
        Args:
            text: 输入文本
            
        Returns:
            (词, 词性) 元组列表
        """
        return list(pseg.cut(text))
    
    def extract_sentences(self, text: str) -> List[str]:
        """提取句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 按标点符号分句
        sentences = re.split(r'[。！？；\n]', text)
        
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

class EntityExtractor:
    """实体抽取器
    
    使用规则和模式匹配的方法从文本中抽取实体
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        
        # 实体识别规则
        self.patterns = self._build_patterns()
        
        # 实体词典
        self.entity_dict = self._build_entity_dict()
        
        # 统计信息
        self.stats = {
            'texts_processed': 0,
            'entities_extracted': 0,
            'processing_time': 0.0
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("实体抽取器初始化完成")
    
    def _build_patterns(self) -> Dict[EntityType, List[str]]:
        """构建实体识别模式
        
        Returns:
            实体类型到正则模式的映射
        """
        patterns = {
            # 股票代码
            EntityType.STOCK: [
                r'\b\d{6}\b',  # 6位数字代码
                r'[A-Z]{2,6}',  # 字母代码
                r'\d{6}\.(SH|SZ)',  # 带交易所后缀
                r'(沪|深)\d{6}',  # 带交易所前缀
            ],
            
            # 金额
            EntityType.MONEY: [
                r'\d+(?:\.\d+)?[万亿千百十]?[元块]',
                r'\$\d+(?:\.\d+)?[KMB]?',
                r'\d+(?:\.\d+)?美元',
                r'\d+(?:\.\d+)?人民币'
            ],
            
            # 百分比
            EntityType.PERCENTAGE: [
                r'\d+(?:\.\d+)?%',
                r'百分之\d+(?:\.\d+)?',
                r'\d+(?:\.\d+)?个百分点'
            ],
            
            # 日期
            EntityType.DATE: [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'\d{4}/\d{1,2}/\d{1,2}',
                r'\d{1,2}月\d{1,2}日',
                r'今天|昨天|明天|前天|后天'
            ],
            
            # 时间周期
            EntityType.TIME_PERIOD: [
                r'\d+年',
                r'\d+个月',
                r'\d+天',
                r'\d+季度',
                r'上半年|下半年',
                r'第[一二三四]季度',
                r'近\d+[年月日]',
                r'过去\d+[年月日]'
            ],
            
            # 财务指标
            EntityType.FINANCIAL_METRIC: [
                r'市盈率|PE',
                r'市净率|PB',
                r'净资产收益率|ROE',
                r'总资产收益率|ROA',
                r'毛利率',
                r'净利率',
                r'资产负债率',
                r'流动比率',
                r'速动比率'
            ],
            
            # 市场指标
            EntityType.MARKET_METRIC: [
                r'上证指数|上证综指',
                r'深证成指|深成指',
                r'创业板指',
                r'科创50',
                r'沪深300',
                r'中证500',
                r'恒生指数',
                r'纳斯达克指数',
                r'道琼斯指数',
                r'标普500'
            ]
        }
        
        return patterns
    
    def _build_entity_dict(self) -> Dict[EntityType, Set[str]]:
        """构建实体词典
        
        Returns:
            实体类型到词汇集合的映射
        """
        entity_dict = {
            # 公司名称（示例）
            EntityType.COMPANY: {
                '中国平安', '招商银行', '贵州茅台', '腾讯控股', '阿里巴巴',
                '工商银行', '建设银行', '农业银行', '中国银行', '交通银行',
                '中国石油', '中国石化', '中国移动', '中国联通', '中国电信',
                '万科A', '保利地产', '华为技术', '小米集团', '字节跳动'
            },
            
            # 行业分类
            EntityType.INDUSTRY: {
                '银行业', '保险业', '证券业', '房地产业', '制造业',
                '信息技术', '医疗保健', '消费品', '能源', '材料',
                '工业', '公用事业', '电信服务', '金融服务'
            },
            
            # 概念板块
            EntityType.CONCEPT: {
                '人工智能', '5G', '新能源汽车', '芯片', '生物医药',
                '云计算', '大数据', '物联网', '区块链', '虚拟现实',
                '新基建', '碳中和', '数字货币', '军工', '航空航天'
            },
            
            # 地区
            EntityType.REGION: {
                '北京', '上海', '广州', '深圳', '杭州', '南京',
                '成都', '重庆', '武汉', '西安', '天津', '苏州',
                '长三角', '珠三角', '京津冀', '粤港澳大湾区'
            },
            
            # 人物职位
            EntityType.EXECUTIVE: {
                '董事长', '总经理', '首席执行官', 'CEO', 'CFO', 'CTO',
                '董事', '监事', '副总经理', '总裁', '副总裁'
            }
        }
        
        return entity_dict
    
    def extract(self, text: str) -> EntityExtractionResult:
        """抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体抽取结果
        """
        start_time = datetime.now()
        
        with self.lock:
            self.stats['texts_processed'] += 1
        
        # 清理文本
        cleaned_text = self.text_processor.clean_text(text)
        
        # 抽取实体
        entities = []
        
        # 1. 基于正则模式的抽取
        pattern_entities = self._extract_by_patterns(cleaned_text)
        entities.extend(pattern_entities)
        
        # 2. 基于词典的抽取
        dict_entities = self._extract_by_dict(cleaned_text)
        entities.extend(dict_entities)
        
        # 3. 基于词性标注的抽取
        pos_entities = self._extract_by_pos(cleaned_text)
        entities.extend(pos_entities)
        
        # 去重和合并
        entities = self._merge_entities(entities)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        with self.lock:
            self.stats['entities_extracted'] += len(entities)
            self.stats['processing_time'] += processing_time
        
        result = EntityExtractionResult(
            text=text,
            entities=entities,
            processing_time=processing_time,
            metadata={
                'method': 'rule_based',
                'patterns_used': len(self.patterns),
                'dict_size': sum(len(words) for words in self.entity_dict.values())
            }
        )
        
        logger.debug(f"实体抽取完成: {len(entities)} 个实体, 耗时 {processing_time:.3f}s")
        
        return result
    
    def _extract_by_patterns(self, text: str) -> List[Entity]:
        """基于正则模式抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity = Entity(
                        text=match.group(),
                        type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8  # 正则匹配的置信度
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_by_dict(self, text: str) -> List[Entity]:
        """基于词典抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        for entity_type, words in self.entity_dict.items():
            for word in words:
                start = 0
                while True:
                    pos = text.find(word, start)
                    if pos == -1:
                        break
                    
                    entity = Entity(
                        text=word,
                        type=entity_type,
                        start_pos=pos,
                        end_pos=pos + len(word),
                        confidence=0.9  # 词典匹配的置信度较高
                    )
                    entities.append(entity)
                    
                    start = pos + 1
        
        return entities
    
    def _extract_by_pos(self, text: str) -> List[Entity]:
        """基于词性标注抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        tokens = self.text_processor.tokenize(text)
        
        current_pos = 0
        for word, pos_tag in tokens:
            # 根据词性标注识别实体
            entity_type = None
            confidence = 0.6  # 词性标注的置信度较低
            
            if pos_tag in ['nr']:  # 人名
                entity_type = EntityType.PERSON
            elif pos_tag in ['ns']:  # 地名
                entity_type = EntityType.REGION
            elif pos_tag in ['nt']:  # 机构名
                entity_type = EntityType.ORGANIZATION
            elif pos_tag in ['m']:  # 数词
                if re.match(r'\d+(?:\.\d+)?', word):
                    entity_type = EntityType.NUMBER
            
            if entity_type:
                # 在原文中查找位置
                start_pos = text.find(word, current_pos)
                if start_pos != -1:
                    entity = Entity(
                        text=word,
                        type=entity_type,
                        start_pos=start_pos,
                        end_pos=start_pos + len(word),
                        confidence=confidence
                    )
                    entities.append(entity)
            
            current_pos += len(word)
        
        return entities
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并重叠的实体
        
        Args:
            entities: 实体列表
            
        Returns:
            合并后的实体列表
        """
        if not entities:
            return entities
        
        # 按位置排序
        entities.sort(key=lambda e: (e.start_pos, e.end_pos))
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # 检查是否重叠
            if (next_entity.start_pos < current.end_pos and 
                next_entity.end_pos > current.start_pos):
                
                # 重叠实体，选择置信度更高的
                if next_entity.confidence > current.confidence:
                    current = next_entity
                # 如果置信度相同，选择更长的
                elif (next_entity.confidence == current.confidence and 
                      len(next_entity.text) > len(current.text)):
                    current = next_entity
            else:
                # 不重叠，添加当前实体
                merged.append(current)
                current = next_entity
        
        # 添加最后一个实体
        merged.append(current)
        
        return merged
    
    def add_entity_pattern(self, entity_type: EntityType, pattern: str):
        """添加实体识别模式
        
        Args:
            entity_type: 实体类型
            pattern: 正则表达式模式
        """
        if entity_type not in self.patterns:
            self.patterns[entity_type] = []
        
        self.patterns[entity_type].append(pattern)
        logger.debug(f"添加实体模式: {entity_type.value} - {pattern}")
    
    def add_entity_words(self, entity_type: EntityType, words: List[str]):
        """添加实体词汇
        
        Args:
            entity_type: 实体类型
            words: 词汇列表
        """
        if entity_type not in self.entity_dict:
            self.entity_dict[entity_type] = set()
        
        self.entity_dict[entity_type].update(words)
        logger.debug(f"添加实体词汇: {entity_type.value} - {len(words)} 个词")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            stats = self.stats.copy()
            
            if stats['texts_processed'] > 0:
                stats['avg_entities_per_text'] = stats['entities_extracted'] / stats['texts_processed']
                stats['avg_processing_time'] = stats['processing_time'] / stats['texts_processed']
            else:
                stats['avg_entities_per_text'] = 0
                stats['avg_processing_time'] = 0
            
            stats['pattern_count'] = sum(len(patterns) for patterns in self.patterns.values())
            stats['dict_size'] = sum(len(words) for words in self.entity_dict.values())
            
            return stats

# 便捷函数
def create_entity_extractor() -> EntityExtractor:
    """创建实体抽取器
    
    Returns:
        实体抽取器实例
    """
    return EntityExtractor()

def create_text_processor() -> TextProcessor:
    """创建文本处理器
    
    Returns:
        文本处理器实例
    """
    return TextProcessor()

def create_entity(text: str, entity_type: EntityType, start_pos: int, end_pos: int, **kwargs) -> Entity:
    """创建实体
    
    Args:
        text: 实体文本
        entity_type: 实体类型
        start_pos: 开始位置
        end_pos: 结束位置
        **kwargs: 其他参数
        
    Returns:
        实体实例
    """
    return Entity(
        text=text,
        type=entity_type,
        start_pos=start_pos,
        end_pos=end_pos,
        **kwargs
    )

def extract_entities_from_text(text: str) -> EntityExtractionResult:
    """从文本中抽取实体（便捷函数）
    
    Args:
        text: 输入文本
        
    Returns:
        实体抽取结果
    """
    extractor = create_entity_extractor()
    return extractor.extract(text)

def create_financial_entity_extractor() -> EntityExtractor:
    """创建金融领域专用的实体抽取器
    
    Returns:
        金融实体抽取器
    """
    extractor = EntityExtractor()
    
    # 添加金融特定的模式
    financial_patterns = {
        EntityType.STOCK: [
            r'[A-Z]{1,5}\.(SH|SZ|HK)',  # 股票代码
            r'\b\d{6}\.(SH|SZ)\b',      # A股代码
            r'\b[0-9]{5}\.(HK)\b'       # 港股代码
        ],
        EntityType.FUND: [
            r'\d{6}基金',
            r'[\u4e00-\u9fa5]+基金',
            r'ETF基金'
        ],
        EntityType.FINANCIAL_METRIC: [
            r'EPS|每股收益',
            r'EBITDA',
            r'现金流量',
            r'营业利润率',
            r'净资产收益率'
        ]
    }
    
    for entity_type, patterns in financial_patterns.items():
        for pattern in patterns:
            extractor.add_entity_pattern(entity_type, pattern)
    
    # 添加金融词汇
    financial_words = {
        EntityType.COMPANY: [
            '中国平安', '招商银行', '贵州茅台', '腾讯控股',
            '阿里巴巴', '工商银行', '建设银行', '农业银行'
        ],
        EntityType.CONCEPT: [
            '新能源', '人工智能', '5G通信', '生物医药',
            '半导体', '新基建', '碳中和', '数字货币'
        ]
    }
    
    for entity_type, words in financial_words.items():
        extractor.add_entity_words(entity_type, words)
    
    return extractor