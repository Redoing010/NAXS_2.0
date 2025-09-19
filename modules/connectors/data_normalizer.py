# 数据标准化器
# 统一不同数据源的数据格式和结构

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

from .base_connector import DataPoint, DataBatch, DataType

class NormalizationType(Enum):
    """标准化类型"""
    FIELD_MAPPING = "field_mapping"        # 字段映射
    VALUE_CONVERSION = "value_conversion"  # 数值转换
    DATE_FORMAT = "date_format"            # 日期格式
    SYMBOL_FORMAT = "symbol_format"        # 股票代码格式
    UNIT_CONVERSION = "unit_conversion"    # 单位转换
    DATA_CLEANING = "data_cleaning"        # 数据清洗

@dataclass
class NormalizationRule:
    """标准化规则"""
    rule_id: str
    rule_type: NormalizationType
    source_pattern: str  # 源数据模式
    target_format: str   # 目标格式
    condition: Optional[str] = None  # 应用条件
    priority: int = 1    # 优先级
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, data_point: DataPoint) -> bool:
        """检查规则是否适用于数据点"""
        if not self.enabled:
            return False
        
        # 检查数据源
        if self.source_pattern and not re.match(self.source_pattern, data_point.source):
            return False
        
        # 检查条件
        if self.condition:
            try:
                # 简单的条件评估
                return eval(self.condition, {'data_point': data_point})
            except Exception:
                return False
        
        return True

class DataNormalizer:
    """数据标准化器"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataNormalizer")
        self.rules: List[NormalizationRule] = []
        self.field_mappings: Dict[str, Dict[str, str]] = {}
        self.value_converters: Dict[str, Callable] = {}
        
        # 初始化默认规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认标准化规则"""
        # 股票代码标准化规则
        self.add_rule(NormalizationRule(
            rule_id="symbol_format_akshare",
            rule_type=NormalizationType.SYMBOL_FORMAT,
            source_pattern="akshare",
            target_format="standard",
            priority=1
        ))
        
        self.add_rule(NormalizationRule(
            rule_id="symbol_format_tushare",
            rule_type=NormalizationType.SYMBOL_FORMAT,
            source_pattern="tushare",
            target_format="standard",
            priority=1
        ))
        
        # 字段映射规则
        self.add_rule(NormalizationRule(
            rule_id="field_mapping_stock_price",
            rule_type=NormalizationType.FIELD_MAPPING,
            source_pattern=".*",
            target_format="standard",
            condition="data_point.data_type == DataType.STOCK_PRICE",
            priority=2
        ))
        
        # 数值转换规则
        self.add_rule(NormalizationRule(
            rule_id="value_conversion_volume",
            rule_type=NormalizationType.VALUE_CONVERSION,
            source_pattern=".*",
            target_format="shares",
            condition="'volume' in data_point.values",
            priority=3
        ))
        
        # 日期格式规则
        self.add_rule(NormalizationRule(
            rule_id="date_format_standard",
            rule_type=NormalizationType.DATE_FORMAT,
            source_pattern=".*",
            target_format="iso",
            priority=4
        ))
        
        # 数据清洗规则
        self.add_rule(NormalizationRule(
            rule_id="data_cleaning_basic",
            rule_type=NormalizationType.DATA_CLEANING,
            source_pattern=".*",
            target_format="clean",
            priority=5
        ))
        
        # 初始化字段映射
        self._init_field_mappings()
        
        # 初始化值转换器
        self._init_value_converters()
    
    def _init_field_mappings(self):
        """初始化字段映射"""
        # 股票价格字段映射
        self.field_mappings['stock_price'] = {
            # AkShare字段映射
            '开盘': 'open',
            '最高': 'high', 
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate',
            
            # TuShare字段映射
            'trade_date': 'date',
            'ts_code': 'symbol',
            'vol': 'volume',
            'pre_close': 'prev_close',
            'pct_chg': 'change_pct',
            
            # 通用映射
            'o': 'open',
            'h': 'high',
            'l': 'low', 
            'c': 'close',
            'v': 'volume',
            'adj_close': 'close',
        }
        
        # 财务数据字段映射
        self.field_mappings['financial'] = {
            # 资产负债表
            'total_assets': 'total_assets',
            'total_liab': 'total_liabilities',
            'total_hldr_eqy_exc_min_int': 'shareholders_equity',
            
            # 利润表
            'total_revenue': 'revenue',
            'n_income': 'net_income',
            'operate_profit': 'operating_profit',
            
            # 现金流量表
            'n_cashflow_act': 'operating_cash_flow',
            'n_cashflow_inv_act': 'investing_cash_flow',
            'n_cashflow_fin_act': 'financing_cash_flow',
        }
        
        # 市场指数字段映射
        self.field_mappings['market_index'] = {
            'index_code': 'symbol',
            'index_name': 'name',
            'close_point': 'close',
            'change_point': 'change_amount',
            'change_rate': 'change_pct',
        }
    
    def _init_value_converters(self):
        """初始化值转换器"""
        # 成交量转换（手 -> 股）
        self.value_converters['volume_to_shares'] = lambda x: x * 100 if x else 0
        
        # 成交额转换（元 -> 万元）
        self.value_converters['amount_to_wan'] = lambda x: x / 10000 if x else 0
        
        # 百分比转换
        self.value_converters['pct_to_decimal'] = lambda x: x / 100 if x else 0
        
        # 价格精度转换
        self.value_converters['price_precision'] = lambda x: round(x, 2) if x else 0
        
        # 日期转换
        self.value_converters['date_to_datetime'] = self._convert_date
        
        # 字符串清理
        self.value_converters['clean_string'] = self._clean_string
        
        # 数值清理
        self.value_converters['clean_numeric'] = self._clean_numeric
    
    def add_rule(self, rule: NormalizationRule):
        """添加标准化规则"""
        self.rules.append(rule)
        # 按优先级排序
        self.rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除标准化规则"""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                return True
        return False
    
    def get_rules(self, rule_type: Optional[NormalizationType] = None) -> List[NormalizationRule]:
        """获取规则列表"""
        if rule_type:
            return [rule for rule in self.rules if rule.rule_type == rule_type]
        return self.rules.copy()
    
    async def normalize(self, data_batch: DataBatch) -> DataBatch:
        """标准化数据批次"""
        try:
            normalized_points = []
            
            for data_point in data_batch.data_points:
                normalized_point = await self.normalize_data_point(data_point)
                normalized_points.append(normalized_point)
            
            # 创建新的数据批次
            normalized_batch = DataBatch(
                data_points=normalized_points,
                batch_id=data_batch.batch_id + "_normalized",
                source=data_batch.source,
                data_type=data_batch.data_type,
                timestamp=data_batch.timestamp,
                metadata={**data_batch.metadata, 'normalized': True}
            )
            
            self.logger.debug(f"Normalized batch {data_batch.batch_id} with {len(normalized_points)} points")
            return normalized_batch
            
        except Exception as e:
            self.logger.error(f"Failed to normalize data batch: {e}")
            raise
    
    async def normalize_data_point(self, data_point: DataPoint) -> DataPoint:
        """标准化单个数据点"""
        try:
            # 复制数据点
            normalized_point = DataPoint(
                timestamp=data_point.timestamp,
                symbol=data_point.symbol,
                data_type=data_point.data_type,
                values=data_point.values.copy(),
                metadata=data_point.metadata.copy(),
                source=data_point.source,
                quality_score=data_point.quality_score
            )
            
            # 应用标准化规则
            for rule in self.rules:
                if rule.matches(data_point):
                    normalized_point = await self._apply_rule(normalized_point, rule)
            
            return normalized_point
            
        except Exception as e:
            self.logger.error(f"Failed to normalize data point: {e}")
            return data_point  # 返回原始数据点
    
    async def _apply_rule(self, data_point: DataPoint, rule: NormalizationRule) -> DataPoint:
        """应用标准化规则"""
        try:
            if rule.rule_type == NormalizationType.SYMBOL_FORMAT:
                data_point.symbol = self._normalize_symbol(data_point.symbol, data_point.source)
                
            elif rule.rule_type == NormalizationType.FIELD_MAPPING:
                data_point.values = self._map_fields(data_point.values, data_point.data_type)
                
            elif rule.rule_type == NormalizationType.VALUE_CONVERSION:
                data_point.values = self._convert_values(data_point.values, data_point.source)
                
            elif rule.rule_type == NormalizationType.DATE_FORMAT:
                data_point.timestamp = self._normalize_date(data_point.timestamp)
                
            elif rule.rule_type == NormalizationType.DATA_CLEANING:
                data_point = self._clean_data_point(data_point)
            
            return data_point
            
        except Exception as e:
            self.logger.warning(f"Failed to apply rule {rule.rule_id}: {e}")
            return data_point
    
    def _normalize_symbol(self, symbol: str, source: str) -> str:
        """标准化股票代码"""
        if not symbol:
            return symbol
        
        # 移除空格和特殊字符
        symbol = symbol.strip().upper()
        
        # 根据数据源进行转换
        if source == 'akshare':
            # AkShare使用6位数字代码
            if '.' in symbol:
                code, exchange = symbol.split('.')
                return f"{code}.{exchange.upper()}"
            else:
                # 根据代码判断交易所
                if symbol.startswith('6'):
                    return f"{symbol}.SH"
                elif symbol.startswith(('0', '3')):
                    return f"{symbol}.SZ"
                    
        elif source == 'tushare':
            # TuShare已经是标准格式
            return symbol
            
        elif source == 'yahoo':
            # Yahoo Finance格式转换
            if symbol.endswith('.SS'):
                return symbol.replace('.SS', '.SH')
            elif symbol.endswith('.SZ'):
                return symbol
        
        return symbol
    
    def _map_fields(self, values: Dict[str, Any], data_type: DataType) -> Dict[str, Any]:
        """映射字段名称"""
        # 根据数据类型选择映射表
        if data_type == DataType.STOCK_PRICE:
            mapping = self.field_mappings.get('stock_price', {})
        elif data_type == DataType.FINANCIAL_REPORT:
            mapping = self.field_mappings.get('financial', {})
        elif data_type == DataType.MARKET_INDEX:
            mapping = self.field_mappings.get('market_index', {})
        else:
            mapping = {}
        
        # 应用映射
        mapped_values = {}
        for key, value in values.items():
            mapped_key = mapping.get(key, key)
            mapped_values[mapped_key] = value
        
        return mapped_values
    
    def _convert_values(self, values: Dict[str, Any], source: str) -> Dict[str, Any]:
        """转换数值"""
        converted_values = {}
        
        for key, value in values.items():
            try:
                # 成交量转换
                if key in ['volume', 'vol', '成交量'] and source in ['akshare', 'tushare']:
                    # AkShare和TuShare的成交量单位可能不同
                    if source == 'akshare':
                        # AkShare成交量单位是手，转换为股
                        converted_values[key] = self.value_converters['volume_to_shares'](value)
                    else:
                        converted_values[key] = value
                
                # 成交额转换
                elif key in ['amount', '成交额']:
                    converted_values[key] = self.value_converters['amount_to_wan'](value)
                
                # 百分比转换
                elif key in ['change_pct', 'pct_chg', '涨跌幅'] and isinstance(value, (int, float)):
                    # 如果值大于1，可能是百分比形式，需要转换为小数
                    if abs(value) > 1:
                        converted_values[key] = self.value_converters['pct_to_decimal'](value)
                    else:
                        converted_values[key] = value
                
                # 价格精度
                elif key in ['open', 'high', 'low', 'close', 'price'] and isinstance(value, (int, float)):
                    converted_values[key] = self.value_converters['price_precision'](value)
                
                # 数值清理
                elif isinstance(value, (int, float)):
                    converted_values[key] = self.value_converters['clean_numeric'](value)
                
                # 字符串清理
                elif isinstance(value, str):
                    converted_values[key] = self.value_converters['clean_string'](value)
                
                else:
                    converted_values[key] = value
                    
            except Exception as e:
                self.logger.warning(f"Failed to convert value {key}={value}: {e}")
                converted_values[key] = value
        
        return converted_values
    
    def _normalize_date(self, timestamp: datetime) -> datetime:
        """标准化日期格式"""
        # 确保时区信息
        if timestamp.tzinfo is None:
            # 假设是中国时区
            import pytz
            china_tz = pytz.timezone('Asia/Shanghai')
            timestamp = china_tz.localize(timestamp)
        
        return timestamp
    
    def _clean_data_point(self, data_point: DataPoint) -> DataPoint:
        """清理数据点"""
        # 移除空值和无效值
        cleaned_values = {}
        
        for key, value in data_point.values.items():
            if value is not None and value != '' and not (isinstance(value, float) and np.isnan(value)):
                cleaned_values[key] = value
        
        data_point.values = cleaned_values
        
        # 更新质量分数
        if len(cleaned_values) < len(data_point.values):
            data_point.quality_score *= 0.9  # 降低质量分数
        
        return data_point
    
    def _convert_date(self, date_value: Any) -> datetime:
        """转换日期"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time())
        elif isinstance(date_value, str):
            try:
                return pd.to_datetime(date_value)
            except Exception:
                return datetime.now()
        else:
            return datetime.now()
    
    def _clean_string(self, value: str) -> str:
        """清理字符串"""
        if not isinstance(value, str):
            return str(value)
        
        # 移除前后空格
        value = value.strip()
        
        # 移除特殊字符
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # 统一换行符
        value = value.replace('\r\n', '\n').replace('\r', '\n')
        
        return value
    
    def _clean_numeric(self, value: Union[int, float]) -> Union[int, float]:
        """清理数值"""
        if pd.isna(value) or np.isinf(value):
            return 0
        
        # 处理异常大的数值
        if abs(value) > 1e15:
            return 0
        
        return value
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """获取标准化统计信息"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'rule_types': list(set(r.rule_type.value for r in self.rules)),
            'field_mappings': {k: len(v) for k, v in self.field_mappings.items()},
            'value_converters': list(self.value_converters.keys()),
        }

# 便捷函数
def create_normalization_rule(
    rule_id: str,
    rule_type: NormalizationType,
    source_pattern: str,
    target_format: str,
    **kwargs
) -> NormalizationRule:
    """创建标准化规则"""
    return NormalizationRule(
        rule_id=rule_id,
        rule_type=rule_type,
        source_pattern=source_pattern,
        target_format=target_format,
        **kwargs
    )

def create_data_normalizer(custom_rules: Optional[List[NormalizationRule]] = None) -> DataNormalizer:
    """创建数据标准化器"""
    normalizer = DataNormalizer()
    
    if custom_rules:
        for rule in custom_rules:
            normalizer.add_rule(rule)
    
    return normalizer