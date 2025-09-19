# 数据质量检查器
# 检查和验证数据质量，确保数据的准确性和完整性

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics

from .base_connector import DataPoint, DataBatch, DataType

class QualityIssueType(Enum):
    """质量问题类型"""
    MISSING_VALUE = "missing_value"          # 缺失值
    INVALID_VALUE = "invalid_value"          # 无效值
    OUTLIER = "outlier"                      # 异常值
    DUPLICATE = "duplicate"                  # 重复数据
    INCONSISTENT = "inconsistent"            # 不一致
    OUT_OF_RANGE = "out_of_range"            # 超出范围
    WRONG_FORMAT = "wrong_format"            # 格式错误
    STALE_DATA = "stale_data"                # 过期数据
    INCOMPLETE = "incomplete"                # 不完整
    SUSPICIOUS = "suspicious"                # 可疑数据

class QualitySeverity(Enum):
    """质量问题严重程度"""
    LOW = "low"          # 低
    MEDIUM = "medium"    # 中
    HIGH = "high"        # 高
    CRITICAL = "critical" # 严重

@dataclass
class QualityIssue:
    """质量问题"""
    issue_type: QualityIssueType
    severity: QualitySeverity
    field: str
    description: str
    value: Any = None
    expected_value: Any = None
    suggestion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityResult:
    """质量检查结果"""
    is_valid: bool
    quality_score: float  # 0-1之间
    total_points: int
    valid_points: int
    issues: List[QualityIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        return len(self.issues) / self.total_points if self.total_points > 0 else 0
    
    @property
    def critical_issues(self) -> List[QualityIssue]:
        """严重问题"""
        return [issue for issue in self.issues if issue.severity == QualitySeverity.CRITICAL]

@dataclass
class QualityRule:
    """质量检查规则"""
    rule_id: str
    rule_name: str
    data_type: Optional[DataType] = None
    fields: Optional[List[str]] = None
    condition: str = ""  # 检查条件
    threshold: Optional[float] = None
    enabled: bool = True
    severity: QualitySeverity = QualitySeverity.MEDIUM
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataQualityChecker")
        self.rules: List[QualityRule] = []
        self.field_ranges: Dict[str, Tuple[float, float]] = {}
        self.required_fields: Dict[DataType, Set[str]] = {}
        
        # 初始化默认规则
        self._init_default_rules()
        self._init_field_ranges()
        self._init_required_fields()
    
    def _init_default_rules(self):
        """初始化默认质量检查规则"""
        # 股票价格数据规则
        self.add_rule(QualityRule(
            rule_id="stock_price_positive",
            rule_name="股票价格必须为正数",
            data_type=DataType.STOCK_PRICE,
            fields=["open", "high", "low", "close"],
            condition="value > 0",
            severity=QualitySeverity.HIGH,
            description="股票价格不能为负数或零"
        ))
        
        self.add_rule(QualityRule(
            rule_id="stock_price_relationship",
            rule_name="股票价格关系检查",
            data_type=DataType.STOCK_PRICE,
            condition="low <= open <= high and low <= close <= high",
            severity=QualitySeverity.HIGH,
            description="最低价应小于等于开盘价和收盘价，最高价应大于等于开盘价和收盘价"
        ))
        
        self.add_rule(QualityRule(
            rule_id="volume_non_negative",
            rule_name="成交量非负检查",
            data_type=DataType.STOCK_PRICE,
            fields=["volume"],
            condition="value >= 0",
            severity=QualitySeverity.MEDIUM,
            description="成交量不能为负数"
        ))
        
        # 财务数据规则
        self.add_rule(QualityRule(
            rule_id="financial_completeness",
            rule_name="财务数据完整性检查",
            data_type=DataType.FINANCIAL_REPORT,
            severity=QualitySeverity.MEDIUM,
            description="财务数据应包含必要字段"
        ))
        
        # 时间序列规则
        self.add_rule(QualityRule(
            rule_id="timestamp_validity",
            rule_name="时间戳有效性检查",
            condition="timestamp is not None and timestamp <= datetime.now()",
            severity=QualitySeverity.HIGH,
            description="时间戳不能为空且不能是未来时间"
        ))
        
        # 重复数据规则
        self.add_rule(QualityRule(
            rule_id="duplicate_detection",
            rule_name="重复数据检测",
            severity=QualitySeverity.MEDIUM,
            description="检测重复的数据点"
        ))
    
    def _init_field_ranges(self):
        """初始化字段取值范围"""
        # 股票价格范围（人民币）
        self.field_ranges.update({
            'open': (0.01, 10000),
            'high': (0.01, 10000),
            'low': (0.01, 10000),
            'close': (0.01, 10000),
            'volume': (0, 1e12),
            'amount': (0, 1e15),
            'change_pct': (-20, 20),  # 涨跌幅限制
            'turnover_rate': (0, 100),  # 换手率
        })
    
    def _init_required_fields(self):
        """初始化必需字段"""
        self.required_fields[DataType.STOCK_PRICE] = {
            'open', 'high', 'low', 'close', 'volume'
        }
        
        self.required_fields[DataType.STOCK_INFO] = {
            'name', 'symbol'
        }
        
        self.required_fields[DataType.FINANCIAL_REPORT] = {
            'revenue', 'net_income'
        }
    
    def add_rule(self, rule: QualityRule):
        """添加质量检查规则"""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除质量检查规则"""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                return True
        return False
    
    def get_rules(self, data_type: Optional[DataType] = None) -> List[QualityRule]:
        """获取规则列表"""
        if data_type:
            return [rule for rule in self.rules 
                   if rule.data_type is None or rule.data_type == data_type]
        return self.rules.copy()
    
    async def check(self, data_batch: DataBatch) -> QualityResult:
        """检查数据批次质量"""
        try:
            total_points = len(data_batch.data_points)
            all_issues = []
            valid_points = 0
            
            # 检查每个数据点
            for i, data_point in enumerate(data_batch.data_points):
                point_issues = await self.check_data_point(data_point)
                
                # 添加位置信息
                for issue in point_issues:
                    issue.metadata['point_index'] = i
                    issue.metadata['symbol'] = data_point.symbol
                    issue.metadata['timestamp'] = data_point.timestamp
                
                all_issues.extend(point_issues)
                
                # 如果没有严重问题，认为是有效点
                critical_issues = [issue for issue in point_issues 
                                 if issue.severity == QualitySeverity.CRITICAL]
                if not critical_issues:
                    valid_points += 1
            
            # 批次级别检查
            batch_issues = await self.check_batch_level(data_batch)
            all_issues.extend(batch_issues)
            
            # 计算质量分数
            quality_score = self._calculate_quality_score(all_issues, total_points)
            
            # 生成统计信息
            statistics = self._generate_statistics(data_batch, all_issues)
            
            # 生成建议
            recommendations = self._generate_recommendations(all_issues)
            
            result = QualityResult(
                is_valid=len([issue for issue in all_issues 
                            if issue.severity == QualitySeverity.CRITICAL]) == 0,
                quality_score=quality_score,
                total_points=total_points,
                valid_points=valid_points,
                issues=all_issues,
                statistics=statistics,
                recommendations=recommendations
            )
            
            self.logger.debug(f"Quality check completed: score={quality_score:.2f}, issues={len(all_issues)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
            raise
    
    async def check_data_point(self, data_point: DataPoint) -> List[QualityIssue]:
        """检查单个数据点质量"""
        issues = []
        
        try:
            # 获取适用的规则
            applicable_rules = [rule for rule in self.rules 
                              if rule.enabled and 
                              (rule.data_type is None or rule.data_type == data_point.data_type)]
            
            # 基本检查
            issues.extend(self._check_basic_validity(data_point))
            
            # 字段检查
            issues.extend(self._check_required_fields(data_point))
            
            # 数值范围检查
            issues.extend(self._check_value_ranges(data_point))
            
            # 业务规则检查
            issues.extend(self._check_business_rules(data_point))
            
            # 异常值检查
            issues.extend(self._check_outliers(data_point))
            
            # 时间有效性检查
            issues.extend(self._check_timestamp_validity(data_point))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Failed to check data point: {e}")
            return [QualityIssue(
                issue_type=QualityIssueType.INVALID_VALUE,
                severity=QualitySeverity.HIGH,
                field="general",
                description=f"检查过程中发生错误: {e}"
            )]
    
    async def check_batch_level(self, data_batch: DataBatch) -> List[QualityIssue]:
        """批次级别检查"""
        issues = []
        
        try:
            # 重复数据检查
            issues.extend(self._check_duplicates(data_batch))
            
            # 数据连续性检查
            issues.extend(self._check_continuity(data_batch))
            
            # 数据一致性检查
            issues.extend(self._check_consistency(data_batch))
            
            # 数据新鲜度检查
            issues.extend(self._check_freshness(data_batch))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Batch level check failed: {e}")
            return []
    
    def _check_basic_validity(self, data_point: DataPoint) -> List[QualityIssue]:
        """基本有效性检查"""
        issues = []
        
        # 检查必要字段
        if not data_point.symbol:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_VALUE,
                severity=QualitySeverity.CRITICAL,
                field="symbol",
                description="股票代码不能为空"
            ))
        
        if not data_point.timestamp:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_VALUE,
                severity=QualitySeverity.CRITICAL,
                field="timestamp",
                description="时间戳不能为空"
            ))
        
        if not data_point.values:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.INCOMPLETE,
                severity=QualitySeverity.CRITICAL,
                field="values",
                description="数据值不能为空"
            ))
        
        return issues
    
    def _check_required_fields(self, data_point: DataPoint) -> List[QualityIssue]:
        """检查必需字段"""
        issues = []
        
        required_fields = self.required_fields.get(data_point.data_type, set())
        
        for field in required_fields:
            if field not in data_point.values or data_point.values[field] is None:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.MISSING_VALUE,
                    severity=QualitySeverity.HIGH,
                    field=field,
                    description=f"必需字段 {field} 缺失"
                ))
        
        return issues
    
    def _check_value_ranges(self, data_point: DataPoint) -> List[QualityIssue]:
        """检查数值范围"""
        issues = []
        
        for field, value in data_point.values.items():
            if field in self.field_ranges and isinstance(value, (int, float)):
                min_val, max_val = self.field_ranges[field]
                
                if value < min_val or value > max_val:
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.OUT_OF_RANGE,
                        severity=QualitySeverity.MEDIUM,
                        field=field,
                        description=f"字段 {field} 值 {value} 超出范围 [{min_val}, {max_val}]",
                        value=value,
                        expected_value=f"[{min_val}, {max_val}]"
                    ))
        
        return issues
    
    def _check_business_rules(self, data_point: DataPoint) -> List[QualityIssue]:
        """检查业务规则"""
        issues = []
        
        if data_point.data_type == DataType.STOCK_PRICE:
            values = data_point.values
            
            # 检查价格关系
            if all(k in values for k in ['open', 'high', 'low', 'close']):
                open_price = values['open']
                high_price = values['high']
                low_price = values['low']
                close_price = values['close']
                
                if not (low_price <= open_price <= high_price):
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT,
                        severity=QualitySeverity.HIGH,
                        field="price_relationship",
                        description=f"开盘价 {open_price} 不在最高价 {high_price} 和最低价 {low_price} 之间"
                    ))
                
                if not (low_price <= close_price <= high_price):
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT,
                        severity=QualitySeverity.HIGH,
                        field="price_relationship",
                        description=f"收盘价 {close_price} 不在最高价 {high_price} 和最低价 {low_price} 之间"
                    ))
            
            # 检查涨跌幅限制
            if 'change_pct' in values:
                change_pct = values['change_pct']
                if abs(change_pct) > 20:  # 涨跌停限制
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.SUSPICIOUS,
                        severity=QualitySeverity.MEDIUM,
                        field="change_pct",
                        description=f"涨跌幅 {change_pct}% 超过正常范围，可能需要验证",
                        value=change_pct
                    ))
        
        return issues
    
    def _check_outliers(self, data_point: DataPoint) -> List[QualityIssue]:
        """检查异常值"""
        issues = []
        
        # 这里可以实现更复杂的异常值检测算法
        # 例如基于历史数据的统计分析
        
        for field, value in data_point.values.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # 简单的异常值检测：检查是否为极端值
                if field in ['open', 'high', 'low', 'close']:
                    if value > 1000 or value < 0.01:  # 股价异常
                        issues.append(QualityIssue(
                            issue_type=QualityIssueType.OUTLIER,
                            severity=QualitySeverity.MEDIUM,
                            field=field,
                            description=f"字段 {field} 值 {value} 可能是异常值",
                            value=value
                        ))
        
        return issues
    
    def _check_timestamp_validity(self, data_point: DataPoint) -> List[QualityIssue]:
        """检查时间戳有效性"""
        issues = []
        
        if data_point.timestamp:
            now = datetime.now()
            
            # 检查未来时间
            if data_point.timestamp > now:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INVALID_VALUE,
                    severity=QualitySeverity.HIGH,
                    field="timestamp",
                    description=f"时间戳 {data_point.timestamp} 是未来时间",
                    value=data_point.timestamp
                ))
            
            # 检查过于陈旧的数据
            if (now - data_point.timestamp).days > 365:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.STALE_DATA,
                    severity=QualitySeverity.LOW,
                    field="timestamp",
                    description=f"数据时间 {data_point.timestamp} 过于陈旧",
                    value=data_point.timestamp
                ))
        
        return issues
    
    def _check_duplicates(self, data_batch: DataBatch) -> List[QualityIssue]:
        """检查重复数据"""
        issues = []
        
        # 创建数据点的唯一标识
        seen = set()
        duplicates = []
        
        for i, data_point in enumerate(data_batch.data_points):
            key = (data_point.symbol, data_point.timestamp, data_point.data_type)
            
            if key in seen:
                duplicates.append((i, data_point))
            else:
                seen.add(key)
        
        for i, data_point in duplicates:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.DUPLICATE,
                severity=QualitySeverity.MEDIUM,
                field="data_point",
                description=f"发现重复数据点: {data_point.symbol} at {data_point.timestamp}",
                metadata={'point_index': i}
            ))
        
        return issues
    
    def _check_continuity(self, data_batch: DataBatch) -> List[QualityIssue]:
        """检查数据连续性"""
        issues = []
        
        # 按股票代码分组
        symbol_groups = {}
        for data_point in data_batch.data_points:
            if data_point.symbol not in symbol_groups:
                symbol_groups[data_point.symbol] = []
            symbol_groups[data_point.symbol].append(data_point)
        
        # 检查每个股票的时间连续性
        for symbol, points in symbol_groups.items():
            if len(points) > 1:
                # 按时间排序
                points.sort(key=lambda p: p.timestamp)
                
                # 检查时间间隔
                for i in range(1, len(points)):
                    time_diff = points[i].timestamp - points[i-1].timestamp
                    
                    # 如果时间间隔过大，可能存在数据缺失
                    if time_diff.days > 7:  # 超过一周
                        issues.append(QualityIssue(
                            issue_type=QualityIssueType.INCOMPLETE,
                            severity=QualitySeverity.LOW,
                            field="timestamp",
                            description=f"股票 {symbol} 在 {points[i-1].timestamp} 和 {points[i].timestamp} 之间可能存在数据缺失"
                        ))
        
        return issues
    
    def _check_consistency(self, data_batch: DataBatch) -> List[QualityIssue]:
        """检查数据一致性"""
        issues = []
        
        # 检查同一股票在同一时间的数据是否一致
        time_symbol_map = {}
        
        for data_point in data_batch.data_points:
            key = (data_point.symbol, data_point.timestamp.date())
            
            if key not in time_symbol_map:
                time_symbol_map[key] = []
            time_symbol_map[key].append(data_point)
        
        for (symbol, date), points in time_symbol_map.items():
            if len(points) > 1:
                # 检查相同字段的值是否一致
                for field in ['close', 'volume']:
                    values = [p.values.get(field) for p in points if field in p.values]
                    if len(set(values)) > 1:
                        issues.append(QualityIssue(
                            issue_type=QualityIssueType.INCONSISTENT,
                            severity=QualitySeverity.MEDIUM,
                            field=field,
                            description=f"股票 {symbol} 在 {date} 的 {field} 字段值不一致: {values}"
                        ))
        
        return issues
    
    def _check_freshness(self, data_batch: DataBatch) -> List[QualityIssue]:
        """检查数据新鲜度"""
        issues = []
        
        now = datetime.now()
        
        # 检查批次中最新数据的时间
        if data_batch.data_points:
            latest_time = max(p.timestamp for p in data_batch.data_points)
            
            # 如果最新数据超过1天，认为数据不够新鲜
            if (now - latest_time).days > 1:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.STALE_DATA,
                    severity=QualitySeverity.LOW,
                    field="batch_freshness",
                    description=f"批次数据不够新鲜，最新数据时间为 {latest_time}",
                    value=latest_time
                ))
        
        return issues
    
    def _calculate_quality_score(self, issues: List[QualityIssue], total_points: int) -> float:
        """计算质量分数"""
        if total_points == 0:
            return 0.0
        
        # 根据问题严重程度计算扣分
        penalty = 0
        for issue in issues:
            if issue.severity == QualitySeverity.CRITICAL:
                penalty += 0.5
            elif issue.severity == QualitySeverity.HIGH:
                penalty += 0.3
            elif issue.severity == QualitySeverity.MEDIUM:
                penalty += 0.1
            else:  # LOW
                penalty += 0.05
        
        # 计算质量分数
        score = max(0, 1 - penalty / total_points)
        return min(1.0, score)
    
    def _generate_statistics(self, data_batch: DataBatch, issues: List[QualityIssue]) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {
            'total_issues': len(issues),
            'issue_types': {},
            'severity_distribution': {},
            'field_issues': {},
            'data_coverage': {},
        }
        
        # 问题类型统计
        for issue in issues:
            issue_type = issue.issue_type.value
            stats['issue_types'][issue_type] = stats['issue_types'].get(issue_type, 0) + 1
        
        # 严重程度统计
        for issue in issues:
            severity = issue.severity.value
            stats['severity_distribution'][severity] = stats['severity_distribution'].get(severity, 0) + 1
        
        # 字段问题统计
        for issue in issues:
            field = issue.field
            stats['field_issues'][field] = stats['field_issues'].get(field, 0) + 1
        
        # 数据覆盖率统计
        if data_batch.data_points:
            all_fields = set()
            for point in data_batch.data_points:
                all_fields.update(point.values.keys())
            
            for field in all_fields:
                non_null_count = sum(1 for point in data_batch.data_points 
                                    if field in point.values and point.values[field] is not None)
                stats['data_coverage'][field] = non_null_count / len(data_batch.data_points)
        
        return stats
    
    def _generate_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 根据问题类型生成建议
        issue_types = set(issue.issue_type for issue in issues)
        
        if QualityIssueType.MISSING_VALUE in issue_types:
            recommendations.append("建议检查数据源配置，确保所有必需字段都被正确提取")
        
        if QualityIssueType.OUTLIER in issue_types:
            recommendations.append("建议实施异常值检测和处理机制")
        
        if QualityIssueType.DUPLICATE in issue_types:
            recommendations.append("建议在数据入库前进行去重处理")
        
        if QualityIssueType.INCONSISTENT in issue_types:
            recommendations.append("建议检查数据源的一致性，可能需要数据标准化")
        
        if QualityIssueType.STALE_DATA in issue_types:
            recommendations.append("建议增加数据更新频率或检查数据源的实时性")
        
        # 根据严重问题数量生成建议
        critical_count = len([i for i in issues if i.severity == QualitySeverity.CRITICAL])
        if critical_count > 0:
            recommendations.append(f"发现 {critical_count} 个严重问题，建议立即处理")
        
        return recommendations

# 便捷函数
def create_quality_rule(
    rule_id: str,
    rule_name: str,
    **kwargs
) -> QualityRule:
    """创建质量检查规则"""
    return QualityRule(
        rule_id=rule_id,
        rule_name=rule_name,
        **kwargs
    )

def create_quality_checker(custom_rules: Optional[List[QualityRule]] = None) -> DataQualityChecker:
    """创建数据质量检查器"""
    checker = DataQualityChecker()
    
    if custom_rules:
        for rule in custom_rules:
            checker.add_rule(rule)
    
    return checker