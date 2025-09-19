# 实时数据质量监控和异常处理模块
# 监控数据完整性、准确性、及时性和一致性

import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import statistics

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats
except ImportError:
    stats = None

logger = logging.getLogger(__name__)

class DataQualityIssueType(Enum):
    """数据质量问题类型"""
    MISSING_DATA = "missing_data"           # 数据缺失
    DUPLICATE_DATA = "duplicate_data"       # 数据重复
    OUTLIER = "outlier"                     # 异常值
    INCONSISTENT = "inconsistent"           # 数据不一致
    STALE_DATA = "stale_data"               # 数据过期
    FORMAT_ERROR = "format_error"           # 格式错误
    RANGE_ERROR = "range_error"             # 范围错误
    CORRELATION_BREAK = "correlation_break" # 相关性异常
    VOLUME_ANOMALY = "volume_anomaly"       # 数据量异常
    LATENCY_HIGH = "latency_high"           # 延迟过高

class AlertSeverity(Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataQualityRule:
    """数据质量规则"""
    rule_id: str
    name: str
    description: str
    data_source: str
    field_name: Optional[str] = None
    rule_type: DataQualityIssueType = DataQualityIssueType.MISSING_DATA
    
    # 规则参数
    threshold: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_format: Optional[str] = None
    max_age_seconds: Optional[int] = None
    correlation_threshold: Optional[float] = None
    reference_field: Optional[str] = None
    
    # 检查配置
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.MEDIUM
    check_interval: int = 60  # 秒
    
    # 统计配置
    window_size: int = 100
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'data_source': self.data_source,
            'field_name': self.field_name,
            'rule_type': self.rule_type.value,
            'threshold': self.threshold,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'expected_format': self.expected_format,
            'max_age_seconds': self.max_age_seconds,
            'correlation_threshold': self.correlation_threshold,
            'reference_field': self.reference_field,
            'enabled': self.enabled,
            'severity': self.severity.value,
            'check_interval': self.check_interval,
            'window_size': self.window_size,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold
        }

@dataclass
class DataQualityIssue:
    """数据质量问题"""
    issue_id: str
    rule_id: str
    issue_type: DataQualityIssueType
    severity: AlertSeverity
    data_source: str
    field_name: Optional[str]
    description: str
    detected_at: datetime
    
    # 问题详情
    affected_records: int = 0
    sample_values: List[Any] = field(default_factory=list)
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    confidence: float = 1.0
    
    # 状态
    status: str = "open"  # open, investigating, resolved, ignored
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_id': self.issue_id,
            'rule_id': self.rule_id,
            'issue_type': self.issue_type.value,
            'severity': self.severity.value,
            'data_source': self.data_source,
            'field_name': self.field_name,
            'description': self.description,
            'detected_at': self.detected_at.isoformat(),
            'affected_records': self.affected_records,
            'sample_values': self.sample_values,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'confidence': self.confidence,
            'status': self.status,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes
        }

@dataclass
class DataSourceMetrics:
    """数据源指标"""
    source_name: str
    last_update: datetime
    record_count: int = 0
    field_count: int = 0
    
    # 质量指标
    completeness: float = 0.0      # 完整性
    accuracy: float = 0.0          # 准确性
    consistency: float = 0.0       # 一致性
    timeliness: float = 0.0        # 及时性
    validity: float = 0.0          # 有效性
    
    # 统计信息
    missing_values: int = 0
    duplicate_records: int = 0
    outlier_count: int = 0
    format_errors: int = 0
    
    # 性能指标
    avg_latency: float = 0.0
    max_latency: float = 0.0
    throughput: float = 0.0
    
    def calculate_overall_quality(self) -> float:
        """计算总体质量分数"""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15
        }
        
        return (
            self.completeness * weights['completeness'] +
            self.accuracy * weights['accuracy'] +
            self.consistency * weights['consistency'] +
            self.timeliness * weights['timeliness'] +
            self.validity * weights['validity']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_name': self.source_name,
            'last_update': self.last_update.isoformat(),
            'record_count': self.record_count,
            'field_count': self.field_count,
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'timeliness': self.timeliness,
            'validity': self.validity,
            'overall_quality': self.calculate_overall_quality(),
            'missing_values': self.missing_values,
            'duplicate_records': self.duplicate_records,
            'outlier_count': self.outlier_count,
            'format_errors': self.format_errors,
            'avg_latency': self.avg_latency,
            'max_latency': self.max_latency,
            'throughput': self.throughput
        }

class DataQualityMonitor:
    """数据质量监控器
    
    实时监控数据质量，检测异常并生成告警
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 规则管理
        self.rules: Dict[str, DataQualityRule] = {}
        self.rule_schedules: Dict[str, datetime] = {}
        
        # 问题跟踪
        self.issues: Dict[str, DataQualityIssue] = {}
        self.issue_history: deque = deque(maxlen=10000)
        
        # 数据源指标
        self.source_metrics: Dict[str, DataSourceMetrics] = {}
        
        # 数据缓存
        self.data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.field_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 监控状态
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.issue_callbacks: List[Callable[[DataQualityIssue], None]] = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 初始化默认规则
        self._init_default_rules()
        
        logger.info("数据质量监控器初始化完成")
    
    def _init_default_rules(self):
        """初始化默认质量规则"""
        default_rules = [
            # 数据缺失检查
            DataQualityRule(
                rule_id="missing_data_check",
                name="数据缺失检查",
                description="检查关键字段的数据缺失情况",
                data_source="*",
                rule_type=DataQualityIssueType.MISSING_DATA,
                threshold=0.05,  # 缺失率超过5%告警
                severity=AlertSeverity.HIGH
            ),
            
            # 数据重复检查
            DataQualityRule(
                rule_id="duplicate_data_check",
                name="数据重复检查",
                description="检查重复记录",
                data_source="*",
                rule_type=DataQualityIssueType.DUPLICATE_DATA,
                threshold=0.01,  # 重复率超过1%告警
                severity=AlertSeverity.MEDIUM
            ),
            
            # 数据过期检查
            DataQualityRule(
                rule_id="stale_data_check",
                name="数据过期检查",
                description="检查数据是否过期",
                data_source="*",
                rule_type=DataQualityIssueType.STALE_DATA,
                max_age_seconds=300,  # 5分钟
                severity=AlertSeverity.HIGH
            ),
            
            # 异常值检查
            DataQualityRule(
                rule_id="outlier_check",
                name="异常值检查",
                description="检查数值字段的异常值",
                data_source="*",
                rule_type=DataQualityIssueType.OUTLIER,
                outlier_method="iqr",
                outlier_threshold=3.0,
                severity=AlertSeverity.MEDIUM
            ),
            
            # 数据量异常检查
            DataQualityRule(
                rule_id="volume_anomaly_check",
                name="数据量异常检查",
                description="检查数据量是否异常",
                data_source="*",
                rule_type=DataQualityIssueType.VOLUME_ANOMALY,
                threshold=0.5,  # 变化超过50%告警
                severity=AlertSeverity.HIGH
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: DataQualityRule) -> bool:
        """添加质量规则"""
        try:
            with self.lock:
                self.rules[rule.rule_id] = rule
                self.rule_schedules[rule.rule_id] = datetime.now()
                
            logger.info(f"添加数据质量规则: {rule.name} ({rule.rule_id})")
            return True
            
        except Exception as e:
            logger.error(f"添加质量规则失败: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除质量规则"""
        try:
            with self.lock:
                if rule_id in self.rules:
                    del self.rules[rule_id]
                    del self.rule_schedules[rule_id]
                    logger.info(f"移除数据质量规则: {rule_id}")
                    return True
                else:
                    logger.warning(f"质量规则不存在: {rule_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"移除质量规则失败: {e}")
            return False
    
    def update_rule(self, rule: DataQualityRule) -> bool:
        """更新质量规则"""
        try:
            with self.lock:
                if rule.rule_id in self.rules:
                    self.rules[rule.rule_id] = rule
                    logger.info(f"更新数据质量规则: {rule.name} ({rule.rule_id})")
                    return True
                else:
                    logger.warning(f"质量规则不存在: {rule.rule_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"更新质量规则失败: {e}")
            return False
    
    async def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            logger.warning("数据质量监控已在运行")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("数据质量监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("数据质量监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        try:
            while self.is_running:
                await self._run_quality_checks()
                await asyncio.sleep(10)  # 每10秒检查一次
                
        except asyncio.CancelledError:
            logger.info("监控循环被取消")
        except Exception as e:
            logger.error(f"监控循环异常: {e}")
    
    async def _run_quality_checks(self):
        """执行质量检查"""
        try:
            current_time = datetime.now()
            
            with self.lock:
                for rule_id, rule in self.rules.items():
                    if not rule.enabled:
                        continue
                    
                    # 检查是否到了执行时间
                    last_check = self.rule_schedules.get(rule_id, datetime.min)
                    if (current_time - last_check).total_seconds() < rule.check_interval:
                        continue
                    
                    # 执行规则检查
                    await self._execute_rule_check(rule)
                    
                    # 更新检查时间
                    self.rule_schedules[rule_id] = current_time
                    
        except Exception as e:
            logger.error(f"质量检查执行失败: {e}")
    
    async def _execute_rule_check(self, rule: DataQualityRule):
        """执行单个规则检查"""
        try:
            # 获取相关数据源的数据
            data_sources = self._get_matching_data_sources(rule.data_source)
            
            for source_name in data_sources:
                data = self._get_source_data(source_name)
                if not data:
                    continue
                
                # 根据规则类型执行检查
                issues = await self._check_rule_violations(rule, source_name, data)
                
                # 处理发现的问题
                for issue in issues:
                    await self._handle_quality_issue(issue)
                    
        except Exception as e:
            logger.error(f"规则检查执行失败 {rule.rule_id}: {e}")
    
    def _get_matching_data_sources(self, pattern: str) -> List[str]:
        """获取匹配的数据源"""
        if pattern == "*":
            return list(self.source_metrics.keys())
        else:
            return [source for source in self.source_metrics.keys() if pattern in source]
    
    def _get_source_data(self, source_name: str) -> Optional[List[Dict[str, Any]]]:
        """获取数据源数据"""
        if source_name in self.data_cache:
            return list(self.data_cache[source_name])
        return None
    
    async def _check_rule_violations(self, rule: DataQualityRule, source_name: str, 
                                   data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查规则违反情况"""
        issues = []
        
        try:
            if rule.rule_type == DataQualityIssueType.MISSING_DATA:
                issues.extend(self._check_missing_data(rule, source_name, data))
            elif rule.rule_type == DataQualityIssueType.DUPLICATE_DATA:
                issues.extend(self._check_duplicate_data(rule, source_name, data))
            elif rule.rule_type == DataQualityIssueType.OUTLIER:
                issues.extend(self._check_outliers(rule, source_name, data))
            elif rule.rule_type == DataQualityIssueType.STALE_DATA:
                issues.extend(self._check_stale_data(rule, source_name, data))
            elif rule.rule_type == DataQualityIssueType.VOLUME_ANOMALY:
                issues.extend(self._check_volume_anomaly(rule, source_name, data))
            elif rule.rule_type == DataQualityIssueType.RANGE_ERROR:
                issues.extend(self._check_range_errors(rule, source_name, data))
                
        except Exception as e:
            logger.error(f"规则违反检查失败 {rule.rule_id}: {e}")
        
        return issues
    
    def _check_missing_data(self, rule: DataQualityRule, source_name: str, 
                          data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查数据缺失"""
        issues = []
        
        if not data:
            return issues
        
        # 检查指定字段或所有字段
        fields_to_check = [rule.field_name] if rule.field_name else data[0].keys()
        
        for field in fields_to_check:
            missing_count = sum(1 for record in data if record.get(field) is None or record.get(field) == '')
            missing_rate = missing_count / len(data)
            
            if missing_rate > (rule.threshold or 0.05):
                issue = DataQualityIssue(
                    issue_id=f"{rule.rule_id}_{source_name}_{field}_{int(datetime.now().timestamp())}",
                    rule_id=rule.rule_id,
                    issue_type=DataQualityIssueType.MISSING_DATA,
                    severity=rule.severity,
                    data_source=source_name,
                    field_name=field,
                    description=f"字段 {field} 缺失率过高: {missing_rate:.2%}",
                    detected_at=datetime.now(),
                    affected_records=missing_count,
                    expected_value=f"缺失率 < {rule.threshold:.2%}",
                    actual_value=f"缺失率 = {missing_rate:.2%}",
                    confidence=0.95
                )
                issues.append(issue)
        
        return issues
    
    def _check_duplicate_data(self, rule: DataQualityRule, source_name: str, 
                            data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查重复数据"""
        issues = []
        
        if len(data) < 2:
            return issues
        
        # 计算重复记录
        seen = set()
        duplicates = []
        
        for record in data:
            # 创建记录的哈希值（排除时间戳字段）
            record_key = tuple(sorted([
                (k, v) for k, v in record.items() 
                if k not in ['timestamp', 'created_at', 'updated_at']
            ]))
            
            if record_key in seen:
                duplicates.append(record)
            else:
                seen.add(record_key)
        
        duplicate_rate = len(duplicates) / len(data)
        
        if duplicate_rate > (rule.threshold or 0.01):
            issue = DataQualityIssue(
                issue_id=f"{rule.rule_id}_{source_name}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                issue_type=DataQualityIssueType.DUPLICATE_DATA,
                severity=rule.severity,
                data_source=source_name,
                field_name=None,
                description=f"重复记录率过高: {duplicate_rate:.2%}",
                detected_at=datetime.now(),
                affected_records=len(duplicates),
                expected_value=f"重复率 < {rule.threshold:.2%}",
                actual_value=f"重复率 = {duplicate_rate:.2%}",
                sample_values=duplicates[:5],  # 提供前5个重复记录作为样本
                confidence=0.9
            )
            issues.append(issue)
        
        return issues
    
    def _check_outliers(self, rule: DataQualityRule, source_name: str, 
                       data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查异常值"""
        issues = []
        
        if not data or len(data) < 10:
            return issues
        
        # 检查数值字段
        numeric_fields = []
        for field in data[0].keys():
            if rule.field_name and field != rule.field_name:
                continue
            
            # 检查字段是否为数值类型
            values = [record.get(field) for record in data if record.get(field) is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                numeric_fields.append(field)
        
        for field in numeric_fields:
            values = [record.get(field) for record in data if record.get(field) is not None]
            
            if len(values) < 10:
                continue
            
            outliers = self._detect_outliers(values, rule.outlier_method, rule.outlier_threshold)
            
            if outliers:
                outlier_rate = len(outliers) / len(values)
                
                if outlier_rate > 0.05:  # 异常值超过5%
                    issue = DataQualityIssue(
                        issue_id=f"{rule.rule_id}_{source_name}_{field}_{int(datetime.now().timestamp())}",
                        rule_id=rule.rule_id,
                        issue_type=DataQualityIssueType.OUTLIER,
                        severity=rule.severity,
                        data_source=source_name,
                        field_name=field,
                        description=f"字段 {field} 异常值过多: {len(outliers)} 个 ({outlier_rate:.2%})",
                        detected_at=datetime.now(),
                        affected_records=len(outliers),
                        sample_values=outliers[:10],  # 提供前10个异常值
                        confidence=0.8
                    )
                    issues.append(issue)
        
        return issues
    
    def _detect_outliers(self, values: List[float], method: str, threshold: float) -> List[float]:
        """检测异常值"""
        outliers = []
        
        try:
            if method == "iqr":
                # 四分位数方法
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                
            elif method == "zscore":
                # Z分数方法
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val > 0:
                    z_scores = [(v - mean_val) / std_val for v in values]
                    outliers = [values[i] for i, z in enumerate(z_scores) if abs(z) > threshold]
                    
            elif method == "modified_zscore":
                # 修正Z分数方法
                median_val = np.median(values)
                mad = np.median([abs(v - median_val) for v in values])
                
                if mad > 0:
                    modified_z_scores = [0.6745 * (v - median_val) / mad for v in values]
                    outliers = [values[i] for i, z in enumerate(modified_z_scores) if abs(z) > threshold]
                    
        except Exception as e:
            logger.error(f"异常值检测失败: {e}")
        
        return outliers
    
    def _check_stale_data(self, rule: DataQualityRule, source_name: str, 
                         data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查过期数据"""
        issues = []
        
        if not data or not rule.max_age_seconds:
            return issues
        
        current_time = datetime.now()
        max_age = timedelta(seconds=rule.max_age_seconds)
        
        # 检查最新数据的时间戳
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'time']
        
        for record in data[-10:]:  # 检查最近10条记录
            record_time = None
            
            for field in timestamp_fields:
                if field in record and record[field]:
                    try:
                        if isinstance(record[field], datetime):
                            record_time = record[field]
                        elif isinstance(record[field], str):
                            record_time = datetime.fromisoformat(record[field].replace('Z', '+00:00'))
                        break
                    except:
                        continue
            
            if record_time and (current_time - record_time) > max_age:
                issue = DataQualityIssue(
                    issue_id=f"{rule.rule_id}_{source_name}_{int(datetime.now().timestamp())}",
                    rule_id=rule.rule_id,
                    issue_type=DataQualityIssueType.STALE_DATA,
                    severity=rule.severity,
                    data_source=source_name,
                    field_name=None,
                    description=f"数据过期: 最新数据时间为 {record_time}, 超过最大允许时间 {max_age}",
                    detected_at=datetime.now(),
                    affected_records=1,
                    expected_value=f"数据时间 < {max_age.total_seconds()} 秒前",
                    actual_value=f"数据时间 = {(current_time - record_time).total_seconds()} 秒前",
                    confidence=0.95
                )
                issues.append(issue)
                break  # 只报告一次过期问题
        
        return issues
    
    def _check_volume_anomaly(self, rule: DataQualityRule, source_name: str, 
                            data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查数据量异常"""
        issues = []
        
        # 获取历史数据量统计
        metrics = self.source_metrics.get(source_name)
        if not metrics:
            return issues
        
        current_count = len(data)
        expected_count = metrics.record_count
        
        if expected_count > 0:
            change_rate = abs(current_count - expected_count) / expected_count
            
            if change_rate > (rule.threshold or 0.5):
                issue = DataQualityIssue(
                    issue_id=f"{rule.rule_id}_{source_name}_{int(datetime.now().timestamp())}",
                    rule_id=rule.rule_id,
                    issue_type=DataQualityIssueType.VOLUME_ANOMALY,
                    severity=rule.severity,
                    data_source=source_name,
                    field_name=None,
                    description=f"数据量异常变化: 当前 {current_count}, 预期 {expected_count}, 变化率 {change_rate:.2%}",
                    detected_at=datetime.now(),
                    affected_records=abs(current_count - expected_count),
                    expected_value=f"数据量变化 < {rule.threshold:.2%}",
                    actual_value=f"数据量变化 = {change_rate:.2%}",
                    confidence=0.85
                )
                issues.append(issue)
        
        return issues
    
    def _check_range_errors(self, rule: DataQualityRule, source_name: str, 
                          data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """检查范围错误"""
        issues = []
        
        if not rule.field_name or rule.min_value is None or rule.max_value is None:
            return issues
        
        out_of_range_records = []
        
        for record in data:
            value = record.get(rule.field_name)
            if value is not None and isinstance(value, (int, float)):
                if value < rule.min_value or value > rule.max_value:
                    out_of_range_records.append(record)
        
        if out_of_range_records:
            error_rate = len(out_of_range_records) / len(data)
            
            issue = DataQualityIssue(
                issue_id=f"{rule.rule_id}_{source_name}_{rule.field_name}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                issue_type=DataQualityIssueType.RANGE_ERROR,
                severity=rule.severity,
                data_source=source_name,
                field_name=rule.field_name,
                description=f"字段 {rule.field_name} 超出范围: {len(out_of_range_records)} 条记录 ({error_rate:.2%})",
                detected_at=datetime.now(),
                affected_records=len(out_of_range_records),
                expected_value=f"值在 [{rule.min_value}, {rule.max_value}] 范围内",
                actual_value=f"{len(out_of_range_records)} 条记录超出范围",
                sample_values=[r.get(rule.field_name) for r in out_of_range_records[:5]],
                confidence=0.9
            )
            issues.append(issue)
        
        return issues
    
    async def _handle_quality_issue(self, issue: DataQualityIssue):
        """处理质量问题"""
        try:
            # 检查是否已存在相同问题
            existing_issue = self._find_similar_issue(issue)
            if existing_issue:
                # 更新现有问题
                existing_issue.affected_records += issue.affected_records
                existing_issue.detected_at = issue.detected_at
                return
            
            # 添加新问题
            with self.lock:
                self.issues[issue.issue_id] = issue
                self.issue_history.append(issue)
            
            # 触发回调
            for callback in self.issue_callbacks:
                try:
                    callback(issue)
                except Exception as e:
                    logger.error(f"问题回调执行失败: {e}")
            
            logger.warning(f"检测到数据质量问题: {issue.description}")
            
        except Exception as e:
            logger.error(f"处理质量问题失败: {e}")
    
    def _find_similar_issue(self, issue: DataQualityIssue) -> Optional[DataQualityIssue]:
        """查找相似问题"""
        for existing_issue in self.issues.values():
            if (existing_issue.rule_id == issue.rule_id and
                existing_issue.data_source == issue.data_source and
                existing_issue.field_name == issue.field_name and
                existing_issue.status == "open" and
                (datetime.now() - existing_issue.detected_at).total_seconds() < 3600):  # 1小时内
                return existing_issue
        return None
    
    def ingest_data(self, source_name: str, records: List[Dict[str, Any]], 
                   timestamp: Optional[datetime] = None):
        """接收数据并更新缓存"""
        try:
            if not records:
                return
            
            timestamp = timestamp or datetime.now()
            
            with self.lock:
                # 更新数据缓存
                for record in records:
                    # 添加时间戳
                    record_with_timestamp = {**record, '_ingested_at': timestamp}
                    self.data_cache[source_name].append(record_with_timestamp)
                
                # 更新数据源指标
                self._update_source_metrics(source_name, records, timestamp)
            
        except Exception as e:
            logger.error(f"数据接收失败 {source_name}: {e}")
    
    def _update_source_metrics(self, source_name: str, records: List[Dict[str, Any]], 
                             timestamp: datetime):
        """更新数据源指标"""
        try:
            if source_name not in self.source_metrics:
                self.source_metrics[source_name] = DataSourceMetrics(
                    source_name=source_name,
                    last_update=timestamp
                )
            
            metrics = self.source_metrics[source_name]
            metrics.last_update = timestamp
            metrics.record_count = len(records)
            
            if records:
                metrics.field_count = len(records[0].keys())
                
                # 计算完整性
                total_fields = len(records) * metrics.field_count
                missing_fields = sum(
                    sum(1 for v in record.values() if v is None or v == '')
                    for record in records
                )
                metrics.completeness = (total_fields - missing_fields) / total_fields if total_fields > 0 else 0
                metrics.missing_values = missing_fields
                
                # 计算及时性
                if '_ingested_at' in records[0]:
                    latencies = []
                    for record in records:
                        if 'timestamp' in record and record['timestamp']:
                            try:
                                record_time = datetime.fromisoformat(str(record['timestamp']).replace('Z', '+00:00'))
                                latency = (timestamp - record_time).total_seconds()
                                latencies.append(latency)
                            except:
                                pass
                    
                    if latencies:
                        metrics.avg_latency = statistics.mean(latencies)
                        metrics.max_latency = max(latencies)
                        # 及时性评分（延迟越小越好）
                        avg_latency_minutes = metrics.avg_latency / 60
                        metrics.timeliness = max(0, 1 - avg_latency_minutes / 60)  # 1小时内为满分
                
                # 简化的准确性和一致性评分
                metrics.accuracy = 0.9  # 需要更复杂的逻辑
                metrics.consistency = 0.85  # 需要更复杂的逻辑
                metrics.validity = 0.9  # 需要更复杂的逻辑
            
        except Exception as e:
            logger.error(f"更新数据源指标失败 {source_name}: {e}")
    
    def add_issue_callback(self, callback: Callable[[DataQualityIssue], None]):
        """添加问题回调函数"""
        self.issue_callbacks.append(callback)
    
    def resolve_issue(self, issue_id: str, resolution_notes: str = "") -> bool:
        """解决问题"""
        try:
            with self.lock:
                if issue_id in self.issues:
                    issue = self.issues[issue_id]
                    issue.status = "resolved"
                    issue.resolved_at = datetime.now()
                    issue.resolution_notes = resolution_notes
                    
                    logger.info(f"问题已解决: {issue_id}")
                    return True
                else:
                    logger.warning(f"问题不存在: {issue_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"解决问题失败: {e}")
            return False
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取质量报告"""
        try:
            with self.lock:
                # 统计问题
                open_issues = [issue for issue in self.issues.values() if issue.status == "open"]
                critical_issues = [issue for issue in open_issues if issue.severity == AlertSeverity.CRITICAL]
                high_issues = [issue for issue in open_issues if issue.severity == AlertSeverity.HIGH]
                
                # 按数据源分组
                issues_by_source = defaultdict(list)
                for issue in open_issues:
                    issues_by_source[issue.data_source].append(issue)
                
                # 按问题类型分组
                issues_by_type = defaultdict(int)
                for issue in open_issues:
                    issues_by_type[issue.issue_type.value] += 1
                
                return {
                    'summary': {
                        'total_sources': len(self.source_metrics),
                        'total_issues': len(self.issues),
                        'open_issues': len(open_issues),
                        'critical_issues': len(critical_issues),
                        'high_issues': len(high_issues),
                        'resolved_issues': len([i for i in self.issues.values() if i.status == "resolved"])
                    },
                    'source_metrics': {name: metrics.to_dict() for name, metrics in self.source_metrics.items()},
                    'issues_by_source': {source: len(issues) for source, issues in issues_by_source.items()},
                    'issues_by_type': dict(issues_by_type),
                    'recent_issues': [issue.to_dict() for issue in list(self.issue_history)[-10:]],
                    'quality_rules': {rule_id: rule.to_dict() for rule_id, rule in self.rules.items()},
                    'overall_quality': self._calculate_overall_quality()
                }
                
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            return {}
    
    def _calculate_overall_quality(self) -> float:
        """计算整体质量分数"""
        if not self.source_metrics:
            return 0.0
        
        total_quality = sum(metrics.calculate_overall_quality() for metrics in self.source_metrics.values())
        return total_quality / len(self.source_metrics)

# 便捷函数
def create_data_quality_monitor(config_file: str = None) -> DataQualityMonitor:
    """创建数据质量监控器
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        数据质量监控器实例
    """
    default_config = {
        'monitoring_interval': 60,
        'max_cache_size': 1000,
        'max_issue_history': 10000,
        'enable_auto_resolution': False,
        'alert_thresholds': {
            'missing_data': 0.05,
            'duplicate_data': 0.01,
            'outlier_rate': 0.05,
            'stale_data_minutes': 5
        }
    }
    
    if config_file:
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
    
    return DataQualityMonitor(default_config)