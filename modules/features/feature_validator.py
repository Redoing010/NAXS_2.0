# 特征验证器 - 验证特征数据的质量和一致性

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """验证级别"""
    INFO = "info"          # 信息
    WARNING = "warning"    # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重错误

class ValidationCategory(Enum):
    """验证类别"""
    DATA_TYPE = "data_type"              # 数据类型
    DATA_RANGE = "data_range"            # 数据范围
    DATA_QUALITY = "data_quality"        # 数据质量
    DATA_CONSISTENCY = "data_consistency" # 数据一致性
    BUSINESS_LOGIC = "business_logic"    # 业务逻辑
    STATISTICAL = "statistical"          # 统计特性

@dataclass
class ValidationResult:
    """验证结果"""
    rule_name: str
    category: ValidationCategory
    level: ValidationLevel
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule_name': self.rule_name,
            'category': self.category.value,
            'level': self.level.value,
            'passed': self.passed,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ValidationReport:
    """验证报告"""
    feature_name: str
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_rules == 0:
            return 0.0
        return self.passed_rules / self.total_rules
    
    @property
    def is_valid(self) -> bool:
        """是否通过验证"""
        # 没有严重错误和错误级别的失败
        critical_errors = [r for r in self.results if not r.passed and r.level == ValidationLevel.CRITICAL]
        errors = [r for r in self.results if not r.passed and r.level == ValidationLevel.ERROR]
        return len(critical_errors) == 0 and len(errors) == 0
    
    def add_result(self, result: ValidationResult):
        """添加验证结果"""
        self.results.append(result)
        self.total_rules += 1
        if result.passed:
            self.passed_rules += 1
        else:
            self.failed_rules += 1
    
    def get_results_by_level(self, level: ValidationLevel) -> List[ValidationResult]:
        """根据级别获取结果"""
        return [r for r in self.results if r.level == level]
    
    def get_results_by_category(self, category: ValidationCategory) -> List[ValidationResult]:
        """根据类别获取结果"""
        return [r for r in self.results if r.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'feature_name': self.feature_name,
            'total_rules': self.total_rules,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'success_rate': self.success_rate,
            'is_valid': self.is_valid,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'timestamp': self.timestamp.isoformat()
        }

class ValidationRule(ABC):
    """验证规则基类"""
    
    def __init__(self, name: str, category: ValidationCategory, 
                 level: ValidationLevel = ValidationLevel.ERROR,
                 description: str = ""):
        self.name = name
        self.category = category
        self.level = level
        self.description = description
    
    @abstractmethod
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        """执行验证
        
        Args:
            data: 特征数据
            metadata: 特征元数据
            
        Returns:
            验证结果
        """
        pass

class DataTypeRule(ValidationRule):
    """数据类型验证规则"""
    
    def __init__(self, expected_type: str, level: ValidationLevel = ValidationLevel.ERROR):
        super().__init__(
            name=f"data_type_{expected_type}",
            category=ValidationCategory.DATA_TYPE,
            level=level,
            description=f"验证数据类型是否为 {expected_type}"
        )
        self.expected_type = expected_type
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            actual_type = str(data.dtype)
            passed = actual_type == self.expected_type or self._is_compatible_type(actual_type, self.expected_type)
            
            message = f"数据类型验证: 期望 {self.expected_type}, 实际 {actual_type}"
            if not passed:
                message += " - 不匹配"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message,
                details={
                    'expected_type': self.expected_type,
                    'actual_type': actual_type
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"数据类型验证失败: {str(e)}",
                details={'error': str(e)}
            )
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """检查类型兼容性"""
        # 数值类型兼容性
        numeric_types = ['int64', 'int32', 'float64', 'float32']
        if expected in numeric_types and actual in numeric_types:
            return True
        
        # 字符串类型兼容性
        string_types = ['object', 'string']
        if expected in string_types and actual in string_types:
            return True
        
        return False

class RangeRule(ValidationRule):
    """数据范围验证规则"""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None,
                 level: ValidationLevel = ValidationLevel.ERROR):
        super().__init__(
            name=f"range_{min_value}_{max_value}",
            category=ValidationCategory.DATA_RANGE,
            level=level,
            description=f"验证数据范围在 [{min_value}, {max_value}] 之间"
        )
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            # 只对数值类型进行范围验证
            if not pd.api.types.is_numeric_dtype(data):
                return ValidationResult(
                    rule_name=self.name,
                    category=self.category,
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="非数值类型，跳过范围验证"
                )
            
            violations = []
            
            if self.min_value is not None:
                below_min = (data < self.min_value).sum()
                if below_min > 0:
                    violations.append(f"{below_min} 个值小于最小值 {self.min_value}")
            
            if self.max_value is not None:
                above_max = (data > self.max_value).sum()
                if above_max > 0:
                    violations.append(f"{above_max} 个值大于最大值 {self.max_value}")
            
            passed = len(violations) == 0
            message = "数据范围验证通过" if passed else f"数据范围验证失败: {'; '.join(violations)}"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message,
                details={
                    'min_value': self.min_value,
                    'max_value': self.max_value,
                    'actual_min': float(data.min()) if len(data) > 0 else None,
                    'actual_max': float(data.max()) if len(data) > 0 else None,
                    'violations': violations
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"数据范围验证失败: {str(e)}",
                details={'error': str(e)}
            )

class NullRule(ValidationRule):
    """空值验证规则"""
    
    def __init__(self, max_null_ratio: float = 0.1, level: ValidationLevel = ValidationLevel.WARNING):
        super().__init__(
            name=f"null_ratio_{max_null_ratio}",
            category=ValidationCategory.DATA_QUALITY,
            level=level,
            description=f"验证空值比例不超过 {max_null_ratio}"
        )
        self.max_null_ratio = max_null_ratio
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            total_count = len(data)
            null_count = data.isnull().sum()
            null_ratio = null_count / total_count if total_count > 0 else 0
            
            passed = null_ratio <= self.max_null_ratio
            message = f"空值比例验证: {null_ratio:.2%} (阈值: {self.max_null_ratio:.2%})"
            if not passed:
                message += " - 超过阈值"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message,
                details={
                    'null_count': int(null_count),
                    'total_count': total_count,
                    'null_ratio': null_ratio,
                    'max_null_ratio': self.max_null_ratio
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"空值验证失败: {str(e)}",
                details={'error': str(e)}
            )

class UniqueRule(ValidationRule):
    """唯一性验证规则"""
    
    def __init__(self, min_unique_ratio: float = 0.8, level: ValidationLevel = ValidationLevel.WARNING):
        super().__init__(
            name=f"unique_ratio_{min_unique_ratio}",
            category=ValidationCategory.DATA_QUALITY,
            level=level,
            description=f"验证唯一值比例不低于 {min_unique_ratio}"
        )
        self.min_unique_ratio = min_unique_ratio
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            total_count = len(data)
            unique_count = data.nunique()
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            passed = unique_ratio >= self.min_unique_ratio
            message = f"唯一性验证: {unique_ratio:.2%} (阈值: {self.min_unique_ratio:.2%})"
            if not passed:
                message += " - 低于阈值"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message,
                details={
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'unique_ratio': unique_ratio,
                    'min_unique_ratio': self.min_unique_ratio
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"唯一性验证失败: {str(e)}",
                details={'error': str(e)}
            )

class OutlierRule(ValidationRule):
    """异常值验证规则"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5, 
                 max_outlier_ratio: float = 0.05, level: ValidationLevel = ValidationLevel.WARNING):
        super().__init__(
            name=f"outlier_{method}_{threshold}",
            category=ValidationCategory.DATA_QUALITY,
            level=level,
            description=f"使用 {method} 方法检测异常值，阈值 {threshold}"
        )
        self.method = method
        self.threshold = threshold
        self.max_outlier_ratio = max_outlier_ratio
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            # 只对数值类型进行异常值检测
            if not pd.api.types.is_numeric_dtype(data):
                return ValidationResult(
                    rule_name=self.name,
                    category=self.category,
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="非数值类型，跳过异常值检测"
                )
            
            # 移除空值
            clean_data = data.dropna()
            if len(clean_data) == 0:
                return ValidationResult(
                    rule_name=self.name,
                    category=self.category,
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="数据为空，跳过异常值检测"
                )
            
            outliers = self._detect_outliers(clean_data)
            outlier_count = len(outliers)
            outlier_ratio = outlier_count / len(clean_data)
            
            passed = outlier_ratio <= self.max_outlier_ratio
            message = f"异常值检测: {outlier_count} 个异常值 ({outlier_ratio:.2%}), 阈值: {self.max_outlier_ratio:.2%}"
            if not passed:
                message += " - 超过阈值"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message,
                details={
                    'method': self.method,
                    'threshold': self.threshold,
                    'outlier_count': outlier_count,
                    'total_count': len(clean_data),
                    'outlier_ratio': outlier_ratio,
                    'max_outlier_ratio': self.max_outlier_ratio,
                    'outlier_indices': outliers.tolist() if len(outliers) < 100 else outliers[:100].tolist()
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"异常值检测失败: {str(e)}",
                details={'error': str(e)}
            )
    
    def _detect_outliers(self, data: pd.Series) -> pd.Index:
        """检测异常值"""
        if self.method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            return data[(data < lower_bound) | (data > upper_bound)].index
        
        elif self.method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return data[z_scores > self.threshold].index
        
        elif self.method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return data[np.abs(modified_z_scores) > self.threshold].index
        
        else:
            raise ValueError(f"不支持的异常值检测方法: {self.method}")

class DistributionRule(ValidationRule):
    """分布验证规则"""
    
    def __init__(self, expected_distribution: str = 'normal', 
                 significance_level: float = 0.05, level: ValidationLevel = ValidationLevel.INFO):
        super().__init__(
            name=f"distribution_{expected_distribution}",
            category=ValidationCategory.STATISTICAL,
            level=level,
            description=f"验证数据是否符合 {expected_distribution} 分布"
        )
        self.expected_distribution = expected_distribution
        self.significance_level = significance_level
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            # 只对数值类型进行分布检验
            if not pd.api.types.is_numeric_dtype(data):
                return ValidationResult(
                    rule_name=self.name,
                    category=self.category,
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="非数值类型，跳过分布检验"
                )
            
            # 移除空值
            clean_data = data.dropna()
            if len(clean_data) < 8:  # 样本量太小
                return ValidationResult(
                    rule_name=self.name,
                    category=self.category,
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="样本量太小，跳过分布检验"
                )
            
            # 执行分布检验
            p_value = self._test_distribution(clean_data)
            passed = p_value > self.significance_level
            
            message = f"分布检验 ({self.expected_distribution}): p-value = {p_value:.4f}"
            if passed:
                message += f" > {self.significance_level} (符合分布)"
            else:
                message += f" <= {self.significance_level} (不符合分布)"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message,
                details={
                    'expected_distribution': self.expected_distribution,
                    'p_value': p_value,
                    'significance_level': self.significance_level,
                    'sample_size': len(clean_data)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"分布检验失败: {str(e)}",
                details={'error': str(e)}
            )
    
    def _test_distribution(self, data: pd.Series) -> float:
        """执行分布检验"""
        try:
            from scipy import stats
            
            if self.expected_distribution == 'normal':
                # Shapiro-Wilk 正态性检验
                if len(data) <= 5000:
                    statistic, p_value = stats.shapiro(data)
                else:
                    # 大样本使用 Kolmogorov-Smirnov 检验
                    statistic, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                return p_value
            
            elif self.expected_distribution == 'uniform':
                # Kolmogorov-Smirnov 均匀分布检验
                statistic, p_value = stats.kstest(data, 'uniform', 
                                                 args=(data.min(), data.max() - data.min()))
                return p_value
            
            else:
                raise ValueError(f"不支持的分布类型: {self.expected_distribution}")
                
        except ImportError:
            # 如果没有 scipy，返回默认值
            return 0.5

class CustomRule(ValidationRule):
    """自定义验证规则"""
    
    def __init__(self, name: str, validation_func: Callable[[pd.Series, Dict[str, Any]], bool],
                 category: ValidationCategory = ValidationCategory.BUSINESS_LOGIC,
                 level: ValidationLevel = ValidationLevel.ERROR,
                 description: str = ""):
        super().__init__(name, category, level, description)
        self.validation_func = validation_func
    
    def validate(self, data: pd.Series, metadata: Dict[str, Any] = None) -> ValidationResult:
        try:
            passed = self.validation_func(data, metadata or {})
            message = f"自定义验证 ({self.name}): {'通过' if passed else '失败'}"
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=self.level,
                passed=passed,
                message=message
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"自定义验证失败: {str(e)}",
                details={'error': str(e)}
            )

class FeatureValidator:
    """特征验证器
    
    提供特征数据的质量验证功能，包括：
    1. 数据类型验证
    2. 数据范围验证
    3. 数据质量验证
    4. 统计特性验证
    5. 自定义业务规则验证
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: List[ValidationRule] = []
        
        # 添加默认规则
        self._add_default_rules()
        
        logger.info(f"特征验证器初始化完成，规则数: {len(self.rules)}")
    
    def _add_default_rules(self):
        """添加默认验证规则"""
        # 数据质量规则
        self.add_rule(NullRule(max_null_ratio=0.1, level=ValidationLevel.WARNING))
        self.add_rule(UniqueRule(min_unique_ratio=0.01, level=ValidationLevel.INFO))
        
        # 异常值检测规则
        self.add_rule(OutlierRule(method='iqr', threshold=1.5, 
                                max_outlier_ratio=0.05, level=ValidationLevel.WARNING))
    
    def add_rule(self, rule: ValidationRule):
        """添加验证规则
        
        Args:
            rule: 验证规则
        """
        self.rules.append(rule)
        logger.debug(f"添加验证规则: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除验证规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否成功移除
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.debug(f"移除验证规则: {rule_name}")
                return True
        return False
    
    def get_rules(self, category: Optional[ValidationCategory] = None) -> List[ValidationRule]:
        """获取验证规则
        
        Args:
            category: 规则类别过滤
            
        Returns:
            验证规则列表
        """
        if category:
            return [rule for rule in self.rules if rule.category == category]
        return self.rules.copy()
    
    def validate_feature(self, feature_name: str, data: pd.Series, 
                        metadata: Dict[str, Any] = None,
                        rules: Optional[List[ValidationRule]] = None) -> ValidationReport:
        """验证特征数据
        
        Args:
            feature_name: 特征名称
            data: 特征数据
            metadata: 特征元数据
            rules: 自定义验证规则，如果为None则使用所有规则
            
        Returns:
            验证报告
        """
        report = ValidationReport(feature_name=feature_name)
        
        # 使用指定规则或所有规则
        validation_rules = rules or self.rules
        
        logger.info(f"开始验证特征: {feature_name}, 规则数: {len(validation_rules)}")
        
        for rule in validation_rules:
            try:
                result = rule.validate(data, metadata)
                report.add_result(result)
                
                if not result.passed and result.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
                    logger.warning(f"验证失败: {rule.name} - {result.message}")
                
            except Exception as e:
                # 规则执行失败
                error_result = ValidationResult(
                    rule_name=rule.name,
                    category=rule.category,
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message=f"规则执行失败: {str(e)}",
                    details={'error': str(e)}
                )
                report.add_result(error_result)
                logger.error(f"验证规则执行失败: {rule.name}, 错误: {str(e)}")
        
        # 生成摘要
        report.summary = self._generate_summary(report)
        
        logger.info(f"特征验证完成: {feature_name}, 成功率: {report.success_rate:.2%}")
        return report
    
    def validate_features(self, features_data: Dict[str, pd.Series],
                         features_metadata: Dict[str, Dict[str, Any]] = None) -> Dict[str, ValidationReport]:
        """批量验证特征
        
        Args:
            features_data: 特征数据字典 {feature_name: data}
            features_metadata: 特征元数据字典 {feature_name: metadata}
            
        Returns:
            验证报告字典 {feature_name: report}
        """
        reports = {}
        features_metadata = features_metadata or {}
        
        logger.info(f"开始批量验证特征，特征数: {len(features_data)}")
        
        for feature_name, data in features_data.items():
            metadata = features_metadata.get(feature_name, {})
            report = self.validate_feature(feature_name, data, metadata)
            reports[feature_name] = report
        
        logger.info(f"批量验证完成，特征数: {len(reports)}")
        return reports
    
    def _generate_summary(self, report: ValidationReport) -> Dict[str, Any]:
        """生成验证摘要
        
        Args:
            report: 验证报告
            
        Returns:
            摘要信息
        """
        summary = {
            'total_rules': report.total_rules,
            'passed_rules': report.passed_rules,
            'failed_rules': report.failed_rules,
            'success_rate': report.success_rate,
            'is_valid': report.is_valid
        }
        
        # 按级别统计
        level_counts = {}
        for level in ValidationLevel:
            results = report.get_results_by_level(level)
            level_counts[level.value] = {
                'total': len(results),
                'passed': len([r for r in results if r.passed]),
                'failed': len([r for r in results if not r.passed])
            }
        summary['level_distribution'] = level_counts
        
        # 按类别统计
        category_counts = {}
        for category in ValidationCategory:
            results = report.get_results_by_category(category)
            category_counts[category.value] = {
                'total': len(results),
                'passed': len([r for r in results if r.passed]),
                'failed': len([r for r in results if not r.passed])
            }
        summary['category_distribution'] = category_counts
        
        # 关键问题
        critical_issues = [r for r in report.results 
                          if not r.passed and r.level == ValidationLevel.CRITICAL]
        error_issues = [r for r in report.results 
                       if not r.passed and r.level == ValidationLevel.ERROR]
        
        summary['critical_issues'] = len(critical_issues)
        summary['error_issues'] = len(error_issues)
        summary['key_issues'] = [r.message for r in critical_issues + error_issues]
        
        return summary
    
    def create_data_type_rule(self, expected_type: str, 
                             level: ValidationLevel = ValidationLevel.ERROR) -> DataTypeRule:
        """创建数据类型验证规则"""
        return DataTypeRule(expected_type, level)
    
    def create_range_rule(self, min_value: Optional[float] = None, 
                         max_value: Optional[float] = None,
                         level: ValidationLevel = ValidationLevel.ERROR) -> RangeRule:
        """创建数据范围验证规则"""
        return RangeRule(min_value, max_value, level)
    
    def create_null_rule(self, max_null_ratio: float = 0.1,
                        level: ValidationLevel = ValidationLevel.WARNING) -> NullRule:
        """创建空值验证规则"""
        return NullRule(max_null_ratio, level)
    
    def create_unique_rule(self, min_unique_ratio: float = 0.8,
                          level: ValidationLevel = ValidationLevel.WARNING) -> UniqueRule:
        """创建唯一性验证规则"""
        return UniqueRule(min_unique_ratio, level)
    
    def create_outlier_rule(self, method: str = 'iqr', threshold: float = 1.5,
                           max_outlier_ratio: float = 0.05,
                           level: ValidationLevel = ValidationLevel.WARNING) -> OutlierRule:
        """创建异常值验证规则"""
        return OutlierRule(method, threshold, max_outlier_ratio, level)
    
    def create_distribution_rule(self, expected_distribution: str = 'normal',
                               significance_level: float = 0.05,
                               level: ValidationLevel = ValidationLevel.INFO) -> DistributionRule:
        """创建分布验证规则"""
        return DistributionRule(expected_distribution, significance_level, level)
    
    def create_custom_rule(self, name: str, validation_func: Callable[[pd.Series, Dict[str, Any]], bool],
                          category: ValidationCategory = ValidationCategory.BUSINESS_LOGIC,
                          level: ValidationLevel = ValidationLevel.ERROR,
                          description: str = "") -> CustomRule:
        """创建自定义验证规则"""
        return CustomRule(name, validation_func, category, level, description)

# 便捷函数
def create_feature_validator(config: Dict[str, Any] = None) -> FeatureValidator:
    """创建特征验证器实例
    
    Args:
        config: 配置字典
        
    Returns:
        特征验证器实例
    """
    return FeatureValidator(config)

def validate_feature_data(feature_name: str, data: pd.Series, 
                         metadata: Dict[str, Any] = None,
                         rules: Optional[List[ValidationRule]] = None) -> ValidationReport:
    """验证特征数据的便捷函数
    
    Args:
        feature_name: 特征名称
        data: 特征数据
        metadata: 特征元数据
        rules: 验证规则列表
        
    Returns:
        验证报告
    """
    validator = create_feature_validator()
    if rules:
        validator.rules = rules
    return validator.validate_feature(feature_name, data, metadata)