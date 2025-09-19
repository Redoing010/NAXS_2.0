# 数据质量检查模块

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from pathlib import Path
import logging
import json
from dataclasses import dataclass, asdict
from .parquet_store import ParquetStore
from .calendar import get_trading_calendar
from .utils import normalize_symbol

logger = logging.getLogger(__name__)


@dataclass
class QualityRule:
    """质量检查规则"""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class QualityIssue:
    """数据质量问题"""
    rule_name: str
    severity: str
    message: str
    symbol: str
    date_range: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    sample_data: Optional[Dict] = None
    timestamp: str = None
    metric_key: Optional[str] = None
    expected_range: Optional[str] = None
    suggestion: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if not self.message:
            parts = [f"规则: {self.rule_name}", f"级别: {self.severity}"]
            if self.metric_key:
                detail = self.metric_key
                if self.value is not None:
                    detail += f" 当前值: {self.value}"
                parts.append(detail)
            if self.threshold is not None:
                parts.append(f"阈值: {self.threshold}")
            if self.expected_range:
                parts.append(f"期望范围: {self.expected_range}")
            if self.suggestion:
                parts.append(f"建议: {self.suggestion}")
            self.message = ' | '.join(parts)


@dataclass
class QualityReport:
    """质量报告"""
    symbol: str
    date_range: str
    total_records: int
    issues: List[QualityIssue]
    metrics: Dict[str, Any]
    generated_at: str
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()
    
    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'error'])
    
    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'warning'])
    
    @property
    def is_healthy(self) -> bool:
        return self.error_count == 0


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, parquet_store: Optional[ParquetStore] = None):
        self.store = parquet_store or ParquetStore()
        self.trading_calendar = get_trading_calendar()
        self.rules = self._init_default_rules()
        
    def _init_default_rules(self) -> List[QualityRule]:
        """初始化默认质量规则"""
        return [
            # 数据完整性规则
            QualityRule(
                name="missing_data",
                description="检查缺失数据比例",
                severity="warning",
                threshold=0.05  # 5%
            ),
            QualityRule(
                name="duplicate_dates",
                description="检查重复日期",
                severity="error"
            ),
            QualityRule(
                name="future_dates",
                description="检查未来日期",
                severity="error"
            ),
            QualityRule(
                name="non_trading_days",
                description="检查非交易日数据",
                severity="warning"
            ),
            
            # 价格数据规则
            QualityRule(
                name="negative_prices",
                description="检查负价格",
                severity="error"
            ),
            QualityRule(
                name="zero_prices",
                description="检查零价格",
                severity="warning"
            ),
            QualityRule(
                name="price_outliers",
                description="检查价格异常值（3σ规则）",
                severity="warning",
                threshold=3.0
            ),
            QualityRule(
                name="ohlc_consistency",
                description="检查OHLC逻辑一致性",
                severity="error"
            ),
            QualityRule(
                name="extreme_returns",
                description="检查极端收益率",
                severity="warning",
                threshold=0.20  # 20%
            ),
            
            # 成交量规则
            QualityRule(
                name="negative_volume",
                description="检查负成交量",
                severity="error"
            ),
            QualityRule(
                name="zero_volume_days",
                description="检查零成交量天数比例",
                severity="warning",
                threshold=0.10  # 10%
            ),
            QualityRule(
                name="volume_outliers",
                description="检查成交量异常值",
                severity="warning",
                threshold=5.0  # 5σ
            ),
            
            # 数据连续性规则
            QualityRule(
                name="data_gaps",
                description="检查数据缺口",
                severity="warning",
                threshold=5  # 连续5个交易日
            ),
            QualityRule(
                name="price_jumps",
                description="检查价格跳跃",
                severity="warning",
                threshold=0.15  # 15%
            ),
        ]
    
    def check_symbol(self, symbol: str, start_date: str, end_date: str, 
                    root_path: str = "data/parquet") -> QualityReport:
        """检查单个股票的数据质量
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            root_path: 数据根路径
            
        Returns:
            质量报告
        """
        try:
            logger.info(f"开始检查 {symbol} 数据质量: {start_date} - {end_date}")
            
            # 读取数据
            df = self.store.read_daily_bars(symbol, start_date, end_date, root_path)
            
            if df.empty:
                return QualityReport(
                    symbol=symbol,
                    date_range=f"{start_date} - {end_date}",
                    total_records=0,
                    issues=[QualityIssue(
                        rule_name="no_data",
                        severity="error",
                        message="未找到数据",
                        symbol=symbol,
                        date_range=f"{start_date} - {end_date}"
                    )],
                    metrics={},
                    generated_at=datetime.now().isoformat()
                )
            
            # 执行所有质量检查
            issues = []
            metrics = {}
            
            for rule in self.rules:
                if not rule.enabled:
                    continue
                    
                try:
                    rule_issues, rule_metrics = self._apply_rule(rule, df, symbol, start_date, end_date)
                    issues.extend(rule_issues)
                    metrics.update(rule_metrics)
                except Exception as e:
                    logger.error(f"执行规则 {rule.name} 失败: {e}")
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity="error",
                        message=f"规则执行失败: {e}",
                        symbol=symbol,
                        date_range=f"{start_date} - {end_date}"
                    ))
            
            # 生成报告
            report = QualityReport(
                symbol=symbol,
                date_range=f"{start_date} - {end_date}",
                total_records=len(df),
                issues=issues,
                metrics=metrics,
                generated_at=datetime.now().isoformat()
            )
            
            logger.info(f"完成 {symbol} 质量检查，发现 {len(issues)} 个问题")
            return report
            
        except Exception as e:
            logger.error(f"检查 {symbol} 数据质量失败: {e}")
            return QualityReport(
                symbol=symbol,
                date_range=f"{start_date} - {end_date}",
                total_records=0,
                issues=[QualityIssue(
                    rule_name="check_failed",
                    severity="error",
                    message=f"质量检查失败: {e}",
                    symbol=symbol,
                    date_range=f"{start_date} - {end_date}"
                )],
                metrics={},
                generated_at=datetime.now().isoformat()
            )
    
    def _apply_rule(self, rule: QualityRule, df: pd.DataFrame, 
                   symbol: str, start_date: str, end_date: str) -> Tuple[List[QualityIssue], Dict]:
        """应用质量规则"""
        issues = []
        metrics = {}
        date_range = f"{start_date} - {end_date}"
        
        if rule.name == "missing_data":
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            metrics["missing_data_ratio"] = missing_ratio
            
            if missing_ratio > rule.threshold:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"缺失数据比例过高: {missing_ratio:.2%}",
                    symbol=symbol,
                    date_range=date_range,
                    value=missing_ratio,
                    threshold=rule.threshold
                ))
        
        elif rule.name == "duplicate_dates":
            duplicates = df.index.duplicated().sum()
            metrics["duplicate_dates_count"] = duplicates
            
            if duplicates > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"发现 {duplicates} 个重复日期",
                    symbol=symbol,
                    date_range=date_range,
                    value=duplicates
                ))
        
        elif rule.name == "future_dates":
            dt_index = df.index
            if isinstance(dt_index, pd.DatetimeIndex):
                if dt_index.tz is not None:
                    # 将时区统一到UTC并移除tz信息，便于和今日日期比较
                    dt_index = dt_index.tz_convert('UTC').tz_localize(None)
            else:
                dt_index = pd.to_datetime(dt_index, errors='coerce')

            today = pd.Timestamp.utcnow().normalize().tz_localize(None)
            future_dates = int((dt_index > today).sum())
            metrics["future_dates_count"] = future_dates

            if future_dates > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"发现 {future_dates} 个未来日期",
                    symbol=symbol,
                    date_range=date_range,
                    value=future_dates
                ))
        
        elif rule.name == "negative_prices":
            price_cols = ['open', 'high', 'low', 'close']
            negative_count = 0
            
            for col in price_cols:
                if col in df.columns:
                    negative = (df[col] < 0).sum()
                    negative_count += negative
            
            metrics["negative_prices_count"] = negative_count
            
            if negative_count > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"发现 {negative_count} 个负价格",
                    symbol=symbol,
                    date_range=date_range,
                    value=negative_count
                ))
        
        elif rule.name == "zero_prices":
            price_cols = ['open', 'high', 'low', 'close']
            zero_count = 0
            
            for col in price_cols:
                if col in df.columns:
                    zeros = (df[col] == 0).sum()
                    zero_count += zeros
            
            metrics["zero_prices_count"] = zero_count
            
            if zero_count > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"发现 {zero_count} 个零价格",
                    symbol=symbol,
                    date_range=date_range,
                    value=zero_count
                ))
        
        elif rule.name == "ohlc_consistency":
            inconsistent_count = 0
            
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # 检查 high >= max(open, close) 和 low <= min(open, close)
                high_invalid = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
                low_invalid = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
                inconsistent_count = high_invalid + low_invalid
            
            metrics["ohlc_inconsistent_count"] = inconsistent_count
            
            if inconsistent_count > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"发现 {inconsistent_count} 个OHLC不一致记录",
                    symbol=symbol,
                    date_range=date_range,
                    value=inconsistent_count
                ))
        
        elif rule.name == "extreme_returns":
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                extreme_returns = (abs(returns) > rule.threshold).sum()
                metrics["extreme_returns_count"] = extreme_returns
                metrics["max_return"] = returns.max()
                metrics["min_return"] = returns.min()
                
                if extreme_returns > 0:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"发现 {extreme_returns} 个极端收益率（>{rule.threshold:.1%}）",
                        symbol=symbol,
                        date_range=date_range,
                        value=extreme_returns,
                        threshold=rule.threshold
                    ))
        
        elif rule.name == "negative_volume":
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                metrics["negative_volume_count"] = negative_volume
                
                if negative_volume > 0:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"发现 {negative_volume} 个负成交量",
                        symbol=symbol,
                        date_range=date_range,
                        value=negative_volume
                    ))
        
        elif rule.name == "zero_volume_days":
            if 'volume' in df.columns:
                zero_volume_days = (df['volume'] == 0).sum()
                zero_volume_ratio = zero_volume_days / len(df)
                metrics["zero_volume_ratio"] = zero_volume_ratio
                
                if zero_volume_ratio > rule.threshold:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"零成交量天数比例过高: {zero_volume_ratio:.2%}",
                        symbol=symbol,
                        date_range=date_range,
                        value=zero_volume_ratio,
                        threshold=rule.threshold
                    ))
        
        return issues, metrics
    
    def check_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str,
                              root_path: str = "data/parquet") -> Dict[str, QualityReport]:
        """检查多个股票的数据质量
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            root_path: 数据根路径
            
        Returns:
            股票代码到质量报告的映射
        """
        reports = {}
        
        for symbol in symbols:
            try:
                report = self.check_symbol(symbol, start_date, end_date, root_path)
                reports[symbol] = report
            except Exception as e:
                logger.error(f"检查 {symbol} 失败: {e}")
                reports[symbol] = QualityReport(
                    symbol=symbol,
                    date_range=f"{start_date} - {end_date}",
                    total_records=0,
                    issues=[QualityIssue(
                        rule_name="check_failed",
                        severity="error",
                        message=f"检查失败: {e}",
                        symbol=symbol,
                        date_range=f"{start_date} - {end_date}"
                    )],
                    metrics={},
                    generated_at=datetime.now().isoformat()
                )
        
        return reports
    
    def generate_summary_report(self, reports: Dict[str, QualityReport]) -> Dict[str, Any]:
        """生成汇总报告
        
        Args:
            reports: 质量报告字典
            
        Returns:
            汇总报告
        """
        summary = {
            'total_symbols': len(reports),
            'healthy_symbols': 0,
            'symbols_with_errors': 0,
            'symbols_with_warnings': 0,
            'total_issues': 0,
            'issue_breakdown': {},
            'top_issues': [],
            'generated_at': datetime.now().isoformat()
        }
        
        issue_counts = {}
        all_issues = []
        
        for symbol, report in reports.items():
            if report.is_healthy:
                summary['healthy_symbols'] += 1
            
            if report.error_count > 0:
                summary['symbols_with_errors'] += 1
            
            if report.warning_count > 0:
                summary['symbols_with_warnings'] += 1
            
            summary['total_issues'] += len(report.issues)
            
            # 统计问题类型
            for issue in report.issues:
                all_issues.append(issue)
                if issue.rule_name not in issue_counts:
                    issue_counts[issue.rule_name] = 0
                issue_counts[issue.rule_name] += 1
        
        # 问题分类统计
        summary['issue_breakdown'] = dict(sorted(issue_counts.items(), 
                                                key=lambda x: x[1], reverse=True))
        
        # 最严重的问题
        error_issues = [i for i in all_issues if i.severity == 'error']
        summary['top_issues'] = sorted(error_issues, 
                                     key=lambda x: x.value or 0, reverse=True)[:10]
        
        return summary
    
    def save_report(self, report: QualityReport, output_path: str):
        """保存质量报告
        
        Args:
            report: 质量报告
            output_path: 输出路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化的字典
            report_dict = asdict(report)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"质量报告已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def save_summary_report(self, summary: Dict[str, Any], output_path: str):
        """保存汇总报告
        
        Args:
            summary: 汇总报告
            output_path: 输出路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"汇总报告已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存汇总报告失败: {e}")
    
    def add_custom_rule(self, rule: QualityRule):
        """添加自定义规则
        
        Args:
            rule: 质量规则
        """
        self.rules.append(rule)
        logger.info(f"添加自定义规则: {rule.name}")
    
    def disable_rule(self, rule_name: str):
        """禁用规则
        
        Args:
            rule_name: 规则名称
        """
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"禁用规则: {rule_name}")
                break
    
    def enable_rule(self, rule_name: str):
        """启用规则
        
        Args:
            rule_name: 规则名称
        """
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"启用规则: {rule_name}")
                break


# 便捷函数
def check_data_quality(symbols: List[str], start_date: str, end_date: str,
                      root_path: str = "data/parquet", 
                      output_dir: str = "reports/dq") -> Dict[str, Any]:
    """数据质量检查的便捷函数
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        root_path: 数据根路径
        output_dir: 报告输出目录
        
    Returns:
        汇总报告
    """
    checker = DataQualityChecker()
    
    # 检查所有股票
    reports = checker.check_multiple_symbols(symbols, start_date, end_date, root_path)
    
    # 生成汇总报告
    summary = checker.generate_summary_report(reports)
    
    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存汇总报告
    checker.save_summary_report(summary, output_path / "summary.json")
    
    # 保存详细报告
    for symbol, report in reports.items():
        checker.save_report(report, output_path / f"{symbol}.json")
    
    return summary