# 编排监控模块 - 实现系统监控、指标收集和告警功能

import logging
import asyncio
import threading
import time
import psutil
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
import json
from collections import defaultdict, deque
import statistics
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"        # 计数器
    GAUGE = "gauge"            # 仪表盘
    HISTOGRAM = "histogram"    # 直方图
    SUMMARY = "summary"        # 摘要
    TIMER = "timer"            # 计时器

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"              # 信息
    WARNING = "warning"        # 警告
    ERROR = "error"            # 错误
    CRITICAL = "critical"      # 严重

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"          # 活跃
    RESOLVED = "resolved"      # 已解决
    SUPPRESSED = "suppressed"  # 已抑制

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"        # 健康
    DEGRADED = "degraded"      # 降级
    UNHEALTHY = "unhealthy"    # 不健康
    UNKNOWN = "unknown"        # 未知

@dataclass
class Metric:
    """指标数据"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'description': self.description,
            'unit': self.unit
        }

@dataclass
class Alert:
    """告警信息"""
    id: str
    name: str
    level: AlertLevel
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.ACTIVE
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'labels': self.labels,
            'metadata': self.metadata,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

@dataclass
class HealthCheck:
    """健康检查"""
    name: str
    status: HealthStatus
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration,
            'metadata': self.metadata
        }

class MetricCollector:
    """指标收集器
    
    收集系统和应用指标，包括：
    1. 系统资源指标
    2. 应用性能指标
    3. 业务指标
    4. 自定义指标
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
        self.collection_interval = self.config.get('collection_interval', 10)  # 秒
        self.running = False
        self.collector_thread = None
        self.lock = threading.RLock()
        
        logger.info(f"指标收集器初始化完成，收集间隔: {self.collection_interval}秒")
    
    def register_metric(self, name: str, metric_type: MetricType, 
                       description: str = "", unit: str = "",
                       labels: Dict[str, str] = None):
        """注册指标
        
        Args:
            name: 指标名称
            metric_type: 指标类型
            description: 描述
            unit: 单位
            labels: 标签
        """
        with self.lock:
            self.metric_definitions[name] = {
                'type': metric_type,
                'description': description,
                'unit': unit,
                'labels': labels or {}
            }
        
        logger.debug(f"注册指标: {name} ({metric_type.value})")
    
    def collect_metric(self, name: str, value: Union[int, float], 
                      labels: Dict[str, str] = None, timestamp: datetime = None):
        """收集指标
        
        Args:
            name: 指标名称
            value: 指标值
            labels: 标签
            timestamp: 时间戳
        """
        if name not in self.metric_definitions:
            logger.warning(f"未注册的指标: {name}")
            return
        
        definition = self.metric_definitions[name]
        metric = Metric(
            name=name,
            metric_type=definition['type'],
            value=value,
            timestamp=timestamp or datetime.now(),
            labels={**definition['labels'], **(labels or {})},
            description=definition['description'],
            unit=definition['unit']
        )
        
        with self.lock:
            self.metrics[name].append(metric)
        
        logger.debug(f"收集指标: {name} = {value}")
    
    def get_metrics(self, name: str, limit: Optional[int] = None) -> List[Metric]:
        """获取指标数据
        
        Args:
            name: 指标名称
            limit: 数量限制
            
        Returns:
            指标数据列表
        """
        with self.lock:
            metrics = list(self.metrics.get(name, []))
            if limit:
                return metrics[-limit:]
            return metrics
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """获取最新指标
        
        Args:
            name: 指标名称
            
        Returns:
            最新指标数据
        """
        with self.lock:
            metric_queue = self.metrics.get(name)
            if metric_queue:
                return metric_queue[-1]
            return None
    
    def get_metric_statistics(self, name: str, duration: timedelta = None) -> Dict[str, Any]:
        """获取指标统计信息
        
        Args:
            name: 指标名称
            duration: 时间范围
            
        Returns:
            统计信息
        """
        with self.lock:
            metrics = list(self.metrics.get(name, []))
            
            if not metrics:
                return {}
            
            # 过滤时间范围
            if duration:
                cutoff_time = datetime.now() - duration
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'latest': values[-1],
                'first_timestamp': metrics[0].timestamp.isoformat(),
                'latest_timestamp': metrics[-1].timestamp.isoformat()
            }
    
    def start_collection(self):
        """启动指标收集"""
        if self.running:
            logger.warning("指标收集已在运行")
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        
        logger.info("指标收集启动")
    
    def stop_collection(self):
        """停止指标收集"""
        if not self.running:
            logger.warning("指标收集未在运行")
            return
        
        self.running = False
        
        if self.collector_thread and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=5)
        
        logger.info("指标收集停止")
    
    def _collection_loop(self):
        """收集循环"""
        logger.info("指标收集循环启动")
        
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"指标收集异常: {str(e)}")
                time.sleep(self.collection_interval)
        
        logger.info("指标收集循环结束")
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.collect_metric('system_cpu_usage_percent', cpu_percent)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            self.collect_metric('system_memory_usage_percent', memory.percent)
            self.collect_metric('system_memory_used_bytes', memory.used)
            self.collect_metric('system_memory_available_bytes', memory.available)
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            self.collect_metric('system_disk_usage_percent', (disk.used / disk.total) * 100)
            self.collect_metric('system_disk_used_bytes', disk.used)
            self.collect_metric('system_disk_free_bytes', disk.free)
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self.collect_metric('system_network_bytes_sent', net_io.bytes_sent)
            self.collect_metric('system_network_bytes_recv', net_io.bytes_recv)
            
            # 进程数量
            process_count = len(psutil.pids())
            self.collect_metric('system_process_count', process_count)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {str(e)}")
    
    def clear_metrics(self, name: Optional[str] = None):
        """清除指标数据
        
        Args:
            name: 指标名称，如果为None则清除所有指标
        """
        with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
            else:
                self.metrics.clear()
        
        logger.info(f"清除指标数据: {name or '所有指标'}")

class AlertRule:
    """告警规则"""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                 level: AlertLevel, message_template: str,
                 cooldown: timedelta = timedelta(minutes=5)):
        self.name = name
        self.condition = condition
        self.level = level
        self.message_template = message_template
        self.cooldown = cooldown
        self.last_triggered = None
    
    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """判断是否应该触发告警
        
        Args:
            context: 上下文数据
            
        Returns:
            是否应该触发
        """
        # 检查冷却时间
        if self.last_triggered:
            if datetime.now() - self.last_triggered < self.cooldown:
                return False
        
        # 检查条件
        try:
            return self.condition(context)
        except Exception as e:
            logger.error(f"告警规则条件检查失败: {self.name}, 错误: {str(e)}")
            return False
    
    def generate_alert(self, context: Dict[str, Any]) -> Alert:
        """生成告警
        
        Args:
            context: 上下文数据
            
        Returns:
            告警对象
        """
        self.last_triggered = datetime.now()
        
        # 格式化消息
        try:
            message = self.message_template.format(**context)
        except Exception as e:
            message = f"告警消息格式化失败: {str(e)}"
        
        return Alert(
            id=str(uuid.uuid4()),
            name=self.name,
            level=self.level,
            message=message,
            source="alert_manager",
            metadata={'context': context}
        )

class AlertManager:
    """告警管理器
    
    管理告警规则和告警通知，包括：
    1. 告警规则管理
    2. 告警触发检查
    3. 告警通知发送
    4. 告警状态管理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.notifiers: List[Callable[[Alert], None]] = []
        self.check_interval = self.config.get('check_interval', 30)  # 秒
        self.running = False
        self.checker_thread = None
        self.lock = threading.RLock()
        
        # 配置通知器
        self._setup_notifiers()
        
        logger.info(f"告警管理器初始化完成，检查间隔: {self.check_interval}秒")
    
    def _setup_notifiers(self):
        """设置通知器"""
        # 邮件通知器
        email_config = self.config.get('email')
        if email_config:
            self.add_notifier(self._create_email_notifier(email_config))
        
        # Webhook通知器
        webhook_config = self.config.get('webhook')
        if webhook_config:
            self.add_notifier(self._create_webhook_notifier(webhook_config))
        
        # 日志通知器（默认）
        self.add_notifier(self._log_notifier)
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则
        
        Args:
            rule: 告警规则
        """
        with self.lock:
            self.rules[rule.name] = rule
        
        logger.debug(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除告警规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否成功移除
        """
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.debug(f"移除告警规则: {rule_name}")
                return True
            return False
    
    def add_notifier(self, notifier: Callable[[Alert], None]):
        """添加通知器
        
        Args:
            notifier: 通知器函数
        """
        self.notifiers.append(notifier)
        logger.debug(f"添加通知器: {notifier.__name__}")
    
    def trigger_alert(self, alert: Alert):
        """触发告警
        
        Args:
            alert: 告警对象
        """
        with self.lock:
            self.alerts[alert.id] = alert
        
        # 发送通知
        for notifier in self.notifiers:
            try:
                notifier(alert)
            except Exception as e:
                logger.error(f"告警通知发送失败: {notifier.__name__}, 错误: {str(e)}")
        
        logger.info(f"触发告警: {alert.name} ({alert.level.value})")
    
    def resolve_alert(self, alert_id: str):
        """解决告警
        
        Args:
            alert_id: 告警ID
        """
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                logger.info(f"解决告警: {alert.name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警
        
        Returns:
            活跃告警列表
        """
        with self.lock:
            return [alert for alert in self.alerts.values() 
                   if alert.status == AlertStatus.ACTIVE]
    
    def get_alerts(self, level: Optional[AlertLevel] = None,
                  status: Optional[AlertStatus] = None,
                  limit: Optional[int] = None) -> List[Alert]:
        """获取告警列表
        
        Args:
            level: 告警级别过滤
            status: 告警状态过滤
            limit: 数量限制
            
        Returns:
            告警列表
        """
        with self.lock:
            alerts = list(self.alerts.values())
            
            if level:
                alerts = [a for a in alerts if a.level == level]
            
            if status:
                alerts = [a for a in alerts if a.status == status]
            
            # 按时间排序
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            if limit:
                return alerts[:limit]
            
            return alerts
    
    def start_checking(self, metric_collector: MetricCollector):
        """启动告警检查
        
        Args:
            metric_collector: 指标收集器
        """
        if self.running:
            logger.warning("告警检查已在运行")
            return
        
        self.running = True
        self.checker_thread = threading.Thread(
            target=self._checking_loop, 
            args=(metric_collector,), 
            daemon=True
        )
        self.checker_thread.start()
        
        logger.info("告警检查启动")
    
    def stop_checking(self):
        """停止告警检查"""
        if not self.running:
            logger.warning("告警检查未在运行")
            return
        
        self.running = False
        
        if self.checker_thread and self.checker_thread.is_alive():
            self.checker_thread.join(timeout=5)
        
        logger.info("告警检查停止")
    
    def _checking_loop(self, metric_collector: MetricCollector):
        """检查循环"""
        logger.info("告警检查循环启动")
        
        while self.running:
            try:
                self._check_rules(metric_collector)
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"告警检查异常: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.info("告警检查循环结束")
    
    def _check_rules(self, metric_collector: MetricCollector):
        """检查告警规则"""
        with self.lock:
            rules = list(self.rules.values())
        
        for rule in rules:
            try:
                # 构建上下文
                context = self._build_context(metric_collector)
                
                # 检查规则
                if rule.should_trigger(context):
                    alert = rule.generate_alert(context)
                    self.trigger_alert(alert)
                    
            except Exception as e:
                logger.error(f"检查告警规则失败: {rule.name}, 错误: {str(e)}")
    
    def _build_context(self, metric_collector: MetricCollector) -> Dict[str, Any]:
        """构建上下文数据"""
        context = {
            'timestamp': datetime.now(),
            'metrics': {}
        }
        
        # 添加最新指标
        for metric_name in metric_collector.metric_definitions.keys():
            latest_metric = metric_collector.get_latest_metric(metric_name)
            if latest_metric:
                context['metrics'][metric_name] = latest_metric.value
        
        return context
    
    def _log_notifier(self, alert: Alert):
        """日志通知器"""
        logger.warning(f"告警通知: [{alert.level.value.upper()}] {alert.name} - {alert.message}")
    
    def _create_email_notifier(self, email_config: Dict[str, Any]) -> Callable[[Alert], None]:
        """创建邮件通知器"""
        def email_notifier(alert: Alert):
            try:
                msg = MIMEMultipart()
                msg['From'] = email_config['from']
                msg['To'] = ', '.join(email_config['to'])
                msg['Subject'] = f"[{alert.level.value.upper()}] {alert.name}"
                
                body = f"""
                告警名称: {alert.name}
                告警级别: {alert.level.value}
                告警消息: {alert.message}
                告警来源: {alert.source}
                告警时间: {alert.timestamp.isoformat()}
                """
                
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
                
                server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
                if email_config.get('use_tls'):
                    server.starttls()
                if email_config.get('username'):
                    server.login(email_config['username'], email_config['password'])
                
                server.send_message(msg)
                server.quit()
                
                logger.debug(f"邮件告警通知发送成功: {alert.name}")
                
            except Exception as e:
                logger.error(f"邮件告警通知发送失败: {str(e)}")
        
        return email_notifier
    
    def _create_webhook_notifier(self, webhook_config: Dict[str, Any]) -> Callable[[Alert], None]:
        """创建Webhook通知器"""
        def webhook_notifier(alert: Alert):
            try:
                payload = {
                    'alert': alert.to_dict(),
                    'timestamp': datetime.now().isoformat()
                }
                
                response = requests.post(
                    webhook_config['url'],
                    json=payload,
                    headers=webhook_config.get('headers', {}),
                    timeout=webhook_config.get('timeout', 10)
                )
                
                response.raise_for_status()
                
                logger.debug(f"Webhook告警通知发送成功: {alert.name}")
                
            except Exception as e:
                logger.error(f"Webhook告警通知发送失败: {str(e)}")
        
        return webhook_notifier

class HealthChecker:
    """健康检查器
    
    执行系统健康检查，包括：
    1. 服务可用性检查
    2. 依赖服务检查
    3. 资源健康检查
    4. 自定义健康检查
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.results: Dict[str, HealthCheck] = {}
        self.check_interval = self.config.get('check_interval', 60)  # 秒
        self.running = False
        self.checker_thread = None
        self.lock = threading.RLock()
        
        # 注册默认健康检查
        self._register_default_checks()
        
        logger.info(f"健康检查器初始化完成，检查间隔: {self.check_interval}秒")
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self.register_check('system_resources', self._check_system_resources)
        self.register_check('disk_space', self._check_disk_space)
        self.register_check('memory_usage', self._check_memory_usage)
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """注册健康检查
        
        Args:
            name: 检查名称
            check_func: 检查函数
        """
        with self.lock:
            self.checks[name] = check_func
        
        logger.debug(f"注册健康检查: {name}")
    
    def unregister_check(self, name: str) -> bool:
        """注销健康检查
        
        Args:
            name: 检查名称
            
        Returns:
            是否成功注销
        """
        with self.lock:
            if name in self.checks:
                del self.checks[name]
                if name in self.results:
                    del self.results[name]
                logger.debug(f"注销健康检查: {name}")
                return True
            return False
    
    def run_check(self, name: str) -> Optional[HealthCheck]:
        """运行单个健康检查
        
        Args:
            name: 检查名称
            
        Returns:
            健康检查结果
        """
        check_func = self.checks.get(name)
        if not check_func:
            logger.warning(f"健康检查不存在: {name}")
            return None
        
        try:
            start_time = time.time()
            result = check_func()
            result.duration = time.time() - start_time
            
            with self.lock:
                self.results[name] = result
            
            logger.debug(f"健康检查完成: {name} - {result.status.value}")
            return result
            
        except Exception as e:
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查执行失败: {str(e)}",
                duration=time.time() - start_time if 'start_time' in locals() else None
            )
            
            with self.lock:
                self.results[name] = result
            
            logger.error(f"健康检查失败: {name}, 错误: {str(e)}")
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """运行所有健康检查
        
        Returns:
            健康检查结果字典
        """
        results = {}
        
        with self.lock:
            check_names = list(self.checks.keys())
        
        for name in check_names:
            result = self.run_check(name)
            if result:
                results[name] = result
        
        return results
    
    def get_health_status(self) -> HealthStatus:
        """获取整体健康状态
        
        Returns:
            整体健康状态
        """
        with self.lock:
            if not self.results:
                return HealthStatus.UNKNOWN
            
            statuses = [result.status for result in self.results.values()]
            
            if any(status == HealthStatus.UNHEALTHY for status in statuses):
                return HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in statuses):
                return HealthStatus.DEGRADED
            elif all(status == HealthStatus.HEALTHY for status in statuses):
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告
        
        Returns:
            健康报告
        """
        with self.lock:
            overall_status = self.get_health_status()
            
            return {
                'overall_status': overall_status.value,
                'timestamp': datetime.now().isoformat(),
                'checks': {name: result.to_dict() for name, result in self.results.items()},
                'summary': {
                    'total_checks': len(self.results),
                    'healthy': len([r for r in self.results.values() if r.status == HealthStatus.HEALTHY]),
                    'degraded': len([r for r in self.results.values() if r.status == HealthStatus.DEGRADED]),
                    'unhealthy': len([r for r in self.results.values() if r.status == HealthStatus.UNHEALTHY]),
                    'unknown': len([r for r in self.results.values() if r.status == HealthStatus.UNKNOWN])
                }
            }
    
    def start_checking(self):
        """启动健康检查"""
        if self.running:
            logger.warning("健康检查已在运行")
            return
        
        self.running = True
        self.checker_thread = threading.Thread(target=self._checking_loop, daemon=True)
        self.checker_thread.start()
        
        logger.info("健康检查启动")
    
    def stop_checking(self):
        """停止健康检查"""
        if not self.running:
            logger.warning("健康检查未在运行")
            return
        
        self.running = False
        
        if self.checker_thread and self.checker_thread.is_alive():
            self.checker_thread.join(timeout=5)
        
        logger.info("健康检查停止")
    
    def _checking_loop(self):
        """检查循环"""
        logger.info("健康检查循环启动")
        
        while self.running:
            try:
                self.run_all_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"健康检查异常: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.info("健康检查循环结束")
    
    def _check_system_resources(self) -> HealthCheck:
        """检查系统资源"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"系统资源使用率过高: CPU {cpu_percent:.1f}%, 内存 {memory_percent:.1f}%"
            elif cpu_percent > 70 or memory_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"系统资源使用率较高: CPU {cpu_percent:.1f}%, 内存 {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"系统资源正常: CPU {cpu_percent:.1f}%, 内存 {memory_percent:.1f}%"
            
            return HealthCheck(
                name='system_resources',
                status=status,
                message=message,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name='system_resources',
                status=HealthStatus.UNHEALTHY,
                message=f"系统资源检查失败: {str(e)}"
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """检查磁盘空间"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"磁盘空间严重不足: {usage_percent:.1f}%"
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"磁盘空间不足: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"磁盘空间正常: {usage_percent:.1f}%"
            
            return HealthCheck(
                name='disk_space',
                status=status,
                message=message,
                metadata={
                    'usage_percent': usage_percent,
                    'total_bytes': disk.total,
                    'used_bytes': disk.used,
                    'free_bytes': disk.free
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name='disk_space',
                status=HealthStatus.UNHEALTHY,
                message=f"磁盘空间检查失败: {str(e)}"
            )
    
    def _check_memory_usage(self) -> HealthCheck:
        """检查内存使用情况"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"内存使用率过高: {memory.percent:.1f}%"
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
                message = f"内存使用率较高: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"内存使用正常: {memory.percent:.1f}%"
            
            return HealthCheck(
                name='memory_usage',
                status=status,
                message=message,
                metadata={
                    'percent': memory.percent,
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'free': memory.free
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name='memory_usage',
                status=HealthStatus.UNHEALTHY,
                message=f"内存使用检查失败: {str(e)}"
            )

class OrchestrationMonitor:
    """编排监控器
    
    整合指标收集、告警管理和健康检查功能，提供：
    1. 统一的监控接口
    2. 监控数据聚合
    3. 监控报告生成
    4. 监控配置管理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化组件
        self.metric_collector = MetricCollector(self.config.get('metrics', {}))
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        self.health_checker = HealthChecker(self.config.get('health', {}))
        
        # 注册默认指标
        self._register_default_metrics()
        
        # 注册默认告警规则
        self._register_default_alert_rules()
        
        logger.info("编排监控器初始化完成")
    
    def _register_default_metrics(self):
        """注册默认指标"""
        # 系统指标
        self.metric_collector.register_metric('system_cpu_usage_percent', MetricType.GAUGE, 'CPU使用率', '%')
        self.metric_collector.register_metric('system_memory_usage_percent', MetricType.GAUGE, '内存使用率', '%')
        self.metric_collector.register_metric('system_memory_used_bytes', MetricType.GAUGE, '已使用内存', 'bytes')
        self.metric_collector.register_metric('system_memory_available_bytes', MetricType.GAUGE, '可用内存', 'bytes')
        self.metric_collector.register_metric('system_disk_usage_percent', MetricType.GAUGE, '磁盘使用率', '%')
        self.metric_collector.register_metric('system_disk_used_bytes', MetricType.GAUGE, '已使用磁盘', 'bytes')
        self.metric_collector.register_metric('system_disk_free_bytes', MetricType.GAUGE, '可用磁盘', 'bytes')
        self.metric_collector.register_metric('system_network_bytes_sent', MetricType.COUNTER, '网络发送字节', 'bytes')
        self.metric_collector.register_metric('system_network_bytes_recv', MetricType.COUNTER, '网络接收字节', 'bytes')
        self.metric_collector.register_metric('system_process_count', MetricType.GAUGE, '进程数量', 'count')
        
        # 应用指标
        self.metric_collector.register_metric('app_requests_total', MetricType.COUNTER, '请求总数', 'count')
        self.metric_collector.register_metric('app_request_duration_seconds', MetricType.HISTOGRAM, '请求耗时', 'seconds')
        self.metric_collector.register_metric('app_errors_total', MetricType.COUNTER, '错误总数', 'count')
        self.metric_collector.register_metric('app_active_connections', MetricType.GAUGE, '活跃连接数', 'count')
    
    def _register_default_alert_rules(self):
        """注册默认告警规则"""
        # CPU使用率告警
        cpu_rule = AlertRule(
            name='high_cpu_usage',
            condition=lambda ctx: ctx['metrics'].get('system_cpu_usage_percent', 0) > 80,
            level=AlertLevel.WARNING,
            message_template='CPU使用率过高: {metrics[system_cpu_usage_percent]:.1f}%'
        )
        self.alert_manager.add_rule(cpu_rule)
        
        # 内存使用率告警
        memory_rule = AlertRule(
            name='high_memory_usage',
            condition=lambda ctx: ctx['metrics'].get('system_memory_usage_percent', 0) > 85,
            level=AlertLevel.WARNING,
            message_template='内存使用率过高: {metrics[system_memory_usage_percent]:.1f}%'
        )
        self.alert_manager.add_rule(memory_rule)
        
        # 磁盘空间告警
        disk_rule = AlertRule(
            name='low_disk_space',
            condition=lambda ctx: ctx['metrics'].get('system_disk_usage_percent', 0) > 90,
            level=AlertLevel.ERROR,
            message_template='磁盘空间不足: {metrics[system_disk_usage_percent]:.1f}%'
        )
        self.alert_manager.add_rule(disk_rule)
    
    def start(self):
        """启动监控"""
        self.metric_collector.start_collection()
        self.alert_manager.start_checking(self.metric_collector)
        self.health_checker.start_checking()
        
        logger.info("编排监控启动")
    
    def stop(self):
        """停止监控"""
        self.metric_collector.stop_collection()
        self.alert_manager.stop_checking()
        self.health_checker.stop_checking()
        
        logger.info("编排监控停止")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告
        
        Returns:
            监控报告
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'health': self.health_checker.get_health_report(),
            'alerts': {
                'active_count': len(self.alert_manager.get_active_alerts()),
                'recent_alerts': [alert.to_dict() for alert in self.alert_manager.get_alerts(limit=10)]
            },
            'metrics': {
                'system_cpu': self.metric_collector.get_latest_metric('system_cpu_usage_percent'),
                'system_memory': self.metric_collector.get_latest_metric('system_memory_usage_percent'),
                'system_disk': self.metric_collector.get_latest_metric('system_disk_usage_percent')
            }
        }
    
    def collect_metric(self, name: str, value: Union[int, float], 
                      labels: Dict[str, str] = None):
        """收集指标
        
        Args:
            name: 指标名称
            value: 指标值
            labels: 标签
        """
        self.metric_collector.collect_metric(name, value, labels)
    
    def trigger_alert(self, name: str, level: AlertLevel, message: str,
                     labels: Dict[str, str] = None):
        """触发告警
        
        Args:
            name: 告警名称
            level: 告警级别
            message: 告警消息
            labels: 标签
        """
        alert = Alert(
            id=str(uuid.uuid4()),
            name=name,
            level=level,
            message=message,
            source="manual",
            labels=labels or {}
        )
        self.alert_manager.trigger_alert(alert)

# 便捷函数
def create_monitor(config: Dict[str, Any] = None) -> OrchestrationMonitor:
    """创建编排监控器实例
    
    Args:
        config: 配置字典
        
    Returns:
        编排监控器实例
    """
    return OrchestrationMonitor(config)

def create_cpu_alert_rule(threshold: float = 80.0) -> AlertRule:
    """创建CPU使用率告警规则
    
    Args:
        threshold: 告警阈值
        
    Returns:
        告警规则
    """
    return AlertRule(
        name='cpu_usage_alert',
        condition=lambda ctx: ctx['metrics'].get('system_cpu_usage_percent', 0) > threshold,
        level=AlertLevel.WARNING,
        message_template=f'CPU使用率超过{threshold}%: {{metrics[system_cpu_usage_percent]:.1f}}%'
    )

def create_memory_alert_rule(threshold: float = 85.0) -> AlertRule:
    """创建内存使用率告警规则
    
    Args:
        threshold: 告警阈值
        
    Returns:
        告警规则
    """
    return AlertRule(
        name='memory_usage_alert',
        condition=lambda ctx: ctx['metrics'].get('system_memory_usage_percent', 0) > threshold,
        level=AlertLevel.WARNING,
        message_template=f'内存使用率超过{threshold}%: {{metrics[system_memory_usage_percent]:.1f}}%'
    )