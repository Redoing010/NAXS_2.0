#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控模块

提供系统性能监控、错误追踪、健康检查等功能
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'active_connections': self.active_connections
        }


@dataclass
class ErrorRecord:
    """错误记录"""
    timestamp: datetime
    level: str
    module: str
    message: str
    exception_type: str = ""
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'module': self.module,
            'message': self.message,
            'exception_type': self.exception_type,
            'stack_trace': self.stack_trace,
            'context': self.context
        }


@dataclass
class HealthStatus:
    """健康状态"""
    is_healthy: bool
    status: str  # 'healthy', 'warning', 'critical'
    checks: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_healthy': self.is_healthy,
            'status': self.status,
            'checks': self.checks,
            'issues': self.issues,
            'last_check': self.last_check.isoformat()
        }


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, 
                 metrics_retention_hours: int = 24,
                 error_retention_hours: int = 72,
                 check_interval_seconds: int = 60):
        self.metrics_retention_hours = metrics_retention_hours
        self.error_retention_hours = error_retention_hours
        self.check_interval_seconds = check_interval_seconds
        
        # 数据存储
        self.metrics_history: deque = deque(maxlen=int(metrics_retention_hours * 3600 / check_interval_seconds))
        self.error_history: deque = deque(maxlen=1000)  # 最多保存1000条错误记录
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.alert_handlers: List[Callable[[str, Dict], None]] = []
        
        # 阈值配置
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'disk_free_gb': 1.0,
            'error_rate_per_hour': 100
        }
        
        # 监控状态
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_network_stats = None
        
        logger.info("系统监控器初始化完成")
    
    def start_monitoring(self):
        """开始监控"""
        if self._monitoring:
            logger.warning("监控已在运行中")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"系统监控已启动，检查间隔: {self.check_interval_seconds}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 检查阈值
                self._check_thresholds(metrics)
                
                # 清理过期数据
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                self.record_error("SystemMonitor", f"监控循环异常: {e}", exception=e)
            
            time.sleep(self.check_interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / 1024 / 1024
        memory_available_mb = memory.available / 1024 / 1024
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # 网络统计
        network_stats = psutil.net_io_counters()
        network_bytes_sent = network_stats.bytes_sent
        network_bytes_recv = network_stats.bytes_recv
        
        # 活跃连接数
        try:
            active_connections = len(psutil.net_connections())
        except (psutil.AccessDenied, OSError):
            active_connections = 0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_connections=active_connections
        )
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """检查阈值并触发告警"""
        alerts = []
        
        # CPU使用率检查
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        # 内存使用率检查
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        # 磁盘使用率检查
        if metrics.disk_usage_percent > self.thresholds['disk_usage_percent']:
            alerts.append(f"磁盘使用率过高: {metrics.disk_usage_percent:.1f}%")
        
        # 磁盘剩余空间检查
        if metrics.disk_free_gb < self.thresholds['disk_free_gb']:
            alerts.append(f"磁盘剩余空间不足: {metrics.disk_free_gb:.1f}GB")
        
        # 错误率检查
        error_rate = self._calculate_error_rate()
        if error_rate > self.thresholds['error_rate_per_hour']:
            alerts.append(f"错误率过高: {error_rate:.1f}/小时")
        
        # 发送告警
        for alert in alerts:
            self._send_alert("threshold_exceeded", {
                'message': alert,
                'metrics': metrics.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
    
    def _calculate_error_rate(self) -> float:
        """计算过去1小时的错误率"""
        if not self.error_history:
            return 0.0
        
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > one_hour_ago
        ]
        
        return len(recent_errors)
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        # 清理过期错误记录
        cutoff_time = datetime.now() - timedelta(hours=self.error_retention_hours)
        while self.error_history and self.error_history[0].timestamp < cutoff_time:
            self.error_history.popleft()
    
    def record_error(self, module: str, message: str, 
                    level: str = "ERROR", exception: Exception = None, 
                    context: Dict[str, Any] = None):
        """记录错误"""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            level=level,
            module=module,
            message=message,
            exception_type=type(exception).__name__ if exception else "",
            stack_trace=str(exception) if exception else "",
            context=context or {}
        )
        
        self.error_history.append(error_record)
        
        # 如果是严重错误，立即发送告警
        if level in ["ERROR", "CRITICAL"]:
            self._send_alert("error_occurred", {
                'error': error_record.to_dict()
            })
    
    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """添加健康检查"""
        self.health_checks[name] = check_func
        logger.info(f"添加健康检查: {name}")
    
    def run_health_checks(self) -> HealthStatus:
        """运行健康检查"""
        checks = {}
        issues = []
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                checks[name] = result
                if not result:
                    issues.append(f"健康检查失败: {name}")
            except Exception as e:
                checks[name] = False
                issues.append(f"健康检查异常: {name} - {e}")
                logger.error(f"健康检查 {name} 异常: {e}")
        
        # 确定整体健康状态
        failed_checks = sum(1 for result in checks.values() if not result)
        total_checks = len(checks)
        
        if failed_checks == 0:
            status = "healthy"
            is_healthy = True
        elif failed_checks <= total_checks * 0.3:  # 30%以下失败为警告
            status = "warning"
            is_healthy = True
        else:
            status = "critical"
            is_healthy = False
        
        return HealthStatus(
            is_healthy=is_healthy,
            status=status,
            checks=checks,
            issues=issues,
            last_check=datetime.now()
        )
    
    def add_alert_handler(self, handler: Callable[[str, Dict], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
        logger.info("添加告警处理器")
    
    def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """发送告警"""
        for handler in self.alert_handlers:
            try:
                handler(alert_type, data)
            except Exception as e:
                logger.error(f"告警处理器异常: {e}")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """获取指定时间范围内的指标历史"""
        if not self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_error_history(self, hours: int = 1) -> List[ErrorRecord]:
        """获取指定时间范围内的错误历史"""
        if not self.error_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        current_metrics = self.get_current_metrics()
        health_status = self.run_health_checks()
        recent_errors = self.get_error_history(hours=1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'health_status': health_status.to_dict(),
            'error_count_last_hour': len(recent_errors),
            'monitoring_active': self._monitoring,
            'thresholds': self.thresholds
        }
    
    def export_metrics(self, file_path: str, hours: int = 24):
        """导出指标数据"""
        metrics_data = [
            metrics.to_dict() 
            for metrics in self.get_metrics_history(hours=hours)
        ]
        
        error_data = [
            error.to_dict() 
            for error in self.get_error_history(hours=hours)
        ]
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'time_range_hours': hours,
            'metrics': metrics_data,
            'errors': error_data,
            'summary': self.get_system_summary()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"监控数据已导出到: {file_path}")


# 全局监控器实例
_monitor_instance = None


def get_system_monitor() -> SystemMonitor:
    """获取系统监控器单例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


# 便捷函数
def start_monitoring():
    """启动系统监控"""
    get_system_monitor().start_monitoring()


def stop_monitoring():
    """停止系统监控"""
    get_system_monitor().stop_monitoring()


def record_error(module: str, message: str, level: str = "ERROR", 
                exception: Exception = None, context: Dict[str, Any] = None):
    """记录错误"""
    get_system_monitor().record_error(module, message, level, exception, context)


def get_health_status() -> HealthStatus:
    """获取健康状态"""
    return get_system_monitor().run_health_checks()


def get_system_summary() -> Dict[str, Any]:
    """获取系统摘要"""
    return get_system_monitor().get_system_summary()