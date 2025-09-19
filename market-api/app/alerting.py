import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import structlog
from .config import settings
from .monitoring import performance_monitor
from .database import db_manager
from .cache import cache_manager

logger = structlog.get_logger(__name__)

class AlertSeverity(Enum):
    """告警严重级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class Alert:
    """告警对象"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'metadata': self.metadata,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by
        }

@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    enabled: bool = True
    cooldown: int = 300  # 5分钟冷却时间
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self) -> bool:
        """检查是否应该触发告警"""
        if not self.enabled:
            return False
        
        # 检查冷却时间
        if self.last_triggered:
            if datetime.now() - self.last_triggered < timedelta(seconds=self.cooldown):
                return False
        
        # 检查条件
        try:
            return self.condition()
        except Exception as e:
            logger.error(f"Alert rule {self.id} condition check failed: {e}")
            return False

class NotificationChannel:
    """通知渠道基类"""
    
    async def send(self, alert: Alert) -> bool:
        """发送通知"""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """邮件通知渠道"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
    
    async def send(self, alert: Alert) -> bool:
        """发送邮件通知"""
        try:
            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = settings.ADMIN_EMAIL if hasattr(settings, 'ADMIN_EMAIL') else 'admin@example.com'
            msg['Subject'] = f"[NAXS Alert] {alert.severity.value.upper()}: {alert.title}"
            
            # 邮件正文
            body = f"""
            告警详情:
            
            标题: {alert.title}
            描述: {alert.description}
            严重级别: {alert.severity.value}
            来源: {alert.source}
            时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            元数据:
            {json.dumps(alert.metadata, indent=2, ensure_ascii=False)}
            
            请及时处理此告警。
            
            NAXS 智能投研系统
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification for alert {alert.id}: {e}")
            return False

class WebhookNotificationChannel(NotificationChannel):
    """Webhook通知渠道"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send(self, alert: Alert) -> bool:
        """发送Webhook通知"""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'NAXS-API'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed for alert {alert.id}: HTTP {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook notification for alert {alert.id}: {e}")
            return False

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.notification_channels: List[NotificationChannel] = []
        self.running = False
        self.check_interval = 30  # 30秒检查一次
        self._task: Optional[asyncio.Task] = None
        
        # 初始化默认规则
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认告警规则"""
        
        # CPU使用率告警
        self.add_rule(AlertRule(
            id="high_cpu_usage",
            name="CPU使用率过高",
            description="系统CPU使用率超过80%",
            condition=lambda: self._check_cpu_usage() > 80,
            severity=AlertSeverity.HIGH,
            cooldown=300
        ))
        
        # 内存使用率告警
        self.add_rule(AlertRule(
            id="high_memory_usage",
            name="内存使用率过高",
            description="系统内存使用率超过85%",
            condition=lambda: self._check_memory_usage() > 85,
            severity=AlertSeverity.HIGH,
            cooldown=300
        ))
        
        # 数据库连接告警
        self.add_rule(AlertRule(
            id="database_connection_failed",
            name="数据库连接失败",
            description="数据库健康检查失败",
            condition=lambda: not db_manager.health_check(),
            severity=AlertSeverity.CRITICAL,
            cooldown=60
        ))
        
        # 缓存连接告警
        self.add_rule(AlertRule(
            id="cache_connection_failed",
            name="缓存连接失败",
            description="Redis缓存连接失败",
            condition=lambda: not self._check_cache_health(),
            severity=AlertSeverity.MEDIUM,
            cooldown=120
        ))
        
        # API响应时间告警
        self.add_rule(AlertRule(
            id="slow_api_response",
            name="API响应时间过长",
            description="API平均响应时间超过2秒",
            condition=lambda: self._check_api_response_time() > 2.0,
            severity=AlertSeverity.MEDIUM,
            cooldown=180
        ))
        
        # 错误率告警
        self.add_rule(AlertRule(
            id="high_error_rate",
            name="错误率过高",
            description="API错误率超过5%",
            condition=lambda: self._check_error_rate() > 0.05,
            severity=AlertSeverity.HIGH,
            cooldown=300
        ))
    
    def _check_cpu_usage(self) -> float:
        """检查CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent()  # 移除interval=1，避免阻塞
        except Exception:
            return 0.0
    
    def _check_memory_usage(self) -> float:
        """检查内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0
    
    def _check_cache_health(self) -> bool:
        """检查缓存健康状态"""
        try:
            # 这里需要异步调用，简化处理
            return True  # 实际应该调用 cache_manager.health_check()
        except Exception:
            return False
    
    def _check_api_response_time(self) -> float:
        """检查API响应时间"""
        try:
            health_status = performance_monitor.get_health_status()
            # 假设有平均响应时间字段
            return health_status.get('avg_response_time', 0.0)
        except Exception:
            return 0.0
    
    def _check_error_rate(self) -> float:
        """检查错误率"""
        try:
            # 从性能监控获取错误率
            # 这里需要实现具体的错误率计算逻辑
            return 0.0
        except Exception:
            return 0.0
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.id] = rule
        logger.info(f"Alert rule added: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Alert rule removed: {rule_id}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        self.notification_channels.append(channel)
        logger.info(f"Notification channel added: {type(channel).__name__}")
    
    async def create_alert(self, title: str, description: str, severity: AlertSeverity, 
                          source: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """创建告警"""
        alert_id = f"{source}_{int(time.time() * 1000)}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        # 发送通知
        await self._send_notifications(alert)
        
        logger.warning(f"Alert created: {alert.title} (ID: {alert_id})")
        return alert
    
    async def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None):
        """解决告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            logger.info(f"Alert resolved: {alert.title} (ID: {alert_id})")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert.title} (ID: {alert_id}) by {acknowledged_by}")
    
    async def _send_notifications(self, alert: Alert):
        """发送通知"""
        for channel in self.notification_channels:
            try:
                await channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send notification via {type(channel).__name__}: {e}")
    
    async def _check_rules(self):
        """检查告警规则"""
        for rule in self.rules.values():
            try:
                if rule.should_trigger():
                    # 创建告警
                    await self.create_alert(
                        title=rule.name,
                        description=rule.description,
                        severity=rule.severity,
                        source=f"rule:{rule.id}",
                        metadata={'rule_id': rule.id}
                    )
                    
                    # 更新最后触发时间
                    rule.last_triggered = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule.id}: {e}")
    
    async def start(self):
        """启动告警管理器"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Alert manager started")
    
    async def stop(self):
        """停止告警管理器"""
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert manager stopped")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                await self._check_rules()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(5)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        alerts = list(self.alerts.values())
        
        stats = {
            'total_alerts': len(alerts),
            'active_alerts': len([a for a in alerts if a.status == AlertStatus.ACTIVE]),
            'resolved_alerts': len([a for a in alerts if a.status == AlertStatus.RESOLVED]),
            'acknowledged_alerts': len([a for a in alerts if a.status == AlertStatus.ACKNOWLEDGED]),
            'by_severity': {
                'critical': len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in alerts if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in alerts if a.severity == AlertSeverity.MEDIUM]),
                'low': len([a for a in alerts if a.severity == AlertSeverity.LOW])
            },
            'rules_count': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
            'notification_channels': len(self.notification_channels)
        }
        
        return stats

# 全局告警管理器实例
alert_manager = AlertManager()

# 初始化函数
async def init_alerting():
    """初始化告警系统"""
    # 添加通知渠道（根据配置）
    if hasattr(settings, 'SMTP_SERVER') and settings.SMTP_SERVER:
        email_channel = EmailNotificationChannel(
            smtp_server=settings.SMTP_SERVER,
            smtp_port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD,
            from_email=settings.SMTP_FROM_EMAIL
        )
        alert_manager.add_notification_channel(email_channel)
    
    if hasattr(settings, 'WEBHOOK_URL') and settings.WEBHOOK_URL:
        webhook_channel = WebhookNotificationChannel(
            webhook_url=settings.WEBHOOK_URL,
            headers={'Content-Type': 'application/json'}
        )
        alert_manager.add_notification_channel(webhook_channel)
    
    # 启动告警管理器
    await alert_manager.start()
    logger.info("Alerting system initialized")

# 清理函数
async def cleanup_alerting():
    """清理告警系统"""
    await alert_manager.stop()
    logger.info("Alerting system cleaned up")