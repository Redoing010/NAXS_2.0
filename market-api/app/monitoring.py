import time
import psutil
import asyncio
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import Request, Response
import structlog
from .config import settings

logger = structlog.get_logger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        # Prometheus指标注册表
        self.registry = CollectorRegistry()
        
        # 请求指标
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # 系统指标
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        # 数据库指标
        self.db_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type'],
            registry=self.registry
        )
        
        # 缓存指标
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # 业务指标
        self.api_errors = Counter(
            'api_errors_total',
            'Total API errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        
        self.concurrent_requests = Gauge(
            'concurrent_requests',
            'Current number of concurrent requests',
            registry=self.registry
        )
        
        # 内部状态
        self._current_requests = 0
        self._start_time = time.time()
        
        # 系统监控任务
        self._monitor_task = None
    
    async def start_monitoring(self):
        """启动系统监控"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._system_monitor_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """停止系统监控"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Performance monitoring stopped")
    
    async def _system_monitor_loop(self):
        """系统监控循环"""
        while True:
            try:
                # 更新系统指标
                self.cpu_usage.set(psutil.cpu_percent())
                
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                self.memory_usage_percent.set(memory.percent)
                
                # 等待下一次监控
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)
    
    def record_request_start(self, request: Request):
        """记录请求开始"""
        self._current_requests += 1
        self.concurrent_requests.set(self._current_requests)
        
        # 在请求状态中存储开始时间
        request.state.start_time = time.time()
    
    def record_request_end(self, request: Request, response: Response):
        """记录请求结束"""
        self._current_requests -= 1
        self.concurrent_requests.set(self._current_requests)
        
        # 计算请求持续时间
        if hasattr(request.state, 'start_time'):
            duration = time.time() - request.state.start_time
            
            # 记录指标
            method = request.method
            endpoint = request.url.path
            status_code = str(response.status_code)
            
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    def record_error(self, error_type: str, endpoint: str):
        """记录API错误"""
        self.api_errors.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()
    
    def record_db_query(self, query_type: str, duration: float):
        """记录数据库查询"""
        self.db_query_duration.labels(query_type=query_type).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """记录缓存命中"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中"""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def update_db_connections(self, active_count: int):
        """更新数据库连接数"""
        self.db_connections_active.set(active_count)
    
    def get_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # 检查系统健康状态
        is_healthy = (
            memory.percent < settings.MAX_MEMORY_USAGE * 100 and
            cpu_percent < 90 and
            self._current_requests < settings.WORKER_CONNECTIONS * 0.8
        )
        
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'uptime': time.time() - self._start_time,
            'cpu_usage': cpu_percent,
            'memory_usage': {
                'used_bytes': memory.used,
                'available_bytes': memory.available,
                'percent': memory.percent
            },
            'concurrent_requests': self._current_requests,
            'timestamp': time.time()
        }

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self):
        # 配置structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # 配置标准库logging
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, settings.LOG_LEVEL.upper()),
        )
    
    def get_logger(self, name: str):
        """获取结构化日志记录器"""
        return structlog.get_logger(name)

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.peak_memory = 0
        self.memory_samples = []
        self.gc_stats = {
            'collections': 0,
            'collected': 0,
            'uncollectable': 0
        }
    
    def sample_memory(self):
        """采样内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        current_memory = memory_info.rss
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # 保留最近100个样本
        self.memory_samples.append({
            'timestamp': time.time(),
            'rss': memory_info.rss,
            'vms': memory_info.vms
        })
        
        if len(self.memory_samples) > 100:
            self.memory_samples.pop(0)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.memory_samples:
            return {}
        
        recent_memory = [s['rss'] for s in self.memory_samples[-10:]]
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        return {
            'current_rss': self.memory_samples[-1]['rss'] if self.memory_samples else 0,
            'peak_rss': self.peak_memory,
            'average_rss': avg_memory,
            'sample_count': len(self.memory_samples),
            'gc_stats': self.gc_stats
        }

# 全局监控实例
performance_monitor = PerformanceMonitor()
structured_logger = StructuredLogger()
memory_profiler = MemoryProfiler()

# 中间件函数
async def monitoring_middleware(request: Request, call_next):
    """监控中间件"""
    # 记录请求开始
    performance_monitor.record_request_start(request)
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 记录请求结束
        performance_monitor.record_request_end(request, response)
        
        return response
        
    except Exception as e:
        # 记录错误
        performance_monitor.record_error(
            error_type=type(e).__name__,
            endpoint=request.url.path
        )
        raise

# 初始化函数
async def init_monitoring():
    """初始化监控"""
    await performance_monitor.start_monitoring()
    logger.info("Monitoring initialized")

# 清理函数
async def cleanup_monitoring():
    """清理监控"""
    await performance_monitor.stop_monitoring()
    logger.info("Monitoring cleaned up")