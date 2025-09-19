import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from circuitbreaker import circuit
import structlog
from .config import settings
from .monitoring import performance_monitor, memory_profiler

logger = structlog.get_logger(__name__)

class RateLimiter:
    """高级限流器"""
    
    def __init__(self):
        self.limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[f"{settings.RATE_LIMIT_REQUESTS}/minute"]
        )
        
        # 自定义限流存储
        self.custom_limits = {}
        self.request_counts = {}
        self.last_reset = time.time()
    
    def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """检查是否触发限流"""
        current_time = time.time()
        
        # 重置窗口
        if current_time - self.last_reset > window:
            self.request_counts.clear()
            self.last_reset = current_time
        
        # 检查请求计数
        count = self.request_counts.get(key, 0)
        if count >= limit:
            return True
        
        # 增加计数
        self.request_counts[key] = count + 1
        return False
    
    def get_rate_limit_info(self, key: str) -> Dict[str, Any]:
        """获取限流信息"""
        count = self.request_counts.get(key, 0)
        remaining = max(0, settings.RATE_LIMIT_REQUESTS - count)
        
        return {
            'limit': settings.RATE_LIMIT_REQUESTS,
            'remaining': remaining,
            'reset_time': self.last_reset + settings.RATE_LIMIT_WINDOW
        }

class CircuitBreakerManager:
    """熔断器管理器"""
    
    def __init__(self):
        self.breakers = {}
        self.stats = {
            'total_calls': 0,
            'failed_calls': 0,
            'circuit_open_count': 0,
            'circuit_half_open_count': 0
        }
    
    def get_circuit_breaker(self, name: str):
        """获取或创建熔断器"""
        if name not in self.breakers:
            @circuit(
                failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                expected_exception=Exception
            )
            def breaker_func(func, *args, **kwargs):
                return func(*args, **kwargs)
            
            self.breakers[name] = breaker_func
        
        return self.breakers[name]
    
    def execute_with_breaker(self, name: str, func: Callable, *args, **kwargs):
        """使用熔断器执行函数"""
        breaker = self.get_circuit_breaker(name)
        
        try:
            self.stats['total_calls'] += 1
            result = breaker(func, *args, **kwargs)
            return result
        except Exception as e:
            self.stats['failed_calls'] += 1
            
            # 检查熔断器状态
            if hasattr(breaker, '_circuit_breaker'):
                state = breaker._circuit_breaker.current_state
                if state == 'open':
                    self.stats['circuit_open_count'] += 1
                elif state == 'half-open':
                    self.stats['circuit_half_open_count'] += 1
            
            raise
    
    def get_breaker_stats(self) -> Dict[str, Any]:
        """获取熔断器统计信息"""
        return {
            **self.stats,
            'active_breakers': len(self.breakers),
            'failure_rate': (
                self.stats['failed_calls'] / self.stats['total_calls']
                if self.stats['total_calls'] > 0 else 0
            )
        }

class RequestTimeoutManager:
    """请求超时管理器"""
    
    def __init__(self):
        self.active_requests = {}
        self.timeout_stats = {
            'total_timeouts': 0,
            'avg_request_time': 0.0,
            'max_request_time': 0.0
        }
    
    async def execute_with_timeout(self, coro, timeout: float = None):
        """执行带超时的协程"""
        if timeout is None:
            timeout = settings.REQUEST_TIMEOUT
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            
            # 更新统计信息
            duration = time.time() - start_time
            self._update_stats(duration)
            
            return result
            
        except asyncio.TimeoutError:
            self.timeout_stats['total_timeouts'] += 1
            logger.warning(f"Request timeout after {timeout}s")
            raise HTTPException(
                status_code=408,
                detail="Request timeout"
            )
    
    def _update_stats(self, duration: float):
        """更新统计信息"""
        if duration > self.timeout_stats['max_request_time']:
            self.timeout_stats['max_request_time'] = duration
        
        # 计算平均请求时间
        current_avg = self.timeout_stats['avg_request_time']
        if current_avg == 0:
            self.timeout_stats['avg_request_time'] = duration
        else:
            self.timeout_stats['avg_request_time'] = (current_avg + duration) / 2

class MemoryGuard:
    """内存保护器"""
    
    def __init__(self):
        self.memory_warnings = 0
        self.memory_errors = 0
        self.last_gc_time = time.time()
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        import psutil
        import gc
        
        memory = psutil.virtual_memory()
        
        if memory.percent > settings.MAX_MEMORY_USAGE * 100:
            self.memory_warnings += 1
            
            # 触发垃圾回收
            if time.time() - self.last_gc_time > 60:  # 1分钟内最多触发一次
                collected = gc.collect()
                self.last_gc_time = time.time()
                
                logger.warning(
                    f"High memory usage: {memory.percent:.1f}%. "
                    f"Triggered GC, collected {collected} objects."
                )
            
            # 如果内存使用率过高，拒绝新请求
            if memory.percent > 95:
                self.memory_errors += 1
                return False
        
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        import psutil
        
        memory = psutil.virtual_memory()
        
        return {
            'memory_percent': memory.percent,
            'memory_warnings': self.memory_warnings,
            'memory_errors': self.memory_errors,
            'last_gc_time': self.last_gc_time
        }

# 全局实例
rate_limiter = RateLimiter()
circuit_breaker_manager = CircuitBreakerManager()
request_timeout_manager = RequestTimeoutManager()
memory_guard = MemoryGuard()

# 中间件函数
async def rate_limit_middleware(request: Request, call_next):
    """限流中间件"""
    client_ip = get_remote_address(request)
    
    # 检查限流
    if rate_limiter.is_rate_limited(
        client_ip, 
        settings.RATE_LIMIT_REQUESTS, 
        settings.RATE_LIMIT_WINDOW
    ):
        rate_info = rate_limiter.get_rate_limit_info(client_ip)
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "limit": rate_info['limit'],
                "remaining": rate_info['remaining'],
                "reset_time": rate_info['reset_time']
            },
            headers={
                "X-RateLimit-Limit": str(rate_info['limit']),
                "X-RateLimit-Remaining": str(rate_info['remaining']),
                "X-RateLimit-Reset": str(int(rate_info['reset_time']))
            }
        )
    
    response = await call_next(request)
    
    # 添加限流头部
    rate_info = rate_limiter.get_rate_limit_info(client_ip)
    response.headers["X-RateLimit-Limit"] = str(rate_info['limit'])
    response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining'])
    response.headers["X-RateLimit-Reset"] = str(int(rate_info['reset_time']))
    
    return response

async def memory_guard_middleware(request: Request, call_next):
    """内存保护中间件"""
    # 检查内存使用情况
    if not memory_guard.check_memory_usage():
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable due to high memory usage",
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )
    
    # 采样内存使用情况
    memory_profiler.sample_memory()
    
    response = await call_next(request)
    return response

async def timeout_middleware(request: Request, call_next):
    """超时中间件"""
    try:
        response = await request_timeout_manager.execute_with_timeout(
            call_next(request),
            timeout=settings.REQUEST_TIMEOUT
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def security_headers_middleware(request: Request, call_next):
    """安全头部中间件"""
    response = await call_next(request)
    
    # 添加安全头部
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

# 异常处理器
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """限流异常处理器"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail)
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # 记录错误指标
    performance_monitor.record_error(
        error_type=type(exc).__name__,
        endpoint=request.url.path
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred" if not settings.DEBUG else str(exc)
        }
    )

# 中间件配置函数
def setup_middleware(app):
    """设置中间件"""
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    
    # 添加中间件（注意顺序）
    app.middleware("http")(security_headers_middleware)
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(memory_guard_middleware)
    app.middleware("http")(timeout_middleware)  # 重新启用超时中间件
    
    # 添加内置中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"] if settings.DEBUG else ["localhost", "127.0.0.1"]
    )
    
    # 添加异常处理器
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Middleware setup completed")