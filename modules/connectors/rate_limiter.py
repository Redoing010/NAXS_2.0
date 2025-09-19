# 频率限制器
# 控制API调用频率，避免触发数据源的限流机制

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict

class RateLimitAlgorithm(Enum):
    """限流算法类型"""
    TOKEN_BUCKET = "token_bucket"        # 令牌桶算法
    LEAKY_BUCKET = "leaky_bucket"        # 漏桶算法
    FIXED_WINDOW = "fixed_window"        # 固定窗口算法
    SLIDING_WINDOW = "sliding_window"    # 滑动窗口算法
    ADAPTIVE = "adaptive"                # 自适应算法

@dataclass
class RateLimit:
    """频率限制配置"""
    requests_per_second: float = 1.0     # 每秒请求数
    requests_per_minute: int = 60        # 每分钟请求数
    requests_per_hour: int = 3600        # 每小时请求数
    requests_per_day: int = 86400        # 每天请求数
    burst_size: int = 10                 # 突发大小
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    
    # 自适应参数
    adaptive_factor: float = 0.8         # 自适应因子
    error_threshold: float = 0.1         # 错误阈值
    recovery_time: int = 300             # 恢复时间（秒）
    
    # 其他配置
    enabled: bool = True
    strict_mode: bool = False            # 严格模式
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitStatus:
    """频率限制状态"""
    remaining_requests: int = 0          # 剩余请求数
    reset_time: Optional[datetime] = None # 重置时间
    retry_after: Optional[int] = None    # 重试等待时间（秒）
    is_limited: bool = False             # 是否被限制
    current_rate: float = 0.0            # 当前速率
    error_rate: float = 0.0              # 错误率
    last_request_time: Optional[datetime] = None

class TokenBucket:
    """令牌桶算法实现"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate                 # 令牌生成速率（每秒）
        self.capacity = capacity         # 桶容量
        self.tokens = capacity           # 当前令牌数
        self.last_refill = time.time()   # 上次填充时间
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """获取令牌"""
        async with self.lock:
            now = time.time()
            
            # 计算需要添加的令牌数
            time_passed = now - self.last_refill
            tokens_to_add = time_passed * self.rate
            
            # 更新令牌数，不超过容量
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # 检查是否有足够的令牌
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, tokens: int = 1) -> float:
        """等待令牌可用，返回等待时间"""
        if await self.acquire(tokens):
            return 0.0
        
        # 计算等待时间
        needed_tokens = tokens - self.tokens
        wait_time = needed_tokens / self.rate
        
        return wait_time
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'tokens': self.tokens,
            'capacity': self.capacity,
            'rate': self.rate,
            'last_refill': self.last_refill,
        }

class SlidingWindow:
    """滑动窗口算法实现"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size   # 窗口大小（秒）
        self.max_requests = max_requests # 最大请求数
        self.requests = deque()          # 请求时间戳队列
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """获取许可"""
        async with self.lock:
            now = time.time()
            
            # 清理过期的请求记录
            while self.requests and now - self.requests[0] > self.window_size:
                self.requests.popleft()
            
            # 检查是否超过限制
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    async def wait_time(self) -> float:
        """计算等待时间"""
        if await self.acquire():
            return 0.0
        
        # 计算最早请求的过期时间
        if self.requests:
            oldest_request = self.requests[0]
            wait_time = self.window_size - (time.time() - oldest_request)
            return max(0, wait_time)
        
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        now = time.time()
        recent_requests = [req for req in self.requests if now - req <= self.window_size]
        
        return {
            'current_requests': len(recent_requests),
            'max_requests': self.max_requests,
            'window_size': self.window_size,
            'remaining_requests': max(0, self.max_requests - len(recent_requests)),
        }

class AdaptiveRateLimiter:
    """自适应频率限制器"""
    
    def __init__(self, base_rate: float, adaptive_factor: float = 0.8):
        self.base_rate = base_rate           # 基础速率
        self.current_rate = base_rate        # 当前速率
        self.adaptive_factor = adaptive_factor # 自适应因子
        self.error_count = 0                 # 错误计数
        self.success_count = 0               # 成功计数
        self.last_adjustment = time.time()   # 上次调整时间
        self.adjustment_interval = 60        # 调整间隔（秒）
        
        # 使用令牌桶实现
        self.token_bucket = TokenBucket(self.current_rate, int(self.current_rate * 2))
    
    async def acquire(self) -> bool:
        """获取许可"""
        return await self.token_bucket.acquire()
    
    async def report_success(self):
        """报告成功"""
        self.success_count += 1
        await self._adjust_rate()
    
    async def report_error(self):
        """报告错误"""
        self.error_count += 1
        await self._adjust_rate()
    
    async def _adjust_rate(self):
        """调整速率"""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        total_requests = self.success_count + self.error_count
        if total_requests == 0:
            return
        
        error_rate = self.error_count / total_requests
        
        # 根据错误率调整速率
        if error_rate > 0.1:  # 错误率过高，降低速率
            self.current_rate *= self.adaptive_factor
        elif error_rate < 0.05:  # 错误率较低，可以提高速率
            self.current_rate = min(self.base_rate, self.current_rate * 1.1)
        
        # 更新令牌桶
        self.token_bucket.rate = self.current_rate
        self.token_bucket.capacity = int(self.current_rate * 2)
        
        # 重置计数器
        self.error_count = 0
        self.success_count = 0
        self.last_adjustment = now
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        total_requests = self.success_count + self.error_count
        error_rate = self.error_count / total_requests if total_requests > 0 else 0
        
        return {
            'base_rate': self.base_rate,
            'current_rate': self.current_rate,
            'error_rate': error_rate,
            'success_count': self.success_count,
            'error_count': self.error_count,
            **self.token_bucket.get_status()
        }

class RateLimiter:
    """频率限制器管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger("RateLimiter")
        self.limiters: Dict[str, Any] = {}  # 限制器实例
        self.configs: Dict[str, RateLimit] = {}  # 配置
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_wait_time': 0.0,
            'last_request_time': None,
        })
    
    def add_limit(self, name: str, rate_limit: RateLimit):
        """添加频率限制"""
        self.configs[name] = rate_limit
        
        # 根据算法创建限制器
        if rate_limit.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            self.limiters[name] = TokenBucket(
                rate=rate_limit.requests_per_second,
                capacity=rate_limit.burst_size
            )
        elif rate_limit.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.limiters[name] = SlidingWindow(
                window_size=60,  # 1分钟窗口
                max_requests=rate_limit.requests_per_minute
            )
        elif rate_limit.algorithm == RateLimitAlgorithm.ADAPTIVE:
            self.limiters[name] = AdaptiveRateLimiter(
                base_rate=rate_limit.requests_per_second,
                adaptive_factor=rate_limit.adaptive_factor
            )
        else:
            # 默认使用令牌桶
            self.limiters[name] = TokenBucket(
                rate=rate_limit.requests_per_second,
                capacity=rate_limit.burst_size
            )
        
        self.logger.info(f"Added rate limit for {name}: {rate_limit.requests_per_second} req/s")
    
    def remove_limit(self, name: str) -> bool:
        """移除频率限制"""
        if name in self.limiters:
            del self.limiters[name]
            del self.configs[name]
            if name in self.stats:
                del self.stats[name]
            self.logger.info(f"Removed rate limit for {name}")
            return True
        return False
    
    async def acquire(self, name: str, tokens: int = 1) -> bool:
        """获取许可"""
        if name not in self.limiters:
            return True  # 没有限制，直接通过
        
        config = self.configs[name]
        if not config.enabled:
            return True
        
        limiter = self.limiters[name]
        self.stats[name]['total_requests'] += 1
        self.stats[name]['last_request_time'] = datetime.now()
        
        # 尝试获取许可
        if hasattr(limiter, 'acquire'):
            if isinstance(limiter, TokenBucket):
                acquired = await limiter.acquire(tokens)
            else:
                acquired = await limiter.acquire()
        else:
            acquired = True
        
        if not acquired:
            self.stats[name]['blocked_requests'] += 1
            self.logger.debug(f"Rate limit exceeded for {name}")
        
        return acquired
    
    async def wait_for_permit(self, name: str, tokens: int = 1) -> float:
        """等待许可，返回等待时间"""
        if name not in self.limiters:
            return 0.0
        
        config = self.configs[name]
        if not config.enabled:
            return 0.0
        
        limiter = self.limiters[name]
        
        # 计算等待时间
        if hasattr(limiter, 'wait_for_token'):
            wait_time = await limiter.wait_for_token(tokens)
        elif hasattr(limiter, 'wait_time'):
            wait_time = await limiter.wait_time()
        else:
            wait_time = 0.0
        
        if wait_time > 0:
            self.stats[name]['total_wait_time'] += wait_time
            self.logger.debug(f"Waiting {wait_time:.2f}s for rate limit {name}")
            await asyncio.sleep(wait_time)
        
        return wait_time
    
    async def acquire_or_wait(self, name: str, tokens: int = 1) -> bool:
        """获取许可或等待"""
        if await self.acquire(name, tokens):
            return True
        
        # 等待许可
        await self.wait_for_permit(name, tokens)
        return True
    
    async def report_success(self, name: str):
        """报告成功（用于自适应算法）"""
        if name in self.limiters:
            limiter = self.limiters[name]
            if hasattr(limiter, 'report_success'):
                await limiter.report_success()
    
    async def report_error(self, name: str):
        """报告错误（用于自适应算法）"""
        if name in self.limiters:
            limiter = self.limiters[name]
            if hasattr(limiter, 'report_error'):
                await limiter.report_error()
    
    def get_status(self, name: str) -> Optional[RateLimitStatus]:
        """获取限制器状态"""
        if name not in self.limiters:
            return None
        
        limiter = self.limiters[name]
        config = self.configs[name]
        stats = self.stats[name]
        
        # 获取限制器特定状态
        limiter_status = {}
        if hasattr(limiter, 'get_status'):
            limiter_status = limiter.get_status()
        
        # 计算剩余请求数
        remaining_requests = 0
        if 'remaining_requests' in limiter_status:
            remaining_requests = limiter_status['remaining_requests']
        elif 'tokens' in limiter_status:
            remaining_requests = int(limiter_status['tokens'])
        
        # 计算当前速率
        current_rate = config.requests_per_second
        if 'current_rate' in limiter_status:
            current_rate = limiter_status['current_rate']
        
        # 计算错误率
        error_rate = 0.0
        if 'error_rate' in limiter_status:
            error_rate = limiter_status['error_rate']
        elif stats['total_requests'] > 0:
            error_rate = stats['blocked_requests'] / stats['total_requests']
        
        return RateLimitStatus(
            remaining_requests=remaining_requests,
            reset_time=None,  # 可以根据算法计算
            retry_after=None,
            is_limited=remaining_requests <= 0,
            current_rate=current_rate,
            error_rate=error_rate,
            last_request_time=stats['last_request_time']
        )
    
    def get_all_status(self) -> Dict[str, RateLimitStatus]:
        """获取所有限制器状态"""
        status = {}
        for name in self.limiters:
            status[name] = self.get_status(name)
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return dict(self.stats)
    
    def update_config(self, name: str, rate_limit: RateLimit):
        """更新配置"""
        if name in self.configs:
            self.remove_limit(name)
        self.add_limit(name, rate_limit)
    
    def reset_stats(self, name: Optional[str] = None):
        """重置统计信息"""
        if name:
            if name in self.stats:
                self.stats[name] = {
                    'total_requests': 0,
                    'blocked_requests': 0,
                    'total_wait_time': 0.0,
                    'last_request_time': None,
                }
        else:
            self.stats.clear()

# 便捷函数
def create_rate_limit(
    requests_per_second: float = 1.0,
    burst_size: int = 10,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    **kwargs
) -> RateLimit:
    """创建频率限制配置"""
    return RateLimit(
        requests_per_second=requests_per_second,
        burst_size=burst_size,
        algorithm=algorithm,
        **kwargs
    )

def create_rate_limiter() -> RateLimiter:
    """创建频率限制器"""
    return RateLimiter()

# 装饰器
def rate_limited(name: str, limiter: RateLimiter):
    """频率限制装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            await limiter.acquire_or_wait(name)
            try:
                result = await func(*args, **kwargs)
                await limiter.report_success(name)
                return result
            except Exception as e:
                await limiter.report_error(name)
                raise
        return wrapper
    return decorator