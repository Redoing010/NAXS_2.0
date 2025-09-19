# 熔断器 - 实现服务调用的熔断保护功能

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"          # 关闭状态（正常）
    OPEN = "open"              # 开启状态（熔断）
    HALF_OPEN = "half_open"    # 半开状态（试探）

class FailureType(Enum):
    """失败类型"""
    TIMEOUT = "timeout"        # 超时
    ERROR = "error"            # 错误
    EXCEPTION = "exception"    # 异常
    SLOW_CALL = "slow_call"    # 慢调用

@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    # 失败阈值配置
    failure_rate_threshold: float = 0.5  # 失败率阈值（0.0-1.0）
    slow_call_rate_threshold: float = 0.5  # 慢调用率阈值（0.0-1.0）
    slow_call_duration_threshold: float = 1.0  # 慢调用时间阈值（秒）
    
    # 最小调用次数
    minimum_number_of_calls: int = 10  # 最小调用次数
    
    # 滑动窗口配置
    sliding_window_size: int = 100  # 滑动窗口大小
    sliding_window_type: str = "count"  # 窗口类型：count（次数）或time（时间）
    
    # 等待时间配置
    wait_duration_in_open_state: float = 60.0  # 开启状态等待时间（秒）
    
    # 半开状态配置
    permitted_number_of_calls_in_half_open_state: int = 10  # 半开状态允许的调用次数
    
    # 自动转换配置
    automatic_transition_from_open_to_half_open_enabled: bool = True
    
    # 忽略的异常类型
    ignored_exceptions: List[str] = field(default_factory=list)
    
    # 记录的异常类型
    recorded_exceptions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'failure_rate_threshold': self.failure_rate_threshold,
            'slow_call_rate_threshold': self.slow_call_rate_threshold,
            'slow_call_duration_threshold': self.slow_call_duration_threshold,
            'minimum_number_of_calls': self.minimum_number_of_calls,
            'sliding_window_size': self.sliding_window_size,
            'sliding_window_type': self.sliding_window_type,
            'wait_duration_in_open_state': self.wait_duration_in_open_state,
            'permitted_number_of_calls_in_half_open_state': self.permitted_number_of_calls_in_half_open_state,
            'automatic_transition_from_open_to_half_open_enabled': self.automatic_transition_from_open_to_half_open_enabled,
            'ignored_exceptions': self.ignored_exceptions,
            'recorded_exceptions': self.recorded_exceptions
        }

@dataclass
class CallResult:
    """调用结果"""
    success: bool
    duration: float
    timestamp: float
    failure_type: Optional[FailureType] = None
    exception: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'duration': self.duration,
            'timestamp': self.timestamp,
            'failure_type': self.failure_type.value if self.failure_type else None,
            'exception': str(self.exception) if self.exception else None
        }

@dataclass
class CircuitBreakerMetrics:
    """熔断器指标"""
    state: CircuitState
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    failure_rate: float = 0.0
    slow_call_rate: float = 0.0
    last_failure_time: Optional[float] = None
    state_transition_time: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state': self.state.value,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'slow_calls': self.slow_calls,
            'failure_rate': self.failure_rate,
            'slow_call_rate': self.slow_call_rate,
            'success_rate': self.success_rate,
            'last_failure_time': self.last_failure_time,
            'state_transition_time': self.state_transition_time
        }

class SlidingWindow:
    """滑动窗口"""
    
    def __init__(self, size: int, window_type: str = "count"):
        self.size = size
        self.window_type = window_type  # "count" or "time"
        self.calls: deque = deque(maxlen=size if window_type == "count" else None)
        self.lock = threading.RLock()
    
    def add_call(self, call_result: CallResult):
        """添加调用结果"""
        with self.lock:
            if self.window_type == "time":
                # 时间窗口：移除过期的调用
                current_time = time.time()
                cutoff_time = current_time - self.size
                
                while self.calls and self.calls[0].timestamp < cutoff_time:
                    self.calls.popleft()
            
            self.calls.append(call_result)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取窗口指标"""
        with self.lock:
            if not self.calls:
                return {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'slow_calls': 0,
                    'failure_rate': 0.0,
                    'slow_call_rate': 0.0
                }
            
            total_calls = len(self.calls)
            successful_calls = sum(1 for call in self.calls if call.success)
            failed_calls = total_calls - successful_calls
            slow_calls = sum(1 for call in self.calls 
                           if call.duration > getattr(self, 'slow_call_threshold', 1.0))
            
            failure_rate = failed_calls / total_calls if total_calls > 0 else 0.0
            slow_call_rate = slow_calls / total_calls if total_calls > 0 else 0.0
            
            return {
                'total_calls': total_calls,
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'slow_calls': slow_calls,
                'failure_rate': failure_rate,
                'slow_call_rate': slow_call_rate
            }
    
    def clear(self):
        """清空窗口"""
        with self.lock:
            self.calls.clear()

class CircuitBreakerException(Exception):
    """熔断器异常"""
    
    def __init__(self, message: str, circuit_breaker_name: str, state: CircuitState):
        super().__init__(message)
        self.circuit_breaker_name = circuit_breaker_name
        self.state = state

class CircuitBreaker:
    """熔断器
    
    实现熔断器模式，提供服务调用的保护机制：
    1. 监控调用成功率和响应时间
    2. 在失败率过高时熔断服务调用
    3. 提供半开状态进行服务恢复探测
    4. 支持自定义失败判断逻辑
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # 状态管理
        self.state = CircuitState.CLOSED
        self.state_transition_time = time.time()
        
        # 滑动窗口
        self.sliding_window = SlidingWindow(
            size=self.config.sliding_window_size,
            window_type=self.config.sliding_window_type
        )
        self.sliding_window.slow_call_threshold = self.config.slow_call_duration_threshold
        
        # 半开状态计数器
        self.half_open_calls = 0
        
        # 指标
        self.metrics = CircuitBreakerMetrics(state=self.state)
        
        # 事件回调
        self.on_state_transition: Optional[Callable[[CircuitState, CircuitState], None]] = None
        self.on_call_not_permitted: Optional[Callable[[], None]] = None
        self.on_success: Optional[Callable[[float], None]] = None
        self.on_error: Optional[Callable[[Exception, float], None]] = None
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"熔断器初始化完成: {self.name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """执行被保护的调用
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            CircuitBreakerException: 熔断器开启时抛出
        """
        # 检查是否允许调用
        if not self._is_call_permitted():
            self._on_call_not_permitted()
            raise CircuitBreakerException(
                f"Circuit breaker '{self.name}' is {self.state.value}",
                self.name,
                self.state
            )
        
        # 执行调用
        start_time = time.time()
        call_result = None
        
        try:
            result = func(*args, **kwargs)
            
            # 调用成功
            duration = time.time() - start_time
            call_result = CallResult(
                success=True,
                duration=duration,
                timestamp=start_time
            )
            
            self._on_success(duration)
            return result
            
        except Exception as e:
            # 调用失败
            duration = time.time() - start_time
            failure_type = self._classify_failure(e, duration)
            
            call_result = CallResult(
                success=False,
                duration=duration,
                timestamp=start_time,
                failure_type=failure_type,
                exception=e
            )
            
            self._on_error(e, duration)
            
            # 如果异常不应该被忽略，则重新抛出
            if not self._should_ignore_exception(e):
                raise
            
        finally:
            if call_result:
                self._record_call_result(call_result)
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """执行被保护的异步调用
        
        Args:
            func: 要调用的异步函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            CircuitBreakerException: 熔断器开启时抛出
        """
        # 检查是否允许调用
        if not self._is_call_permitted():
            self._on_call_not_permitted()
            raise CircuitBreakerException(
                f"Circuit breaker '{self.name}' is {self.state.value}",
                self.name,
                self.state
            )
        
        # 执行异步调用
        start_time = time.time()
        call_result = None
        
        try:
            result = await func(*args, **kwargs)
            
            # 调用成功
            duration = time.time() - start_time
            call_result = CallResult(
                success=True,
                duration=duration,
                timestamp=start_time
            )
            
            self._on_success(duration)
            return result
            
        except Exception as e:
            # 调用失败
            duration = time.time() - start_time
            failure_type = self._classify_failure(e, duration)
            
            call_result = CallResult(
                success=False,
                duration=duration,
                timestamp=start_time,
                failure_type=failure_type,
                exception=e
            )
            
            self._on_error(e, duration)
            
            # 如果异常不应该被忽略，则重新抛出
            if not self._should_ignore_exception(e):
                raise
            
        finally:
            if call_result:
                self._record_call_result(call_result)
    
    def _is_call_permitted(self) -> bool:
        """检查是否允许调用"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # 检查是否可以转换到半开状态
                if (self.config.automatic_transition_from_open_to_half_open_enabled and
                    time.time() - self.state_transition_time >= self.config.wait_duration_in_open_state):
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                # 半开状态下允许有限次数的调用
                return self.half_open_calls < self.config.permitted_number_of_calls_in_half_open_state
            
            return False
    
    def _record_call_result(self, call_result: CallResult):
        """记录调用结果"""
        with self.lock:
            # 添加到滑动窗口
            self.sliding_window.add_call(call_result)
            
            # 更新指标
            self._update_metrics()
            
            # 检查状态转换
            self._check_state_transition(call_result)
    
    def _update_metrics(self):
        """更新指标"""
        window_metrics = self.sliding_window.get_metrics()
        
        self.metrics.total_calls = window_metrics['total_calls']
        self.metrics.successful_calls = window_metrics['successful_calls']
        self.metrics.failed_calls = window_metrics['failed_calls']
        self.metrics.slow_calls = window_metrics['slow_calls']
        self.metrics.failure_rate = window_metrics['failure_rate']
        self.metrics.slow_call_rate = window_metrics['slow_call_rate']
        
        if window_metrics['failed_calls'] > 0:
            self.metrics.last_failure_time = time.time()
    
    def _check_state_transition(self, call_result: CallResult):
        """检查状态转换"""
        if self.state == CircuitState.CLOSED:
            self._check_transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._check_transition_from_half_open(call_result)
    
    def _check_transition_to_open(self):
        """检查是否应该转换到开启状态"""
        # 检查最小调用次数
        if self.metrics.total_calls < self.config.minimum_number_of_calls:
            return
        
        # 检查失败率阈值
        if self.metrics.failure_rate >= self.config.failure_rate_threshold:
            self._transition_to_open()
            return
        
        # 检查慢调用率阈值
        if self.metrics.slow_call_rate >= self.config.slow_call_rate_threshold:
            self._transition_to_open()
            return
    
    def _check_transition_from_half_open(self, call_result: CallResult):
        """检查半开状态的转换"""
        self.half_open_calls += 1
        
        if not call_result.success:
            # 调用失败，转换到开启状态
            self._transition_to_open()
        elif self.half_open_calls >= self.config.permitted_number_of_calls_in_half_open_state:
            # 达到允许的调用次数，检查是否可以转换到关闭状态
            if self.metrics.failure_rate < self.config.failure_rate_threshold:
                self._transition_to_closed()
            else:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """转换到开启状态"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_transition_time = time.time()
        self.half_open_calls = 0
        self.metrics.state = self.state
        self.metrics.state_transition_time = self.state_transition_time
        
        logger.warning(f"熔断器开启: {self.name} ({old_state.value} -> {self.state.value})")
        
        if self.on_state_transition:
            self.on_state_transition(old_state, self.state)
    
    def _transition_to_half_open(self):
        """转换到半开状态"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_transition_time = time.time()
        self.half_open_calls = 0
        self.metrics.state = self.state
        self.metrics.state_transition_time = self.state_transition_time
        
        logger.info(f"熔断器半开: {self.name} ({old_state.value} -> {self.state.value})")
        
        if self.on_state_transition:
            self.on_state_transition(old_state, self.state)
    
    def _transition_to_closed(self):
        """转换到关闭状态"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_transition_time = time.time()
        self.half_open_calls = 0
        self.metrics.state = self.state
        self.metrics.state_transition_time = self.state_transition_time
        
        # 清空滑动窗口
        self.sliding_window.clear()
        
        logger.info(f"熔断器关闭: {self.name} ({old_state.value} -> {self.state.value})")
        
        if self.on_state_transition:
            self.on_state_transition(old_state, self.state)
    
    def _classify_failure(self, exception: Exception, duration: float) -> FailureType:
        """分类失败类型"""
        if isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif duration > self.config.slow_call_duration_threshold:
            return FailureType.SLOW_CALL
        else:
            return FailureType.ERROR
    
    def _should_ignore_exception(self, exception: Exception) -> bool:
        """检查是否应该忽略异常"""
        exception_name = exception.__class__.__name__
        
        # 检查忽略列表
        if exception_name in self.config.ignored_exceptions:
            return True
        
        # 检查记录列表（如果配置了记录列表，只记录列表中的异常）
        if self.config.recorded_exceptions:
            return exception_name not in self.config.recorded_exceptions
        
        return False
    
    def _on_success(self, duration: float):
        """成功回调"""
        if self.on_success:
            self.on_success(duration)
    
    def _on_error(self, exception: Exception, duration: float):
        """错误回调"""
        if self.on_error:
            self.on_error(exception, duration)
    
    def _on_call_not_permitted(self):
        """调用不被允许回调"""
        if self.on_call_not_permitted:
            self.on_call_not_permitted()
    
    def reset(self):
        """重置熔断器"""
        with self.lock:
            old_state = self.state
            self._transition_to_closed()
            logger.info(f"熔断器重置: {self.name}")
    
    def force_open(self):
        """强制开启熔断器"""
        with self.lock:
            self._transition_to_open()
            logger.info(f"熔断器强制开启: {self.name}")
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """获取指标"""
        with self.lock:
            return self.metrics
    
    def get_state(self) -> CircuitState:
        """获取当前状态"""
        return self.state
    
    def is_closed(self) -> bool:
        """是否关闭状态"""
        return self.state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """是否开启状态"""
        return self.state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """是否半开状态"""
        return self.state == CircuitState.HALF_OPEN

class CircuitBreakerRegistry:
    """熔断器注册表
    
    管理多个熔断器实例
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
        
        logger.info("熔断器注册表初始化完成")
    
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """获取或创建熔断器
        
        Args:
            name: 熔断器名称
            config: 熔断器配置
            
        Returns:
            熔断器实例
        """
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
                logger.debug(f"创建熔断器: {name}")
            
            return self.circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """获取熔断器
        
        Args:
            name: 熔断器名称
            
        Returns:
            熔断器实例
        """
        return self.circuit_breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """移除熔断器
        
        Args:
            name: 熔断器名称
            
        Returns:
            是否成功移除
        """
        with self.lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                logger.debug(f"移除熔断器: {name}")
                return True
            return False
    
    def list_all(self) -> List[str]:
        """列出所有熔断器名称
        
        Returns:
            熔断器名称列表
        """
        return list(self.circuit_breakers.keys())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有熔断器的指标
        
        Returns:
            指标字典
        """
        with self.lock:
            return {
                name: cb.get_metrics().to_dict()
                for name, cb in self.circuit_breakers.items()
            }
    
    def reset_all(self):
        """重置所有熔断器"""
        with self.lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            logger.info("重置所有熔断器")

# 全局熔断器注册表
_global_registry = CircuitBreakerRegistry()

# 便捷函数
def create_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """创建熔断器实例
    
    Args:
        name: 熔断器名称
        config: 熔断器配置
        
    Returns:
        熔断器实例
    """
    return CircuitBreaker(name, config)

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """获取或创建熔断器（使用全局注册表）
    
    Args:
        name: 熔断器名称
        config: 熔断器配置
        
    Returns:
        熔断器实例
    """
    return _global_registry.get_or_create(name, config)

def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """熔断器装饰器
    
    Args:
        name: 熔断器名称
        config: 熔断器配置
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        cb = get_circuit_breaker(name, config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await cb.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator

def create_default_config(**kwargs) -> CircuitBreakerConfig:
    """创建默认配置
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        熔断器配置
    """
    return CircuitBreakerConfig(**kwargs)

def create_fast_fail_config() -> CircuitBreakerConfig:
    """创建快速失败配置"""
    return CircuitBreakerConfig(
        failure_rate_threshold=0.3,
        minimum_number_of_calls=5,
        sliding_window_size=20,
        wait_duration_in_open_state=30.0
    )

def create_slow_recovery_config() -> CircuitBreakerConfig:
    """创建慢恢复配置"""
    return CircuitBreakerConfig(
        failure_rate_threshold=0.7,
        minimum_number_of_calls=20,
        sliding_window_size=100,
        wait_duration_in_open_state=120.0
    )