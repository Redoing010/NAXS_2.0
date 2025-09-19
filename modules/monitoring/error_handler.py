#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理模块

提供统一的异常处理、重试机制、错误恢复等功能
"""

import time
import logging
import functools
import traceback
from typing import Dict, List, Optional, Callable, Any, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """重试策略"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 60.0  # 最大延迟（秒）
    backoff_multiplier: float = 2.0  # 退避乘数
    jitter: bool = True  # 是否添加随机抖动
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class ErrorContext:
    """错误上下文"""
    module: str
    function: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    attempt: int = 1
    max_attempts: int = 1
    start_time: datetime = field(default_factory=datetime.now)
    last_error: Optional[Exception] = None
    error_history: List[Exception] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5  # 失败阈值
    recovery_timeout: float = 60.0  # 恢复超时（秒）
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 3  # 半开状态下成功阈值


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(f"熔断器开启，拒绝调用 {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout
    
    def _on_success(self):
        """成功回调"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """失败回调"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_stats = defaultdict(int)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_callbacks: Dict[Type[Exception], List[Callable]] = defaultdict(list)
        self.global_error_callback: Optional[Callable] = None
        self._lock = threading.Lock()
        
        logger.info("错误处理器初始化完成")
    
    def retry(self, config: RetryConfig = None):
        """重试装饰器"""
        if config is None:
            config = RetryConfig()
        
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(func, config, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_retry(self, func: Callable, config: RetryConfig, *args, **kwargs):
        """执行带重试的函数"""
        context = ErrorContext(
            module=func.__module__,
            function=func.__name__,
            args=args,
            kwargs=kwargs,
            max_attempts=config.max_attempts
        )
        
        for attempt in range(1, config.max_attempts + 1):
            context.attempt = attempt
            
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"{func.__name__} 在第 {attempt} 次尝试后成功")
                return result
                
            except Exception as e:
                context.last_error = e
                context.error_history.append(e)
                
                # 检查是否应该停止重试
                if any(isinstance(e, exc_type) for exc_type in config.stop_on_exceptions):
                    logger.error(f"{func.__name__} 遇到停止重试的异常: {e}")
                    self._handle_error(e, context)
                    raise e
                
                # 检查是否应该重试
                should_retry = any(isinstance(e, exc_type) for exc_type in config.retry_on_exceptions)
                
                if not should_retry or attempt == config.max_attempts:
                    logger.error(f"{func.__name__} 在 {attempt} 次尝试后最终失败: {e}")
                    self._handle_error(e, context)
                    raise e
                
                # 计算延迟时间
                delay = self._calculate_delay(config, attempt)
                logger.warning(f"{func.__name__} 第 {attempt} 次尝试失败: {e}，{delay:.1f}秒后重试")
                
                time.sleep(delay)
        
        # 理论上不会到达这里
        raise RuntimeError("重试逻辑异常")
    
    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """计算延迟时间"""
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
        else:
            delay = config.base_delay
        
        # 限制最大延迟
        delay = min(delay, config.max_delay)
        
        # 添加随机抖动
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50%-100%的随机延迟
        
        return delay
    
    def circuit_breaker(self, name: str, config: CircuitBreakerConfig = None):
        """熔断器装饰器"""
        if config is None:
            config = CircuitBreakerConfig()
        
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                breaker = self._get_circuit_breaker(name, config)
                return breaker.call(func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _get_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """获取熔断器"""
        if name not in self.circuit_breakers:
            with self._lock:
                if name not in self.circuit_breakers:
                    self.circuit_breakers[name] = CircuitBreaker(config)
        return self.circuit_breakers[name]
    
    def safe_execute(self, func: Callable, *args, 
                    default_return=None, 
                    log_errors: bool = True,
                    **kwargs):
        """安全执行函数，捕获所有异常"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"安全执行 {func.__name__} 失败: {e}")
            
            context = ErrorContext(
                module=func.__module__,
                function=func.__name__,
                args=args,
                kwargs=kwargs
            )
            self._handle_error(e, context)
            return default_return
    
    def _handle_error(self, error: Exception, context: ErrorContext):
        """处理错误"""
        # 更新错误统计
        error_type = type(error).__name__
        with self._lock:
            self.error_stats[error_type] += 1
        
        # 调用特定异常的回调
        for exc_type, callbacks in self.error_callbacks.items():
            if isinstance(error, exc_type):
                for callback in callbacks:
                    try:
                        callback(error, context)
                    except Exception as cb_error:
                        logger.error(f"错误回调执行失败: {cb_error}")
        
        # 调用全局错误回调
        if self.global_error_callback:
            try:
                self.global_error_callback(error, context)
            except Exception as cb_error:
                logger.error(f"全局错误回调执行失败: {cb_error}")
        
        # 记录到监控系统
        try:
            from .system_monitor import record_error
            record_error(
                module=context.module,
                message=f"{context.function}: {str(error)}",
                level="ERROR",
                exception=error,
                context={
                    'function': context.function,
                    'attempt': context.attempt,
                    'max_attempts': context.max_attempts,
                    'args_count': len(context.args),
                    'kwargs_keys': list(context.kwargs.keys())
                }
            )
        except ImportError:
            pass  # 监控模块不可用
    
    def add_error_callback(self, exception_type: Type[Exception], callback: Callable):
        """添加错误回调"""
        self.error_callbacks[exception_type].append(callback)
        logger.info(f"添加 {exception_type.__name__} 错误回调")
    
    def set_global_error_callback(self, callback: Callable):
        """设置全局错误回调"""
        self.global_error_callback = callback
        logger.info("设置全局错误回调")
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        with self._lock:
            return dict(self.error_stats)
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """获取熔断器状态"""
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count,
                'last_failure_time': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
            }
        return status
    
    def reset_circuit_breaker(self, name: str):
        """重置熔断器"""
        if name in self.circuit_breakers:
            breaker = self.circuit_breakers[name]
            with breaker._lock:
                breaker.state = CircuitBreakerState.CLOSED
                breaker.failure_count = 0
                breaker.success_count = 0
                breaker.last_failure_time = None
            logger.info(f"熔断器 {name} 已重置")
    
    def clear_error_stats(self):
        """清除错误统计"""
        with self._lock:
            self.error_stats.clear()
        logger.info("错误统计已清除")


# 全局错误处理器实例
_error_handler_instance = None


def get_error_handler() -> ErrorHandler:
    """获取错误处理器单例"""
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = ErrorHandler()
    return _error_handler_instance


# 便捷装饰器
def retry(max_attempts: int = 3, 
         base_delay: float = 1.0,
         strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
         retry_on: List[Type[Exception]] = None,
         stop_on: List[Type[Exception]] = None):
    """重试装饰器"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        retry_on_exceptions=retry_on or [Exception],
        stop_on_exceptions=stop_on or []
    )
    return get_error_handler().retry(config)


def circuit_breaker(name: str, 
                   failure_threshold: int = 5,
                   recovery_timeout: float = 60.0,
                   expected_exception: Type[Exception] = Exception):
    """熔断器装饰器"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    return get_error_handler().circuit_breaker(name, config)


def safe_execute(func: Callable, *args, default_return=None, **kwargs):
    """安全执行函数"""
    return get_error_handler().safe_execute(func, *args, default_return=default_return, **kwargs)


def handle_exceptions(*exception_types: Type[Exception]):
    """异常处理装饰器"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                logger.error(f"{func.__name__} 发生异常: {e}")
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__,
                    args=args,
                    kwargs=kwargs
                )
                get_error_handler()._handle_error(e, context)
                raise e
        return wrapper
    return decorator