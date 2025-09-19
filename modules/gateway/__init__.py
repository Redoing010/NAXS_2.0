# API网关和服务发现模块

from .api_gateway import (
    APIGateway,
    Route,
    HTTPMethod,
    RouteType,
    Middleware,
    LoggingMiddleware,
    CORSMiddleware,
    AuthMiddleware,
    RateLimiter,
    create_api_gateway,
    create_route,
    create_rate_limiter
)

from .service_discovery import (
    ServiceRegistry,
    ServiceInstance,
    ServiceHealth,
    ServiceStatus,
    LoadBalanceStrategy,
    HealthChecker,
    LoadBalancer,
    RoundRobinLoadBalancer,
    WeightedRoundRobinLoadBalancer,
    RandomLoadBalancer,
    create_service_registry,
    create_service_instance,
    create_load_balancer
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    FailureType,
    CallResult,
    CircuitBreakerMetrics,
    CircuitBreakerException,
    SlidingWindow,
    create_circuit_breaker,
    get_circuit_breaker,
    circuit_breaker,
    create_default_config,
    create_fast_fail_config,
    create_slow_recovery_config
)

from .proxy import (
    ReverseProxy,
    ProxyManager,
    UpstreamServer,
    ProxyConfig,
    ProxyRequest,
    ProxyResponse,
    LoadBalancer as ProxyLoadBalancer,
    HealthChecker as ProxyHealthChecker,
    RequestCache,
    ProxyMethod,
    HealthStatus,
    LoadBalanceStrategy as ProxyLoadBalanceStrategy,
    create_upstream_server,
    create_proxy_config,
    create_reverse_proxy,
    create_proxy_manager
)

__all__ = [
    # API网关
    'APIGateway',
    'Route',
    'HTTPMethod',
    'RouteType',
    'Middleware',
    'LoggingMiddleware',
    'CORSMiddleware',
    'AuthMiddleware',
    'RateLimiter',
    'create_api_gateway',
    'create_route',
    'create_rate_limiter',
    
    # 服务发现
    'ServiceRegistry',
    'ServiceInstance',
    'ServiceHealth',
    'ServiceStatus',
    'LoadBalanceStrategy',
    'HealthChecker',
    'LoadBalancer',
    'RoundRobinLoadBalancer',
    'WeightedRoundRobinLoadBalancer',
    'RandomLoadBalancer',
    'create_service_registry',
    'create_service_instance',
    'create_load_balancer',
    
    # 熔断器
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerRegistry',
    'CircuitState',
    'FailureType',
    'CallResult',
    'CircuitBreakerMetrics',
    'CircuitBreakerException',
    'SlidingWindow',
    'create_circuit_breaker',
    'get_circuit_breaker',
    'circuit_breaker',
    'create_default_config',
    'create_fast_fail_config',
    'create_slow_recovery_config',
    
    # 代理
    'ReverseProxy',
    'ProxyManager',
    'UpstreamServer',
    'ProxyConfig',
    'ProxyRequest',
    'ProxyResponse',
    'ProxyLoadBalancer',
    'ProxyHealthChecker',
    'RequestCache',
    'ProxyMethod',
    'HealthStatus',
    'ProxyLoadBalanceStrategy',
    'create_upstream_server',
    'create_proxy_config',
    'create_reverse_proxy',
    'create_proxy_manager'
]

__version__ = '1.0.0'
__author__ = 'NAXS Team'
__description__ = 'API Gateway and Service Discovery for microservices architecture'