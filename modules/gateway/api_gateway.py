# API网关 - 实现统一的API管理和路由功能

import logging
import asyncio
import time
import json
import re
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Pattern
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict, deque
from urllib.parse import urlparse, parse_qs
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web_middlewares import cors_handler
import jwt
import hashlib
import threading

logger = logging.getLogger(__name__)

class HTTPMethod(Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class RouteType(Enum):
    """路由类型"""
    EXACT = "exact"        # 精确匹配
    PREFIX = "prefix"      # 前缀匹配
    REGEX = "regex"        # 正则匹配
    WILDCARD = "wildcard"  # 通配符匹配

@dataclass
class Route:
    """路由定义"""
    id: str
    path: str
    methods: List[HTTPMethod]
    target_url: str
    route_type: RouteType = RouteType.EXACT
    priority: int = 0
    timeout: float = 30.0
    retries: int = 0
    retry_delay: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    middleware: List[str] = field(default_factory=list)
    auth_required: bool = False
    rate_limit: Optional[Dict[str, Any]] = None
    circuit_breaker: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    # 编译后的正则表达式（用于正则路由）
    _compiled_pattern: Optional[Pattern] = field(default=None, init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.route_type == RouteType.REGEX:
            try:
                self._compiled_pattern = re.compile(self.path)
            except re.error as e:
                logger.error(f"路由正则表达式编译失败: {self.path}, 错误: {str(e)}")
                raise ValueError(f"无效的正则表达式: {self.path}")
    
    def matches(self, path: str, method: HTTPMethod) -> bool:
        """检查路由是否匹配
        
        Args:
            path: 请求路径
            method: HTTP方法
            
        Returns:
            是否匹配
        """
        if not self.enabled:
            return False
        
        if method not in self.methods:
            return False
        
        if self.route_type == RouteType.EXACT:
            return path == self.path
        elif self.route_type == RouteType.PREFIX:
            return path.startswith(self.path)
        elif self.route_type == RouteType.REGEX:
            if self._compiled_pattern:
                return bool(self._compiled_pattern.match(path))
            return False
        elif self.route_type == RouteType.WILDCARD:
            # 简单的通配符匹配（* 匹配任意字符）
            pattern = self.path.replace('*', '.*')
            return bool(re.match(pattern, path))
        
        return False
    
    def extract_path_params(self, path: str) -> Dict[str, str]:
        """提取路径参数
        
        Args:
            path: 请求路径
            
        Returns:
            路径参数字典
        """
        params = {}
        
        if self.route_type == RouteType.REGEX and self._compiled_pattern:
            match = self._compiled_pattern.match(path)
            if match:
                params.update(match.groupdict())
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'path': self.path,
            'methods': [m.value for m in self.methods],
            'target_url': self.target_url,
            'route_type': self.route_type.value,
            'priority': self.priority,
            'timeout': self.timeout,
            'retries': self.retries,
            'retry_delay': self.retry_delay,
            'headers': self.headers,
            'query_params': self.query_params,
            'middleware': self.middleware,
            'auth_required': self.auth_required,
            'rate_limit': self.rate_limit,
            'circuit_breaker': self.circuit_breaker,
            'metadata': self.metadata,
            'enabled': self.enabled
        }

class Middleware(ABC):
    """中间件基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def process_request(self, request: web.Request, route: Route) -> Optional[web.Response]:
        """处理请求
        
        Args:
            request: HTTP请求
            route: 匹配的路由
            
        Returns:
            如果返回Response则中断后续处理，否则继续
        """
        pass
    
    @abstractmethod
    async def process_response(self, request: web.Request, response: web.Response, 
                             route: Route) -> web.Response:
        """处理响应
        
        Args:
            request: HTTP请求
            response: HTTP响应
            route: 匹配的路由
            
        Returns:
            处理后的响应
        """
        pass

class LoggingMiddleware(Middleware):
    """日志中间件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("logging", config)
        self.log_requests = self.config.get('log_requests', True)
        self.log_responses = self.config.get('log_responses', True)
        self.log_headers = self.config.get('log_headers', False)
    
    async def process_request(self, request: web.Request, route: Route) -> Optional[web.Response]:
        """记录请求日志"""
        if self.log_requests:
            log_data = {
                'method': request.method,
                'path': request.path_qs,
                'remote': request.remote,
                'route_id': route.id,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.log_headers:
                log_data['headers'] = dict(request.headers)
            
            logger.info(f"请求: {json.dumps(log_data, ensure_ascii=False)}")
        
        # 记录请求开始时间
        request['_start_time'] = time.time()
        return None
    
    async def process_response(self, request: web.Request, response: web.Response, 
                             route: Route) -> web.Response:
        """记录响应日志"""
        if self.log_responses:
            duration = time.time() - request.get('_start_time', 0)
            
            log_data = {
                'method': request.method,
                'path': request.path_qs,
                'status': response.status,
                'duration_ms': round(duration * 1000, 2),
                'route_id': route.id,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.log_headers:
                log_data['response_headers'] = dict(response.headers)
            
            logger.info(f"响应: {json.dumps(log_data, ensure_ascii=False)}")
        
        return response

class CORSMiddleware(Middleware):
    """CORS中间件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("cors", config)
        self.allow_origins = self.config.get('allow_origins', ['*'])
        self.allow_methods = self.config.get('allow_methods', ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
        self.allow_headers = self.config.get('allow_headers', ['*'])
        self.allow_credentials = self.config.get('allow_credentials', False)
        self.max_age = self.config.get('max_age', 86400)
    
    async def process_request(self, request: web.Request, route: Route) -> Optional[web.Response]:
        """处理CORS预检请求"""
        if request.method == 'OPTIONS':
            response = web.Response()
            self._add_cors_headers(response, request)
            return response
        
        return None
    
    async def process_response(self, request: web.Request, response: web.Response, 
                             route: Route) -> web.Response:
        """添加CORS头"""
        self._add_cors_headers(response, request)
        return response
    
    def _add_cors_headers(self, response: web.Response, request: web.Request):
        """添加CORS头"""
        origin = request.headers.get('Origin')
        
        if '*' in self.allow_origins or (origin and origin in self.allow_origins):
            response.headers['Access-Control-Allow-Origin'] = origin or '*'
        
        response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allow_methods)
        response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allow_headers)
        
        if self.allow_credentials:
            response.headers['Access-Control-Allow-Credentials'] = 'true'
        
        response.headers['Access-Control-Max-Age'] = str(self.max_age)

class AuthMiddleware(Middleware):
    """认证中间件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("auth", config)
        self.jwt_secret = self.config.get('jwt_secret', 'default_secret')
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.auth_header = self.config.get('auth_header', 'Authorization')
        self.auth_scheme = self.config.get('auth_scheme', 'Bearer')
    
    async def process_request(self, request: web.Request, route: Route) -> Optional[web.Response]:
        """验证认证"""
        if not route.auth_required:
            return None
        
        auth_header = request.headers.get(self.auth_header)
        if not auth_header:
            return web.json_response(
                {'error': 'Missing authorization header'}, 
                status=401
            )
        
        try:
            scheme, token = auth_header.split(' ', 1)
            if scheme != self.auth_scheme:
                raise ValueError(f"Invalid auth scheme: {scheme}")
            
            # 验证JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            request['user'] = payload
            
        except (ValueError, jwt.InvalidTokenError) as e:
            return web.json_response(
                {'error': f'Invalid token: {str(e)}'}, 
                status=401
            )
        
        return None
    
    async def process_response(self, request: web.Request, response: web.Response, 
                             route: Route) -> web.Response:
        """处理响应"""
        return response

class RateLimiter:
    """限流器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.requests_per_minute = config.get('requests_per_minute', 60)
        self.requests_per_hour = config.get('requests_per_hour', 1000)
        self.burst_size = config.get('burst_size', 10)
        
        # 使用滑动窗口计数器
        self.minute_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.requests_per_minute))
        self.hour_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.requests_per_hour))
        self.lock = threading.RLock()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """检查是否允许请求
        
        Args:
            client_id: 客户端标识
            
        Returns:
            (是否允许, 限流信息)
        """
        current_time = time.time()
        
        with self.lock:
            # 清理过期的记录
            self._cleanup_expired_records(client_id, current_time)
            
            minute_window = self.minute_windows[client_id]
            hour_window = self.hour_windows[client_id]
            
            # 检查分钟级限制
            if len(minute_window) >= self.requests_per_minute:
                return False, {
                    'error': 'Rate limit exceeded (per minute)',
                    'limit': self.requests_per_minute,
                    'window': 'minute',
                    'reset_time': int(current_time + 60)
                }
            
            # 检查小时级限制
            if len(hour_window) >= self.requests_per_hour:
                return False, {
                    'error': 'Rate limit exceeded (per hour)',
                    'limit': self.requests_per_hour,
                    'window': 'hour',
                    'reset_time': int(current_time + 3600)
                }
            
            # 记录请求
            minute_window.append(current_time)
            hour_window.append(current_time)
            
            return True, {
                'remaining_minute': self.requests_per_minute - len(minute_window),
                'remaining_hour': self.requests_per_hour - len(hour_window),
                'reset_minute': int(current_time + 60),
                'reset_hour': int(current_time + 3600)
            }
    
    def _cleanup_expired_records(self, client_id: str, current_time: float):
        """清理过期记录"""
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        minute_window = self.minute_windows[client_id]
        hour_window = self.hour_windows[client_id]
        
        # 清理分钟窗口
        while minute_window and minute_window[0] < minute_cutoff:
            minute_window.popleft()
        
        # 清理小时窗口
        while hour_window and hour_window[0] < hour_cutoff:
            hour_window.popleft()

class RateLimitMiddleware(Middleware):
    """限流中间件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("rate_limit", config)
        self.global_limiter = RateLimiter(self.config.get('global', {}))
        self.route_limiters: Dict[str, RateLimiter] = {}
    
    async def process_request(self, request: web.Request, route: Route) -> Optional[web.Response]:
        """检查限流"""
        client_id = self._get_client_id(request)
        
        # 检查全局限流
        allowed, info = self.global_limiter.is_allowed(client_id)
        if not allowed:
            return web.json_response(info, status=429)
        
        # 检查路由级限流
        if route.rate_limit:
            if route.id not in self.route_limiters:
                self.route_limiters[route.id] = RateLimiter(route.rate_limit)
            
            route_limiter = self.route_limiters[route.id]
            allowed, info = route_limiter.is_allowed(client_id)
            if not allowed:
                return web.json_response(info, status=429)
        
        return None
    
    async def process_response(self, request: web.Request, response: web.Response, 
                             route: Route) -> web.Response:
        """添加限流头"""
        client_id = self._get_client_id(request)
        _, info = self.global_limiter.is_allowed(client_id)
        
        if 'remaining_minute' in info:
            response.headers['X-RateLimit-Remaining'] = str(info['remaining_minute'])
            response.headers['X-RateLimit-Reset'] = str(info['reset_minute'])
        
        return response
    
    def _get_client_id(self, request: web.Request) -> str:
        """获取客户端ID"""
        # 优先使用用户ID
        user = request.get('user')
        if user and 'user_id' in user:
            return f"user:{user['user_id']}"
        
        # 使用IP地址
        return f"ip:{request.remote}"

class APIGateway:
    """API网关
    
    提供统一的API管理功能，包括：
    1. 路由管理和匹配
    2. 请求转发和负载均衡
    3. 中间件支持
    4. 认证和授权
    5. 限流和熔断
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.routes: Dict[str, Route] = {}
        self.middleware: Dict[str, Middleware] = {}
        self.app = web.Application()
        self.client_session: Optional[ClientSession] = None
        
        # 配置参数
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8080)
        self.client_timeout = ClientTimeout(total=self.config.get('client_timeout', 30))
        
        # 注册默认中间件
        self._register_default_middleware()
        
        # 设置路由处理器
        self.app.router.add_route('*', '/{path:.*}', self._handle_request)
        
        logger.info(f"API网关初始化完成，监听地址: {self.host}:{self.port}")
    
    def _register_default_middleware(self):
        """注册默认中间件"""
        # 日志中间件
        logging_config = self.config.get('logging', {})
        self.register_middleware(LoggingMiddleware(logging_config))
        
        # CORS中间件
        cors_config = self.config.get('cors', {})
        if cors_config.get('enabled', True):
            self.register_middleware(CORSMiddleware(cors_config))
        
        # 认证中间件
        auth_config = self.config.get('auth', {})
        if auth_config.get('enabled', False):
            self.register_middleware(AuthMiddleware(auth_config))
        
        # 限流中间件
        rate_limit_config = self.config.get('rate_limit', {})
        if rate_limit_config.get('enabled', False):
            self.register_middleware(RateLimitMiddleware(rate_limit_config))
    
    def register_middleware(self, middleware: Middleware):
        """注册中间件
        
        Args:
            middleware: 中间件实例
        """
        self.middleware[middleware.name] = middleware
        logger.debug(f"注册中间件: {middleware.name}")
    
    def unregister_middleware(self, name: str) -> bool:
        """注销中间件
        
        Args:
            name: 中间件名称
            
        Returns:
            是否成功注销
        """
        if name in self.middleware:
            del self.middleware[name]
            logger.debug(f"注销中间件: {name}")
            return True
        return False
    
    def add_route(self, route: Route):
        """添加路由
        
        Args:
            route: 路由定义
        """
        self.routes[route.id] = route
        logger.debug(f"添加路由: {route.id} - {route.path}")
    
    def remove_route(self, route_id: str) -> bool:
        """移除路由
        
        Args:
            route_id: 路由ID
            
        Returns:
            是否成功移除
        """
        if route_id in self.routes:
            route = self.routes[route_id]
            del self.routes[route_id]
            logger.debug(f"移除路由: {route_id} - {route.path}")
            return True
        return False
    
    def get_route(self, route_id: str) -> Optional[Route]:
        """获取路由
        
        Args:
            route_id: 路由ID
            
        Returns:
            路由定义
        """
        return self.routes.get(route_id)
    
    def list_routes(self) -> List[Route]:
        """列出所有路由
        
        Returns:
            路由列表
        """
        return list(self.routes.values())
    
    def find_route(self, path: str, method: HTTPMethod) -> Optional[Route]:
        """查找匹配的路由
        
        Args:
            path: 请求路径
            method: HTTP方法
            
        Returns:
            匹配的路由
        """
        # 按优先级排序
        sorted_routes = sorted(self.routes.values(), key=lambda r: r.priority, reverse=True)
        
        for route in sorted_routes:
            if route.matches(path, method):
                return route
        
        return None
    
    async def _handle_request(self, request: web.Request) -> web.Response:
        """处理请求"""
        try:
            # 查找匹配的路由
            method = HTTPMethod(request.method)
            route = self.find_route(request.path, method)
            
            if not route:
                return web.json_response(
                    {'error': 'Route not found'}, 
                    status=404
                )
            
            # 执行请求中间件
            for middleware_name in route.middleware:
                middleware = self.middleware.get(middleware_name)
                if middleware:
                    response = await middleware.process_request(request, route)
                    if response:
                        return response
            
            # 执行全局中间件
            for middleware in self.middleware.values():
                if middleware.name not in route.middleware:
                    response = await middleware.process_request(request, route)
                    if response:
                        return response
            
            # 转发请求
            response = await self._forward_request(request, route)
            
            # 执行响应中间件（逆序）
            middleware_list = list(self.middleware.values())
            for middleware in reversed(middleware_list):
                response = await middleware.process_response(request, response, route)
            
            return response
            
        except Exception as e:
            logger.error(f"请求处理异常: {str(e)}")
            return web.json_response(
                {'error': 'Internal server error'}, 
                status=500
            )
    
    async def _forward_request(self, request: web.Request, route: Route) -> web.Response:
        """转发请求"""
        if not self.client_session:
            self.client_session = ClientSession(timeout=self.client_timeout)
        
        # 构建目标URL
        target_url = self._build_target_url(request, route)
        
        # 准备请求头
        headers = dict(request.headers)
        headers.update(route.headers)
        
        # 移除hop-by-hop头
        hop_by_hop_headers = {
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        }
        for header in hop_by_hop_headers:
            headers.pop(header, None)
        
        # 准备请求体
        body = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            body = await request.read()
        
        # 执行请求（带重试）
        for attempt in range(route.retries + 1):
            try:
                async with self.client_session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    timeout=ClientTimeout(total=route.timeout)
                ) as resp:
                    # 读取响应体
                    response_body = await resp.read()
                    
                    # 构建响应
                    response = web.Response(
                        body=response_body,
                        status=resp.status,
                        headers=resp.headers
                    )
                    
                    return response
                    
            except Exception as e:
                if attempt < route.retries:
                    logger.warning(f"请求失败，准备重试: {target_url}, 尝试次数: {attempt + 1}/{route.retries + 1}, 错误: {str(e)}")
                    await asyncio.sleep(route.retry_delay)
                else:
                    logger.error(f"请求失败: {target_url}, 错误: {str(e)}")
                    return web.json_response(
                        {'error': f'Upstream service error: {str(e)}'}, 
                        status=502
                    )
        
        # 不应该到达这里
        return web.json_response(
            {'error': 'Unexpected error'}, 
            status=500
        )
    
    def _build_target_url(self, request: web.Request, route: Route) -> str:
        """构建目标URL"""
        # 提取路径参数
        path_params = route.extract_path_params(request.path)
        
        # 构建目标路径
        target_path = request.path
        if route.route_type == RouteType.PREFIX:
            # 移除前缀
            target_path = request.path[len(route.path):]
            if not target_path.startswith('/'):
                target_path = '/' + target_path
        
        # 替换路径参数
        for param_name, param_value in path_params.items():
            target_path = target_path.replace(f'{{{param_name}}}', param_value)
        
        # 构建查询参数
        query_params = dict(parse_qs(request.query_string))
        query_params.update(route.query_params)
        
        # 构建完整URL
        target_url = route.target_url.rstrip('/') + target_path
        
        if query_params:
            query_string = '&'.join(f"{k}={v[0] if isinstance(v, list) else v}" 
                                  for k, v in query_params.items())
            target_url += '?' + query_string
        
        return target_url
    
    async def start(self):
        """启动网关"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"API网关启动成功，监听地址: http://{self.host}:{self.port}")
    
    async def stop(self):
        """停止网关"""
        if self.client_session:
            await self.client_session.close()
        
        await self.app.shutdown()
        await self.app.cleanup()
        
        logger.info("API网关停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            'total_routes': len(self.routes),
            'enabled_routes': len([r for r in self.routes.values() if r.enabled]),
            'middleware_count': len(self.middleware),
            'routes_by_type': {
                route_type.value: len([r for r in self.routes.values() if r.route_type == route_type])
                for route_type in RouteType
            },
            'routes_by_method': {
                method.value: len([r for r in self.routes.values() if method in r.methods])
                for method in HTTPMethod
            }
        }

# 便捷函数
def create_api_gateway(config: Dict[str, Any] = None) -> APIGateway:
    """创建API网关实例
    
    Args:
        config: 配置字典
        
    Returns:
        API网关实例
    """
    return APIGateway(config)

def create_route(route_id: str, path: str, target_url: str,
                methods: List[str] = None, route_type: str = "exact",
                **kwargs) -> Route:
    """创建路由
    
    Args:
        route_id: 路由ID
        path: 路径模式
        target_url: 目标URL
        methods: HTTP方法列表
        route_type: 路由类型
        **kwargs: 其他参数
        
    Returns:
        路由对象
    """
    http_methods = [HTTPMethod(m.upper()) for m in (methods or ['GET'])]
    route_type_enum = RouteType(route_type.lower())
    
    return Route(
        id=route_id,
        path=path,
        methods=http_methods,
        target_url=target_url,
        route_type=route_type_enum,
        **kwargs
    )

def create_exact_route(route_id: str, path: str, target_url: str,
                      methods: List[str] = None, **kwargs) -> Route:
    """创建精确匹配路由"""
    return create_route(route_id, path, target_url, methods, "exact", **kwargs)

def create_prefix_route(route_id: str, path: str, target_url: str,
                       methods: List[str] = None, **kwargs) -> Route:
    """创建前缀匹配路由"""
    return create_route(route_id, path, target_url, methods, "prefix", **kwargs)

def create_regex_route(route_id: str, pattern: str, target_url: str,
                      methods: List[str] = None, **kwargs) -> Route:
    """创建正则匹配路由"""
    return create_route(route_id, pattern, target_url, methods, "regex", **kwargs)