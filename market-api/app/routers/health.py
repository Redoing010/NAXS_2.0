import sys
import os
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
import structlog

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新的监控模块
try:
    from ..database import db_manager, get_db
    from ..cache import cache_manager
    from ..monitoring import performance_monitor, memory_profiler
    from ..middleware import memory_guard, circuit_breaker_manager
except ImportError as e:
    logging.warning(f"新监控模块导入失败: {e}")

# 尝试导入旧的监控模块（向后兼容）
try:
    from modules.monitoring.system_monitor import get_system_monitor
    from modules.monitoring.error_handler import get_error_handler
    from modules.monitoring.health_checks import get_system_health_summary, APIHealthChecker
except ImportError as e:
    logging.warning(f"旧监控模块导入失败: {e}")
    get_system_monitor = None
    get_error_handler = None
    get_system_health_summary = None
    APIHealthChecker = None

router = APIRouter(prefix="/health", tags=["健康检查"])
logger = structlog.get_logger(__name__)


@router.get("/", summary="基础健康检查")
async def health():
    """基础健康检查"""
    return {
        "status": "ok", 
        "timestamp": time.time(),
        "service": "NAXS Market API",
        "version": "2.0.0"
    }


@router.get("/detailed", summary="详细健康检查")
async def detailed_health():
    """详细健康检查 - 包含所有组件状态"""
    try:
        current_time = time.time()
        
        # 基础健康数据
        health_data = {
            "status": "ok",
            "timestamp": current_time,
            "api_status": "running",
            "uptime": current_time - performance_monitor._start_time if hasattr(performance_monitor, '_start_time') else 0
        }
        
        # 检查各个组件
        components = {}
        
        # 数据库健康检查
        try:
            db_healthy = db_manager.health_check()
            db_stats = db_manager.get_connection_stats()
            components["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "stats": db_stats
            }
        except Exception as e:
            components["database"] = {
                "status": "error",
                "error": str(e)
            }
        
        # 缓存健康检查
        try:
            cache_health = await cache_manager.health_check()
            cache_stats = cache_manager.get_stats()
            components["cache"] = {
                "status": "healthy" if cache_health.get('redis_cache', False) else "degraded",
                "redis_available": cache_health.get('redis_cache', False),
                "memory_cache_available": cache_health.get('memory_cache', True),
                "stats": cache_stats
            }
        except Exception as e:
            components["cache"] = {
                "status": "error",
                "error": str(e)
            }
        
        # 内存健康检查
        try:
            memory_healthy = memory_guard.check_memory_usage()
            memory_stats = memory_guard.get_memory_stats()
            memory_profile = memory_profiler.get_memory_stats()
            components["memory"] = {
                "status": "healthy" if memory_healthy else "warning",
                "guard_stats": memory_stats,
                "profile_stats": memory_profile
            }
        except Exception as e:
            components["memory"] = {
                "status": "error",
                "error": str(e)
            }
        
        # 熔断器状态
        try:
            breaker_stats = circuit_breaker_manager.get_breaker_stats()
            components["circuit_breakers"] = {
                "status": "healthy",
                "stats": breaker_stats
            }
        except Exception as e:
            components["circuit_breakers"] = {
                "status": "error",
                "error": str(e)
            }
        
        # 系统性能指标
        try:
            system_health = performance_monitor.get_health_status()
            components["performance"] = {
                "status": system_health["status"],
                "metrics": system_health
            }
        except Exception as e:
            components["performance"] = {
                "status": "error",
                "error": str(e)
            }
        
        # 判断整体健康状态
        unhealthy_components = [
            name for name, comp in components.items() 
            if comp["status"] in ["unhealthy", "error"]
        ]
        
        if unhealthy_components:
            health_data["status"] = "degraded" if len(unhealthy_components) < len(components) / 2 else "unhealthy"
            health_data["unhealthy_components"] = unhealthy_components
        
        health_data["components"] = components
        
        # 兼容旧监控模块
        if get_system_health_summary:
            try:
                legacy_health = get_system_health_summary()
                health_data["legacy_monitoring"] = legacy_health
            except Exception as e:
                logger.warning(f"获取旧监控数据失败: {e}")
        
        return health_data
        
    except Exception as e:
        logger.error(f"详细健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/system", summary="系统健康状态")
async def system_health():
    """系统级健康状态检查"""
    try:
        system_status = performance_monitor.get_health_status()
        
        # 添加额外的系统信息
        import psutil
        
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free
            } if os.name != 'nt' else {
                "total": psutil.disk_usage('C:').total,
                "used": psutil.disk_usage('C:').used,
                "free": psutil.disk_usage('C:').free
            }
        }
        
        return {
            **system_status,
            "system_info": system_info,
            "legacy_available": get_system_health_summary is not None
        }
        
    except Exception as e:
        logger.error(f"系统健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统健康检查失败: {str(e)}")


@router.get("/metrics", summary="性能指标")
async def current_metrics():
    """当前系统性能指标"""
    try:
        # 获取新监控系统的指标
        performance_stats = performance_monitor.get_health_status()
        cache_stats = cache_manager.get_stats()
        db_stats = db_manager.get_connection_stats()
        memory_stats = memory_profiler.get_memory_stats()
        
        metrics = {
            "performance": performance_stats,
            "cache": cache_stats,
            "database": db_stats,
            "memory": memory_stats,
            "timestamp": time.time()
        }
        
        # 如果旧监控系统可用，也包含其数据
        if get_system_monitor:
            try:
                monitor = get_system_monitor()
                legacy_metrics = monitor.get_system_summary()
                metrics["legacy_metrics"] = legacy_metrics
            except Exception as e:
                logger.warning(f"获取旧监控指标失败: {e}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {str(e)}")


@router.get("/errors", summary="错误统计")
async def error_statistics():
    """系统错误统计信息"""
    try:
        # 获取熔断器统计
        breaker_stats = circuit_breaker_manager.get_breaker_stats()
        
        # 获取内存保护统计
        memory_stats = memory_guard.get_memory_stats()
        
        error_data = {
            "circuit_breakers": breaker_stats,
            "memory_protection": memory_stats,
            "timestamp": time.time()
        }
        
        # 如果旧错误处理器可用，包含其数据
        if get_error_handler:
            try:
                error_handler = get_error_handler()
                legacy_errors = {
                    "error_stats": error_handler.get_error_stats(),
                    "circuit_breakers": error_handler.get_circuit_breaker_status()
                }
                error_data["legacy_errors"] = legacy_errors
            except Exception as e:
                logger.warning(f"获取旧错误统计失败: {e}")
        
        return error_data
        
    except Exception as e:
        logger.error(f"获取错误统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取错误统计失败: {str(e)}")


@router.get("/database", summary="数据库健康检查")
async def database_health():
    """数据库连接和性能检查"""
    try:
        # 执行数据库健康检查
        is_healthy = db_manager.health_check()
        connection_stats = db_manager.get_connection_stats()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connection_stats": connection_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/cache", summary="缓存健康检查")
async def cache_health():
    """缓存系统健康检查"""
    try:
        # 执行缓存健康检查
        health_status = await cache_manager.health_check()
        cache_stats = cache_manager.get_stats()
        
        return {
            "status": "healthy" if health_status.get('redis_cache', False) else "degraded",
            "redis_available": health_status.get('redis_cache', False),
            "memory_cache_available": health_status.get('memory_cache', True),
            "stats": cache_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"缓存健康检查失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }





