import asyncio
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from .config import settings
from .database import init_database, cleanup_database, db_manager
from .cache import init_cache, cleanup_cache, cache_manager
from .monitoring import (
    init_monitoring, 
    cleanup_monitoring, 
    performance_monitor,
    monitoring_middleware
)
from .middleware import (
    setup_middleware,
    rate_limiter,
    circuit_breaker_manager,
    memory_guard
)
from .alerting import init_alerting, cleanup_alerting, alert_manager
from .routers import health, spot, minute, placeholders, bars, admin, system

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting NAXS Market API...")
    
    try:
        # 初始化数据库
        init_database()
        logger.info("Database initialized")
        
        # 初始化缓存
        await init_cache()
        logger.info("Cache initialized")
        
        # 初始化监控
        await init_monitoring()
        logger.info("Monitoring initialized")
        
        # 初始化告警系统
        await init_alerting()
        logger.info("Alerting system initialized")
        
        logger.info("NAXS Market API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # 关闭时清理资源
        logger.info("Shutting down NAXS Market API...")
        
        try:
            await cleanup_alerting()
            await cleanup_monitoring()
            await cleanup_cache()
            cleanup_database()
            logger.info("NAXS Market API shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NAXS智能投研系统 - 高性能市场数据API",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加监控中间件
app.middleware("http")(monitoring_middleware)

# 设置其他中间件
setup_middleware(app)

# 设置Prometheus监控
if settings.ENABLE_METRICS:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(app)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(system.router, tags=["系统监控"])
app.include_router(spot.router, tags=["现货数据"])
app.include_router(minute.router, tags=["分钟数据"])
app.include_router(bars.router, tags=["K线数据"])
app.include_router(placeholders.router, tags=["占位符"])
app.include_router(admin.router, tags=["管理接口"])

@app.get("/", summary="API根路径")
async def root():
    """API根路径，返回基本信息"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics" if settings.ENABLE_METRICS else None,
        "status": "running"
    }

@app.get("/metrics", response_class=PlainTextResponse, summary="Prometheus指标")
async def metrics():
    """返回Prometheus格式的指标数据"""
    if not settings.ENABLE_METRICS:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics not enabled"}
        )
    
    return performance_monitor.get_metrics()

@app.get("/system/stats", summary="系统统计信息")
async def system_stats():
    """获取系统统计信息"""
    return {
        "database": db_manager.get_connection_stats(),
        "cache": cache_manager.get_stats(),
        "rate_limiter": rate_limiter.get_rate_limit_info("system"),
        "circuit_breaker": circuit_breaker_manager.get_breaker_stats(),
        "memory": memory_guard.get_memory_stats(),
        "health": performance_monitor.get_health_status(),
        "alerts": alert_manager.get_alert_stats()
    }

@app.get("/system/health", summary="系统健康检查")
async def system_health():
    """详细的系统健康检查"""
    health_status = performance_monitor.get_health_status()
    
    # 检查各个组件的健康状态
    components = {
        "database": db_manager.health_check(),
        "cache": await cache_manager.health_check(),
        "memory": memory_guard.check_memory_usage()
    }
    
    # 判断整体健康状态
    is_healthy = all(components.values()) and health_status['status'] == 'healthy'
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "components": components,
        "system": health_status,
        "timestamp": health_status['timestamp']
    }

@app.get("/system/alerts", summary="系统告警")
async def system_alerts():
    """获取系统告警信息"""
    active_alerts = alert_manager.get_active_alerts()
    alert_stats = alert_manager.get_alert_stats()
    
    return {
        "active_alerts": [alert.to_dict() for alert in active_alerts],
        "stats": alert_stats,
        "timestamp": time.time()
    }

@app.post("/system/alerts/{alert_id}/acknowledge", summary="确认告警")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = "system"):
    """确认告警"""
    await alert_manager.acknowledge_alert(alert_id, acknowledged_by)
    return {"message": "Alert acknowledged", "alert_id": alert_id}

@app.post("/system/alerts/{alert_id}/resolve", summary="解决告警")
async def resolve_alert(alert_id: str):
    """解决告警"""
    await alert_manager.resolve_alert(alert_id)
    return {"message": "Alert resolved", "alert_id": alert_id}

@app.post("/system/alerts/test", summary="测试告警")
async def test_alert():
    """创建测试告警"""
    from .alerting import AlertSeverity
    
    alert = await alert_manager.create_alert(
        title="测试告警",
        description="这是一个测试告警，用于验证告警系统功能",
        severity=AlertSeverity.LOW,
        source="manual_test",
        metadata={"test": True, "created_by": "api"}
    )
    
    return {"message": "Test alert created", "alert": alert.to_dict()}





