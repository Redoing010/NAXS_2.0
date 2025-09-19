import time
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import psutil
import structlog

# 导入监控模块
try:
    from ..monitoring import performance_monitor, memory_profiler
    from ..database import db_manager
    from ..cache import cache_manager
    from ..middleware import memory_guard, circuit_breaker_manager
except ImportError as e:
    print(f"监控模块导入失败: {e}")

router = APIRouter(prefix="/system", tags=["系统监控"])
logger = structlog.get_logger(__name__)

# 数据模型
class SystemMetrics(BaseModel):
    api: Dict[str, float]
    database: Dict[str, Any]
    system: Dict[str, float]
    websocket: Dict[str, Any]

class TestResult(BaseModel):
    id: str
    name: str
    status: str
    duration: float
    timestamp: str
    details: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# 全局变量存储测试结果
test_results_storage: List[TestResult] = []
websocket_stats = {
    "connected": True,
    "active_connections": 0,
    "message_rate": 0
}

@router.get("/metrics", response_model=SystemMetrics, summary="获取系统指标")
async def get_system_metrics():
    """获取系统性能指标"""
    try:
        # API性能指标 - 使用模拟数据或从Prometheus指标计算
        health_status = performance_monitor.get_health_status()
        api_metrics = {
            "responseTime": 150.5,  # 模拟平均响应时间
            "successRate": 98.5,   # 模拟成功率
            "throughput": 45.2,    # 模拟吞吐量
            "errorRate": 1.5       # 模拟错误率
        }
        
        # 数据库指标
        db_stats = db_manager.get_connection_stats()
        db_health = db_manager.health_check()
        database_metrics = {
            "connectionCount": db_stats.get('active_connections', 0),
            "queryTime": db_stats.get('avg_query_time', 0),
            "cacheHitRate": 85.5,  # 模拟缓存命中率
            "status": "healthy" if db_health else "error"
        }
        
        # 系统资源指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('C:' if os.name == 'nt' else '/')
        
        system_metrics = {
            "cpuUsage": cpu_percent,
            "memoryUsage": memory.percent,
            "diskUsage": (disk.used / disk.total) * 100,
            "uptime": time.time() - performance_monitor._start_time if hasattr(performance_monitor, '_start_time') else 0
        }
        
        # WebSocket指标（模拟）
        websocket_metrics = {
            "connected": websocket_stats["connected"],
            "activeConnections": websocket_stats["active_connections"],
            "messageRate": websocket_stats["message_rate"]
        }
        
        return SystemMetrics(
            api=api_metrics,
            database=database_metrics,
            system=system_metrics,
            websocket=websocket_metrics
        )
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {str(e)}")

@router.get("/tests/results", response_model=List[TestResult], summary="获取测试结果")
async def get_test_results():
    """获取测试结果列表"""
    try:
        # 如果没有测试结果，返回一些示例数据
        if not test_results_storage:
            sample_results = [
                TestResult(
                    id="test_1",
                    name="API响应时间测试",
                    status="passed",
                    duration=150.5,
                    timestamp=datetime.now().isoformat(),
                    details="所有API端点响应时间均在200ms以内",
                    metrics={
                        "requests": 100,
                        "failures": 0,
                        "avgResponseTime": 145.2,
                        "maxResponseTime": 198.7
                    }
                ),
                TestResult(
                    id="test_2",
                    name="数据库连接测试",
                    status="passed",
                    duration=89.3,
                    timestamp=(datetime.now() - timedelta(minutes=5)).isoformat(),
                    details="数据库连接池正常，查询性能良好",
                    metrics={
                        "requests": 50,
                        "failures": 0,
                        "avgResponseTime": 45.6,
                        "maxResponseTime": 89.3
                    }
                ),
                TestResult(
                    id="test_3",
                    name="压力测试",
                    status="running",
                    duration=0,
                    timestamp=datetime.now().isoformat(),
                    details="正在执行1000并发用户压力测试"
                ),
                TestResult(
                    id="test_4",
                    name="缓存性能测试",
                    status="passed",
                    duration=234.1,
                    timestamp=(datetime.now() - timedelta(minutes=10)).isoformat(),
                    details="Redis缓存命中率达到85%以上",
                    metrics={
                        "requests": 200,
                        "failures": 2,
                        "avgResponseTime": 12.5,
                        "maxResponseTime": 45.8
                    }
                ),
                TestResult(
                    id="test_5",
                    name="内存泄漏检测",
                    status="failed",
                    duration=1205.7,
                    timestamp=(datetime.now() - timedelta(minutes=15)).isoformat(),
                    details="检测到潜在的内存泄漏问题",
                    metrics={
                        "requests": 1000,
                        "failures": 15,
                        "avgResponseTime": 180.3,
                        "maxResponseTime": 2500.1
                    }
                )
            ]
            test_results_storage.extend(sample_results)
        
        return test_results_storage
        
    except Exception as e:
        logger.error(f"获取测试结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取测试结果失败: {str(e)}")

@router.post("/tests/run/{test_type}", summary="运行测试")
async def run_tests(test_type: str, background_tasks: BackgroundTasks):
    """运行指定类型的测试"""
    try:
        # 创建新的测试结果
        test_id = f"test_{int(time.time())}_{test_type}"
        test_name = {
            "all": "完整系统测试",
            "api": "API性能测试",
            "stress": "压力测试",
            "integration": "集成测试"
        }.get(test_type, f"{test_type}测试")
        
        # 添加运行中的测试结果
        running_test = TestResult(
            id=test_id,
            name=test_name,
            status="running",
            duration=0,
            timestamp=datetime.now().isoformat(),
            details=f"正在执行{test_name}..."
        )
        
        test_results_storage.append(running_test)
        
        # 在后台运行测试
        background_tasks.add_task(execute_test, test_id, test_type)
        
        return {
            "message": f"已启动{test_name}",
            "test_id": test_id,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"启动测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动测试失败: {str(e)}")

async def execute_test(test_id: str, test_type: str):
    """执行测试的后台任务"""
    try:
        # 模拟测试执行时间
        test_duration = {
            "all": 30,
            "api": 10,
            "stress": 60,
            "integration": 20
        }.get(test_type, 15)
        
        await asyncio.sleep(test_duration)
        
        # 更新测试结果
        for i, result in enumerate(test_results_storage):
            if result.id == test_id:
                # 模拟测试结果
                import random
                success = random.choice([True, True, True, False])  # 75%成功率
                
                test_results_storage[i] = TestResult(
                    id=test_id,
                    name=result.name,
                    status="passed" if success else "failed",
                    duration=test_duration * 1000 + random.uniform(-500, 500),
                    timestamp=result.timestamp,
                    details="测试完成" if success else "测试失败，发现性能问题",
                    metrics={
                        "requests": random.randint(50, 500),
                        "failures": 0 if success else random.randint(1, 10),
                        "avgResponseTime": random.uniform(50, 200),
                        "maxResponseTime": random.uniform(200, 1000)
                    }
                )
                break
                
    except Exception as e:
        logger.error(f"执行测试失败: {e}")
        # 更新为失败状态
        for i, result in enumerate(test_results_storage):
            if result.id == test_id:
                test_results_storage[i] = TestResult(
                    id=test_id,
                    name=result.name,
                    status="failed",
                    duration=0,
                    timestamp=result.timestamp,
                    details=f"测试执行失败: {str(e)}"
                )
                break

@router.get("/status", summary="系统状态概览")
async def get_system_status():
    """获取系统状态概览"""
    try:
        # 获取基本系统信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('C:' if os.name == 'nt' else '/')
        
        # 获取网络统计
        network = psutil.net_io_counters()
        
        # 获取进程信息
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free": disk.free
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "process": {
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads()
            },
            "database": {
                "healthy": db_manager.health_check(),
                "connections": db_manager.get_connection_stats()
            }
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@router.delete("/tests/results", summary="清空测试结果")
async def clear_test_results():
    """清空所有测试结果"""
    try:
        global test_results_storage
        test_results_storage.clear()
        
        return {
            "message": "测试结果已清空",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"清空测试结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空测试结果失败: {str(e)}")

@router.get("/websocket/stats", summary="WebSocket统计")
async def get_websocket_stats():
    """获取WebSocket连接统计"""
    try:
        return {
            "connected": websocket_stats["connected"],
            "active_connections": websocket_stats["active_connections"],
            "message_rate": websocket_stats["message_rate"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取WebSocket统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取WebSocket统计失败: {str(e)}")

@router.post("/websocket/simulate", summary="模拟WebSocket活动")
async def simulate_websocket_activity(connections: int = 10, message_rate: int = 50):
    """模拟WebSocket活动用于测试"""
    try:
        global websocket_stats
        websocket_stats["active_connections"] = connections
        websocket_stats["message_rate"] = message_rate
        websocket_stats["connected"] = True
        
        return {
            "message": "WebSocket活动模拟已启动",
            "connections": connections,
            "message_rate": message_rate,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"模拟WebSocket活动失败: {e}")
        raise HTTPException(status_code=500, detail=f"模拟WebSocket活动失败: {str(e)}")