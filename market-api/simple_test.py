#!/usr/bin/env python3
"""
简化版API测试服务器
用于验证基本功能
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import psutil
import uvicorn

app = FastAPI(title="NAXS Test API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": "NAXS Test API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "NAXS Test API"
    }

@app.get("/system/stats")
async def system_stats():
    try:
        # 获取系统信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                }
            },
            "api": {
                "status": "running",
                "uptime": time.time() - start_time
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/performance/test")
async def performance_test():
    """性能测试端点"""
    start = time.time()
    
    # 模拟一些计算
    result = sum(i * i for i in range(10000))
    
    end = time.time()
    
    return {
        "computation_result": result,
        "execution_time_ms": (end - start) * 1000,
        "timestamp": time.time()
    }

# 记录启动时间
start_time = time.time()

if __name__ == "__main__":
    print("🚀 Starting NAXS Test API Server...")
    uvicorn.run(
        "simple_test:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )