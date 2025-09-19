#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAXS系统启动脚本

集成监控、错误处理和健康检查功能的系统启动器
"""

import sys
import os
import logging
import signal
import time
from pathlib import Path
from typing import Optional
import threading

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入监控模块
try:
    from modules.monitoring.system_monitor import get_system_monitor, start_monitoring, stop_monitoring
    from modules.monitoring.error_handler import get_error_handler
    from modules.monitoring.health_checks import (
        get_health_registry, register_health_check, 
        APIHealthChecker, DatabaseHealthChecker
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"警告: 监控模块不可用: {e}")
    MONITORING_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemManager:
    """系统管理器"""
    
    def __init__(self):
        self.monitoring_started = False
        self.api_process = None
        self.shutdown_event = threading.Event()
        
        # 创建必要目录
        self._create_directories()
        
        # 初始化监控（如果可用）
        if MONITORING_AVAILABLE:
            self._setup_monitoring()
        
        logger.info("系统管理器初始化完成")
    
    def _create_directories(self):
        """创建必要目录"""
        directories = [
            "data",
            "data/parquet",
            "data/qlib", 
            "data/cache",
            "logs",
            "reports"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("系统目录创建完成")
    
    def _setup_monitoring(self):
        """设置监控"""
        try:
            # 获取监控器实例
            monitor = get_system_monitor()
            error_handler = get_error_handler()
            
            # 设置告警处理器
            def alert_handler(alert_type: str, data: dict):
                logger.warning(f"系统告警 [{alert_type}]: {data.get('message', 'Unknown')}")
            
            monitor.add_alert_handler(alert_handler)
            
            # 设置全局错误回调
            def global_error_callback(error: Exception, context):
                logger.error(f"全局错误 [{context.module}.{context.function}]: {error}")
            
            error_handler.set_global_error_callback(global_error_callback)
            
            # 注册自定义健康检查
            self._register_custom_health_checks()
            
            logger.info("监控系统设置完成")
            
        except Exception as e:
            logger.error(f"监控设置失败: {e}")
    
    def _register_custom_health_checks(self):
        """注册自定义健康检查"""
        try:
            # API健康检查
            api_checker = APIHealthChecker()
            register_health_check("api_health", api_checker.check_api_health)
            register_health_check("api_data_endpoints", api_checker.check_data_endpoints)
            register_health_check("api_admin_endpoints", api_checker.check_admin_endpoints)
            
            # 数据库健康检查
            db_checker = DatabaseHealthChecker()
            register_health_check("parquet_files", db_checker.check_parquet_files)
            register_health_check("qlib_data", db_checker.check_qlib_data)
            
            # 自定义业务检查
            def check_data_freshness():
                """检查数据新鲜度"""
                try:
                    from datetime import datetime, timedelta
                    import os
                    
                    # 检查最近是否有数据更新
                    parquet_dir = Path("data/parquet")
                    if not parquet_dir.exists():
                        return True  # 没有数据不算错误
                    
                    # 查找最新的parquet文件
                    parquet_files = list(parquet_dir.rglob("*.parquet"))
                    if not parquet_files:
                        return True  # 没有文件不算错误
                    
                    # 检查最新文件的修改时间
                    latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
                    file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
                    
                    # 如果数据超过7天没更新，认为不新鲜
                    return file_age.days < 7
                    
                except Exception:
                    return True  # 检查失败不算错误
            
            register_health_check("data_freshness", check_data_freshness)
            
            logger.info("自定义健康检查注册完成")
            
        except Exception as e:
            logger.error(f"注册健康检查失败: {e}")
    
    def start_monitoring(self):
        """启动监控"""
        if not MONITORING_AVAILABLE:
            logger.warning("监控模块不可用，跳过监控启动")
            return
        
        try:
            start_monitoring()
            self.monitoring_started = True
            logger.info("系统监控已启动")
        except Exception as e:
            logger.error(f"启动监控失败: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        if not MONITORING_AVAILABLE or not self.monitoring_started:
            return
        
        try:
            stop_monitoring()
            self.monitoring_started = False
            logger.info("系统监控已停止")
        except Exception as e:
            logger.error(f"停止监控失败: {e}")
    
    def start_api_server(self):
        """启动API服务器"""
        try:
            import subprocess
            
            # 启动FastAPI服务器
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "market-api.app.main:app",
                "--host", "0.0.0.0",
                "--port", "3001",
                "--reload"
            ]
            
            logger.info("启动API服务器...")
            self.api_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待服务器启动
            time.sleep(3)
            
            if self.api_process.poll() is None:
                logger.info("API服务器启动成功 (PID: {})".format(self.api_process.pid))
            else:
                logger.error("API服务器启动失败")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"启动API服务器失败: {e}")
            return False
    
    def stop_api_server(self):
        """停止API服务器"""
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=10)
                logger.info("API服务器已停止")
            except subprocess.TimeoutExpired:
                self.api_process.kill()
                logger.warning("强制终止API服务器")
            except Exception as e:
                logger.error(f"停止API服务器失败: {e}")
    
    def run_health_check(self):
        """运行健康检查"""
        if not MONITORING_AVAILABLE:
            logger.info("监控模块不可用，跳过健康检查")
            return
        
        try:
            from modules.monitoring.health_checks import get_system_health_summary
            
            health_summary = get_system_health_summary()
            
            logger.info("=== 系统健康检查 ===")
            logger.info(f"整体状态: {health_summary['overall_status']}")
            logger.info(f"健康状态: {'✅' if health_summary['is_healthy'] else '❌'}")
            logger.info(f"检查项目: {health_summary['passed_checks']}/{health_summary['total_checks']} 通过")
            
            if health_summary['failed_check_names']:
                logger.warning(f"失败项目: {', '.join(health_summary['failed_check_names'])}")
            
            logger.info("====================")
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
    
    def start_system(self):
        """启动完整系统"""
        logger.info("开始启动NAXS系统")
        
        # 1. 启动监控
        self.start_monitoring()
        
        # 2. 运行健康检查
        self.run_health_check()
        
        # 3. 启动API服务器
        if not self.start_api_server():
            logger.error("API服务器启动失败，系统启动中止")
            return False
        
        # 4. 等待一段时间后再次检查
        time.sleep(5)
        self.run_health_check()
        
        logger.info("🎉 NAXS系统启动完成")
        logger.info("API服务地址: http://localhost:3001")
        logger.info("API文档地址: http://localhost:3001/docs")
        logger.info("健康检查: http://localhost:3001/health")
        
        return True
    
    def stop_system(self):
        """停止系统"""
        logger.info("开始停止NAXS系统")
        
        # 设置停止事件
        self.shutdown_event.set()
        
        # 停止API服务器
        self.stop_api_server()
        
        # 停止监控
        self.stop_monitoring()
        
        logger.info("NAXS系统已停止")
    
    def wait_for_shutdown(self):
        """等待关闭信号"""
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号")


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，准备关闭系统")
    if 'system_manager' in globals():
        system_manager.stop_system()
    sys.exit(0)


def main():
    """主函数"""
    global system_manager
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 创建系统管理器
        system_manager = SystemManager()
        
        # 启动系统
        if system_manager.start_system():
            # 等待关闭信号
            system_manager.wait_for_shutdown()
        else:
            logger.error("系统启动失败")
            return 1
            
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        return 1
    finally:
        if 'system_manager' in locals():
            system_manager.stop_system()
    
    return 0


if __name__ == '__main__':
    sys.exit(main