#!/usr/bin/env python3
"""
NAXS Market API 部署和运维脚本

提供自动化部署、健康检查、性能监控等功能
"""

import os
import sys
import time
import json
import subprocess
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional
import psutil

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, config_file: str = "deploy_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.api_url = self.config.get('api_url', 'http://localhost:8000')
        
    def load_config(self) -> Dict:
        """加载部署配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            default_config = {
                "api_url": "http://localhost:8000",
                "workers": 4,
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "info",
                "reload": False,
                "environment": "production",
                "database_url": "sqlite:///./naxs.db",
                "redis_url": "redis://localhost:6379/0",
                "enable_metrics": True,
                "health_check_timeout": 30,
                "deployment_timeout": 300
            }
            
            # 保存默认配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            return default_config
    
    def install_dependencies(self) -> bool:
        """安装依赖包"""
        print("📦 Installing dependencies...")
        
        try:
            # 升级pip
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # 安装requirements
            if os.path.exists('requirements.txt'):
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                             check=True, capture_output=True)
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ requirements.txt not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """设置环境变量"""
        print("🔧 Setting up environment...")
        
        env_vars = {
            'DATABASE_URL': self.config.get('database_url'),
            'REDIS_URL': self.config.get('redis_url'),
            'LOG_LEVEL': self.config.get('log_level', 'info').upper(),
            'ENABLE_METRICS': str(self.config.get('enable_metrics', True)).lower(),
            'ENVIRONMENT': self.config.get('environment', 'production')
        }
        
        # 创建.env文件
        env_content = []
        for key, value in env_vars.items():
            if value:
                env_content.append(f"{key}={value}")
                os.environ[key] = str(value)
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_content))
        
        print("✅ Environment setup completed")
        return True
    
    def start_server(self, background: bool = False) -> Optional[subprocess.Popen]:
        """启动服务器"""
        print("🚀 Starting NAXS Market API server...")
        
        cmd = [
            'uvicorn',
            'app.main:app',
            '--host', self.config.get('host', '0.0.0.0'),
            '--port', str(self.config.get('port', 8000)),
            '--workers', str(self.config.get('workers', 4)),
            '--log-level', self.config.get('log_level', 'info')
        ]
        
        if self.config.get('reload', False):
            cmd.append('--reload')
        
        try:
            if background:
                # 后台运行
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"✅ Server started in background (PID: {process.pid})")
                return process
            else:
                # 前台运行
                subprocess.run(cmd, check=True)
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start server: {e}")
            return None
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            return None
    
    def health_check(self, timeout: int = 30) -> bool:
        """健康检查"""
        print("🏥 Performing health check...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        print("✅ Health check passed")
                        return True
                
            except requests.RequestException:
                pass
            
            print("⏳ Waiting for server to be ready...")
            time.sleep(2)
        
        print("❌ Health check failed - server not responding")
        return False
    
    def detailed_health_check(self) -> Dict:
        """详细健康检查"""
        print("🔍 Performing detailed health check...")
        
        try:
            response = requests.get(f"{self.api_url}/health/detailed", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                print("📊 Health Check Results:")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  API Status: {data.get('api_status', 'unknown')}")
                
                components = data.get('components', {})
                for component, info in components.items():
                    status = info.get('status', 'unknown')
                    emoji = '✅' if status == 'healthy' else '⚠️' if status == 'degraded' else '❌'
                    print(f"  {component}: {emoji} {status}")
                
                return data
            else:
                print(f"❌ Detailed health check failed: HTTP {response.status_code}")
                return {}
                
        except requests.RequestException as e:
            print(f"❌ Detailed health check failed: {e}")
            return {}
    
    def performance_check(self) -> Dict:
        """性能检查"""
        print("⚡ Performing performance check...")
        
        try:
            # 系统统计
            response = requests.get(f"{self.api_url}/system/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                
                print("📈 Performance Metrics:")
                
                # 数据库性能
                db_stats = stats.get('database', {})
                print(f"  Database Connections: {db_stats.get('current_active', 0)}")
                print(f"  Database Queries: {db_stats.get('total_queries', 0)}")
                
                # 缓存性能
                cache_stats = stats.get('cache', {})
                print(f"  Cache Hit Rate: {cache_stats.get('memory_hit_rate', 0):.2%}")
                print(f"  Cache Size: {cache_stats.get('memory_cache_size', 0)}")
                
                # 系统健康
                health = stats.get('health', {})
                print(f"  CPU Usage: {health.get('cpu_usage', 0):.1f}%")
                print(f"  Memory Usage: {health.get('memory_usage', {}).get('percent', 0):.1f}%")
                print(f"  Concurrent Requests: {health.get('concurrent_requests', 0)}")
                
                return stats
            else:
                print(f"❌ Performance check failed: HTTP {response.status_code}")
                return {}
                
        except requests.RequestException as e:
            print(f"❌ Performance check failed: {e}")
            return {}
    
    def run_stress_test(self, users: int = 100, duration: int = 60) -> bool:
        """运行压力测试"""
        print(f"🔥 Running stress test ({users} users, {duration}s)...")
        
        try:
            # 检查locust是否安装
            subprocess.run(['locust', '--version'], 
                         check=True, capture_output=True)
            
            # 运行压力测试
            cmd = [
                'locust',
                '-f', 'stress_test.py',
                '--host', self.api_url,
                '--users', str(users),
                '--spawn-rate', str(min(users // 10, 10)),
                '--run-time', f"{duration}s",
                '--headless',
                '--html', 'stress_test_report.html'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Stress test completed successfully")
                print("📊 Report saved to: stress_test_report.html")
                return True
            else:
                print(f"❌ Stress test failed: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError:
            print("❌ Locust not installed. Install with: pip install locust")
            return False
        except FileNotFoundError:
            print("❌ stress_test.py not found")
            return False
    
    def backup_database(self) -> bool:
        """备份数据库"""
        print("💾 Backing up database...")
        
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            backup_dir = Path('backups')
            backup_dir.mkdir(exist_ok=True)
            
            db_url = self.config.get('database_url', 'sqlite:///./naxs.db')
            
            if db_url.startswith('sqlite'):
                # SQLite备份
                db_file = db_url.replace('sqlite:///', '')
                if os.path.exists(db_file):
                    backup_file = backup_dir / f"naxs_backup_{timestamp}.db"
                    subprocess.run(['cp', db_file, str(backup_file)], check=True)
                    print(f"✅ Database backed up to: {backup_file}")
                    return True
                else:
                    print(f"❌ Database file not found: {db_file}")
                    return False
            else:
                # PostgreSQL/MySQL备份需要额外工具
                print("⚠️ Non-SQLite database backup not implemented")
                return False
                
        except Exception as e:
            print(f"❌ Database backup failed: {e}")
            return False
    
    def monitor_system(self, duration: int = 60) -> None:
        """系统监控"""
        print(f"📊 Monitoring system for {duration} seconds...")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 系统资源
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # API健康状态
                try:
                    response = requests.get(f"{self.api_url}/system/stats", timeout=5)
                    if response.status_code == 200:
                        stats = response.json()
                        health = stats.get('health', {})
                        
                        print(f"\r🖥️  CPU: {cpu_percent:5.1f}% | "
                              f"💾 Memory: {memory.percent:5.1f}% | "
                              f"🔗 Requests: {health.get('concurrent_requests', 0):3d} | "
                              f"⏱️  Uptime: {health.get('uptime', 0):6.0f}s", end="")
                    else:
                        print(f"\r❌ API not responding (HTTP {response.status_code})", end="")
                        
                except requests.RequestException:
                    print(f"\r❌ API not responding (Connection error)", end="")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        print("\n✅ Monitoring completed")
    
    def deploy(self, skip_deps: bool = False, skip_tests: bool = False) -> bool:
        """完整部署流程"""
        print("🚀 Starting NAXS Market API deployment...")
        
        # 1. 安装依赖
        if not skip_deps:
            if not self.install_dependencies():
                return False
        
        # 2. 设置环境
        if not self.setup_environment():
            return False
        
        # 3. 备份数据库
        self.backup_database()
        
        # 4. 启动服务器
        process = self.start_server(background=True)
        if not process:
            return False
        
        try:
            # 5. 健康检查
            if not self.health_check(self.config.get('health_check_timeout', 30)):
                return False
            
            # 6. 详细健康检查
            self.detailed_health_check()
            
            # 7. 性能检查
            self.performance_check()
            
            # 8. 运行测试
            if not skip_tests:
                print("🧪 Running basic tests...")
                # 这里可以添加基本的API测试
                
            print("\n🎉 Deployment completed successfully!")
            print(f"🌐 API URL: {self.api_url}")
            print(f"📚 API Docs: {self.api_url}/docs")
            print(f"📊 Metrics: {self.api_url}/metrics")
            
            return True
            
        finally:
            # 清理
            if process and process.poll() is None:
                print("🧹 Cleaning up...")
                process.terminate()
                process.wait(timeout=10)

def main():
    parser = argparse.ArgumentParser(description='NAXS Market API Deployment Manager')
    parser.add_argument('command', choices=[
        'deploy', 'start', 'health', 'performance', 'stress', 'monitor', 'backup'
    ], help='Command to execute')
    
    parser.add_argument('--config', default='deploy_config.json', 
                       help='Configuration file path')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip tests during deployment')
    parser.add_argument('--users', type=int, default=100, 
                       help='Number of users for stress test')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration in seconds for stress test or monitoring')
    
    args = parser.parse_args()
    
    manager = DeploymentManager(args.config)
    
    if args.command == 'deploy':
        success = manager.deploy(args.skip_deps, args.skip_tests)
        sys.exit(0 if success else 1)
        
    elif args.command == 'start':
        manager.start_server(background=False)
        
    elif args.command == 'health':
        if manager.health_check():
            manager.detailed_health_check()
        
    elif args.command == 'performance':
        manager.performance_check()
        
    elif args.command == 'stress':
        success = manager.run_stress_test(args.users, args.duration)
        sys.exit(0 if success else 1)
        
    elif args.command == 'monitor':
        manager.monitor_system(args.duration)
        
    elif args.command == 'backup':
        success = manager.backup_database()
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()