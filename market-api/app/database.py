from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
import asyncio
import logging
import time
from .config import settings

logger = logging.getLogger(__name__)

# Database Base
Base = declarative_base()

class DatabaseManager:
    """数据库连接池管理器"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connection_count = 0
        self._active_connections = set()
        self._connection_stats = {
            'total_created': 0,
            'total_closed': 0,
            'current_active': 0,
            'peak_active': 0,
            'total_queries': 0,
            'avg_query_time': 0.0
        }
    
    def initialize(self):
        """初始化数据库连接池"""
        try:
            # 根据数据库类型选择合适的连接池
            if settings.DATABASE_URL.startswith('sqlite'):
                # SQLite 使用 StaticPool
                self.engine = create_engine(
                    settings.DATABASE_URL,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": settings.DB_POOL_TIMEOUT
                    },
                    echo=settings.DB_ECHO,
                    pool_pre_ping=True,
                    pool_recycle=settings.DB_POOL_RECYCLE
                )
            else:
                # PostgreSQL/MySQL 使用 QueuePool
                self.engine = create_engine(
                    settings.DATABASE_URL,
                    poolclass=QueuePool,
                    pool_size=settings.DB_POOL_SIZE,
                    max_overflow=settings.DB_MAX_OVERFLOW,
                    pool_timeout=settings.DB_POOL_TIMEOUT,
                    pool_recycle=settings.DB_POOL_RECYCLE,
                    pool_pre_ping=True,
                    echo=settings.DB_ECHO
                )
            
            # 注册连接事件监听器
            self._register_connection_events()
            
            # 创建会话工厂
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(f"Database initialized with pool_size={settings.DB_POOL_SIZE}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _register_connection_events(self):
        """注册连接事件监听器"""
        
        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self._connection_count += 1
            self._connection_stats['total_created'] += 1
            self._connection_stats['current_active'] += 1
            
            if self._connection_stats['current_active'] > self._connection_stats['peak_active']:
                self._connection_stats['peak_active'] = self._connection_stats['current_active']
            
            connection_id = id(dbapi_connection)
            self._active_connections.add(connection_id)
            
            logger.debug(f"Database connection created: {connection_id}")
        
        @event.listens_for(self.engine, "close")
        def on_close(dbapi_connection, connection_record):
            connection_id = id(dbapi_connection)
            if connection_id in self._active_connections:
                self._active_connections.remove(connection_id)
                self._connection_stats['total_closed'] += 1
                self._connection_stats['current_active'] -= 1
                
                logger.debug(f"Database connection closed: {connection_id}")
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
            self._connection_stats['total_queries'] += 1
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            
            # 更新平均查询时间
            current_avg = self._connection_stats['avg_query_time']
            total_queries = self._connection_stats['total_queries']
            self._connection_stats['avg_query_time'] = (
                (current_avg * (total_queries - 1) + total_time) / total_queries
            )
    
    @contextmanager
    def get_db_session(self) -> Generator[Session, None, None]:
        """获取数据库会话上下文管理器"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_connection_stats(self) -> dict:
        """获取连接池统计信息"""
        pool_stats = {}
        
        if self.engine and hasattr(self.engine, 'pool'):
            pool = self.engine.pool
            
            # 检查连接池类型并获取相应的统计信息
            if hasattr(pool, 'size'):
                # QueuePool 有这些方法
                pool_stats.update({
                    'pool_size': pool.size(),
                    'checked_in': pool.checkedin(),
                    'checked_out': pool.checked_out(),
                    'overflow': pool.overflow() if hasattr(pool, 'overflow') else 0,
                })
            else:
                # StaticPool 或其他类型的连接池
                pool_stats.update({
                    'pool_size': 1,  # StaticPool 通常只有一个连接
                    'checked_in': 0,
                    'checked_out': len(self._active_connections),
                    'overflow': 0,
                })
        else:
            pool_stats = {
                'pool_size': 0,
                'checked_in': 0,
                'checked_out': 0,
                'overflow': 0,
            }
        
        return {
            **self._connection_stats,
            **pool_stats
        }
    
    def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            with self.get_db_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def cleanup_connections(self):
        """清理空闲连接"""
        if self.engine:
            try:
                self.engine.dispose()
                logger.info("Database connections cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup connections: {e}")

# 全局数据库管理器实例
db_manager = DatabaseManager()

# 依赖注入函数
def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入函数"""
    with db_manager.get_db_session() as session:
        yield session

# 初始化函数
def init_database():
    """初始化数据库"""
    db_manager.initialize()

# 清理函数
def cleanup_database():
    """清理数据库连接"""
    db_manager.cleanup_connections()