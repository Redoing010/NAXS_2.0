# 图存储模块
# 实现知识图谱的持久化存储、备份和恢复功能

import logging
import json
import sqlite3
import pickle
import gzip
import os
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from pathlib import Path
import shutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class StorageType(Enum):
    """存储类型"""
    SQLITE = "sqlite"              # SQLite数据库
    JSON = "json"                  # JSON文件
    PICKLE = "pickle"              # Python pickle
    MEMORY = "memory"              # 内存存储
    HYBRID = "hybrid"              # 混合存储

class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"                  # 无压缩
    GZIP = "gzip"                  # GZIP压缩
    LZMA = "lzma"                  # LZMA压缩

@dataclass
class StorageConfig:
    """存储配置"""
    storage_type: StorageType = StorageType.SQLITE
    storage_path: str = "./data/knowledge_graph"
    compression: CompressionType = CompressionType.NONE
    auto_backup: bool = True
    backup_interval: int = 3600  # 秒
    max_backups: int = 10
    batch_size: int = 1000
    enable_wal: bool = True  # SQLite WAL模式
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'storage_type': self.storage_type.value,
            'storage_path': self.storage_path,
            'compression': self.compression.value,
            'auto_backup': self.auto_backup,
            'backup_interval': self.backup_interval,
            'max_backups': self.max_backups,
            'batch_size': self.batch_size,
            'enable_wal': self.enable_wal
        }

@dataclass
class StorageStats:
    """存储统计"""
    total_nodes: int = 0
    total_edges: int = 0
    total_knowledge: int = 0
    storage_size: int = 0  # 字节
    last_backup: Optional[datetime] = None
    last_access: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'total_knowledge': self.total_knowledge,
            'storage_size': self.storage_size,
            'storage_size_mb': round(self.storage_size / 1024 / 1024, 2),
            'last_backup': self.last_backup.isoformat() if self.last_backup else None,
            'last_access': self.last_access.isoformat() if self.last_access else None
        }

class SQLiteStorage:
    """SQLite存储实现"""
    
    def __init__(self, db_path: str, enable_wal: bool = True):
        self.db_path = db_path
        self.enable_wal = enable_wal
        self.lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"SQLite存储初始化完成: {db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            # 启用WAL模式
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            
            # 创建节点表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    properties TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建边表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    properties TEXT,
                    weight REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_id) REFERENCES nodes (id)
                )
            """)
            
            # 创建知识表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    metadata TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges (type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_subject ON knowledge (subject)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_predicate ON knowledge (predicate)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_object ON knowledge (object)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge (type)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> bool:
        """保存节点"""
        try:
            with self.lock, self._get_connection() as conn:
                properties_json = json.dumps(properties, ensure_ascii=False)
                
                conn.execute("""
                    INSERT OR REPLACE INTO nodes (id, type, properties, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (node_id, node_type, properties_json))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存节点失败: {e}")
            return False
    
    def save_edge(self, edge_id: str, source_id: str, target_id: str, 
                 edge_type: str, properties: Dict[str, Any], weight: float = 1.0) -> bool:
        """保存边"""
        try:
            with self.lock, self._get_connection() as conn:
                properties_json = json.dumps(properties, ensure_ascii=False)
                
                conn.execute("""
                    INSERT OR REPLACE INTO edges (id, source_id, target_id, type, properties, weight, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (edge_id, source_id, target_id, edge_type, properties_json, weight))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存边失败: {e}")
            return False
    
    def save_knowledge(self, knowledge_id: str, knowledge_type: str, subject: str,
                      predicate: str, object_val: str, confidence: float = 1.0,
                      source: str = "", metadata: Dict[str, Any] = None,
                      tags: Set[str] = None) -> bool:
        """保存知识"""
        try:
            with self.lock, self._get_connection() as conn:
                metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
                tags_json = json.dumps(list(tags or set()), ensure_ascii=False)
                
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge 
                    (id, type, subject, predicate, object, confidence, source, metadata, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (knowledge_id, knowledge_type, subject, predicate, object_val, 
                      confidence, source, metadata_json, tags_json))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存知识失败: {e}")
            return False
    
    def load_nodes(self, node_type: str = None) -> List[Dict[str, Any]]:
        """加载节点"""
        try:
            with self.lock, self._get_connection() as conn:
                if node_type:
                    cursor = conn.execute(
                        "SELECT * FROM nodes WHERE type = ? ORDER BY created_at",
                        (node_type,)
                    )
                else:
                    cursor = conn.execute("SELECT * FROM nodes ORDER BY created_at")
                
                nodes = []
                for row in cursor:
                    node = dict(row)
                    node['properties'] = json.loads(node['properties']) if node['properties'] else {}
                    nodes.append(node)
                
                return nodes
        except Exception as e:
            logger.error(f"加载节点失败: {e}")
            return []
    
    def load_edges(self, edge_type: str = None) -> List[Dict[str, Any]]:
        """加载边"""
        try:
            with self.lock, self._get_connection() as conn:
                if edge_type:
                    cursor = conn.execute(
                        "SELECT * FROM edges WHERE type = ? ORDER BY created_at",
                        (edge_type,)
                    )
                else:
                    cursor = conn.execute("SELECT * FROM edges ORDER BY created_at")
                
                edges = []
                for row in cursor:
                    edge = dict(row)
                    edge['properties'] = json.loads(edge['properties']) if edge['properties'] else {}
                    edges.append(edge)
                
                return edges
        except Exception as e:
            logger.error(f"加载边失败: {e}")
            return []
    
    def load_knowledge(self, knowledge_type: str = None) -> List[Dict[str, Any]]:
        """加载知识"""
        try:
            with self.lock, self._get_connection() as conn:
                if knowledge_type:
                    cursor = conn.execute(
                        "SELECT * FROM knowledge WHERE type = ? ORDER BY created_at",
                        (knowledge_type,)
                    )
                else:
                    cursor = conn.execute("SELECT * FROM knowledge ORDER BY created_at")
                
                knowledge_list = []
                for row in cursor:
                    knowledge = dict(row)
                    knowledge['metadata'] = json.loads(knowledge['metadata']) if knowledge['metadata'] else {}
                    knowledge['tags'] = set(json.loads(knowledge['tags'])) if knowledge['tags'] else set()
                    knowledge_list.append(knowledge)
                
                return knowledge_list
        except Exception as e:
            logger.error(f"加载知识失败: {e}")
            return []
    
    def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        try:
            with self.lock, self._get_connection() as conn:
                # 删除相关的边
                conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                # 删除节点
                conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"删除节点失败: {e}")
            return False
    
    def delete_edge(self, edge_id: str) -> bool:
        """删除边"""
        try:
            with self.lock, self._get_connection() as conn:
                conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"删除边失败: {e}")
            return False
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        try:
            with self.lock, self._get_connection() as conn:
                conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"删除知识失败: {e}")
            return False
    
    def get_stats(self) -> StorageStats:
        """获取存储统计"""
        try:
            with self.lock, self._get_connection() as conn:
                # 统计节点数
                cursor = conn.execute("SELECT COUNT(*) FROM nodes")
                total_nodes = cursor.fetchone()[0]
                
                # 统计边数
                cursor = conn.execute("SELECT COUNT(*) FROM edges")
                total_edges = cursor.fetchone()[0]
                
                # 统计知识数
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
                total_knowledge = cursor.fetchone()[0]
                
                # 文件大小
                storage_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return StorageStats(
                    total_nodes=total_nodes,
                    total_edges=total_edges,
                    total_knowledge=total_knowledge,
                    storage_size=storage_size,
                    last_access=datetime.now()
                )
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return StorageStats()

class JSONStorage:
    """JSON文件存储实现"""
    
    def __init__(self, storage_path: str, compression: CompressionType = CompressionType.NONE):
        self.storage_path = Path(storage_path)
        self.compression = compression
        self.lock = threading.RLock()
        
        # 确保目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.nodes_file = self.storage_path / "nodes.json"
        self.edges_file = self.storage_path / "edges.json"
        self.knowledge_file = self.storage_path / "knowledge.json"
        
        if compression == CompressionType.GZIP:
            self.nodes_file = self.nodes_file.with_suffix(".json.gz")
            self.edges_file = self.edges_file.with_suffix(".json.gz")
            self.knowledge_file = self.knowledge_file.with_suffix(".json.gz")
        
        logger.info(f"JSON存储初始化完成: {storage_path}")
    
    def _save_json(self, data: Any, file_path: Path) -> bool:
        """保存JSON数据"""
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            if self.compression == CompressionType.GZIP:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(json_str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            return True
        except Exception as e:
            logger.error(f"保存JSON失败: {e}")
            return False
    
    def _load_json(self, file_path: Path) -> Any:
        """加载JSON数据"""
        try:
            if not file_path.exists():
                return []
            
            if self.compression == CompressionType.GZIP:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON失败: {e}")
            return []
    
    def save_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """保存节点"""
        with self.lock:
            return self._save_json(nodes, self.nodes_file)
    
    def save_edges(self, edges: List[Dict[str, Any]]) -> bool:
        """保存边"""
        with self.lock:
            return self._save_json(edges, self.edges_file)
    
    def save_knowledge(self, knowledge_list: List[Dict[str, Any]]) -> bool:
        """保存知识"""
        with self.lock:
            return self._save_json(knowledge_list, self.knowledge_file)
    
    def load_nodes(self) -> List[Dict[str, Any]]:
        """加载节点"""
        with self.lock:
            return self._load_json(self.nodes_file)
    
    def load_edges(self) -> List[Dict[str, Any]]:
        """加载边"""
        with self.lock:
            return self._load_json(self.edges_file)
    
    def load_knowledge(self) -> List[Dict[str, Any]]:
        """加载知识"""
        with self.lock:
            return self._load_json(self.knowledge_file)
    
    def get_stats(self) -> StorageStats:
        """获取存储统计"""
        try:
            nodes = self.load_nodes()
            edges = self.load_edges()
            knowledge = self.load_knowledge()
            
            # 计算文件大小
            storage_size = 0
            for file_path in [self.nodes_file, self.edges_file, self.knowledge_file]:
                if file_path.exists():
                    storage_size += file_path.stat().st_size
            
            return StorageStats(
                total_nodes=len(nodes),
                total_edges=len(edges),
                total_knowledge=len(knowledge),
                storage_size=storage_size,
                last_access=datetime.now()
            )
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return StorageStats()

class GraphStorage:
    """图存储管理器
    
    统一管理不同类型的存储后端
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.storage_backend = None
        self.backup_thread = None
        self.lock = threading.RLock()
        
        # 初始化存储后端
        self._init_storage_backend()
        
        # 启动自动备份
        if config.auto_backup:
            self._start_auto_backup()
        
        logger.info(f"图存储管理器初始化完成: {config.storage_type.value}")
    
    def _init_storage_backend(self):
        """初始化存储后端"""
        if self.config.storage_type == StorageType.SQLITE:
            db_path = os.path.join(self.config.storage_path, "graph.db")
            self.storage_backend = SQLiteStorage(db_path, self.config.enable_wal)
        elif self.config.storage_type == StorageType.JSON:
            self.storage_backend = JSONStorage(self.config.storage_path, self.config.compression)
        else:
            raise ValueError(f"不支持的存储类型: {self.config.storage_type}")
    
    def _start_auto_backup(self):
        """启动自动备份"""
        def backup_worker():
            while True:
                try:
                    threading.Event().wait(self.config.backup_interval)
                    self.create_backup()
                except Exception as e:
                    logger.error(f"自动备份失败: {e}")
        
        self.backup_thread = threading.Thread(target=backup_worker, daemon=True)
        self.backup_thread.start()
        logger.info(f"自动备份已启动，间隔: {self.config.backup_interval}秒")
    
    def save_graph_data(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                       knowledge_list: List[Dict[str, Any]]) -> bool:
        """保存图数据
        
        Args:
            nodes: 节点列表
            edges: 边列表
            knowledge_list: 知识列表
            
        Returns:
            是否保存成功
        """
        try:
            with self.lock:
                if isinstance(self.storage_backend, SQLiteStorage):
                    # SQLite存储
                    success = True
                    
                    # 保存节点
                    for node in nodes:
                        if not self.storage_backend.save_node(
                            node['id'], node['type'], node.get('properties', {})
                        ):
                            success = False
                    
                    # 保存边
                    for edge in edges:
                        if not self.storage_backend.save_edge(
                            edge['id'], edge['source_id'], edge['target_id'],
                            edge['type'], edge.get('properties', {}), edge.get('weight', 1.0)
                        ):
                            success = False
                    
                    # 保存知识
                    for knowledge in knowledge_list:
                        if not self.storage_backend.save_knowledge(
                            knowledge['id'], knowledge['type'], knowledge['subject'],
                            knowledge['predicate'], knowledge['object'],
                            knowledge.get('confidence', 1.0), knowledge.get('source', ''),
                            knowledge.get('metadata', {}), knowledge.get('tags', set())
                        ):
                            success = False
                    
                    return success
                    
                elif isinstance(self.storage_backend, JSONStorage):
                    # JSON存储
                    return (self.storage_backend.save_nodes(nodes) and
                           self.storage_backend.save_edges(edges) and
                           self.storage_backend.save_knowledge(knowledge_list))
                
                return False
        except Exception as e:
            logger.error(f"保存图数据失败: {e}")
            return False
    
    def load_graph_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """加载图数据
        
        Returns:
            (节点列表, 边列表, 知识列表)
        """
        try:
            with self.lock:
                if isinstance(self.storage_backend, SQLiteStorage):
                    nodes = self.storage_backend.load_nodes()
                    edges = self.storage_backend.load_edges()
                    knowledge_list = self.storage_backend.load_knowledge()
                elif isinstance(self.storage_backend, JSONStorage):
                    nodes = self.storage_backend.load_nodes()
                    edges = self.storage_backend.load_edges()
                    knowledge_list = self.storage_backend.load_knowledge()
                else:
                    return [], [], []
                
                logger.info(f"加载图数据完成: {len(nodes)} 节点, {len(edges)} 边, {len(knowledge_list)} 知识")
                return nodes, edges, knowledge_list
        except Exception as e:
            logger.error(f"加载图数据失败: {e}")
            return [], [], []
    
    def create_backup(self, backup_name: str = None) -> bool:
        """创建备份
        
        Args:
            backup_name: 备份名称
            
        Returns:
            是否备份成功
        """
        try:
            if not backup_name:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_dir = Path(self.config.storage_path) / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制存储文件
            source_dir = Path(self.config.storage_path)
            for item in source_dir.iterdir():
                if item.is_file() and item.name != "backups":
                    shutil.copy2(item, backup_dir / item.name)
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            logger.info(f"备份创建完成: {backup_name}")
            return True
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False
    
    def restore_backup(self, backup_name: str) -> bool:
        """恢复备份
        
        Args:
            backup_name: 备份名称
            
        Returns:
            是否恢复成功
        """
        try:
            backup_dir = Path(self.config.storage_path) / "backups" / backup_name
            if not backup_dir.exists():
                logger.error(f"备份不存在: {backup_name}")
                return False
            
            # 恢复文件
            source_dir = Path(self.config.storage_path)
            for item in backup_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, source_dir / item.name)
            
            # 重新初始化存储后端
            self._init_storage_backend()
            
            logger.info(f"备份恢复完成: {backup_name}")
            return True
        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份
        
        Returns:
            备份列表
        """
        try:
            backups_dir = Path(self.config.storage_path) / "backups"
            if not backups_dir.exists():
                return []
            
            backups = []
            for backup_dir in backups_dir.iterdir():
                if backup_dir.is_dir():
                    # 计算备份大小
                    size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
                    
                    backups.append({
                        'name': backup_dir.name,
                        'created_at': datetime.fromtimestamp(backup_dir.stat().st_ctime),
                        'size': size,
                        'size_mb': round(size / 1024 / 1024, 2)
                    })
            
            # 按创建时间排序
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            return backups
        except Exception as e:
            logger.error(f"列出备份失败: {e}")
            return []
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        try:
            backups = self.list_backups()
            if len(backups) > self.config.max_backups:
                # 删除最旧的备份
                for backup in backups[self.config.max_backups:]:
                    backup_dir = Path(self.config.storage_path) / "backups" / backup['name']
                    shutil.rmtree(backup_dir)
                    logger.debug(f"删除旧备份: {backup['name']}")
        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")
    
    def get_storage_stats(self) -> StorageStats:
        """获取存储统计
        
        Returns:
            存储统计信息
        """
        if hasattr(self.storage_backend, 'get_stats'):
            return self.storage_backend.get_stats()
        else:
            return StorageStats()
    
    def optimize_storage(self) -> bool:
        """优化存储
        
        Returns:
            是否优化成功
        """
        try:
            if isinstance(self.storage_backend, SQLiteStorage):
                # SQLite优化
                with self.storage_backend._get_connection() as conn:
                    conn.execute("VACUUM")
                    conn.execute("ANALYZE")
                    conn.commit()
                
                logger.info("SQLite存储优化完成")
                return True
            
            return True
        except Exception as e:
            logger.error(f"存储优化失败: {e}")
            return False
    
    def close(self):
        """关闭存储"""
        try:
            # 停止自动备份
            if self.backup_thread and self.backup_thread.is_alive():
                # 这里应该有一个停止标志，简化实现
                pass
            
            logger.info("图存储已关闭")
        except Exception as e:
            logger.error(f"关闭存储失败: {e}")

# 便捷函数
def create_storage_config(storage_type: StorageType = StorageType.SQLITE,
                         storage_path: str = "./data/knowledge_graph",
                         compression: CompressionType = CompressionType.NONE,
                         auto_backup: bool = True) -> StorageConfig:
    """创建存储配置
    
    Args:
        storage_type: 存储类型
        storage_path: 存储路径
        compression: 压缩类型
        auto_backup: 是否自动备份
        
    Returns:
        存储配置
    """
    return StorageConfig(
        storage_type=storage_type,
        storage_path=storage_path,
        compression=compression,
        auto_backup=auto_backup
    )

def create_graph_storage(config: StorageConfig = None) -> GraphStorage:
    """创建图存储
    
    Args:
        config: 存储配置
        
    Returns:
        图存储实例
    """
    if config is None:
        config = create_storage_config()
    
    return GraphStorage(config)