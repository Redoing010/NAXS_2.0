# 特征注册表 - 管理特征的元数据和版本信息

import logging
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class FeatureStatus(Enum):
    """特征状态枚举"""
    DRAFT = "draft"                  # 草稿
    ACTIVE = "active"                # 活跃
    DEPRECATED = "deprecated"        # 已弃用
    ARCHIVED = "archived"            # 已归档

class DataQuality(Enum):
    """数据质量等级"""
    UNKNOWN = "unknown"              # 未知
    POOR = "poor"                    # 差
    FAIR = "fair"                    # 一般
    GOOD = "good"                    # 良好
    EXCELLENT = "excellent"          # 优秀

@dataclass
class FeatureLineage:
    """特征血缘信息"""
    upstream_features: List[str] = field(default_factory=list)  # 上游特征
    downstream_features: List[str] = field(default_factory=list)  # 下游特征
    source_tables: List[str] = field(default_factory=list)  # 源表
    transformation_logic: str = ""  # 转换逻辑
    dependencies: List[str] = field(default_factory=list)  # 依赖项

@dataclass
class FeatureStatistics:
    """特征统计信息"""
    count: int = 0
    null_count: int = 0
    unique_count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)  # 分位数
    histogram: Dict[str, int] = field(default_factory=dict)  # 直方图
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class FeatureMetadata:
    """特征元数据"""
    # 基本信息
    name: str
    display_name: str = ""
    description: str = ""
    feature_type: str = "numerical"
    data_type: str = "float64"
    
    # 版本信息
    version: str = "1.0.0"
    status: FeatureStatus = FeatureStatus.DRAFT
    
    # 所有权信息
    owner: str = ""
    team: str = ""
    contact_email: str = ""
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None
    
    # 标签和分类
    tags: List[str] = field(default_factory=list)
    category: str = ""
    domain: str = ""
    
    # 数据质量
    quality_score: Optional[float] = None
    quality_level: DataQuality = DataQuality.UNKNOWN
    quality_issues: List[str] = field(default_factory=list)
    
    # 业务信息
    business_definition: str = ""
    use_cases: List[str] = field(default_factory=list)
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # 技术信息
    source_system: str = ""
    update_frequency: str = ""
    latency_requirements: str = ""
    
    # 血缘信息
    lineage: FeatureLineage = field(default_factory=FeatureLineage)
    
    # 统计信息
    statistics: FeatureStatistics = field(default_factory=FeatureStatistics)
    
    # 配置信息
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 验证规则
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # 访问控制
    access_groups: List[str] = field(default_factory=list)
    is_pii: bool = False  # 是否包含个人身份信息
    
    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.update_timestamp()
    
    def remove_tag(self, tag: str):
        """移除标签"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.update_timestamp()
    
    def update_statistics(self, stats: FeatureStatistics):
        """更新统计信息"""
        self.statistics = stats
        self.update_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理枚举类型
        data['status'] = self.status.value
        data['quality_level'] = self.quality_level.value
        # 处理日期时间
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.last_used_at:
            data['last_used_at'] = self.last_used_at.isoformat()
        data['statistics']['last_updated'] = self.statistics.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """从字典创建"""
        # 处理枚举类型
        if 'status' in data:
            data['status'] = FeatureStatus(data['status'])
        if 'quality_level' in data:
            data['quality_level'] = DataQuality(data['quality_level'])
        
        # 处理日期时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'last_used_at' in data and data['last_used_at']:
            data['last_used_at'] = datetime.fromisoformat(data['last_used_at'])
        
        # 处理嵌套对象
        if 'lineage' in data and isinstance(data['lineage'], dict):
            data['lineage'] = FeatureLineage(**data['lineage'])
        
        if 'statistics' in data and isinstance(data['statistics'], dict):
            stats_data = data['statistics']
            if 'last_updated' in stats_data and isinstance(stats_data['last_updated'], str):
                stats_data['last_updated'] = datetime.fromisoformat(stats_data['last_updated'])
            data['statistics'] = FeatureStatistics(**stats_data)
        
        return cls(**data)

class FeatureRegistry:
    """特征注册表
    
    管理特征的元数据、版本信息和血缘关系，提供：
    1. 特征元数据管理
    2. 版本控制
    3. 血缘追踪
    4. 搜索和发现
    5. 数据质量监控
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry_path = Path(config.get('registry_path', './feature_registry'))
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self.features: Dict[str, FeatureMetadata] = {}
        self.feature_versions: Dict[str, List[str]] = {}  # feature_name -> [versions]
        self.tags_index: Dict[str, Set[str]] = {}  # tag -> {feature_names}
        self.category_index: Dict[str, Set[str]] = {}  # category -> {feature_names}
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载现有注册表
        self._load_registry()
        
        logger.info(f"特征注册表初始化完成，路径: {self.registry_path}")
    
    def _get_feature_key(self, name: str, version: str = None) -> str:
        """生成特征键"""
        if version:
            return f"{name}@{version}"
        return name
    
    def _get_feature_file_path(self, name: str, version: str = None) -> Path:
        """获取特征文件路径"""
        key = self._get_feature_key(name, version)
        return self.registry_path / f"{key}.json"
    
    def _load_registry(self):
        """加载注册表"""
        try:
            with self.lock:
                # 扫描注册表目录
                for file_path in self.registry_path.glob("*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        metadata = FeatureMetadata.from_dict(data)
                        key = self._get_feature_key(metadata.name, metadata.version)
                        self.features[key] = metadata
                        
                        # 更新版本索引
                        if metadata.name not in self.feature_versions:
                            self.feature_versions[metadata.name] = []
                        if metadata.version not in self.feature_versions[metadata.name]:
                            self.feature_versions[metadata.name].append(metadata.version)
                        
                        # 更新标签索引
                        for tag in metadata.tags:
                            if tag not in self.tags_index:
                                self.tags_index[tag] = set()
                            self.tags_index[tag].add(metadata.name)
                        
                        # 更新分类索引
                        if metadata.category:
                            if metadata.category not in self.category_index:
                                self.category_index[metadata.category] = set()
                            self.category_index[metadata.category].add(metadata.name)
                        
                    except Exception as e:
                        logger.error(f"加载特征文件失败: {file_path}, 错误: {str(e)}")
                
                logger.info(f"加载特征注册表完成，特征数: {len(self.features)}")
                
        except Exception as e:
            logger.error(f"加载注册表失败: {str(e)}")
    
    def _save_feature(self, metadata: FeatureMetadata):
        """保存特征元数据"""
        try:
            file_path = self._get_feature_file_path(metadata.name, metadata.version)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"保存特征元数据失败: {str(e)}")
            raise
    
    def register_feature(self, metadata: FeatureMetadata, overwrite: bool = False) -> bool:
        """注册特征
        
        Args:
            metadata: 特征元数据
            overwrite: 是否覆盖已存在的特征
            
        Returns:
            注册是否成功
        """
        try:
            with self.lock:
                key = self._get_feature_key(metadata.name, metadata.version)
                
                # 检查是否已存在
                if key in self.features and not overwrite:
                    logger.warning(f"特征已存在: {key}")
                    return False
                
                # 更新时间戳
                if key not in self.features:
                    metadata.created_at = datetime.now()
                metadata.update_timestamp()
                
                # 保存到文件
                self._save_feature(metadata)
                
                # 更新内存缓存
                old_metadata = self.features.get(key)
                self.features[key] = metadata
                
                # 更新版本索引
                if metadata.name not in self.feature_versions:
                    self.feature_versions[metadata.name] = []
                if metadata.version not in self.feature_versions[metadata.name]:
                    self.feature_versions[metadata.name].append(metadata.version)
                    # 按版本排序
                    self.feature_versions[metadata.name].sort()
                
                # 更新标签索引
                if old_metadata:
                    # 移除旧标签
                    for tag in old_metadata.tags:
                        if tag in self.tags_index:
                            self.tags_index[tag].discard(metadata.name)
                            if not self.tags_index[tag]:
                                del self.tags_index[tag]
                
                for tag in metadata.tags:
                    if tag not in self.tags_index:
                        self.tags_index[tag] = set()
                    self.tags_index[tag].add(metadata.name)
                
                # 更新分类索引
                if old_metadata and old_metadata.category:
                    if old_metadata.category in self.category_index:
                        self.category_index[old_metadata.category].discard(metadata.name)
                        if not self.category_index[old_metadata.category]:
                            del self.category_index[old_metadata.category]
                
                if metadata.category:
                    if metadata.category not in self.category_index:
                        self.category_index[metadata.category] = set()
                    self.category_index[metadata.category].add(metadata.name)
                
                logger.info(f"特征注册成功: {key}")
                return True
                
        except Exception as e:
            logger.error(f"特征注册失败: {str(e)}")
            return False
    
    def get_feature(self, name: str, version: str = None) -> Optional[FeatureMetadata]:
        """获取特征元数据
        
        Args:
            name: 特征名称
            version: 特征版本，如果为None则返回最新版本
            
        Returns:
            特征元数据
        """
        try:
            with self.lock:
                if version:
                    key = self._get_feature_key(name, version)
                    return self.features.get(key)
                else:
                    # 获取最新版本
                    if name in self.feature_versions:
                        latest_version = self.feature_versions[name][-1]
                        key = self._get_feature_key(name, latest_version)
                        return self.features.get(key)
                    else:
                        # 尝试直接获取（兼容旧版本）
                        return self.features.get(name)
                
        except Exception as e:
            logger.error(f"获取特征元数据失败: {str(e)}")
            return None
    
    def list_features(self, status: Optional[FeatureStatus] = None, 
                     category: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> List[FeatureMetadata]:
        """列出特征
        
        Args:
            status: 特征状态过滤
            category: 分类过滤
            tags: 标签过滤
            
        Returns:
            特征元数据列表
        """
        try:
            with self.lock:
                features = list(self.features.values())
                
                # 状态过滤
                if status:
                    features = [f for f in features if f.status == status]
                
                # 分类过滤
                if category:
                    features = [f for f in features if f.category == category]
                
                # 标签过滤
                if tags:
                    features = [f for f in features if any(tag in f.tags for tag in tags)]
                
                return features
                
        except Exception as e:
            logger.error(f"列出特征失败: {str(e)}")
            return []
    
    def search_features(self, query: str, fields: Optional[List[str]] = None) -> List[FeatureMetadata]:
        """搜索特征
        
        Args:
            query: 搜索查询
            fields: 搜索字段，默认搜索名称、描述、标签
            
        Returns:
            匹配的特征元数据列表
        """
        try:
            if not fields:
                fields = ['name', 'display_name', 'description', 'tags']
            
            query_lower = query.lower()
            results = []
            
            with self.lock:
                for metadata in self.features.values():
                    match = False
                    
                    for field in fields:
                        if field == 'name' and query_lower in metadata.name.lower():
                            match = True
                            break
                        elif field == 'display_name' and query_lower in metadata.display_name.lower():
                            match = True
                            break
                        elif field == 'description' and query_lower in metadata.description.lower():
                            match = True
                            break
                        elif field == 'tags' and any(query_lower in tag.lower() for tag in metadata.tags):
                            match = True
                            break
                    
                    if match:
                        results.append(metadata)
                
                return results
                
        except Exception as e:
            logger.error(f"搜索特征失败: {str(e)}")
            return []
    
    def get_feature_versions(self, name: str) -> List[str]:
        """获取特征的所有版本
        
        Args:
            name: 特征名称
            
        Returns:
            版本列表
        """
        with self.lock:
            return self.feature_versions.get(name, [])
    
    def delete_feature(self, name: str, version: str = None) -> bool:
        """删除特征
        
        Args:
            name: 特征名称
            version: 特征版本，如果为None则删除所有版本
            
        Returns:
            删除是否成功
        """
        try:
            with self.lock:
                if version:
                    # 删除指定版本
                    key = self._get_feature_key(name, version)
                    if key in self.features:
                        metadata = self.features[key]
                        
                        # 删除文件
                        file_path = self._get_feature_file_path(name, version)
                        if file_path.exists():
                            file_path.unlink()
                        
                        # 更新内存缓存
                        del self.features[key]
                        
                        # 更新版本索引
                        if name in self.feature_versions:
                            if version in self.feature_versions[name]:
                                self.feature_versions[name].remove(version)
                                if not self.feature_versions[name]:
                                    del self.feature_versions[name]
                        
                        # 更新标签索引
                        for tag in metadata.tags:
                            if tag in self.tags_index:
                                self.tags_index[tag].discard(name)
                                if not self.tags_index[tag]:
                                    del self.tags_index[tag]
                        
                        # 更新分类索引
                        if metadata.category and metadata.category in self.category_index:
                            self.category_index[metadata.category].discard(name)
                            if not self.category_index[metadata.category]:
                                del self.category_index[metadata.category]
                        
                        logger.info(f"特征删除成功: {key}")
                        return True
                    else:
                        logger.warning(f"特征不存在: {key}")
                        return False
                else:
                    # 删除所有版本
                    if name in self.feature_versions:
                        versions = self.feature_versions[name].copy()
                        success = True
                        for v in versions:
                            if not self.delete_feature(name, v):
                                success = False
                        return success
                    else:
                        logger.warning(f"特征不存在: {name}")
                        return False
                
        except Exception as e:
            logger.error(f"删除特征失败: {str(e)}")
            return False
    
    def update_feature_status(self, name: str, status: FeatureStatus, version: str = None) -> bool:
        """更新特征状态
        
        Args:
            name: 特征名称
            status: 新状态
            version: 特征版本
            
        Returns:
            更新是否成功
        """
        try:
            metadata = self.get_feature(name, version)
            if metadata:
                metadata.status = status
                metadata.update_timestamp()
                return self.register_feature(metadata, overwrite=True)
            else:
                logger.warning(f"特征不存在: {name}@{version}")
                return False
                
        except Exception as e:
            logger.error(f"更新特征状态失败: {str(e)}")
            return False
    
    def update_feature_statistics(self, name: str, statistics: FeatureStatistics, 
                                version: str = None) -> bool:
        """更新特征统计信息
        
        Args:
            name: 特征名称
            statistics: 统计信息
            version: 特征版本
            
        Returns:
            更新是否成功
        """
        try:
            metadata = self.get_feature(name, version)
            if metadata:
                metadata.update_statistics(statistics)
                return self.register_feature(metadata, overwrite=True)
            else:
                logger.warning(f"特征不存在: {name}@{version}")
                return False
                
        except Exception as e:
            logger.error(f"更新特征统计信息失败: {str(e)}")
            return False
    
    def get_features_by_tag(self, tag: str) -> List[FeatureMetadata]:
        """根据标签获取特征
        
        Args:
            tag: 标签
            
        Returns:
            特征元数据列表
        """
        try:
            with self.lock:
                if tag in self.tags_index:
                    feature_names = self.tags_index[tag]
                    return [self.get_feature(name) for name in feature_names if self.get_feature(name)]
                else:
                    return []
                
        except Exception as e:
            logger.error(f"根据标签获取特征失败: {str(e)}")
            return []
    
    def get_features_by_category(self, category: str) -> List[FeatureMetadata]:
        """根据分类获取特征
        
        Args:
            category: 分类
            
        Returns:
            特征元数据列表
        """
        try:
            with self.lock:
                if category in self.category_index:
                    feature_names = self.category_index[category]
                    return [self.get_feature(name) for name in feature_names if self.get_feature(name)]
                else:
                    return []
                
        except Exception as e:
            logger.error(f"根据分类获取特征失败: {str(e)}")
            return []
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签"""
        with self.lock:
            return list(self.tags_index.keys())
    
    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        with self.lock:
            return list(self.category_index.keys())
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        with self.lock:
            total_features = len(self.features)
            unique_features = len(self.feature_versions)
            
            status_counts = {}
            for metadata in self.features.values():
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_features': total_features,
                'unique_features': unique_features,
                'total_tags': len(self.tags_index),
                'total_categories': len(self.category_index),
                'status_distribution': status_counts,
                'registry_path': str(self.registry_path)
            }
    
    def export_registry(self, export_path: str) -> bool:
        """导出注册表
        
        Args:
            export_path: 导出路径
            
        Returns:
            导出是否成功
        """
        try:
            export_data = {
                'features': [metadata.to_dict() for metadata in self.features.values()],
                'export_time': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"注册表导出成功: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"注册表导出失败: {str(e)}")
            return False
    
    def import_registry(self, import_path: str, overwrite: bool = False) -> bool:
        """导入注册表
        
        Args:
            import_path: 导入路径
            overwrite: 是否覆盖已存在的特征
            
        Returns:
            导入是否成功
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            features_data = import_data.get('features', [])
            success_count = 0
            
            for feature_data in features_data:
                try:
                    metadata = FeatureMetadata.from_dict(feature_data)
                    if self.register_feature(metadata, overwrite=overwrite):
                        success_count += 1
                except Exception as e:
                    logger.error(f"导入特征失败: {feature_data.get('name', 'unknown')}, 错误: {str(e)}")
            
            logger.info(f"注册表导入完成: {success_count}/{len(features_data)} 个特征")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"注册表导入失败: {str(e)}")
            return False

# 便捷函数
def create_feature_registry(config: Dict[str, Any] = None) -> FeatureRegistry:
    """创建特征注册表实例
    
    Args:
        config: 配置字典
        
    Returns:
        特征注册表实例
    """
    if config is None:
        config = {
            'registry_path': './feature_registry'
        }
    
    return FeatureRegistry(config)