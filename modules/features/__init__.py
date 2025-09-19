# Feature Store 特征仓库模块
# 提供特征存储、管理和检索功能

from .feature_store import (
    FeatureStore,
    Feature,
    FeatureGroup,
    FeatureType,
    StorageBackend,
    BaseStorage,
    MemoryStorage,
    RedisStorage,
    create_feature_store
)

from .feature_registry import (
    FeatureRegistry,
    FeatureMetadata,
    FeatureStatus,
    DataQuality,
    FeatureLineage,
    FeatureStatistics,
    create_feature_registry
)

from .feature_validator import (
    FeatureValidator,
    ValidationRule,
    ValidationResult,
    ValidationReport,
    ValidationLevel,
    ValidationCategory,
    DataTypeRule,
    RangeRule,
    NullRule,
    UniqueRule,
    OutlierRule,
    DistributionRule,
    CustomRule,
    create_feature_validator,
    validate_feature_data
)

from .feature_engine import (
    FeatureEngine,
    FeatureComputer,
    FeatureConfig,
    ComputeResult,
    FeatureCategory,
    ComputeMode,
    PriceFeatureComputer,
    VolumeFeatureComputer,
    TechnicalFeatureComputer,
    CrossSectionalFeatureComputer,
    InteractionFeatureComputer,
    create_feature_engine,
    compute_basic_features
)

__all__ = [
    # Feature Store
    'FeatureStore',
    'Feature',
    'FeatureGroup', 
    'FeatureType',
    'StorageBackend',
    'BaseStorage',
    'MemoryStorage',
    'RedisStorage',
    'create_feature_store',
    
    # Feature Registry
    'FeatureRegistry',
    'FeatureMetadata',
    'FeatureStatus',
    'DataQuality',
    'FeatureLineage',
    'FeatureStatistics',
    'create_feature_registry',
    
    # Feature Validator
    'FeatureValidator',
    'ValidationRule',
    'ValidationResult',
    'ValidationReport',
    'ValidationLevel',
    'ValidationCategory',
    'DataTypeRule',
    'RangeRule',
    'NullRule',
    'UniqueRule',
    'OutlierRule',
    'DistributionRule',
    'CustomRule',
    'create_feature_validator',
    'validate_feature_data',
    
    # Feature Engine
    'FeatureEngine',
    'FeatureComputer',
    'FeatureConfig',
    'ComputeResult',
    'FeatureCategory',
    'ComputeMode',
    'PriceFeatureComputer',
    'VolumeFeatureComputer',
    'TechnicalFeatureComputer',
    'CrossSectionalFeatureComputer',
    'InteractionFeatureComputer',
    'create_feature_engine',
    'compute_basic_features'
]

__version__ = '1.0.0'
__author__ = 'NAXS Team'
__description__ = 'Feature Store for quantitative research and trading'