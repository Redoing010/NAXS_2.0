# NAXS Dynamic Factors Module
# 动态因子权重系统 - 市场状态识别与权重自适应调整

from .regime_detector import (
    MarketRegime,
    RegimeState,
    RegimeTransition,
    RegimeDetector
)
from .weight_optimizer import (
    OptimizationMethod,
    RebalanceFrequency,
    OptimizationConfig,
    WeightUpdate,
    PerformanceMetrics,
    WeightOptimizer,
    create_optimization_config,
    create_weight_optimizer
)
from .factor_manager import (
    FactorCategory,
    FactorStatus,
    FactorMetadata,
    FactorValue,
    FactorPerformance,
    BaseFactor,
    FactorManager
)
from .online_learner import OnlineLearner
from .market_state import (
    MarketPhase,
    VolatilityRegime,
    TrendDirection,
    LiquidityCondition,
    MarketIndicators,
    MarketState,
    StateTransition,
    MarketStateAnalyzer,
    create_market_state_analyzer
)
from .factor_evaluator import (
    EvaluationMetric,
    FactorQuality,
    EvaluationConfig,
    FactorMetrics,
    FactorRanking,
    EvaluationResult,
    FactorEvaluator,
    create_evaluation_config,
    create_factor_evaluator
)
from .enhanced_factor_system import (
    AlertLevel,
    MarketRegime as EnhancedMarketRegime,
    FactorAlert,
    SystemMetrics,
    EnhancedFactorSystem,
    MarketRegimeDetector,
    MeanVarianceOptimizer,
    OnlineGradientOptimizer,
    ThompsonSamplingOptimizer,
    create_enhanced_factor_system
)

__version__ = "1.0.0"

__all__ = [
    # 市场状态识别
    "MarketRegime",
    "RegimeState",
    "RegimeTransition",
    "RegimeDetector",
    
    # 权重优化
    "OptimizationMethod",
    "RebalanceFrequency",
    "OptimizationConfig",
    "WeightUpdate",
    "PerformanceMetrics",
    "WeightOptimizer",
    "create_optimization_config",
    "create_weight_optimizer",
    
    # 因子管理
    "FactorCategory",
    "FactorStatus",
    "FactorMetadata",
    "FactorValue",
    "FactorPerformance",
    "BaseFactor",
    "FactorManager",
    
    # 在线学习
    "OnlineLearner",
    
    # 市场状态分析
    "MarketPhase",
    "VolatilityRegime",
    "TrendDirection",
    "LiquidityCondition",
    "MarketIndicators",
    "MarketState",
    "StateTransition",
    "MarketStateAnalyzer",
    "create_market_state_analyzer",
    
    # 因子评估
    "EvaluationMetric",
    "FactorQuality",
    "EvaluationConfig",
    "FactorMetrics",
    "FactorRanking",
    "EvaluationResult",
    "FactorEvaluator",
    "create_evaluation_config",
    "create_factor_evaluator",
    
    # 增强因子评估系统
    "AlertLevel",
    "EnhancedMarketRegime",
    "FactorAlert",
    "SystemMetrics",
    "EnhancedFactorSystem",
    "MarketRegimeDetector",
    "MeanVarianceOptimizer",
    "OnlineGradientOptimizer",
    "ThompsonSamplingOptimizer",
    "create_enhanced_factor_system",
]