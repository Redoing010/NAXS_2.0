"""
Qlib 适配层：提供数据集构建、模型训练与打分的统一入口。

模块职责（最小骨架）：
- build_dataset(cfg): 基于 Alpha158/Alpha360 生成 DatasetH
- train_model(cfg):   使用 LGBModel 训练并保存模型
- predict_scores(cfg): 用同一处理器打分，输出 (datetime, instrument, score)

注意：本目录为 Python 运行时，仅承担量化内核职责；NAXS Orchestrator 通过 API 调用。
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"


