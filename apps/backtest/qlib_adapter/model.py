from __future__ import annotations

import json
import os
from typing import Any, Dict


def train_model(cfg: Dict[str, Any]):
    """
    使用 Qlib 的 LGBModel 训练模型并返回已训练实例。

    cfg 关键项：
    - dataset: 由 dataset.build_dataset 生成的 DatasetH
    - work_dir: 输出目录（保存模型与指标）
    - lgb_params: LightGBM 参数（可缺省）
    """
    dataset = cfg["dataset"]
    work_dir = os.path.abspath(cfg.get("work_dir", "apps/backtest/out"))
    os.makedirs(work_dir, exist_ok=True)

    from qlib.utils import init_instance_by_config

    lgb_params = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.02,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": os.cpu_count() or 4,
        },
    }
    user_params = cfg.get("lgb_params")
    if isinstance(user_params, dict):
        # 浅合并用户自定义参数
        lgb_params = {**lgb_params, **user_params}
        if "kwargs" in user_params:
            lgb_params["kwargs"] = {**lgb_params["kwargs"], **user_params["kwargs"]}

    model = init_instance_by_config(lgb_params)
    model.fit(dataset)

    # 简要保存模型信息
    meta = {
        "model_class": lgb_params["class"],
        "work_dir": work_dir,
    }
    with open(os.path.join(work_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model


