from __future__ import annotations

import os
from typing import Any, Dict, Tuple


def _ensure_qlib_initialized(cfg: Dict[str, Any]) -> None:
    try:
        import qlib
    except Exception as exc:  # pragma: no cover - 环境依赖
        raise RuntimeError(
            "未安装 qlib，请先安装并确保 Python 版本为 3.8-3.12"
        ) from exc

    provider_uri = cfg.get(
        "provider_uri",
        os.path.expanduser("~/.qlib/qlib_data/cn_data"),
    )
    region = cfg.get("region", "cn")
    if not getattr(qlib, "_inited", False):  # type: ignore[attr-defined]
        qlib.init(provider_uri=provider_uri, region=region)


def build_dataset(cfg: Dict[str, Any]):
    """
    基于 Alpha158 构建 DatasetH。

    cfg 示例（可缺省使用默认值）：
    {
      "region": "cn",
      "provider_uri": "~/.qlib/qlib_data/cn_data",
      "instruments": "csi300",
      "start_time": "2018-01-01",
      "end_time": "2021-12-31",
      "fit_start_time": "2018-01-01",
      "fit_end_time": "2020-06-30",
      "label": "Ref($close, -1) / $close - 1"
    }
    """
    _ensure_qlib_initialized(cfg)

    from qlib.utils import init_instance_by_config

    instruments = cfg.get("instruments", "csi300")
    start_time = cfg.get("start_time", "2018-01-01")
    end_time = cfg.get("end_time", "2021-12-31")
    fit_start_time = cfg.get("fit_start_time", start_time)
    fit_end_time = cfg.get("fit_end_time", "2020-06-30")
    label = cfg.get("label", "Ref($close, -1) / $close - 1")

    handler_config: Dict[str, Any] = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
            "instruments": instruments,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
            "label": [label],
        },
    }

    dataset_config: Dict[str, Any] = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (fit_start_time, fit_end_time),
                "valid": ("2020-07-01", "2020-12-31"),
                "test": ("2021-01-01", end_time),
            },
        },
    }

    dataset = init_instance_by_config(dataset_config)
    return dataset


