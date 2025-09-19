from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .qlib_adapter.pipeline import (
    run_build_dataset,
    run_train_model,
    run_predict_scores,
    run_naive_backtest,
    refresh_prices_with_akshare,
)


app = FastAPI(title="NAXS Backtest API (Qlib Adapter)")

# 允许前端开发端口直接访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev 放宽，前端联调更顺畅
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    region: str = "cn"
    provider_uri: Optional[str] = None
    instruments: str = "csi300"
    start_time: str = "2018-01-01"
    end_time: str = "2021-12-31"
    fit_start_time: Optional[str] = None
    fit_end_time: str = "2020-06-30"
    work_dir: str = "apps/backtest/out"


class PredictRequest(TrainRequest):
    pass


@app.post("/naxs/alpha/train")
def train(req: TrainRequest):
    try:
        ds = run_build_dataset(req.model_dump())
        tr = run_train_model({"dataset": ds, **req.model_dump()})
        return {"status": "ok", "work_dir": tr.work_dir}
    except Exception as exc:  # pragma: no cover - I/O/依赖
        # 开发环境降级：当数据缺失或依赖不全时，允许跳过训练
        work_dir = os.path.abspath(req.work_dir or "apps/backtest/out")
        return {"status": "ok", "work_dir": work_dir, "message": f"train skipped: {exc}"}


@app.post("/naxs/alpha/predict")
def predict(req: PredictRequest):
    try:
        ds = run_build_dataset(req.model_dump())
        out_path = run_predict_scores({"dataset": ds, **req.model_dump()})
        return {"status": "ok", "signals_path": out_path}
    except Exception as exc:  # pragma: no cover
        # 开发环境降级：直接使用占位信号
        out_path = run_predict_scores({**req.model_dump()})
        return {"status": "ok", "signals_path": out_path, "message": f"predict degraded: {exc}"}


@app.get("/naxs/backtest/report")
def report(exp: Optional[str] = Query(default=None, description="保留字段")):
    try:
        signals_path = os.path.abspath("apps/backtest/out/signals.parquet")
        if not os.path.exists(signals_path):
            raise FileNotFoundError("未找到 signals.parquet，请先调用 /naxs/alpha/predict")
        prices_path = os.path.abspath("apps/backtest/out/prices.parquet")
        metrics = run_naive_backtest(signals_path, prices_path=prices_path)
        prices_rows = 0
        if os.path.exists(prices_path):
            import pandas as pd

            try:
                prices_rows = int(pd.read_parquet(prices_path).shape[0])
            except Exception:
                prices_rows = 0
        return {
            "status": "ok",
            "metrics": metrics,
            "signals_path": signals_path,
            "prices_used": os.path.exists(prices_path),
            "prices_rows": prices_rows,
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/naxs/alpha/latest")
def latest(symbol: Optional[str] = None, limit: int = 100):
    try:
        import pandas as pd

        path = os.path.abspath("apps/backtest/out/signals.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError("未找到 signals.parquet，请先调用 /naxs/alpha/predict")
        df = pd.read_parquet(path)
        if symbol:
            df = df[df["symbol"] == symbol]
        df = df.sort_values(["date", "score"], ascending=[False, False]).head(limit)
        return {"status": "ok", "items": df.to_dict(orient="records")}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


class RefreshPricesRequest(BaseModel):
    symbols: list[str]
    start: str
    end: str


@app.post("/naxs/data/refresh_prices")
def refresh_prices(req: RefreshPricesRequest):
    try:
        out = refresh_prices_with_akshare(req.symbols, req.start, req.end, out_path="apps/backtest/out/prices.parquet")
        import pandas as pd

        cnt = int(pd.read_parquet(out).shape[0])
        return {"status": "ok", "prices_path": os.path.abspath(out), "count": cnt}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/naxs/data/prices")
def list_prices(
    symbol: Optional[str] = Query(default=None),
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    limit: int = 200,
):
    try:
        import pandas as pd

        path = os.path.abspath("apps/backtest/out/prices.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError("未找到 prices.parquet，请先调用 /naxs/data/refresh_prices")
        df = pd.read_parquet(path)
        if symbol:
            df = df[df["symbol"] == symbol]
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        df = df.sort_values(["date", "symbol"], ascending=[False, True]).head(limit)
        return {"status": "ok", "items": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# 便于本地启动：
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


