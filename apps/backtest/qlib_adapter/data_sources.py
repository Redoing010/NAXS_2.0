from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import pandas as pd


def _normalize_symbol_to_numeric(symbol: str) -> Tuple[str, str]:
    """
    将符号标准化，返回 (market, numeric_code)。
    支持输入如 SH600000 / SZ000001 / 600000.SH / 000001.SZ / 600000 等。
    """
    s = symbol.strip().upper()
    if "." in s:
        code, market = s.split(".")
        if market in ("SH", "SZ"):
            return market, code
    if s.startswith("SH") or s.startswith("SZ"):
        return s[:2], s[2:]
    # fallback: 未指明市场，默认 SH
    return "SH", s


def fetch_akshare_prices(symbols: Iterable[str], start_date: str, end_date: str, out_path: str) -> str:
    """
    使用 AkShare 抓取 A 股日线收盘价，保存为 parquet。

    - start_date/end_date 格式: YYYY-MM-DD 或 YYYYMMDD（均可）
    - 输出列: date, symbol, close
    """
    import akshare as ak  # 延迟导入，避免环境未装时报错

    def ymd(s: str) -> str:
        t = s.replace("-", "")
        if len(t) != 8:
            raise ValueError(f"非法日期: {s}")
        return t

    start = ymd(start_date)
    end = ymd(end_date)

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        market, code = _normalize_symbol_to_numeric(sym)
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
            if df is None or df.empty:
                continue
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df = df[["date", "close"]].copy()
            df["date"] = pd.to_datetime(df["date"])  # type: ignore[assignment]
            df["symbol"] = f"{market}{code}"
            frames.append(df)
        except Exception:
            # 网络或个股异常，跳过但不中断
            continue

    if not frames:
        raise RuntimeError("AkShare 未获取到任何行情数据")

    all_prices = pd.concat(frames, ignore_index=True)
    all_prices = all_prices.sort_values(["symbol", "date"])  # type: ignore[arg-type]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_prices.to_parquet(out_path, index=False)
    return out_path


