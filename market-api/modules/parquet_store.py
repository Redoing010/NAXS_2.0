import os
from typing import Iterable, Optional
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def daily_path(root: str, code: str, freq: str = "D") -> str:
    return os.path.join(root, freq.upper(), f"{code}.parquet")


def write_daily(root: str, code: str, df: pd.DataFrame, freq: str = "D") -> str:
    path = daily_path(root, code, freq)
    ensure_dir(os.path.dirname(path))
    # Incremental append by merging on datetime
    if os.path.exists(path):
        old = pd.read_parquet(path)
        merged = (
            pd.concat([old, df])
            .drop_duplicates(subset=["datetime"], keep="last")
            .sort_values("datetime")
        )
        merged.to_parquet(path, index=False)
    else:
        df.sort_values("datetime").to_parquet(path, index=False)
    return path


def read_code(code: str, start: Optional[str], end: Optional[str], root: str, freq: str = "D") -> pd.DataFrame:
    path = daily_path(root, code, freq)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if start:
        df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["datetime"] <= pd.Timestamp(end, tz="UTC")]
    return df





