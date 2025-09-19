import argparse
import os
from datetime import datetime
from typing import List

import akshare as ak
import pandas as pd

from modules.parquet_store import write_daily


def fetch_daily(code: str, start: str, end: str) -> pd.DataFrame:
    # akshare expects symbol without suffix in some endpoints; try EM style
    # Prefer EastMoney daily adjusted free; fallback to pro if needed
    # Here we use stock_zh_a_hist for stability
    symbol = code[:6]
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="")
    df = df.rename(
        columns={
            "日期": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    keep = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    df = df[keep]
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    # EastMoney volumes are in shares; keep as-is
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="stock")
    ap.add_argument("--codes", action="append", required=True, help="e.g. 000831.SZ; can repeat")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--freq", default="D")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for code in args.codes:
        df = fetch_daily(code, args.start, args.end)
        path = write_daily(args.out, code, df, freq=args.freq)
        print(f"wrote {code} -> {path} rows={len(df)}")


if __name__ == "__main__":
    main()





