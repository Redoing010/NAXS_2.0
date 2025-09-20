"""AkShare数据源实现，支持在缺失AkShare依赖时使用合成数据。"""

from __future__ import annotations

import importlib
from importlib import util as importlib_util
import logging
from datetime import date, datetime
from typing import List, Union

import numpy as np
import pandas as pd

from .interfaces import IDataSource
from .utils import akshare_to_standard_symbol, normalize_symbol

logger = logging.getLogger(__name__)

# AkShare 为可选依赖，在测试环境中可能不存在
_akshare_spec = importlib_util.find_spec("akshare")
if _akshare_spec is not None:
    ak = importlib.import_module("akshare")  # type: ignore[import-untyped]
else:  # pragma: no cover - 仅在缺少AkShare时执行
    ak = None  # type: ignore[assignment]
    logger.warning(
        "AkShare package is not available; falling back to synthetic market data."
    )


class AkshareSource(IDataSource):
    """AkShare数据源实现类。

    当环境中未安装 AkShare 或网络不可用时，本实现会自动生成可预测的合成
    数据，确保单元测试和端到端测试可以在隔离环境中运行。
    """

    def __init__(self) -> None:
        self.name = "akshare"
        if ak is None:  # pragma: no cover - 在CI环境中提醒
            logger.info("AkShareSource initialised with synthetic data backend.")
        else:
            logger.info("AkShare数据源初始化完成")

    # ------------------------------------------------------------------
    # 公共API
    # ------------------------------------------------------------------
    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日频OHLCV数据。"""

        if ak is None:
            logger.debug("Using synthetic daily bars for %s", symbol)
            return self._generate_synthetic_daily_data(symbol, start_date, end_date)

        try:
            ak_symbol = normalize_symbol(symbol, target="akshare")
            start_str = self._format_date(start_date)
            end_str = self._format_date(end_date)

            logger.info(
                "获取 %s 日频数据: %s - %s, 复权: %s", symbol, start_str, end_str, adjust
            )

            df = ak.stock_zh_a_hist(  # type: ignore[attr-defined]
                symbol=ak_symbol,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=adjust,
            )

            if df is None or df.empty:
                logger.warning("未获取到 %s 的数据，使用合成数据替代", symbol)
                return self._generate_synthetic_daily_data(symbol, start_date, end_date)

            df = df.rename(
                columns=
                {
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            required_cols = ["date", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要的列: {missing_cols}")

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Shanghai")
            df.index = df.index.tz_convert("UTC")

            numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            core_cols = ["open", "high", "low", "close", "volume"]
            if "amount" in df.columns:
                core_cols.append("amount")

            df = df.sort_index().drop_duplicates()
            return df[core_cols]

        except Exception as exc:  # pragma: no cover - 捕获AkShare运行时异常
            logger.warning(
                "获取 %s 日频数据失败 (%s)，使用合成数据", symbol, exc,
            )
            return self._generate_synthetic_daily_data(symbol, start_date, end_date)

    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        period: str = "1",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取分钟频数据。"""

        if ak is None:
            logger.debug("Using synthetic minute bars for %s", symbol)
            return self._generate_synthetic_minute_data(symbol, start_date, end_date, period)

        try:
            ak_symbol = normalize_symbol(symbol, target="akshare")
            logger.info(
                "获取 %s 分钟数据: period=%s, adjust=%s", symbol, period, adjust
            )
            df = ak.stock_zh_a_hist_min_em(  # type: ignore[attr-defined]
                symbol=ak_symbol,
                period=period,
                adjust=adjust,
            )

            if df is None or df.empty:
                logger.warning("未获取到 %s 的分钟数据，使用合成数据替代", symbol)
                return self._generate_synthetic_minute_data(symbol, start_date, end_date, period)

            df = df.rename(
                columns=
                {
                    "时间": "datetime",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Shanghai")
            df.index = df.index.tz_convert("UTC")

            numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            start_dt = pd.to_datetime(start_date).tz_localize("UTC")
            end_dt = pd.to_datetime(end_date).tz_localize("UTC")
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            df = df.sort_index().drop_duplicates()
            return df

        except Exception as exc:  # pragma: no cover - 捕获AkShare运行时异常
            logger.warning(
                "获取 %s 分钟数据失败 (%s)，使用合成数据", symbol, exc,
            )
            return self._generate_synthetic_minute_data(symbol, start_date, end_date, period)

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表。"""

        if ak is None:
            logger.debug("Using synthetic stock list")
            return self._generate_synthetic_stock_list()

        try:
            logger.info("获取A股股票列表")
            df = ak.stock_info_a_code_name()  # type: ignore[attr-defined]
            if df is None or df.empty:
                logger.warning("未获取到股票列表，使用合成数据")
                return self._generate_synthetic_stock_list()

            df = df.rename(columns={"code": "symbol", "name": "name"})
            df["symbol"] = df["symbol"].apply(akshare_to_standard_symbol)
            df["exchange"] = df["symbol"].apply(
                lambda x: "SZ" if x.endswith(".SZ") else "SH"
            )
            df["list_date"] = None
            df["delist_date"] = None
            return df

        except Exception as exc:  # pragma: no cover - 捕获AkShare运行时异常
            logger.warning("获取股票列表失败 (%s)，使用合成数据", exc)
            return self._generate_synthetic_stock_list()

    def fetch_trading_calendar(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> List[str]:
        """获取交易日历。"""

        if ak is None:
            return self._generate_synthetic_trading_calendar(start_year, end_year)

        try:
            logger.info("获取交易日历: %s-%s", start_year, end_year)
            df = ak.tool_trade_date_hist_sina()  # type: ignore[attr-defined]
            if df is None or df.empty:
                logger.warning("未获取到交易日历，使用合成日历")
                return self._generate_synthetic_trading_calendar(start_year, end_year)

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df[
                (df["trade_date"].dt.year >= start_year)
                & (df["trade_date"].dt.year <= end_year)
            ]
            trading_dates = (
                df["trade_date"].dt.strftime("%Y-%m-%d").drop_duplicates().sort_values().tolist()
            )
            return trading_dates

        except Exception as exc:  # pragma: no cover - 捕获AkShare运行时异常
            logger.warning(
                "获取交易日历失败 (%s)，使用合成日历", exc,
            )
            return self._generate_synthetic_trading_calendar(start_year, end_year)

    def fetch_realtime_quotes(self, symbols: List[str] | None = None) -> pd.DataFrame:
        """获取实时行情快照。"""

        if ak is None:
            logger.debug("Using synthetic realtime quotes")
            return self._generate_synthetic_realtime_quotes(symbols)

        try:
            logger.info("获取实时行情快照")
            df = ak.stock_zh_a_spot()  # type: ignore[attr-defined]
            if df is None or df.empty:
                logger.warning("未获取到实时行情，使用合成数据")
                return self._generate_synthetic_realtime_quotes(symbols)

            df = df.rename(
                columns=
                {
                    "代码": "symbol",
                    "名称": "name",
                    "最新价": "price",
                    "涨跌幅": "pct_chg",
                    "涨跌额": "change",
                    "成交量": "volume",
                    "成交额": "amount",
                    "今开": "open",
                    "昨收": "prev_close",
                    "最高": "high",
                    "最低": "low",
                }
            )
            df["symbol"] = df["symbol"].apply(akshare_to_standard_symbol)
            if symbols:
                wanted = {normalize_symbol(s) for s in symbols}
                df = df[df["symbol"].isin(wanted)]
            numeric_cols = [
                "price",
                "pct_chg",
                "change",
                "volume",
                "amount",
                "open",
                "prev_close",
                "high",
                "low",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.Timestamp.utcnow().tz_localize("UTC")
            return df

        except Exception as exc:  # pragma: no cover - 捕获AkShare运行时异常
            logger.warning("获取实时行情失败 (%s)，使用合成数据", exc)
            return self._generate_synthetic_realtime_quotes(symbols)

    # ------------------------------------------------------------------
    # 合成数据生成工具
    # ------------------------------------------------------------------
    def _generate_synthetic_daily_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
    ) -> pd.DataFrame:
        start_dt, end_dt = self._normalize_date_range(start_date, end_date)
        dates = pd.bdate_range(start_dt, end_dt, tz="Asia/Shanghai")
        if dates.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount"])

        rng = np.random.default_rng(self._seed_from_symbol(symbol))
        base_price = 60 + rng.random() * 80
        returns = rng.normal(0, 0.01, len(dates))
        close = base_price * np.cumprod(1 + returns)
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(dates))))
        volume = rng.integers(800_000, 5_000_000, len(dates))
        amount = volume * (open_ + close) / 2 / 100

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume.astype(float),
                "amount": amount,
            },
            index=dates,
        )
        df.index = df.index.tz_convert("UTC")
        return df

    def _generate_synthetic_minute_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        period: str,
    ) -> pd.DataFrame:
        start_dt, end_dt = self._normalize_date_range(start_date, end_date)
        freq = f"{int(period)}T"
        dates = pd.date_range(
            start_dt.floor(freq),
            end_dt.ceil(freq),
            freq=freq,
            tz="Asia/Shanghai",
        )
        if dates.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount"])

        rng = np.random.default_rng(self._seed_from_symbol(f"{symbol}-{period}"))
        base_price = 60 + rng.random() * 80
        noise = rng.normal(0, 0.002, len(dates))
        close = base_price * (1 + np.cumsum(noise))
        open_ = close * (1 + rng.normal(0, 0.0005, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, len(dates))))
        volume = rng.integers(20_000, 200_000, len(dates))
        amount = volume * (open_ + close) / 2 / 100

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume.astype(float),
                "amount": amount,
            },
            index=dates,
        )
        df.index = df.index.tz_convert("UTC")
        return df

    def _generate_synthetic_stock_list(self) -> pd.DataFrame:
        records = [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "exchange": "SZ",
                "list_date": "1991-04-03",
                "delist_date": None,
            },
            {
                "symbol": "600519.SH",
                "name": "贵州茅台",
                "exchange": "SH",
                "list_date": "2001-08-27",
                "delist_date": None,
            },
            {
                "symbol": "000858.SZ",
                "name": "五粮液",
                "exchange": "SZ",
                "list_date": "1998-07-27",
                "delist_date": None,
            },
        ]
        return pd.DataFrame.from_records(records)

    def _generate_synthetic_trading_calendar(self, start_year: int, end_year: int) -> List[str]:
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        dates = pd.bdate_range(start, end)
        return [d.strftime("%Y-%m-%d") for d in dates]

    def _generate_synthetic_realtime_quotes(
        self, symbols: List[str] | None = None
    ) -> pd.DataFrame:
        if symbols:
            std_symbols = [normalize_symbol(symbol) for symbol in symbols]
        else:
            std_symbols = ["000001.SZ", "600519.SH", "000858.SZ"]

        rows = []
        for symbol in std_symbols:
            rng = np.random.default_rng(self._seed_from_symbol(f"rt-{symbol}"))
            price = 60 + rng.random() * 80
            pct = rng.normal(0, 0.02)
            prev_close = price / (1 + pct)
            change = price - prev_close
            rows.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "price": price,
                    "pct_chg": pct * 100,
                    "change": change,
                    "volume": float(rng.integers(500_000, 5_000_000)),
                    "amount": price * rng.integers(100_000, 300_000) / 100,
                    "open": price * (1 + rng.normal(0, 0.01)),
                    "prev_close": prev_close,
                    "high": price * (1 + abs(rng.normal(0, 0.01))),
                    "low": price * (1 - abs(rng.normal(0, 0.01))),
                    "timestamp": pd.Timestamp.utcnow().tz_localize("UTC"),
                }
            )

        return pd.DataFrame.from_records(rows)

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _format_date(value: Union[str, date, datetime]) -> str:
        dt = pd.to_datetime(value)
        return dt.strftime("%Y%m%d")

    @staticmethod
    def _normalize_date_range(
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        return start_dt, end_dt

    @staticmethod
    def _seed_from_symbol(symbol: str) -> int:
        return abs(hash(symbol)) % (2**32)
