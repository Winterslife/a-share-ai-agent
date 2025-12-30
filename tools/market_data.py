from __future__ import annotations

import pandas as pd
import akshare as ak
import ssl
import requests
from pathlib import Path
from typing import List

from core.models import Bar, Quote
from core.logging import cache_dir

# Workaround for SSL errors: Monkeypatch requests to disable verification
orig_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return orig_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request

# Disable urllib3 warnings about unverified HTTPS requests
requests.packages.urllib3.disable_warnings()

# Also try to fix SSL context for urllib/ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def get_bars(ticker: str, freq: str = "1d", lookback: int = 120, use_cache: bool = True) -> List[Bar]:
    """
    Fetch daily history using ak.stock_zh_a_hist.
    freq supports: "1d" (must), "1m"/"5m" (NotImplementedError for v0.1.0)
    """
    if freq != "1d":
        raise NotImplementedError(f"Frequency {freq} is not supported in v0.1.0")

    cache_path = cache_dir() / f"{ticker}_{freq}_{lookback}.csv"

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        # Fetch daily history using ak.stock_zh_a_hist
        # Note: ticker should be 6 digits for akshare
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        
        if df.empty:
            raise ValueError(f"No data for ticker {ticker}")

        # Map dataframe columns to Bar
        # Standard columns from ak.stock_zh_a_hist:
        # 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        column_map = {
            "日期": "ts",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount"
        }
        df = df.rename(columns=column_map)
        
        # Keep only necessary columns
        df = df[list(column_map.values())]
        
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert ts to string if it's date objects
        df["ts"] = df["ts"].astype(str)

        # Sort ascending by ts
        df = df.sort_values("ts", ascending=True)
        
        # Trim to last `lookback` bars
        df = df.tail(lookback)
        
        # Cache as CSV
        df.to_csv(cache_path, index=False)

    # Convert to list of Bar models
    bars = [Bar(**row) for _, row in df.iterrows()]
    return bars

def get_quote(ticker: str) -> Quote:
    """
    Get current quote for a ticker.
    Uses get_bars(ticker, "1d", lookback=2) to approximate.
    """
    bars = get_bars(ticker, "1d", lookback=2, use_cache=False)
    
    if not bars:
        raise ValueError(f"Could not fetch data for ticker {ticker}")

    last_bar = bars[-1]
    price = last_bar.close
    
    chg_pct = 0.0
    if len(bars) >= 2:
        prev_close = bars[-2].close
        if prev_close != 0:
            chg_pct = (price - prev_close) / prev_close * 100

    return Quote(
        ticker=ticker,
        price=price,
        chg_pct=chg_pct,
        amount=last_bar.amount,
        timestamp=f"{last_bar.ts} 15:00:00"
    )
