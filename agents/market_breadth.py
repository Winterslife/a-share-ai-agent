from __future__ import annotations

import pandas as pd
import akshare as ak
from typing import Dict, Any, Literal
from core.models import now_ts_str, Bar
from tools.market_data import get_bars
from tools.indicators import calc_indicators

def get_market_regime(index_ticker: str = "sh000300", lookback: int = 60) -> dict:
    """
    Classifies the overall market regime based on index trend and market breadth.
    """
    index_close = 0.0
    index_ma20 = None
    up_ratio = None
    limit_up = None
    limit_down = None
    
    # 1. Fetch Index Data
    try:
        # Use ak.stock_zh_index_daily_em for index data
        df_index = ak.stock_zh_index_daily_em(symbol=index_ticker)
        if not df_index.empty:
            df_index = df_index.tail(lookback).copy()
            # Map columns to Bar-like structure for calc_indicators
            df_index = df_index.rename(columns={"date": "ts"})
            
            # Convert to Bar objects
            bars = [Bar(**row) for _, row in df_index.iterrows()]
            index_close = bars[-1].close
            indicators = calc_indicators(bars)
            index_ma20 = indicators.ma20
    except Exception as e:
        # Fallback if index fetch fails
        print(f"Index fetch error: {e}")

    # 2. Fetch Market Breadth (Up Ratio)
    try:
        # ak.stock_zh_a_spot_em returns a snapshot of all A-shares
        df_spot = ak.stock_zh_a_spot_em()
        if not df_spot.empty:
            # Columns: 代码, 名称, 最新价, 涨跌幅, ...
            total_count = len(df_spot)
            up_count = len(df_spot[df_spot["涨跌幅"] > 0])
            up_ratio = up_count / total_count if total_count > 0 else None
            
            # Limit up/down (approximate by 9.8% for A-shares)
            limit_up = len(df_spot[df_spot["涨跌幅"] >= 9.8])
            limit_down = len(df_spot[df_spot["涨跌幅"] <= -9.8])
    except Exception as e:
        # Fallback if spot data fails
        pass

    # 3. Regime Logic
    regime: Literal["BULL", "NEUTRAL", "DEFENSIVE"] = "NEUTRAL"
    
    if up_ratio is not None and index_ma20 is not None:
        if up_ratio > 0.60 and index_close >= index_ma20:
            regime = "BULL"
        elif up_ratio < 0.30 or index_close < index_ma20 * 0.995:
            regime = "DEFENSIVE"
        else:
            regime = "NEUTRAL"
    elif index_ma20 is not None:
        # Fallback if up_ratio is missing
        if index_close >= index_ma20:
            regime = "NEUTRAL" # Or mild BULL
        else:
            regime = "DEFENSIVE"
    else:
        # Total fallback
        regime = "NEUTRAL"

    # 4. Policy Mapping
    policy_map = {
        "BULL": {"candidate_top_n": 15, "risk_level": "high"},
        "NEUTRAL": {"candidate_top_n": 10, "risk_level": "mid"},
        "DEFENSIVE": {"candidate_top_n": 5, "risk_level": "low"},
    }
    
    policy = policy_map[regime]

    return {
        "regime": regime,
        "metrics": {
            "up_ratio": round(up_ratio, 4) if up_ratio is not None else None,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "index_close": round(index_close, 2),
            "index_ma20": round(index_ma20, 2) if index_ma20 is not None else None,
            "timestamp": now_ts_str()
        },
        "policy": policy
    }
