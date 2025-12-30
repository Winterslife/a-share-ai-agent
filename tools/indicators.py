from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List
from core.models import Bar, Indicators

def calc_indicators(bars: List[Bar]) -> Indicators:
    """
    Calculate technical indicators from a list of Bar objects.
    """
    if not bars:
        return Indicators()

    df = pd.DataFrame([b.model_dump() for b in bars])
    
    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # MA
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()

    # RSI (Simplified)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = exp1 - exp2
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

    # Bollinger Bands
    df["boll_mid"] = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["boll_up"] = df["boll_mid"] + (std * 2)
    df["boll_low"] = df["boll_mid"] - (std * 2)

    # ATR
    high_low = df["high"] - df["low"]
    high_cp = np.abs(df["high"] - df["close"].shift())
    low_cp = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()

    # Vol Ratio
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    vol_ma5 = df["volume"].rolling(window=5).mean()
    df["vol_ratio_5_20"] = vol_ma5 / df["vol_ma20"]

    # Get last row
    last = df.iloc[-1]
    
    return Indicators(
        ma5=float(last["ma5"]) if not pd.isna(last["ma5"]) else None,
        ma20=float(last["ma20"]) if not pd.isna(last["ma20"]) else None,
        rsi14=float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None,
        macd_dif=float(last["macd_dif"]) if not pd.isna(last["macd_dif"]) else None,
        macd_dea=float(last["macd_dea"]) if not pd.isna(last["macd_dea"]) else None,
        macd_hist=float(last["macd_hist"]) if not pd.isna(last["macd_hist"]) else None,
        boll_up=float(last["boll_up"]) if not pd.isna(last["boll_up"]) else None,
        boll_mid=float(last["boll_mid"]) if not pd.isna(last["boll_mid"]) else None,
        boll_low=float(last["boll_low"]) if not pd.isna(last["boll_low"]) else None,
        atr14=float(last["atr14"]) if not pd.isna(last["atr14"]) else None,
        vol_ma20=float(last["vol_ma20"]) if not pd.isna(last["vol_ma20"]) else None,
        vol_ratio_5_20=float(last["vol_ratio_5_20"]) if not pd.isna(last["vol_ratio_5_20"]) else None,
    )
