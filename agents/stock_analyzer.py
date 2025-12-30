from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Literal
from core.models import PortfolioInput, Quote, Bar, Indicators, StockSnapshot, now_ts_str
from tools.market_data import get_bars, get_quote
from tools.indicators import calc_indicators

def _bars_to_df(bars: List[Bar]) -> pd.DataFrame:
    """Converts a list of Bar objects to a pandas DataFrame."""
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame([b.model_dump() for b in bars])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _safe_float(x: Any) -> float | None:
    """Safely converts a value to float or None."""
    try:
        val = float(x)
        return val if not np.isnan(val) else None
    except (TypeError, ValueError):
        return None

def _pct(a: float, b: float) -> float:
    """Computes percentage change from b to a."""
    if b == 0:
        return 0.0
    return (a / b - 1) * 100

def _trend_from_indicators(df: pd.DataFrame, indicators: Indicators) -> Literal["up", "range", "down"]:
    """Determines trend based on MA5, MA20 and MA20 slope."""
    if indicators.ma5 is None or indicators.ma20 is None:
        return "range"
    
    # Calculate MA20 slope (current vs 5 days ago)
    if len(df) >= 6:
        ma20_series = df["close"].rolling(window=20).mean()
        ma20_today = ma20_series.iloc[-1]
        ma20_5_days_ago = ma20_series.iloc[-6]
        
        if indicators.ma5 > indicators.ma20 and ma20_today > ma20_5_days_ago:
            return "up"
        elif indicators.ma5 < indicators.ma20 and ma20_today < ma20_5_days_ago:
            return "down"
    
    return "range"

def _structure_from_levels(close: float, indicators: Indicators) -> Literal["healthy", "fragile", "broken"]:
    """Determines structure based on price relative to MA20 and ATR."""
    if indicators.ma20 is None:
        return "fragile"
    
    if close >= indicators.ma20:
        return "healthy"
    
    # If below MA20, check if it's within 0.5 * ATR14
    if indicators.atr14 is not None:
        if close >= indicators.ma20 - 0.5 * indicators.atr14:
            return "fragile"
        else:
            return "broken"
    
    return "broken"

def _heat_from_amount(df: pd.DataFrame, current_amount: Optional[float]) -> Literal["high", "mid", "low"]:
    """Determines heat based on amount percentile over last 20 days."""
    if current_amount is None or len(df) < 20:
        return "mid"
    
    amounts = df["amount"].tail(20).dropna()
    if len(amounts) < 5:
        return "mid"
    
    p70 = np.percentile(amounts, 70)
    p30 = np.percentile(amounts, 30)
    
    if current_amount >= p70:
        return "high"
    elif current_amount <= p30:
        return "low"
    else:
        return "mid"

def _location_from_range(close: float, df: pd.DataFrame) -> Literal["low", "mid", "high"]:
    """Determines location based on price position within last 20-day range."""
    if len(df) < 2:
        return "mid"
    
    recent_df = df.tail(20)
    low_val = recent_df["close"].min()
    high_val = recent_df["close"].max()
    
    if high_val == low_val:
        return "mid"
    
    range_val = high_val - low_val
    if close <= low_val + 0.33 * range_val:
        return "low"
    elif close >= low_val + 0.66 * range_val:
        return "high"
    else:
        return "mid"

def _flow_proxy(close_today: float, close_yesterday: float, volume_today: float, vol_ma20: Optional[float]) -> Literal["inflow", "neutral", "outflow"]:
    """Determines flow based on price change and volume relative to 20-day mean."""
    if vol_ma20 is None or vol_ma20 == 0:
        return "neutral"
    
    if close_today > close_yesterday and volume_today > vol_ma20:
        return "inflow"
    elif close_today < close_yesterday and volume_today > vol_ma20:
        return "outflow"
    else:
        return "neutral"

def analyze_stock(
    ticker: str,
    freq: str = "1d",
    lookback: int = 120,
    portfolio: PortfolioInput | None = None,
) -> dict:
    """
    Analyze a single stock and return a structured report.
    """
    # 1. Fetch Data
    bars = get_bars(ticker, freq, lookback)
    if not bars:
        raise ValueError(f"No data found for ticker {ticker}")
    
    quote = get_quote(ticker)
    indicators = calc_indicators(bars)
    df = _bars_to_df(bars)
    
    # 2. Determine Snapshot State
    last_close = quote.price
    
    # Consistency check: is quote for a new day or the same day as the last bar?
    quote_date = quote.timestamp.split(" ")[0]
    last_bar_date = df["ts"].iloc[-1]
    
    if quote_date > last_bar_date:
        # Quote is a new day (intraday or new close)
        prev_close = df["close"].iloc[-1]
        # If quote has amount, we could estimate volume, but for now use last bar vol as proxy or 0
        last_vol = df["volume"].iloc[-1] 
    else:
        # Quote is the same day as last bar
        prev_close = df["close"].iloc[-2] if len(df) >= 2 else last_close
        last_vol = df["volume"].iloc[-1]

    vol_ma20_val = indicators.vol_ma20

    trend = _trend_from_indicators(df, indicators)
    structure = _structure_from_levels(last_close, indicators)
    heat = _heat_from_amount(df, quote.amount)
    location = _location_from_range(last_close, df)
    flow = _flow_proxy(last_close, prev_close, last_vol, vol_ma20_val)
    
    snapshot = StockSnapshot(
        ticker=ticker,
        trend=trend,
        flow=flow,
        heat=heat,
        structure=structure,
        event="neu",
        location=location,
        timestamp=now_ts_str()
    )
    
    # 3. Key Levels
    if trend == "up":
        support = max(indicators.ma20 or 0, df["low"].tail(5).min())
        resistance = df["high"].tail(20).max()
    elif trend == "down":
        support = df["low"].tail(10).min()
        resistance = indicators.ma20 or df["high"].tail(20).max()
    else: # range
        support = df["low"].tail(10).min()
        resistance = df["high"].tail(20).max()
    
    stop_loss = support - 0.8 * indicators.atr14 if indicators.atr14 else support * 0.97
    tp1 = last_close * 1.06
    tp2 = max(resistance, last_close * 1.10)
    
    # 4. Report Generation
    summary = f"{ticker} 当前处于{trend}趋势，位置{location}。市场热度{heat}，资金流向{flow}。技术结构{structure}。"
    
    status_desc = {
        "trend": f"{trend} (基于MA5/MA20及斜率)",
        "heat": f"{heat} (基于成交额百分位)",
        "structure": f"{structure} (基于价格与MA20关系)",
        "location": f"{location} (基于20日高低位区间)",
        "flow": f"{flow} (基于量价配合)"
    }
    
    key_levels = {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": [round(tp1, 2), round(tp2, 2)]
    }
    
    risks = []
    if structure == "broken":
        risks.append("技术形态已破位，需警惕进一步下跌风险。")
    if location == "high":
        risks.append("股价处于近期高位，追高风险较大。")
    if flow == "outflow":
        risks.append("资金呈现流出迹象，短期承压。")
    if not risks:
        risks.append("暂无明显重大技术风险。")
    
    actions = []
    if portfolio:
        pos_limit = 0.25 if portfolio.style == "aggressive" else 0.15
        pnl_pct = _pct(last_close, portfolio.cost_price) if portfolio.cost_price else 0.0
        
        if portfolio.position >= pos_limit:
            actions.append(f"当前仓位 ({portfolio.position:.2%}) 已达上限，建议观望或逢高减仓。")
        elif trend == "up" and structure == "healthy":
            actions.append("趋势向好且结构健康，可考虑在支撑位附近分批建仓。")
            
        if portfolio.cost_price:
            if pnl_pct > 5 and last_close >= resistance * 0.98:
                actions.append(f"当前盈利 {pnl_pct:.2f}% 且接近阻力位，建议部分止盈。")
            elif pnl_pct < -5 and structure == "broken":
                actions.append(f"当前亏损 {pnl_pct:.2f}% 且形态破位，建议严格执行止损。")
    else:
        if trend == "up" and location != "high":
            actions.append("趋势上行且位置尚可，可关注回调买入机会。")
        elif structure == "broken":
            actions.append("形态走坏，建议暂时回避。")
        else:
            actions.append("建议继续观察，等待更明确的信号。")
            
    actions.append("注：以上分析仅供参考，不构成投资建议。入市有风险，投资需谨慎。")
    
    return {
        "meta": {
            "ticker": ticker,
            "freq": freq,
            "lookback": lookback,
            "timestamp": now_ts_str()
        },
        "quote": quote.model_dump(),
        "indicators": indicators.model_dump(),
        "snapshot": snapshot.model_dump(),
        "report": {
            "summary": summary,
            "status": status_desc,
            "key_levels": key_levels,
            "risks": risks,
            "actions": actions
        }
    }
