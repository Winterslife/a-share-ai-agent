from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set

# 1. Enum-like state strings
STATES = {
    "OFF",
    "BASE_BUILDING",
    "BASE_READY",
    "BREAKOUT_READY",
    "MOMENTUM_HOT",
    "INVALIDATED",
    "COOLDOWN"
}

# 2. Dataclass for memory
@dataclass
class StateMemory:
    last_state: str = "OFF"
    last_state_change: Optional[str] = None
    base_low: Optional[float] = None
    resistance: Optional[float] = None
    ratio_mid_seen_date: Optional[str] = None
    cooldown_until: Optional[str] = None
    base_vol_benchmark: Optional[float] = None

# 3. Feature computation
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features required for FSM from daily bars DataFrame.
    Expects columns: ts, open, high, low, close, volume, amount.
    """
    df = df.copy()
    
    # Ensure numeric
    cols = ["open", "high", "low", "close", "volume", "amount"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    # Moving Averages
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    
    # MA20 Slope (5 days ago)
    df["ma20_slope"] = df["ma20"].diff() # Daily change
    # To match "ma20_today - ma20_5days_ago", we can use diff(5) or just use daily slope trend
    # The prompt says: ma20_slope = ma20_today - ma20_5days_ago
    df["ma20_slope_5d"] = df["ma20"].diff(5)
    
    # LLV / HHV
    df["llv10"] = df["close"].rolling(window=10).min()
    df["llv20"] = df["close"].rolling(window=20).min()
    df["hhv20"] = df["close"].rolling(window=20).max()
    
    # Prev HHV/LLV (Shifted) for breakout/support logic
    df["prev_hhv20"] = df["high"].rolling(window=20).max().shift(1)
    df["prev_llv20"] = df["low"].rolling(window=20).min().shift(1)
    
    # ATR14
    high_low = df["high"] - df["low"]
    high_cp = np.abs(df["high"] - df["close"].shift())
    low_cp = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()
    
    # Volume Metrics
    if "amount" in df.columns:
        # base_vol = median(amount_last_10)
        df["base_vol"] = df["amount"].rolling(window=10).median()
        # trial_ratio = amount / base_vol
        df["trial_ratio"] = df["amount"] / df["base_vol"]
        
        # amt_ma20
        df["amt_ma20"] = df["amount"].rolling(window=20).mean()
        
        # cnt_amt_ok_5: count in last 5 days where amount >= 0.80 * base_vol
        # Changed from 0.65 * amt_ma20 to 0.80 * base_vol to be more robust to outliers
        amt_ok = (df["amount"] >= 0.80 * df["base_vol"]).astype(int)
        df["cnt_amt_ok_5"] = amt_ok.rolling(window=5).sum()
    else:
        df["base_vol"] = np.nan
        df["trial_ratio"] = np.nan
        df["amt_ma20"] = np.nan
        df["cnt_amt_ok_5"] = np.nan
        
    return df

# 4. Guard functions
def _get_val(row: pd.Series, key: str, default=None):
    return row[key] if key in row and pd.notna(row[key]) else default

def base_building_hint(row: pd.Series) -> bool:
    # llv10 >= llv20 OR abs(ma20_slope) small OR close >= ma20*0.95
    llv10 = _get_val(row, "llv10")
    llv20 = _get_val(row, "llv20")
    ma20_slope = _get_val(row, "ma20_slope_5d", 0) # Use 5d slope
    close = _get_val(row, "close")
    ma20 = _get_val(row, "ma20")
    
    cond1 = (llv10 is not None and llv20 is not None and llv10 >= llv20)
    cond2 = (abs(ma20_slope) <= 0.02 * close) if close else False # heuristic for "small"
    cond3 = (close >= ma20 * 0.95) if (close and ma20) else False
    
    return cond1 or cond2 or cond3

def structure_ok(row: pd.Series) -> bool:
    # close >= 0.97*ma20 OR ma20_slope >= 0
    close = _get_val(row, "close")
    ma20 = _get_val(row, "ma20")
    ma20_slope = _get_val(row, "ma20_slope_5d", 0)
    
    # Fallback if MA20 missing: close >= MA10
    if ma20 is None:
        ma10 = _get_val(row, "ma10")
        if ma10 is not None and close is not None:
            return close >= ma10
        return True # Relax if no data
        
    cond1 = (close >= 0.97 * ma20)
    cond2 = (ma20_slope >= 0)
    return cond1 or cond2

def base_ready_v2(row: pd.Series, history_df: pd.DataFrame, idx: int) -> Tuple[bool, List[str]]:
    reasons = []
    
    # A) llv10 >= llv20
    llv10 = _get_val(row, "llv10")
    llv20 = _get_val(row, "llv20")
    if llv10 is None or llv20 is None or llv10 < llv20:
        return False, []
    reasons.append(f"LLV10({llv10:.2f})>=LLV20({llv20:.2f})")
    
    # B) structure_ok
    if not structure_ok(row):
        return False, []
    reasons.append(f"结构修复:C={row['close']:.2f},MA20={_get_val(row,'ma20',0):.2f}")
    
    # C) volume_not_broken
    # cnt_amt_ok_5 >= 4 (if amount exists)
    if pd.notna(row.get("amount")):
        cnt = _get_val(row, "cnt_amt_ok_5", 0)
        if cnt < 4:
            return False, []
        reasons.append(f"量能稳健:cnt={cnt}/5")
    else:
        reasons.append("量能缺失(跳过)")
        
    # D) trial_volume_seen: in last 7 days, exists day where 1.25 <= trial_ratio <= 1.80
    # We need to look back 7 days from current idx
    start_idx = max(0, idx - 6)
    recent_rows = history_df.iloc[start_idx : idx + 1]
    
    trial_seen = False
    max_ratio = 0.0
    if "trial_ratio" in recent_rows.columns:
        # Filter valid ratios
        valid_ratios = recent_rows["trial_ratio"].dropna()
        for r in valid_ratios:
            if 1.25 <= r <= 1.80:
                trial_seen = True
            max_ratio = max(max_ratio, r)
            
    if not trial_seen:
        # If amount is missing, we might skip this or fail. 
        # Prompt says "If amount missing: skip volume guards".
        if pd.isna(row.get("amount")):
            pass
        else:
            return False, []
    else:
        reasons.append(f"试探抬头:max_ratio={max_ratio:.2f}")
        
    return True, reasons

def breakout_ready(row: pd.Series, history_df: pd.DataFrame, idx: int) -> Tuple[bool, List[str]]:
    reasons = []
    close = _get_val(row, "close")
    prev_hhv20 = _get_val(row, "prev_hhv20")
    ma20 = _get_val(row, "ma20")
    ma20_slope = _get_val(row, "ma20_slope_5d", 0)
    
    # Trigger 1: close >= 0.98 * prev_hhv20 (using shifted HHV)
    if close and prev_hhv20 and close >= 0.98 * prev_hhv20:
        reasons.append(f"逼近前高:C={close:.2f},HHV={prev_hhv20:.2f}")
        return True, reasons
        
    # Trigger 2: vol_ratio_trend
    # trial_ratio rising 2 consecutive days and last >= 1.2
    if "trial_ratio" in history_df.columns and idx >= 2:
        r_today = history_df["trial_ratio"].iloc[idx]
        r_prev = history_df["trial_ratio"].iloc[idx-1]
        r_prev2 = history_df["trial_ratio"].iloc[idx-2]
        
        if pd.notna(r_today) and pd.notna(r_prev) and pd.notna(r_prev2):
            if r_today > r_prev > r_prev2 and r_today >= 1.2:
                reasons.append(f"量能递增:ratio={r_today:.2f}")
                return True, reasons
                
    # Trigger 3: close >= ma20 and ma20_slope > 0
    if close and ma20 and close >= ma20 and ma20_slope > 0:
        reasons.append(f"趋势向上:slope={ma20_slope:.3f}")
        return True, reasons
        
    return False, []

def momentum_hot(row: pd.Series) -> Tuple[bool, List[str]]:
    reasons = []
    close = _get_val(row, "close")
    prev_hhv20 = _get_val(row, "prev_hhv20")
    trial_ratio = _get_val(row, "trial_ratio", 0)
    
    # Trigger 1: close > prev_hhv20 AND trial_ratio >= 1.1
    if close and prev_hhv20 and close > prev_hhv20 and trial_ratio >= 1.1:
        reasons.append(f"突破确立:C>HHV,ratio={trial_ratio:.2f}")
        return True, reasons
        
    # Trigger 2: trial_ratio > 1.80
    if trial_ratio > 1.80:
        reasons.append(f"放量攻击:ratio={trial_ratio:.2f}")
        return True, reasons
        
    return False, []

def invalidated(row: pd.Series, memory: StateMemory, history_df: pd.DataFrame, idx: int) -> Tuple[bool, List[str]]:
    reasons = []
    llv10 = _get_val(row, "llv10")
    llv20 = _get_val(row, "llv20")
    close = _get_val(row, "close")
    atr14 = _get_val(row, "atr14")
    
    # 1. llv10 < llv20
    if llv10 is not None and llv20 is not None and llv10 < llv20:
        reasons.append("创新低(LLV10<LLV20)")
        return True, reasons
        
    # 2. close < base_low - 0.8*atr14
    if memory.base_low is not None and close is not None:
        threshold = memory.base_low * 0.97
        if atr14 is not None:
            threshold = memory.base_low - 0.8 * atr14
        
        if close < threshold:
            reasons.append(f"跌破支撑:C={close:.2f}<{threshold:.2f}")
            return True, reasons
            
    # 3. structure_ok false for 3 consecutive days AND volume weak
    # Check last 3 days
    if idx >= 2:
        # We need to check structure_ok for idx, idx-1, idx-2
        # This is a bit expensive if we re-evaluate, but okay for daily loop
        s_ok_0 = structure_ok(history_df.iloc[idx])
        s_ok_1 = structure_ok(history_df.iloc[idx-1])
        s_ok_2 = structure_ok(history_df.iloc[idx-2])
        
        if not s_ok_0 and not s_ok_1 and not s_ok_2:
            trial_ratio = _get_val(row, "trial_ratio", 1.0)
            if pd.notna(row.get("amount")) and trial_ratio < 0.9:
                reasons.append("结构持续走坏且缩量")
                return True, reasons
                
    return False, []

# 5. Main FSM update
def update_state(
    prev_state: str,
    memory: StateMemory,
    row: pd.Series,
    history_df: pd.DataFrame,
    idx: int,
    cooldown_days: int = 7,
) -> Tuple[str, StateMemory, List[str], Dict]:
    
    today_ts = str(row["ts"])
    new_state = prev_state
    reasons = []
    
    # Extras to return
    extras = {
        "llv10": _get_val(row, "llv10"),
        "llv20": _get_val(row, "llv20"),
        "base_vol": _get_val(row, "base_vol"),
        "trial_ratio": _get_val(row, "trial_ratio"),
        "cnt_amt_ok_5": _get_val(row, "cnt_amt_ok_5"),
        "hhv20": _get_val(row, "hhv20"),
        "ma20": _get_val(row, "ma20"),
        "ma20_slope": _get_val(row, "ma20_slope_5d"),
        "base_low": memory.base_low,
        "resistance": memory.resistance
    }

    # Handle COOLDOWN
    if prev_state == "COOLDOWN":
        if memory.cooldown_until and today_ts >= memory.cooldown_until:
            new_state = "OFF"
            reasons.append("冷却期结束")
        else:
            return "COOLDOWN", memory, ["冷却中"], extras

    # Handle INVALIDATED -> COOLDOWN
    if prev_state == "INVALIDATED":
        new_state = "COOLDOWN"
        # Simple date math approximation or just rely on string comparison for next check
        # For simplicity in this MVP, we won't do complex date math, just assume next day is cooldown
        # But prompt asks for cooldown_days. 
        # We'll just set it. Since we are running daily loop, we can't easily add days to string without datetime.
        # We will rely on the caller or just set a flag. 
        # Actually, let's try to parse date.
        try:
            dt = pd.to_datetime(today_ts)
            cooldown_dt = dt + pd.Timedelta(days=cooldown_days)
            memory.cooldown_until = cooldown_dt.strftime("%Y-%m-%d")
        except:
            memory.cooldown_until = "9999-12-31"
            
        return "COOLDOWN", memory, ["进入冷却"], extras

    # Check Invalidation for active states
    if prev_state in ["BASE_BUILDING", "BASE_READY", "BREAKOUT_READY", "MOMENTUM_HOT"]:
        is_inv, inv_reasons = invalidated(row, memory, history_df, idx)
        if is_inv:
            new_state = "INVALIDATED"
            memory.last_state = prev_state
            memory.last_state_change = today_ts
            return new_state, memory, inv_reasons, extras

    # State Transitions
    if prev_state == "OFF":
        if base_building_hint(row):
            new_state = "BASE_BUILDING"
            reasons.append("初现企稳迹象")
            
    elif prev_state == "BASE_BUILDING":
        is_ready, ready_reasons = base_ready_v2(row, history_df, idx)
        if is_ready:
            new_state = "BASE_READY"
            reasons.extend(ready_reasons)
            # Freeze base_low using prev_llv20 (shifted) to avoid lookahead bias or current day noise
            memory.base_low = _get_val(row, "prev_llv20")
            memory.base_vol_benchmark = _get_val(row, "base_vol")
            
    elif prev_state == "BASE_READY":
        is_breakout, brk_reasons = breakout_ready(row, history_df, idx)
        if is_breakout:
            new_state = "BREAKOUT_READY"
            reasons.extend(brk_reasons)
            # Freeze resistance using prev_hhv20
            memory.resistance = _get_val(row, "prev_hhv20")
            
    elif prev_state == "BREAKOUT_READY":
        is_hot, hot_reasons = momentum_hot(row)
        if is_hot:
            new_state = "MOMENTUM_HOT"
            reasons.extend(hot_reasons)
            
    elif prev_state == "MOMENTUM_HOT":
        # Stays unless invalidated
        pass

    if new_state != prev_state:
        memory.last_state_change = today_ts
        
    # Update memory for trial ratio seen
    trial_ratio = _get_val(row, "trial_ratio")
    if trial_ratio and 1.25 <= trial_ratio <= 1.80:
        memory.ratio_mid_seen_date = today_ts

    return new_state, memory, reasons, extras

def run_fsm_for_ticker(df: pd.DataFrame) -> Dict:
    """
    Run FSM over the dataframe and return the final state and details.
    """
    df_features = compute_features(df)
    memory = StateMemory()
    state = "OFF"
    last_reasons = []
    last_extras = {}
    
    # Iterate through history to simulate FSM
    # We need at least some history for features to be valid
    # Start from index where features are likely ready (e.g. 20)
    start_idx = 20
    if len(df_features) < start_idx:
        start_idx = 0
        
    for i in range(start_idx, len(df_features)):
        row = df_features.iloc[i]
        state, memory, reasons, extras = update_state(state, memory, row, df_features, i)
        if reasons:
            last_reasons = reasons
        last_extras = extras
        
    return {
        "state": state,
        "reasons": last_reasons,
        "extras": last_extras,
        "memory": memory
    }
