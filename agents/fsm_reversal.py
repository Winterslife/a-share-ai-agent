# agents/fsm_reversal.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# States
# -----------------------------
OFF = "OFF"
BASE_BUILDING = "BASE_BUILDING"
BASE_READY = "BASE_READY"
BREAKOUT_READY = "BREAKOUT_READY"
MOMENTUM_HOT = "MOMENTUM_HOT"
INVALIDATED = "INVALIDATED"
COOLDOWN = "COOLDOWN"

ACTIVE_STATES = {BASE_BUILDING, BASE_READY, BREAKOUT_READY, MOMENTUM_HOT}


# -----------------------------
# Config
# -----------------------------
@dataclass
class FSMConfig:
    # Liquidity (Relaxed from 3e7 to 1e7 for bottom stocks)
    MIN_AMT: float = 1e7

    # Cooldown
    cooldown_days: int = 7

    # Oversold (Relaxed thresholds)
    bias60_th: float = -0.10  # Was -0.18
    rsi14_th: float = 45.0    # Was 35.0

    # Slope thresholds
    slope_atr_k: float = 0.25
    atr_fallback_k: float = 0.005  # if atr missing: 0.005*close

    # Volume / trial
    dryup_k: float = 0.75
    amt_ok_k: float = 0.80
    dryup_cnt_min: int = 1  # Relaxed from 3 to 1 to capture slow bulls
    cnt_amt_ok_5_min: int = 4
    trial_lo: float = 1.15
    trial_hi: float = 1.80
    trial_min_valid_samples: int = 3

    # Breakout / momentum
    near_prev_hhv_k: float = 0.98
    trial_up_last_min: float = 1.2
    momentum_trial_min: float = 1.1
    momentum_trial_hot: float = 1.8

    # Invalidation
    base_low_atr_k: float = 1.0
    structure_false_days: int = 3
    invalid_trial_max: float = 0.9


# -----------------------------
# Memory
# -----------------------------
@dataclass
class StateMemory:
    last_state: str = OFF
    last_state_change: Optional[str] = None  # YYYY-MM-DD

    base_low: Optional[float] = None         # frozen when entering BASE_READY (prev_llv20)
    resistance: Optional[float] = None       # frozen when entering BREAKOUT_READY (prev_hhv20)

    ratio_mid_seen_date: Optional[str] = None
    cooldown_until: Optional[str] = None     # YYYY-MM-DD

    # internal (for invalidated rule 3)
    structure_false_streak: int = 0


# -----------------------------
# Utils
# -----------------------------
def _isfinite(x: Any) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: Any, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}" if _isfinite(x) else "NaN"


def _to_dt_series(ts: pd.Series) -> Tuple[pd.Series, List[str]]:
    reasons: List[str] = []
    try:
        dt = pd.to_datetime(ts, errors="coerce")
    except Exception:
        dt = pd.to_datetime(pd.Series([None] * len(ts)), errors="coerce")
        reasons.append("ts parse failed: fallback NaT")

    if dt.isna().any():
        reasons.append("ts contains NaT after parsing (some rows degraded)")
    return dt, reasons


def _date_str(dt: Any) -> Optional[str]:
    try:
        t = pd.to_datetime(dt, errors="coerce")
        if pd.isna(t):
            return None
        return t.strftime("%Y-%m-%d")
    except Exception:
        return None


def _safe_atr(row: pd.Series, cfg: FSMConfig) -> float:
    atr = _f(row.get("atr14", np.nan))
    if np.isfinite(atr):
        return atr
    close = _f(row.get("close", np.nan))
    if np.isfinite(close):
        return cfg.atr_fallback_k * close
    return float("nan")


# -----------------------------
# Feature computation (SPEC)
# -----------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features required for reversal FSM.

    - safe downgrade: missing columns => create NaN
    - sort by ts ascending
    - rolling uses min_periods to reduce false triggers
    """
    d = df.copy()

    # Ensure required columns exist
    for c in ["ts", "open", "high", "low", "close", "amount", "volume"]:
        if c not in d.columns:
            d[c] = np.nan

    # Parse ts and sort
    d["ts_dt"], _ = _to_dt_series(d["ts"])
    d = d.sort_values("ts_dt", kind="mergesort").reset_index(drop=True)

    # Numeric coercion
    for c in ["open", "high", "low", "close", "amount", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    close = d["close"]
    high = d["high"]
    low = d["low"]
    amount = d["amount"]

    # MA
    d["ma10"] = close.rolling(10, min_periods=5).mean()
    d["ma20"] = close.rolling(20, min_periods=10).mean()
    d["ma60"] = close.rolling(60, min_periods=30).mean()
    d["ma20_slope_5d"] = d["ma20"].diff(5)

    # LLV/HHV
    d["llv10"] = low.rolling(10, min_periods=5).min()
    d["llv20"] = low.rolling(20, min_periods=10).min()
    d["hhv20"] = high.rolling(20, min_periods=10).max()
    d["prev_llv20"] = d["llv20"].shift(1)
    d["prev_hhv20"] = d["hhv20"].shift(1)

    # ATR14 (TR rolling mean)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr14"] = tr.rolling(14, min_periods=7).mean()

    # Amount / volume metrics
    d["base_vol"] = amount.rolling(10, min_periods=5).median()
    d["trial_ratio"] = amount / d["base_vol"]

    # dryup_flag: NaN stays NaN (do NOT treat as False)
    dryup_flag = np.where(
        amount.notna() & d["base_vol"].notna(),
        amount <= 0.75 * d["base_vol"],
        np.nan,
    )
    d["dryup_flag"] = pd.Series(dryup_flag, index=d.index).astype("float")

    d["dryup_cnt_10"] = d["dryup_flag"].rolling(10, min_periods=5).sum()

    amt_ok = np.where(
        amount.notna() & d["base_vol"].notna(),
        amount >= 0.80 * d["base_vol"],
        np.nan,
    )
    d["amt_ok"] = pd.Series(amt_ok, index=d.index).astype("float")
    d["cnt_amt_ok_5"] = d["amt_ok"].rolling(5, min_periods=3).sum()

    # Oversold
    d["bias60"] = (close - d["ma60"]) / d["ma60"]

    # Wilder RSI14
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    d["rsi14"] = 100 - (100 / (1 + rs))

    # Liquidity helper (for OversoldGate)
    d["amt_mean_20"] = amount.rolling(20, min_periods=10).mean()

    return d


# -----------------------------
# Guards (return bool + reasons)
# -----------------------------
def oversold_gate(row: pd.Series, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    bias60 = row.get("bias60", np.nan)
    rsi14 = row.get("rsi14", np.nan)
    amt20 = row.get("amt_mean_20", np.nan)

    if not _isfinite(bias60) or not _isfinite(rsi14):
        return False, ["OversoldGate: 样本不足/缺数据(bias60/rsi14)"]

    if not _isfinite(amt20):
        return False, ["OversoldGate: 样本不足/缺数据(amount_20mean)"]

    if float(amt20) < cfg.MIN_AMT:
        return False, [f"OversoldGate: 低流动性 amount20mean={_fmt(amt20,0)} < MIN_AMT={cfg.MIN_AMT:.0f}"]

    ok = (float(bias60) <= cfg.bias60_th) and (float(rsi14) <= cfg.rsi14_th)
    if ok:
        reasons.append(f"OversoldGate: bias60={_fmt(bias60,3)}, rsi14={_fmt(rsi14,1)}")
    else:
        reasons.append(f"OversoldGate fail: bias60={_fmt(bias60,3)}, rsi14={_fmt(rsi14,1)}")
    return ok, reasons


def base_building_hint(row: pd.Series, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    close = row.get("close", np.nan)
    if not _isfinite(close):
        return False, ["base_building_hint: close 缺失"]

    llv10 = row.get("llv10", np.nan)
    llv20 = row.get("llv20", np.nan)
    ma20 = row.get("ma20", np.nan)
    slope = row.get("ma20_slope_5d", np.nan)
    atr = _safe_atr(row, cfg)

    c1 = _isfinite(llv10) and _isfinite(llv20) and float(llv10) >= float(llv20)
    if c1:
        reasons.append("base_building_hint: LLV10>=LLV20(不再创新低)")

    c2 = _isfinite(slope) and _isfinite(atr) and abs(float(slope)) <= cfg.slope_atr_k * float(atr)
    if c2:
        reasons.append(f"base_building_hint: |ma20_slope_5d|<=0.25*ATR (slope={_fmt(slope)}, atr={_fmt(atr)})")

    c3 = False
    if _isfinite(ma20) and _isfinite(slope) and _isfinite(atr):
        c3 = (float(close) >= 0.95 * float(ma20)) and (float(slope) >= -cfg.slope_atr_k * float(atr))
    if c3:
        reasons.append("base_building_hint: close>=0.95*MA20 且 MA20_slope 不强负")

    ok = bool(c1 or c2 or c3)
    if not ok:
        reasons.append("base_building_hint: 无明显企稳迹象")
    return ok, reasons


def structure_ok(row: pd.Series, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    close = row.get("close", np.nan)
    if not _isfinite(close):
        return False, ["structure_ok: close 缺失"]

    ma20 = row.get("ma20", np.nan)
    ma10 = row.get("ma10", np.nan)
    slope = row.get("ma20_slope_5d", np.nan)

    ma_ref = ma20 if _isfinite(ma20) else ma10
    if not _isfinite(ma_ref):
        return False, ["structure_ok: ma20/ma10 缺失(更保守)"]

    c1 = float(close) >= 0.97 * float(ma_ref)
    c2 = _isfinite(slope) and float(slope) >= 0.0
    ok = bool(c1 or c2)

    if ok:
        reasons.append(f"structure_ok: close>=0.97*MA(ref) 或 slope>=0 (close={_fmt(close)}, ma_ref={_fmt(ma_ref)}, slope={_fmt(slope)})")
    else:
        reasons.append(f"structure_ok fail: close={_fmt(close)}, ma_ref={_fmt(ma_ref)}, slope={_fmt(slope)})")
    return ok, reasons


def base_ready_v2(df_feat: pd.DataFrame, i: int, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    row = df_feat.iloc[i]
    reasons: List[str] = []

    # A) llv10 >= llv20
    llv10 = row.get("llv10", np.nan)
    llv20 = row.get("llv20", np.nan)
    if not (_isfinite(llv10) and _isfinite(llv20) and float(llv10) >= float(llv20)):
        return False, ["base_ready_v2 fail: LLV10<LLV20 或样本不足"]
    reasons.append("LLV10>=LLV20")

    # B) structure_ok
    sok, sok_r = structure_ok(row, cfg)
    reasons.extend(sok_r)
    if not sok:
        return False, reasons

    # C) cnt_amt_ok_5 >= 4 (NaN => allow with reason)
    cnt = row.get("cnt_amt_ok_5", np.nan)
    if _isfinite(cnt):
        if float(cnt) < cfg.cnt_amt_ok_5_min:
            return False, reasons + [f"base_ready_v2 fail: cnt_amt_ok_5={_fmt(cnt,1)}/5 < {cfg.cnt_amt_ok_5_min}"]
        reasons.append(f"cnt_amt_ok_5={_fmt(cnt,1)}/5")
    else:
        reasons.append("cnt_amt_ok_5 NaN: 量能样本不足放行")

    # D) dryup_cnt_10 >= 3 (NaN => allow with reason)
    dry = row.get("dryup_cnt_10", np.nan)
    if _isfinite(dry):
        if float(dry) < cfg.dryup_cnt_min:
            return False, reasons + [f"base_ready_v2 fail: dryup_cnt_10={_fmt(dry,1)}/10 < {cfg.dryup_cnt_min}"]
        reasons.append(f"dryup_cnt_10={_fmt(dry,1)}/10")
    else:
        reasons.append("dryup_cnt_10 NaN: 地量样本不足放行")

    # E) last 7 days: exists trial_ratio in [1.15,1.80]
    start = max(0, i - 6)
    tr = df_feat.loc[start:i, "trial_ratio"] if "trial_ratio" in df_feat.columns else pd.Series([], dtype=float)
    tr_valid = tr.dropna()

    if len(tr_valid) < cfg.trial_min_valid_samples:
        reasons.append(f"trial_ratio有效样本不足放行(n={len(tr_valid)})")
        if len(tr_valid) > 0:
            reasons.append(f"trial_ratio统计 max={_fmt(tr_valid.max(),2)} (仅统计)")
        return True, reasons

    seen = ((tr_valid >= cfg.trial_lo) & (tr_valid <= cfg.trial_hi)).any()
    if not seen:
        return False, reasons + [f"base_ready_v2 fail: 近7天未见 trial_ratio in [{cfg.trial_lo},{cfg.trial_hi}], max={_fmt(tr_valid.max(),2)}"]
    reasons.append(f"trial_ratio_seen max={_fmt(tr_valid.max(),2)}")
    return True, reasons


def breakout_ready(df_feat: pd.DataFrame, i: int, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    row = df_feat.iloc[i]
    reasons: List[str] = []

    close = row.get("close", np.nan)
    prev_hhv20 = row.get("prev_hhv20", np.nan)
    ma20 = row.get("ma20", np.nan)
    slope = row.get("ma20_slope_5d", np.nan)

    # 1) close >= 0.98*prev_hhv20 (NO lookahead)
    c1 = _isfinite(close) and _isfinite(prev_hhv20) and float(close) >= cfg.near_prev_hhv_k * float(prev_hhv20)
    if c1:
        reasons.append(f"逼近前高: close={_fmt(close)}, prev_hhv20={_fmt(prev_hhv20)}")

    # 2) trial_ratio increasing for 2~3 days, last>=1.2
    c2 = False
    if "trial_ratio" in df_feat.columns:
        tr = df_feat.loc[max(0, i - 2):i, "trial_ratio"].dropna()
        if len(tr) >= 3:
            c2 = (tr.iloc[0] < tr.iloc[1] < tr.iloc[2]) and (tr.iloc[2] >= cfg.trial_up_last_min)
        elif len(tr) == 2:
            c2 = (tr.iloc[0] < tr.iloc[1]) and (tr.iloc[1] >= cfg.trial_up_last_min)
        if c2:
            reasons.append("量比递增: 连续2~3天上行且末日>=1.2")

    # 3) close>=ma20 and slope>0
    c3 = _isfinite(close) and _isfinite(ma20) and _isfinite(slope) and (float(close) >= float(ma20)) and (float(slope) > 0)
    if c3:
        reasons.append("趋势向上: close>=MA20 且 MA20_slope>0")

    ok = bool(c1 or c2 or c3)
    if not ok:
        reasons.append("breakout_ready: 条件未满足")
    return ok, reasons


def momentum_hot(row: pd.Series, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    close = row.get("close", np.nan)
    prev_hhv20 = row.get("prev_hhv20", np.nan)
    tr = row.get("trial_ratio", np.nan)

    c1 = _isfinite(close) and _isfinite(prev_hhv20) and _isfinite(tr) and (float(close) >= float(prev_hhv20)) and (float(tr) >= cfg.momentum_trial_min)
    c2 = _isfinite(tr) and float(tr) > cfg.momentum_trial_hot

    ok = bool(c1 or c2)
    if ok:
        reasons.append(f"momentum_hot: close={_fmt(close)}, prev_hhv20={_fmt(prev_hhv20)}, trial_ratio={_fmt(tr,2)}")
    return ok, reasons


def invalidated(df_feat: pd.DataFrame, i: int, mem: StateMemory, cfg: FSMConfig) -> Tuple[bool, List[str]]:
    row = df_feat.iloc[i]
    reasons: List[str] = []

    llv10 = row.get("llv10", np.nan)
    llv20 = row.get("llv20", np.nan)
    close = row.get("close", np.nan)
    tr = row.get("trial_ratio", np.nan)

    # 1) llv10 < llv20
    if _isfinite(llv10) and _isfinite(llv20) and float(llv10) < float(llv20):
        return True, ["Invalidated: LLV10<LLV20(再次创新低)"]

    # 2) close < base_low - 1.0*atr14 (base_low frozen at BASE_READY: prev_llv20)
    if mem.base_low is not None and _isfinite(close):
        atr = _safe_atr(row, cfg)
        if _isfinite(atr) and _isfinite(mem.base_low):
            if float(close) < float(mem.base_low) - cfg.base_low_atr_k * float(atr):
                return True, [f"Invalidated: close fell below base_low-ATR (close={_fmt(close)}, base_low={_fmt(mem.base_low)}, atr={_fmt(atr)})"]

    # 3) structure_ok 连续3天 False 且 trial_ratio < 0.9 (amount缺失 => trial_ratio NaN => 不触发)
    if mem.structure_false_streak >= cfg.structure_false_days and _isfinite(tr) and float(tr) < cfg.invalid_trial_max:
        return True, [f"Invalidated: structure_ok连续{mem.structure_false_streak}天False 且 trial_ratio<{cfg.invalid_trial_max} (trial_ratio={_fmt(tr,2)})"]

    return False, reasons


# -----------------------------
# FSM core
# -----------------------------
def run_fsm(df: pd.DataFrame, cfg: Optional[FSMConfig] = None) -> pd.DataFrame:
    """
    Run FSM and return full trajectory:
      columns: ts, state, reasons(list[str]), extras(dict)
    """
    cfg = cfg or FSMConfig()
    feat = compute_features(df)

    mem = StateMemory()
    state = OFF

    out: List[Dict[str, Any]] = []

    for i in range(len(feat)):
        row = feat.iloc[i]
        day_dt = row.get("ts_dt", pd.NaT)
        day = _date_str(day_dt) or "UNKNOWN"

        reasons: List[str] = []
        extras = _build_extras(row, mem)

        # update structure_false_streak
        sok, _ = structure_ok(row, cfg)
        mem.structure_false_streak = 0 if sok else (mem.structure_false_streak + 1)

        # COOLDOWN -> OFF
        if state == COOLDOWN:
            if mem.cooldown_until is None:
                reasons.append("COOLDOWN: cooldown_until缺失，降级回OFF")
                state = OFF
            else:
                cd = pd.to_datetime(mem.cooldown_until, errors="coerce")
                dd = pd.to_datetime(day_dt, errors="coerce")
                if pd.isna(cd) or pd.isna(dd):
                    reasons.append("COOLDOWN: 日期解析失败，降级回OFF")
                    state = OFF
                elif dd >= cd:
                    reasons.append(f"COOLDOWN结束: today={day} >= cooldown_until={mem.cooldown_until}")
                    state = OFF
                else:
                    reasons.append(f"冷却中: until {mem.cooldown_until}")

            extras = _build_extras(row, mem)
            out.append({"ts": day, "state": state, "reasons": reasons, "extras": extras})
            mem.last_state = state
            mem.last_state_change = day if day != "UNKNOWN" else mem.last_state_change
            continue

        # Active states invalidation check
        if state in ACTIVE_STATES:
            inv, inv_r = invalidated(feat, i, mem, cfg)
            if inv:
                # output INVALIDATED today
                reasons.extend(inv_r)
                out.append({"ts": day, "state": INVALIDATED, "reasons": reasons, "extras": _build_extras(row, mem)})

                # immediately enter cooldown for next day
                dd = pd.to_datetime(day_dt, errors="coerce")
                if not pd.isna(dd):
                    cd_until = (dd + pd.Timedelta(days=int(cfg.cooldown_days))).normalize()
                    mem.cooldown_until = cd_until.strftime("%Y-%m-%d")
                    reasons2 = [f"Enter COOLDOWN: cooldown_until={mem.cooldown_until}"]
                else:
                    mem.cooldown_until = None
                    reasons2 = ["Enter COOLDOWN: ts invalid, cooldown_until=None"]

                state = COOLDOWN
                mem.last_state = state
                mem.last_state_change = day if day != "UNKNOWN" else mem.last_state_change

                # also record a COOLDOWN row for same day?（不建议）
                # spec tests通常看 next day；这里不额外插行
                continue

        # Transitions
        if state == OFF:
            og, og_r = oversold_gate(row, cfg)
            bb, bb_r = base_building_hint(row, cfg)
            reasons.extend(og_r)
            reasons.extend(bb_r)
            if og and bb:
                state = BASE_BUILDING
                reasons.append("Transition: OFF->BASE_BUILDING")

        elif state == BASE_BUILDING:
            br, br_r = base_ready_v2(feat, i, cfg)
            reasons.extend(br_r)
            if br:
                state = BASE_READY
                # freeze base_low = prev_llv20
                base_low = row.get("prev_llv20", np.nan)
                if _isfinite(base_low):
                    mem.base_low = float(base_low)
                    reasons.append(f"Freeze base_low=prev_llv20={_fmt(mem.base_low)}")
                else:
                    mem.base_low = None
                    reasons.append("Freeze base_low failed: prev_llv20 NaN")
                reasons.append("Transition: BASE_BUILDING->BASE_READY")

        elif state == BASE_READY:
            bo, bo_r = breakout_ready(feat, i, cfg)
            reasons.extend(bo_r)
            if bo:
                state = BREAKOUT_READY
                # freeze resistance = prev_hhv20
                res = row.get("prev_hhv20", np.nan)
                if _isfinite(res):
                    mem.resistance = float(res)
                    reasons.append(f"Freeze resistance=prev_hhv20={_fmt(mem.resistance)}")
                else:
                    mem.resistance = None
                    reasons.append("Freeze resistance failed: prev_hhv20 NaN")
                reasons.append("Transition: BASE_READY->BREAKOUT_READY")
                
                # Check immediate jump to MOMENTUM_HOT
                mh, mh_r = momentum_hot(row, cfg)
                if mh:
                    state = MOMENTUM_HOT
                    reasons.extend(mh_r)
                    reasons.append("Transition: BREAKOUT_READY->MOMENTUM_HOT (Same Day)")

        elif state == BREAKOUT_READY:
            # Check decay/stale
            # If close < 0.94 * resistance AND trial_ratio < 1.0 -> Downgrade to BASE_READY
            close = row.get("close", np.nan)
            tr = row.get("trial_ratio", np.nan)
            if mem.resistance and _isfinite(close) and float(close) < 0.94 * mem.resistance:
                if _isfinite(tr) and float(tr) < 1.0:
                    state = BASE_READY
                    reasons.append(f"Downgrade: Stale Breakout (C={_fmt(close)} < 0.94*Res={_fmt(mem.resistance)})")
            
            if state == BREAKOUT_READY: # If not downgraded
                mh, mh_r = momentum_hot(row, cfg)
                if mh:
                    state = MOMENTUM_HOT
                    reasons.extend(mh_r)
                    reasons.append("Transition: BREAKOUT_READY->MOMENTUM_HOT")
                else:
                    reasons.extend(mh_r)

        elif state == MOMENTUM_HOT:
            reasons.append("MOMENTUM_HOT: tracking only")

        # memory bookkeeping
        if day != "UNKNOWN":
            mem.last_state_change = day
        mem.last_state = state

        # ratio_mid_seen_date: only if truly within [1.15,1.80] today
        tr = row.get("trial_ratio", np.nan)
        if _isfinite(tr) and (cfg.trial_lo <= float(tr) <= cfg.trial_hi) and day != "UNKNOWN":
            mem.ratio_mid_seen_date = day

        extras = _build_extras(row, mem)
        out.append({"ts": day, "state": state, "reasons": reasons, "extras": extras})

    return pd.DataFrame(out)


def run_fsm_for_ticker(df: pd.DataFrame, cfg: Optional[FSMConfig] = None) -> Dict[str, Any]:
    """
    Convenience entry for candidate_pool.

    Returns:
      {state, reasons, extras, memory}
    """
    cfg = cfg or FSMConfig()
    traj = run_fsm(df, cfg=cfg)
    if traj.empty:
        mem = StateMemory()
        return {"state": OFF, "reasons": ["empty df"], "extras": {}, "memory": asdict(mem)}

    last = traj.iloc[-1].to_dict()
    extras = last.get("extras", {}) or {}
    mem_snap = extras.get("_memory_snapshot", {}) or {}

    return {
        "state": last.get("state", OFF),
        "reasons": last.get("reasons", []),
        "extras": extras,
        "memory": mem_snap,
    }


def _build_extras(row: pd.Series, mem: StateMemory) -> Dict[str, Any]:
    keys = [
        "llv10", "llv20", "prev_llv20", "prev_hhv20",
        "ma20", "ma60", "bias60", "rsi14", "atr14",
        "base_vol", "trial_ratio", "cnt_amt_ok_5", "dryup_cnt_10",
    ]
    extras: Dict[str, Any] = {}
    for k in keys:
        v = row.get(k, np.nan)
        extras[k] = float(v) if _isfinite(v) else None

    extras["base_low"] = mem.base_low
    extras["resistance"] = mem.resistance

    extras["_memory_snapshot"] = asdict(mem)
    return extras
