from __future__ import annotations
import akshare as ak
import pandas as pd
from typing import List, Dict, Any
from agents.market_breadth import get_market_regime
from core.models import now_ts_str
from tools.market_data import get_bars
from agents.state_machine import run_fsm_for_ticker

def generate_candidate_pool(
    top_n: int | None = None, 
    lookback: int = 120, 
    enforce_diversification: bool = False,
    mode: str = "simple"
) -> Dict[str, Any]:
    """
    Generate a list of candidate stocks based on market regime and technical filters.
    mode: "simple" (default) or "fsm"
    """
    # 1. If top_n is not provided, get it from market regime policy
    if top_n is None:
        regime_data = get_market_regime()
        top_n = regime_data.get("policy", {}).get("candidate_top_n", 10)
    
    try:
        # 2. Fetch spot data for basic filtering
        df_spot = ak.stock_zh_a_spot_em()
        # Filter out ST and New stocks (simplified)
        df_filtered = df_spot[~df_spot['名称'].str.contains("ST|N|C")]
        
        candidates = []
        groups = {"BASE_READY": 0, "BREAKOUT_READY": 0, "MOMENTUM_HOT": 0}

        if mode == "fsm":
            # For FSM mode, we need to scan more stocks to find those in specific states.
            # Since we can't scan all 5000+, we'll take a larger slice of active stocks.
            # Strategy: Take top 100 by turnover (active) + top 100 by gain (strong)
            # to increase chances of finding candidates in different stages.
            # For MVP speed, we limit to top 50 gainers + top 50 turnover.
            
            # Sort by turnover (成交额) if available, else just gainers
            if '成交额' in df_filtered.columns:
                df_active = df_filtered.sort_values(by='成交额', ascending=False).head(50)
                df_gainers = df_filtered.sort_values(by='涨跌幅', ascending=False).head(50)
                df_scan = pd.concat([df_active, df_gainers]).drop_duplicates(subset=['代码'])
            else:
                df_scan = df_filtered.sort_values(by='涨跌幅', ascending=False).head(100)
            
            print(f"Scanning {len(df_scan)} stocks with FSM...")
            
            for _, row in df_scan.iterrows():
                ticker = row['代码']
                try:
                    bars = get_bars(ticker, lookback=lookback, use_cache=True)
                    if len(bars) < 30:
                        continue
                        
                    df_bars = pd.DataFrame([b.model_dump() for b in bars])
                    fsm_result = run_fsm_for_ticker(df_bars)
                    state = fsm_result["state"]
                    
                    if state in ["BASE_READY", "BREAKOUT_READY", "MOMENTUM_HOT"]:
                        score = 0
                        if state == "BASE_READY":
                            score = 70
                            groups["BASE_READY"] += 1
                        elif state == "BREAKOUT_READY":
                            score = 80
                            groups["BREAKOUT_READY"] += 1
                        elif state == "MOMENTUM_HOT":
                            score = 60
                            groups["MOMENTUM_HOT"] += 1
                            
                        candidates.append({
                            "ticker": ticker,
                            "name": row['名称'],
                            "price": row['最新价'],
                            "pct_chg": row['涨跌幅'],
                            "state": state,
                            "score_total": score,
                            "reasons": fsm_result["reasons"],
                            "extras": fsm_result["extras"]
                        })
                except Exception as e:
                    # Skip individual failures
                    continue
            
            # Sort by score descending
            candidates.sort(key=lambda x: x["score_total"], reverse=True)
            # Limit to top_n
            candidates = candidates[:top_n]
            
        else:
            # Simple mode: Top gainers
            df_sorted = df_filtered.sort_values(by='涨跌幅', ascending=False)
            for _, row in df_sorted.head(top_n).iterrows():
                candidates.append({
                    "ticker": row['代码'],
                    "name": row['名称'],
                    "price": row['最新价'],
                    "pct_chg": row['涨跌幅'],
                    "reason": "Top gainer in current spot data"
                })
            
        return {
            "status": "success",
            "timestamp": now_ts_str(),
            "parameters": {
                "top_n": top_n,
                "lookback": lookback,
                "enforce_diversification": enforce_diversification,
                "mode": mode
            },
            "candidates": candidates,
            "count": len(candidates),
            "groups": groups if mode == "fsm" else None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": now_ts_str()
        }
