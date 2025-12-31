import pytest
import pandas as pd
import numpy as np
from agents.fsm_reversal import (
    compute_features, 
    run_fsm, 
    run_fsm_for_ticker, 
    check_breakout_ready,
    StateMemory
)

@pytest.fixture
def basic_df():
    dates = pd.date_range(start="2025-01-01", periods=100)
    df = pd.DataFrame({
        "ts": dates,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.0,
        "volume": 10000,
        "amount": 100000.0
    })
    # Make some price movement
    df["close"] = np.linspace(10, 20, 100)
    df["high"] = df["close"] + 1
    df["low"] = df["close"] - 1
    return df

def test_no_crash_with_missing_amount(basic_df):
    """Test that FSM runs without crashing when amount is missing or NaN."""
    df = basic_df.copy()
    df["amount"] = np.nan
    
    # Should not raise
    res = run_fsm_for_ticker(df)
    assert res["state"] in ["OFF", "BASE_BUILDING", "BASE_READY", "BREAKOUT_READY", "MOMENTUM_HOT", "INVALIDATED", "COOLDOWN"]
    
    # Test with missing column
    df_no_col = basic_df.drop(columns=["amount"])
    res2 = run_fsm_for_ticker(df_no_col)
    assert res2["state"] is not None

def test_ts_mixed_types():
    """Test robustness against mixed timestamp types."""
    df = pd.DataFrame({
        "ts": ["2025-01-01", pd.Timestamp("2025-01-02"), "2025-01-03"],
        "open": [10, 10, 10],
        "high": [11, 11, 11],
        "low": [9, 9, 9],
        "close": [10, 10, 10],
        "volume": [100, 100, 100],
        "amount": [1000, 1000, 1000]
    })
    
    # Should sort and run
    df_feat = compute_features(df)
    assert len(df_feat) == 3
    # Check if sorted correctly (implied by no crash and valid index)
    
    res = run_fsm_for_ticker(df)
    assert res["state"] == "OFF" # Not enough data, should stay OFF

def test_no_lookahead_breakout(basic_df):
    """Ensure breakout check uses prev_hhv20, not current hhv20."""
    df = basic_df.copy()
    # Construct a scenario:
    # Day N: High spikes to 100. HHV20 becomes 100.
    # Day N: Close is 99.
    # If using current HHV20 (100), 99 >= 0.98*100 is True.
    # If using prev HHV20 (say 50), 99 >= 0.98*50 is True.
    # Wait, we want to ensure it uses PREV.
    # Let's make prev HHV low (10), and current High huge (100).
    # If it uses current HHV (100), and Close is 11 (small rise), 11 < 0.98*100 (False).
    # If it uses prev HHV (10), 11 >= 0.98*10 (True).
    
    # So: Prev HHV = 10. Today High = 100. Close = 11.
    # Expected: True (Breakout of prev high).
    # If it used current high (100), it would fail.
    
    # We need enough history for HHV20 to be valid.
    # Let's mock the row and history.
    
    row = pd.Series({
        "close": 11.0,
        "prev_hhv20": 10.0,
        "hhv20": 100.0, # Current high pushed this up
        "ma20": 9.0,
        "ma20_slope_5d": 0.1
    })
    
    # Mock history not needed for this specific trigger if row has data
    history_df = pd.DataFrame() 
    
    is_brk, reasons = check_breakout_ready(row, history_df, 0)
    
    # Should be True because 11 >= 0.98 * 10
    assert is_brk is True
    assert any("逼近前高" in r for r in reasons)
    assert "HHV=10.00" in reasons[0] # Should reference prev_hhv20

def test_cooldown_logic(basic_df):
    """Test INVALIDATED -> COOLDOWN -> OFF transition."""
    # We can test this by mocking the state transitions or running a crafted sequence.
    # Let's run a crafted sequence where we force INVALIDATED.
    
    # Create a DF that triggers Oversold -> Base -> Ready -> Invalidated
    # This is hard to craft perfectly with synthetic data in short time.
    # Instead, let's test the update_state logic directly.
    
    from agents.fsm_reversal import update_state
    
    memory = StateMemory()
    row = pd.Series({"ts": "2025-01-01", "close": 10})
    history = pd.DataFrame()
    
    # 1. Force INVALIDATED
    state, mem, _, _ = update_state("INVALIDATED", memory, row, history, 0, cooldown_days=2)
    assert state == "COOLDOWN"
    assert mem.cooldown_until == "2025-01-03" # 1 + 2 days
    
    # 2. Next day: 2025-01-02. Should still be COOLDOWN.
    row2 = pd.Series({"ts": "2025-01-02", "close": 10})
    state, mem, _, _ = update_state("COOLDOWN", mem, row2, history, 1)
    assert state == "COOLDOWN"
    
    # 3. Target day: 2025-01-03. Should exit COOLDOWN.
    row3 = pd.Series({"ts": "2025-01-03", "close": 10})
    state, mem, reasons, extras = update_state("COOLDOWN", mem, row3, history, 2)
    assert state == "OFF"
    assert "冷却期结束" in reasons

def test_min_periods_prevents_false_ready(basic_df):
    """Ensure we don't get signals when data is insufficient."""
    # Use first few rows where rolling windows (e.g. 20, 60) are NaN
    df = basic_df.iloc[:15] # < 20
    res = run_fsm_for_ticker(df)
    assert res["state"] == "OFF"
    
    # Even if we force values that look like signals, NaNs should propagate or guards should fail
    # e.g. bias60 will be NaN. OversoldGate should return False.
