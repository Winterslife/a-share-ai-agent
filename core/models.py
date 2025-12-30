from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

def now_ts_str() -> str:
    """Returns current local timestamp string in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class PortfolioInput(BaseModel):
    ticker: str
    position: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_price: Optional[float] = None
    holding_days: Optional[int] = None
    style: Literal["aggressive", "neutral"] = "neutral"

    model_config = ConfigDict(populate_by_name=True)

class Quote(BaseModel):
    ticker: str
    price: float
    chg_pct: float
    amount: Optional[float] = None
    turnover: Optional[float] = None
    vol_ratio: Optional[float] = None
    timestamp: str = Field(default_factory=now_ts_str)

class Bar(BaseModel):
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    amount: Optional[float] = None

class Indicators(BaseModel):
    ma5: Optional[float] = None
    ma20: Optional[float] = None
    rsi14: Optional[float] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_hist: Optional[float] = None
    boll_up: Optional[float] = None
    boll_mid: Optional[float] = None
    boll_low: Optional[float] = None
    atr14: Optional[float] = None
    vol_ma20: Optional[float] = None
    vol_ratio_5_20: Optional[float] = None  # mean(volume last 5) / mean(volume last 20)

class StockSnapshot(BaseModel):
    ticker: str
    trend: Literal["up", "range", "down"]
    flow: Literal["inflow", "neutral", "outflow"]
    heat: Literal["high", "mid", "low"]
    structure: Literal["healthy", "fragile", "broken"]
    event: Literal["pos", "neu", "neg"]
    location: Literal["low", "mid", "high"]
    features: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=now_ts_str)

class Candidate(BaseModel):
    ticker: str
    score_total: float
    reasons: List[str]
    sector: Optional[str] = None

class TradePlan(BaseModel):
    ticker: str
    participate: bool
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    max_position: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: List[Dict[str, Any]] = Field(default_factory=list)
    invalidation_rules: List[str] = Field(default_factory=list)
    intraday_triggers: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=now_ts_str)

    @model_validator(mode='after')
    def check_entry_zone(self) -> TradePlan:
        low = self.entry_zone_low
        high = self.entry_zone_high
        if low is not None and high is not None:
            if low > high:
                raise ValueError(f"entry_zone_low ({low}) must be <= entry_zone_high ({high})")
        return self
