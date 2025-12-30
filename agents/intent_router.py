from __future__ import annotations

import re

def route_intent(user_query: str) -> str:
    """
    Simple rule-based intent router for user queries.
    """
    candidate_keywords = ["选股", "机会", "候选", "池", "扫描"]
    alert_keywords = ["盯盘", "提醒", "预警", "报警"]
    analysis_keywords = ["分析", "怎么看", "策略", "止损", "止盈", "持仓"]

    if any(kw in user_query for kw in candidate_keywords):
        return "candidate_pool"
    
    if any(kw in user_query for kw in alert_keywords):
        return "alerts"
    
    if any(kw in user_query for kw in analysis_keywords):
        return "stock_analysis"
    
    return "unknown"

def extract_ticker(text: str) -> str | None:
    """
    Extract 6-digit A-share code if present.
    """
    # Use lookbehind and lookahead to ensure exactly 6 digits, 
    # avoiding \b which fails between Chinese characters and digits.
    match = re.search(r'(?<!\d)\d{6}(?!\d)', text)
    return match.group(0) if match else None
