from __future__ import annotations

import argparse
import json
import sys
import re
from core.logging import new_run_id, save_run_json
from agents.stock_analyzer import analyze_stock
from agents.intent_router import route_intent, extract_ticker
from core.models import PortfolioInput

def handle_analyze(args):
    ticker = args.ticker
    freq = args.freq
    lookback = args.lookback
    run_id = new_run_id("analyze")

    # Optional portfolio input
    portfolio = None
    if args.position is not None or args.cost is not None or args.holding_days is not None:
        portfolio = PortfolioInput(
            ticker=ticker,
            position=args.position if args.position is not None else 0.0,
            cost_price=args.cost,
            holding_days=args.holding_days,
            style=args.style
        )

    try:
        # Use the stock analyzer agent
        result = analyze_stock(
            ticker=ticker,
            freq=freq,
            lookback=lookback,
            portfolio=portfolio
        )
        
        # Add run_id to meta
        result["meta"]["run_id"] = run_id
        
        # Print payload as pretty JSON
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Save payload
        save_run_json(run_id, result)
        
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

def handle_demo_query(args):
    text = args.text
    print(f"Query: {text}")
    
    intent = route_intent(text)
    ticker = extract_ticker(text)
    
    print(f"Detected Intent: {intent}")
    print(f"Detected Ticker: {ticker}")
    
    if intent == "stock_analysis" and ticker:
        print(f"Routing to stock analysis for {ticker}...")
        
        class MockArgs:
            def __init__(self, ticker):
                self.ticker = ticker
                self.freq = "1d"
                self.lookback = 120
                self.position = None
                self.cost = None
                self.holding_days = None
                self.style = "neutral"
        
        # Simple extraction for position and cost from text
        pos_match = re.search(r'仓位\s*(\d+(\.\d+)?)', text)
        cost_match = re.search(r'成本\s*(\d+(\.\d+)?)', text)
        
        m_args = MockArgs(ticker)
        if pos_match:
            m_args.position = float(pos_match.group(1))
        if cost_match:
            m_args.cost = float(cost_match.group(1))
            
        handle_analyze(m_args)
    else:
        print("Intent not supported for demo or ticker not found.")

def main():
    parser = argparse.ArgumentParser(description="A-Share Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a ticker")
    analyze_parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., 000001)")
    analyze_parser.add_argument("--freq", default="1d", help="Frequency (default: 1d)")
    analyze_parser.add_argument("--lookback", type=int, default=120, help="Lookback period (default: 120)")
    
    # Portfolio arguments
    analyze_parser.add_argument("--position", type=float, help="Current position (0.0-1.0)")
    analyze_parser.add_argument("--cost", type=float, help="Average cost price")
    analyze_parser.add_argument("--style", choices=["aggressive", "neutral"], default="neutral", help="Trading style")
    analyze_parser.add_argument("--holding-days", type=int, help="Number of days holding the stock")

    # Demo Query command
    demo_parser = subparsers.add_parser("demo_query", help="Demo natural language query")
    demo_parser.add_argument("--text", required=True, help="User query text")
    
    args = parser.parse_args()

    if args.command == "analyze":
        handle_analyze(args)
    elif args.command == "demo_query":
        handle_demo_query(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
