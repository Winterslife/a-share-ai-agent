from __future__ import annotations

import argparse
import json
import sys
import re
from core.logging import new_run_id, save_run_json
from agents.stock_analyzer import analyze_stock
from agents.intent_router import route_intent, extract_ticker
from agents.market_breadth import get_market_regime
from agents.candidate_pool import generate_candidate_pool
from core.models import PortfolioInput

def handle_candidates(args):
    top = args.top
    lookback = args.lookback
    diversify = bool(args.diversify)
    mode = args.mode
    run_id = new_run_id("candidates")

    try:
        result = generate_candidate_pool(
            top_n=top,
            lookback=lookback,
            enforce_diversification=diversify,
            mode=mode
        )
        result["meta"] = {"run_id": run_id, "type": "candidate_pool"}
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        save_run_json(run_id, result)
    except Exception as e:
        print(f"Error during candidate generation: {e}", file=sys.stderr)
        sys.exit(1)

def handle_market(args):
    index_ticker = args.index
    lookback = args.lookback
    run_id = new_run_id("market")

    try:
        result = get_market_regime(index_ticker=index_ticker, lookback=lookback)
        result["meta"] = {"run_id": run_id, "type": "market_breadth"}
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        save_run_json(run_id, result)
    except Exception as e:
        print(f"Error during market analysis: {e}", file=sys.stderr)
        sys.exit(1)

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
    
    if intent == "market_breadth":
        print("Routing to market breadth analysis...")
        class MockArgs:
            def __init__(self):
                self.index = "sh000300"
                self.lookback = 60
        handle_market(MockArgs())
    elif intent == "candidate_pool":
        print("Routing to candidate pool generation...")
        class MockArgs:
            def __init__(self):
                self.top = None
                self.lookback = 120
                self.diversify = 0
                self.mode = "simple" # Default to simple for demo query
        handle_candidates(MockArgs())
    elif intent == "stock_analysis" and ticker:
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

    # Market command
    market_parser = subparsers.add_parser("market", help="Analyze market breadth")
    market_parser.add_argument("--index", default="sh000300", help="Index ticker (default: sh000300)")
    market_parser.add_argument("--lookback", type=int, default=60, help="Lookback period (default: 60)")

    # Candidates command
    candidates_parser = subparsers.add_parser("candidates", help="Generate candidate pool")
    candidates_parser.add_argument("--top", type=int, help="Number of top candidates (optional)")
    candidates_parser.add_argument("--lookback", type=int, default=120, help="Lookback period (default: 120)")
    candidates_parser.add_argument("--diversify", type=int, choices=[0, 1], default=0, help="Enforce diversification (0 or 1)")
    candidates_parser.add_argument("--mode", default="simple", choices=["simple", "fsm", "reversal"], help="Generation mode (simple/fsm/reversal)")

    # Demo Query command
    demo_parser = subparsers.add_parser("demo_query", help="Demo natural language query")
    demo_parser.add_argument("--text", required=True, help="User query text")
    
    args = parser.parse_args()

    if args.command == "analyze":
        handle_analyze(args)
    elif args.command == "market":
        handle_market(args)
    elif args.command == "candidates":
        handle_candidates(args)
    elif args.command == "demo_query":
        handle_demo_query(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
