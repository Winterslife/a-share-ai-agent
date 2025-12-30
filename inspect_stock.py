import argparse
import pandas as pd
from tools.market_data import get_bars
from tools.indicators import calc_strategy_metrics

def main():
    parser = argparse.ArgumentParser(description="Inspect stock data for strategy validation")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g., 000048)")
    parser.add_argument("--lookback", type=int, default=30, help="Rows to show")
    args = parser.parse_args()

    print(f"Fetching data for {args.ticker}...")
    try:
        # 1. Get Bars (fetching more history to ensure MAs are calculated)
        bars = get_bars(args.ticker, lookback=200)
        
        # 2. Calculate Strategy Metrics
        df = calc_strategy_metrics(bars)
        
        # 3. Select requested columns
        cols = [
            "ts", "close", 
            "llv_close_10", "llv_close_20", 
            "ma20", "ma20_slope",
            "amount", "amt_ma5", "amt_ma20", "amt_vol_ratio",
            "volume"
        ]
        
        # Filter columns that exist
        display_cols = [c for c in cols if c in df.columns]
        
        # 4. Show last N rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        
        print(df[display_cols].tail(args.lookback))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
