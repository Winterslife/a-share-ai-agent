# A-Share Agent

A modular A-share trading agent system with rule-based analysis and market breadth monitoring.

## Features

- **Stock Analysis (v0.2.0)**: Rule-based analysis of trend, structure, heat, location, and flow.
- **Market Breadth (v0.3.0)**: Market regime classification (BULL/NEUTRAL/DEFENSIVE) based on index trend and up/down ratio.
- **Intent Router**: Natural language query routing to specific agents.
- **CLI Interface**: Easy-to-use command line interface for analysis and queries.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Market Analysis
```bash
python app/cli.py market --index sh000300
```

### Candidate Generation
```bash
python app/cli.py candidates --top 10 --diversify 1
```

### Stock Analysis
```bash
python app/cli.py analyze --ticker 000001 --position 0.5 --cost 10.5
```

### Natural Language Query
```bash
python app/cli.py demo_query --text "现在市场环境怎么样？"
python app/cli.py demo_query --text "帮我选股"
python app/cli.py demo_query --text "分析一下 600000，我持仓 0.3，成本 8.5"
```

## Project Structure

- `agents/`: Agent implementations (Stock Analyzer, Market Breadth, Intent Router).
- `app/`: CLI and application entry points.
- `core/`: Core models and logging utilities.
- `tools/`: Market data fetching and technical indicators.
- `runs/`: JSON artifacts of analysis runs.
