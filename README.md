# Dip, DCA & Chill

> Monthly Machine versus Cash Goblin — settled by historical data, not vibes.

A humorous but mathematically rigorous ETF timing dashboard comparing Dollar-Cost Averaging (DCA) vs waiting-for-the-dip strategies across 10 major indices.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Generate demo fixtures (synthetic data, no real market data)
uv run python data/fixtures/generate.py

# Run the app
uv run streamlit run app.py
```

## Features

- **Strategy Lab**: Head-to-head DCA vs Wait-for-Dip backtest with full metrics
- **Market Arcade**: Browse 10 assets with real-time drawdown status
- **Drawdown labelling**: From "Barely a dip" to "Financial Mariana Trench"
- **No-lookahead bias**: Signal uses prior close, trade at next close
- **Offline demo mode**: Synthetic fixtures if live data unavailable
- **Reproducible results**: Fixed seeds, deterministic logic

## Asset Universe

Nasdaq-100, Nasdaq Composite, S&P 500, STOXX 600, DAX, AEX, SMI,
MSCI World (IWDA), FTSE All-World (VWCE), Emerging Markets (IS3N)

## Tech Stack

Python 3.12 · Streamlit · Plotly · pandas · numpy · scipy · yfinance · DuckDB · Pydantic

## Development

```bash
make install      # Install with dev extras
make fixtures     # Generate synthetic demo data
make test         # Run pytest
make lint         # Ruff check
make run          # Start Streamlit
make docker-build # Build Docker image
```

## Disclaimer

Educational and informational purposes only. Not investment advice.
See [DISCLAIMER.md](DISCLAIMER.md) for full terms.

## License

MIT — see [LICENSE](LICENSE)
