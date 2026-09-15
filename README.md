DoWeHaveADip is a live-data investment simulator that compares buying the dip vs dollar-cost averaging using real market prices.

_(screenshot coming soon)_

## Usage

1. Visit the app at [deploy URL here]
2. Pick your market or ETF and enter your monthly contribution
3. See the historical analysis — no registration, no install required

## Data policy

- All prices come from live market sources (Yahoo Finance via yfinance)
- No synthetic demo data, CSV files, or offline fallbacks are served
- Data includes freshness timestamps on every page
- If data is temporarily unavailable, the app shows an error with a Retry button rather than stale data
- The only market data provider is Yahoo Finance; no automatic failover between providers
- Data labeled "delayed" or "closed market" reflects the actual market status, never "real-time" unless truly fresh

## What this is / is not

**This is:**
- An educational tool to explore historical investment strategies
- A backtester using real historical adjusted prices

**This is not:**
- Investment advice, a trading system, or a prediction tool
- A source of real-time data for active trading decisions

## Pages

| Page | Description |
|------|-------------|
| Market Arcade | Browse all available assets and their current drawdown status |
| Dip-O-Meter | Assess whether today's price level is historically a dip |
| Strategy Lab | Head-to-head backtest comparing DCA, Wait-for-Dip, and Dip Buffet strategies |
| Historical Scenario Lab | Parameter sweep and historical scenario simulation across dip thresholds |
| Exit Lab | Model and compare exit strategies on top of a completed accumulation backtest |
| Currency Reality Check | Decompose how FX movements affected your returns on foreign-denominated assets |
| Interest Rates | Official ECB and SNB policy rates — not retail savings rates — as an opportunity-cost baseline |
| Methodology and Data Health | Formulas, assumptions, and live data freshness report |

## Developer setup

```bash
git clone git@github.com:tzijlstra-acn/DoWeHaveADip.git
cd DoWeHaveADip
uv sync --extra dev
uv run streamlit run app.py
```

Run tests:

```bash
uv run pytest -q
```

Lint and typecheck:

```bash
uv run ruff check . && uv run mypy src/dipdca
```

## Deployment

Deployed on Streamlit Community Cloud.

| Setting | Value |
|---------|-------|
| Repository | `tzijlstra-acn/DoWeHaveADip` |
| Branch | `master` |
| Entry point | `app.py` |
| Python version | 3.12 (pinned via `uv.lock`) |

No API keys are required for the default Yahoo Finance provider.

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `MARKET_DATA_TIMEOUT_SECONDS` | Request timeout for live price fetches |
| `MARKET_QUOTE_TTL_SECONDS` | Cache lifetime for current-quote responses |
| `MARKET_HISTORY_TTL_SECONDS` | Cache lifetime for historical price responses |

## Disclaimer

Educational and informational purposes only. Not investment advice.
See [DISCLAIMER.md](DISCLAIMER.md) for full terms.
