# Methodology

## Total Return Index

All price series are converted to a total return index using dividend-adjusted prices:

```
TR_base(t) = [AdjClose(t) * FX(t)] / [AdjClose(t0) * FX(t0)]
```

- `AdjClose`: Yahoo Finance `auto_adjust=True` (dividend-reinvested)
- `FX`: ECB reference rates (units of base currency per EUR)
- `t0`: Simulation start date

## No-Lookahead Guarantee

Signal computation uses only prior completed information:

1. **Signal**: Drawdown at `t-1` (yesterday's close)
2. **Execution**: Trade at `t` (today's close)
3. **Contributions**: Arrive on payday, invest on next trading day

## DCA Strategy

- Monthly contribution arrives on `payday`
- Deployed immediately on the next trading day
- No drawdown condition required
- Cash earns overnight rate between arrival and deployment (≤ 1 trading day)

## Wait-for-Dip Strategy

- Monthly contribution accumulates in cash
- Deployed when `drawdown(t-1) ≤ threshold`
- Force-deployed after `max_wait_months` regardless of drawdown
- Cash earns full overnight rate while waiting

## Tiered-Dip Strategy

- Multiple thresholds with partial deployment fractions
- Tiers reset only after a new all-time high is confirmed
- Prevents double-dipping within the same episode

## Cost Model

```
net_invested = gross_amount - fixed_fee - gross_amount * pct_fee - gross_amount * slippage
```

## Cash Interest

Day-by-day compounding using calendar days:
```
B(t) = B(t-1) * (1 + r_annual) ^ (days_elapsed / 365)
```

## XIRR

Newton-Raphson via `scipy.optimize.brentq`:
```
Σ CF_i / (1 + r)^(d_i / 365) = 0
```

Investments are negative; terminal wealth is a positive receipt.

## Sharpe & Sortino

Annualized using sqrt(252) trading days:
```
Sharpe = (mean(excess_returns) / std(excess_returns)) * sqrt(252)
Sortino = (mean(excess_returns) / std(downside_returns)) * sqrt(252)
```
