# Dip-or-DCA — Claude Code instructions

## Drawdown benchmark and all-time high

The drawdown signal must be calculated from the selected reference index that the
investment product follows.

Examples:

- a Nasdaq-100 ETF uses the Nasdaq-100 Index as its drawdown benchmark;
- an S&P 500 ETF uses the S&P 500 Index;
- an MSCI World ETF uses the selected MSCI World benchmark series.

Do not calculate the signal from:

- the ETF's own market price;
- the investor's portfolio value;
- the savings balance;
- the base-currency-converted portfolio;
- the strategy's previous purchase prices.

Model the benchmark separately from the investable instrument:

```python
@dataclass(frozen=True)
class MarketDefinition:
    benchmark_symbol: str
    benchmark_name: str
    benchmark_currency: str

    instrument_symbol: str
    instrument_name: str
    instrument_currency: str

    base_currency: str
```

The benchmark index determines:

- the running all-time high;
- the current drawdown;
- threshold crossings;
- drawdown episode state;
- signal dates.

The investable instrument determines:

- execution prices;
- units purchased;
- transaction fees;
- tracking difference;
- final investment value.

### ATH definition

For daily data, define the benchmark ATH as the highest completed benchmark closing
level observed up to and including date t:

```python
benchmark_ath_t = benchmark_close.loc[:t].max()
```

The benchmark drawdown is:

```python
benchmark_drawdown_t = (
    benchmark_close_t / benchmark_ath_t - 1.0
)
```

Use closing levels consistently.

Do not mix:

- an intraday index high with a closing index level;
- an ETF ATH with an index close;
- a price-index ATH with a total-return-index current value.

The selected benchmark series must remain consistent throughout the simulation.

### Price index versus total-return index

Make the benchmark-series type explicit:

```python
benchmark_return_type: Literal[
    "price_index",
    "net_total_return",
    "gross_total_return",
]
```

The default signal benchmark should be the standard headline price index unless the
user explicitly chooses a total-return index.

Display the selected convention clearly, for example:

```
Drawdown benchmark:
Nasdaq-100 Price Index

ATH convention:
Highest historical daily close
```

Do not silently switch between price and total-return versions because they have
different ATHs and drawdowns.

### Required historical context

The running ATH must use benchmark history from before the strategy evaluation
start date.

Example:

```
Benchmark history begins: January 1, 2000
Strategy evaluation begins: January 1, 2010
```

The drawdown on January 1, 2010 must still be measured against the highest benchmark
close reached before that date.

Do not reset the benchmark ATH at the beginning of each backtest or rolling
evaluation window.

Separate:

- `benchmark_history_start`
- `simulation_start`
- `simulation_end`

If the data does not cover the complete history of the index, do not label the result
an absolute all-time high.

Use wording such as:

```
Highest observed benchmark close since data begins
```

and show the benchmark data start date.

### Signal and execution separation

The index creates the signal, but the ETF or other investable vehicle executes the
purchase.

Required sequence:

1. Observe the completed reference-index close at date t.
2. Update the reference-index ATH.
3. Calculate the reference-index drawdown.
4. Detect any threshold crossing.
5. Schedule an investment order.
6. Execute using the investable instrument at the next available legal execution point.

For example:

```python
signal_date = t
signal_drawdown = benchmark_drawdown_t

execution_date = next_common_execution_date_after(t)
execution_price = instrument_price.loc[execution_date]
```

Never execute at the index level itself because an index is not directly tradable.

Never use the ETF's next-day movement to determine whether the index signal was valid.

Once the reference index generates the signal, execute the pending order at the next
legal instrument price even if the index rebounds before execution.

### Calendar alignment

The benchmark index and investable instrument may have different missing dates or
exchange calendars.

A valid execution date must:

- occur strictly after the signal date;
- have a valid investable-instrument price;
- have a valid FX rate when currency conversion is required.

Use last-known-prior data only for FX or other explicitly permitted auxiliary series.

Do not forward-fill an ETF execution price across a non-trading day and treat it as
a trade.

### Currency treatment

The benchmark drawdown should be calculated in the benchmark's native published
index level.

Do not convert the benchmark index to EUR before calculating its ATH or drawdown
unless the product explicitly tracks a currency-adjusted or currency-hedged index.

Currency conversion belongs to the investment execution and valuation layer:

```python
instrument_price_base = (
    instrument_price_native
    * native_to_base_fx_multiplier
)
```

Thus:

- the native reference index determines whether a dip occurred;
- the investable instrument and FX rate determine how many units the investor buys.

For a currency-hedged product, use the exact hedged benchmark it contractually
follows when that series is available.

### Required ledger fields

Add:

- `benchmark_name`
- `benchmark_close`
- `benchmark_ath`
- `benchmark_drawdown`
- `benchmark_series_type`
- `signal_date`
- `signal_threshold`
- `instrument_symbol`
- `instrument_native_price`
- `instrument_base_currency_price`
- `execution_date`

This must make it possible to audit:

- which index generated the signal;
- what its ATH was;
- what drawdown triggered the order;
- which instrument was purchased;
- at what price and date it was purchased.
