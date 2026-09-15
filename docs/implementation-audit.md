# Implementation Audit — DoWeHaveADip

Baseline captured before `feat/live-friends-release` changes.

## Baseline Command Results

| Command | Result |
|---|---|
| `uv sync --extra dev` | 76 packages resolved; clean |
| `uv run ruff check .` | All checks passed |
| `uv run pytest -q --tb=short` | 119 passed, 0 failures, 38 warnings |
| `uv run mypy src/dipdca` | 29 errors — all `import-untyped` for `pandas`/`yfinance` (no logic errors) |
| `uv run streamlit run app.py` | Not tested at baseline (requires network + browser) |

---

## Dead Code Confirmed

| Item | File | Evidence |
|---|---|---|
| `DuckDB cache` | `src/dipdca/data/cache.py` | No provider or page calls `read_cached` or `write_cache` |
| `PriceData.demo_mode` | `src/dipdca/models.py` | Always `False`; no branch reads it to serve fixture data |
| `StrategyResult.demo_mode` | `src/dipdca/models.py` | Always `False`; unused |
| `DEMO_MODE` env var | `.env.example` | No `os.environ.get("DEMO_MODE")` call anywhere |
| `AssetConfig.fallback_symbols` | `src/dipdca/models.py` | Populated in YAML but no provider iterates fallbacks |
| `BaseProvider` ABC | `src/dipdca/data/providers/base.py` | `YahooProvider` does not extend it |
| `cash.py` `accrue_over_period` | `src/dipdca/quant/cash.py` | Not called from backtest engine (inline logic used instead) |
| `quant/total_return.py` | `src/dipdca/quant/total_return.py` | `normalize_total_return` not called from any page |
| `quant/bootstrap.py` | `src/dipdca/quant/bootstrap.py` | Not called from `monte_carlo.py` or any page |

---

## Production Fixture / Local-File Usage

**Result: No production page loads synthetic fixture data.**

- `data/fixtures/*.parquet` committed to repo and regenerated in CI, but no page reads them.
- `pages/8_Methodology_and_Data_Health.py` checks fixture file *existence* (metadata display only).
- Fixture generation step in CI is wasted — no test reads the parquet output.

---

## Confirmed Bugs

| # | Severity | File | Line | Category | Description |
|---|---|---|---|---|---|
| 1 | BLOCKER | `app.py` | 105 | Leap-day crash | `date(_end.year - 1, _end.month, _end.day)` raises `ValueError` on Feb 29 |
| 2 | BLOCKER | `pages/1_Market_Arcade.py` | 36 | Leap-day crash | `date(end.year - 3, end.month, end.day)` raises `ValueError` on Feb 29 |
| 3 | HIGH | `pages/2_Dip_O_Meter.py` | 260 | Stat correctness | `int(p_beat * n_obs)` truncates (not rounds); Wilson CI computed from wrong success count |
| 4 | HIGH | `pages/2_Dip_O_Meter.py` | 357–371 | Session state | Bootstrap results stored without fingerprint; stale results shown if params change without re-run |
| 5 | HIGH | `pages/4_Threshold_Lab.py` | 497–499 | Label mismatch | `p5_wealth`/`p95_wealth` (5th/95th pct) displayed as "P10 (Worst)"/"P90 (Best)" |
| 6 | HIGH | `pages/4_Threshold_Lab.py` | 684 | Session state | Deep-history sweep cache stored without fingerprint; stale if ticker changes |
| 7 | MEDIUM | `pages/6_Currency_Reality_Check.py` | 132, 134 | Deprecated API | `.fillna(method="bfill")` removed in pandas 3.0; use `.bfill()` |
| 8 | MEDIUM | `pages/6_Currency_Reality_Check.py` | 193 | f-string missing prefix | Third string in `st.info(...)` not an f-string; `{asset_currency}`/`{base_currency}` render as literal text |
| 9 | MEDIUM | `pages/6_Currency_Reality_Check.py` | 136–138 | FX assumption | Non-EUR base currencies fall through to 1:1 assumption (no EUR triangulation) |
| 10 | HIGH | `src/dipdca/quant/monte_carlo.py` | 309–316 | Stat correctness | DCA baseline applies full cumulative return to all contributions regardless of investment month (inflates DCA) |
| 11 | HIGH | `src/dipdca/quant/monte_carlo.py` | 286–289 | Reproducibility | `np.random.randint` uses unseeded global RNG; bootstrap results differ on each run |
| 12 | LOW | `pages/7_Savings_Rates.py` | 96 | Deprecated API | `freq="M"` deprecated alias in pandas 2.2+; use `"ME"` |

---

## CI / Infrastructure Issues

| Issue | Severity |
|---|---|
| CI targets `main`/`develop`; canonical branch is `master` | BLOCKER |
| `uv sync --extra dev` (not `--frozen`) — non-reproducible | HIGH |
| `uv.lock` not copied into Dockerfile | HIGH |
| `data/fixtures/*.parquet` copied into Docker image | MEDIUM |
| Fixture generation run at Docker build time (silently ignored on failure) | MEDIUM |
| No mypy step in CI | MEDIUM |
| No Streamlit smoke test in CI | MEDIUM |
| Pytest runs twice (steps 7 and 8) | LOW |
| Docker health check uses `curl` (not installed in slim image) | HIGH |
| `requests` (used in `fear_greed.py`) not declared in `pyproject.toml` | LOW |

---

## Architecture Gaps

| Gap | Impact |
|---|---|
| All pages import `YahooProvider` directly (no DI/factory) | Blocks provider swap, mocking in tests |
| FX conversion missing from backtest pipeline (USD ETF wealth labeled "EUR") | Misleading to users |
| `@st.cache_data` inside `deep_history.py` (library module) | Cannot import in tests without Streamlit session |
| No TTL caching on quote/history calls (except deep history) | Excessive API calls on every page interaction |
| Broad `except Exception: pass` in 11 production locations | Silent failures; bugs invisible |
