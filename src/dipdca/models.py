"""Core Pydantic data models for dip-or-dca."""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetConfig(BaseModel):
    """Configuration for a single asset (index + ETF pair)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    display_name: str
    index_symbol: str | None = None
    etf_symbol: str
    quote_currency: str
    category: str = "equity"
    is_hedged: bool = False
    data_provider: str = "yahoo"
    fallback_symbols: list[str] = Field(default_factory=list)
    notes: str = ""


class CountryProfile(BaseModel):
    """Country-specific profile for simulation defaults."""

    code: str
    display_name: str
    base_currency: str
    timezone: str
    cash_rate_source: str
    tax_rate_default: float = 0.0
    notes: str = ""


class SimulationParams(BaseModel):
    """Parameters for a single backtest simulation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    monthly_contribution: float = Field(gt=0, description="EUR/CHF contributed each month")
    payday: int = Field(ge=1, le=28, description="Day of month salary arrives")
    initial_investment: float = Field(ge=0, default=0.0)
    country: str = "NL"
    base_currency: str = "EUR"
    start_date: date = date(2010, 1, 1)
    end_date: date = date(2024, 12, 31)
    dip_threshold: float = Field(
        le=0.0, default=-0.05, description="Drawdown threshold to trigger buy (negative)"
    )
    max_wait_months: int = Field(ge=1, default=24, description="Max months to hold cash before force-deploy")
    fixed_fee: float = Field(ge=0, default=0.0, description="Fixed transaction cost per trade")
    pct_fee: float = Field(ge=0, default=0.001, description="Percentage transaction cost per trade")
    slippage: float = Field(ge=0, default=0.001, description="Slippage as fraction of price")
    tax_rate: float = Field(ge=0, lt=1, default=0.0, description="Capital gains tax rate")
    cash_rate_override: float | None = Field(
        default=None, description="Override annual cash interest rate (e.g. 0.04 = 4%)"
    )
    deployment_pct: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Fraction of deployable cash to invest when threshold triggers"
    )
    cash_buffer_months: int = Field(
        ge=0, default=0, description="Months of contributions kept as emergency buffer (never invested)"
    )
    deploy_spread_months: int = Field(
        ge=1, default=1, description="1 = lump sum, >1 = spread deployment over N months"
    )

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class StrategyResult(BaseModel):
    """Summary metrics from a backtest run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_name: str
    ending_wealth: float
    total_contributions: float
    ending_cash: float
    ending_market_value: float
    pnl: float
    xirr: float | None = None
    cagr: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float
    time_in_market_pct: float
    n_deployments: int
    total_fees: float
    total_cash_interest: float
    demo_mode: bool = False
    as_of: date | None = None


class PriceData(BaseModel):
    """Validated price data container."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    is_total_return: bool
    source: str
    symbol: str
    as_of: date
    demo_mode: bool = False

    @field_validator("df")
    @classmethod
    def validate_columns(cls, v: pd.DataFrame) -> pd.DataFrame:
        required = {"adj_close"}
        missing = required - set(v.columns)
        if missing:
            raise ValueError(f"PriceData missing columns: {missing}")
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("PriceData.df must have DatetimeIndex")
        return v


class CashFlowRecord(BaseModel):
    """Single cash flow event for XIRR calculation."""

    dt: date
    amount: float  # negative = outflow (investment), positive = inflow (receipt)
    description: str = ""


class DataHealthStatus(BaseModel):
    """Data freshness and quality status for a symbol."""

    symbol: str
    last_date: date | None = None
    n_rows: int = 0
    has_adj_close: bool = False
    is_stale: bool = False
    stale_threshold_days: int = 5
    source: str = "unknown"
    error: str | None = None

    @property
    def age_days(self) -> int | None:
        if self.last_date is None:
            return None
        return (date.today() - self.last_date).days
