"""Typed result objects for live market data responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class DataFreshness:
    """Metadata about the provenance and freshness of a price dataset."""

    provider: str
    symbol: str
    observed_at: datetime
    retrieved_at: datetime
    status: Literal["fresh", "delayed", "closed_market"]
    label: str

    def caption(self) -> str:
        """Short human-readable freshness label for display."""
        observed = self.observed_at.strftime("%Y-%m-%d %H:%M UTC")
        retrieved = self.retrieved_at.strftime("%Y-%m-%d %H:%M UTC")
        return f"Source: {self.provider} · as of {observed} · fetched {retrieved} · {self.label}"


@dataclass
class PriceDataResult:
    """A validated price DataFrame paired with its freshness metadata.

    The frame has:
    - A sorted, unique DatetimeIndex
    - Lowercase columns: close, adj_close (required), open/high/low/volume (when available)
    """

    frame: pd.DataFrame
    freshness: DataFreshness
