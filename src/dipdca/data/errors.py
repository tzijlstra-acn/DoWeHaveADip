"""Domain exceptions for live market data failures."""

from __future__ import annotations


class LiveDataUnavailable(Exception):
    """Raised when a live market data source cannot be reached or returns unusable data."""


class RateLimited(LiveDataUnavailable):
    """Raised when the provider signals a rate limit (HTTP 429 or equivalent)."""


class InvalidProviderResponse(LiveDataUnavailable):
    """Raised when provider data fails validation (missing columns, empty, bad format)."""


class StaleLiveData(LiveDataUnavailable):
    """Raised when the most recent observation is too old to be considered fresh."""
