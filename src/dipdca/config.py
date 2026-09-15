"""Configuration loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
CACHE_DIR = DATA_DIR / "cache"


@lru_cache(maxsize=1)
def load_assets_config() -> list[dict]:
    """Load asset definitions from assets.yaml."""
    path = CONFIG_DIR / "assets.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["assets"]


@lru_cache(maxsize=1)
def load_countries_config() -> list[dict]:
    """Load country profiles from countries.yaml."""
    path = CONFIG_DIR / "countries.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["countries"]


def get_asset_by_id(asset_id: str) -> dict | None:
    """Return asset config dict by id, or None if not found."""
    for asset in load_assets_config():
        if asset["id"] == asset_id:
            return asset
    return None


def get_country_by_code(code: str) -> dict | None:
    """Return country profile dict by code, or None if not found."""
    for country in load_countries_config():
        if country["code"] == code:
            return country
    return None
