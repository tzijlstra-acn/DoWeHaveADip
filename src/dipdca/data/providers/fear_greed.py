"""CNN Fear & Greed Index provider with VIX-based proxy fallback."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import requests

FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


@dataclass
class FearGreedReading:
    value: float  # 0-100
    label: str  # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    as_of: datetime.date
    is_live: bool


def fetch_current_fear_greed() -> FearGreedReading | None:
    """
    Attempt to fetch live CNN F&G index.
    Falls back to None on any error (caller handles fallback).
    """
    try:
        resp = requests.get(
            FEAR_GREED_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        score = float(data["fear_and_greed"]["score"])
        label = data["fear_and_greed"]["rating"]
        return FearGreedReading(
            value=score,
            label=label.title(),
            as_of=datetime.date.today(),
            is_live=True,
        )
    except Exception:
        return None


def score_to_label(score: float) -> str:
    if score <= 25:
        return "Extreme Fear"
    elif score <= 45:
        return "Fear"
    elif score <= 55:
        return "Neutral"
    elif score <= 75:
        return "Greed"
    else:
        return "Extreme Greed"


def score_to_color(score: float) -> str:
    if score <= 25:
        return "#FF4B6B"
    elif score <= 45:
        return "#F47920"
    elif score <= 55:
        return "#9CA3AF"
    elif score <= 75:
        return "#00C896"
    else:
        return "#FFD700"


def score_to_deployment_multiplier(score: float) -> float:
    """
    Convert F&G score to a deployment conviction multiplier.
    Extreme fear -> deploy more aggressively.
    Extreme greed -> deploy more cautiously.

    Returns multiplier to apply to deployment_pct:
    - score <= 15 (Extreme Fear): 2.0x (deploy double your normal fraction)
    - score <= 25 (Fear): 1.5x
    - score 25-55 (Fear to Neutral): 1.0x (no change)
    - score 55-75 (Greed): 0.75x
    - score > 75 (Extreme Greed): 0.5x (deploy half your normal fraction)
    """
    if score <= 15:
        return 2.0
    elif score <= 25:
        return 1.5
    elif score <= 55:
        return 1.0
    elif score <= 75:
        return 0.75
    else:
        return 0.5
