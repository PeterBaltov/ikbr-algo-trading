"""Utility functions for stock-based strategies.
"""

from __future__ import annotations

import pandas as pd


def rsi(series: pd.Series, length: int) -> pd.Series:
    """Compute the Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def t3(series: pd.Series, length: int, b: float = 0.7) -> pd.Series:
    """Apply T3 smoothing to a series."""
    e1 = series.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    e4 = e3.ewm(span=length, adjust=False).mean()
    e5 = e4.ewm(span=length, adjust=False).mean()
    e6 = e5.ewm(span=length, adjust=False).mean()
    c1 = -b ** 3
    c2 = 3 * b ** 2 + 3 * b ** 3
    c3 = -6 * b ** 2 - 3 * b - 3 * b ** 3
    c4 = 1 + 3 * b + b ** 3 + 3 * b ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def compute_bx_trender(
    close: pd.Series,
    fast_span: int = 5,
    slow_span: int = 20,
    rsi_length: int = 15,
    t3_length: int = 5,
) -> pd.Series:
    """Compute the BX Trender oscillator for a series of close prices."""
    ema_fast = close.ewm(span=fast_span, adjust=False).mean()
    ema_slow = close.ewm(span=slow_span, adjust=False).mean()
    diff = ema_fast - ema_slow
    short = rsi(diff, rsi_length) - 50
    long = rsi(close.ewm(span=slow_span, adjust=False).mean(), rsi_length) - 50
    _ = long  # unused but left for completeness
    short_smoothed = t3(short, t3_length)
    return short_smoothed


def bx_trender_signal(values: pd.Series) -> str | None:
    """Return BUY or SELL when the oscillator crosses zero."""
    if len(values) < 2 or values.iloc[-2] is None:
        return None
    prev = values.iloc[-2]
    curr = values.iloc[-1]
    if prev <= 0 and curr > 0:
        return "BUY"
    if prev >= 0 and curr < 0:
        return "SELL"
    return None
