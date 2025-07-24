"""
Enums and types for the ThetaGang strategy framework.

This module defines the core enumerations used throughout the strategy system
including signal types, strategy categories, timeframes, and execution status.
"""

from enum import Enum, auto
from typing import Literal


class StrategySignal(Enum):
    """Signals that a strategy can emit."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    
    def __str__(self) -> str:
        return self.value


class StrategyType(Enum):
    """Types of trading strategies."""
    OPTIONS = "options"          # Options-only strategies (wheel, iron condor, etc.)
    STOCKS = "stocks"            # Stock-only strategies (momentum, trend following)
    MIXED = "mixed"              # Hybrid strategies (covered calls, protective puts)
    HEDGING = "hedging"          # Risk management strategies (VIX calls, etc.)
    CASH_MANAGEMENT = "cash"     # Cash optimization strategies
    
    def __str__(self) -> str:
        return self.value


class TimeFrame(Enum):
    """Time frames for strategy execution and data analysis."""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m" 
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    
    @property
    def seconds(self) -> int:
        """Return the timeframe in seconds."""
        mapping = {
            TimeFrame.TICK: 0,
            TimeFrame.SECOND_1: 1,
            TimeFrame.SECOND_5: 5,
            TimeFrame.SECOND_15: 15,
            TimeFrame.SECOND_30: 30,
            TimeFrame.MINUTE_1: 60,
            TimeFrame.MINUTE_5: 300,
            TimeFrame.MINUTE_15: 900,
            TimeFrame.MINUTE_30: 1800,
            TimeFrame.HOUR_1: 3600,
            TimeFrame.HOUR_4: 14400,
            TimeFrame.DAY_1: 86400,
            TimeFrame.WEEK_1: 604800,
            TimeFrame.MONTH_1: 2592000,  # Approximate
        }
        return mapping[self]
    
    def __str__(self) -> str:
        return self.value


class StrategyStatus(Enum):
    """Status of strategy execution."""
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()
    COMPLETED = auto()


class OrderSide(Enum):
    """Order side for trading."""
    BUY = "BUY"
    SELL = "SELL"
    
    def __str__(self) -> str:
        return self.value


class PositionSide(Enum):
    """Position side for tracking."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"
    
    def __str__(self) -> str:
        return self.value


# Type aliases for better type hints
StrategyName = str
SymbolName = str
ContractId = int
OrderId = int 
