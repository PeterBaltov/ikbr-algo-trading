"""
MoneyTrailz Strategy Framework

This package provides a flexible, extensible framework for implementing
various trading strategies including options, stocks, and hybrid approaches.
"""

from .base import BaseStrategy, StrategyResult, StrategyContext
from .enums import StrategySignal, StrategyType, TimeFrame, StrategyStatus
from .exceptions import StrategyError, StrategyConfigError, StrategyExecutionError
from .interfaces import IStrategyConfig, IMarketData, IIndicator
from .registry.registry import StrategyRegistry, get_registry, register_strategy

__all__ = [
    # Core classes
    "BaseStrategy",
    "StrategyResult",
    "StrategyContext",
    
    # Enums
    "StrategySignal",
    "StrategyType", 
    "TimeFrame",
    "StrategyStatus",
    
    # Exceptions
    "StrategyError",
    "StrategyConfigError",
    "StrategyExecutionError",
    
    # Interfaces
    "IStrategyConfig",
    "IMarketData", 
    "IIndicator",
    
    # Registry
    "StrategyRegistry",
    "get_registry",
    "register_strategy",
] 
