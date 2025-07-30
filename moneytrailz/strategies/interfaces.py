"""
Protocol interfaces for the ThetaGang strategy framework.

This module defines the interfaces (protocols) that components must implement
to work with the strategy framework. These interfaces ensure type safety and
provide clear contracts for strategy components.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd
from ib_async import Contract, Order

from .enums import StrategySignal, StrategyType, TimeFrame


@runtime_checkable
class IStrategyConfig(Protocol):
    """Interface for strategy configuration objects."""
    
    @property
    def strategy_name(self) -> str:
        """Name of the strategy."""
        ...
    
    @property
    def strategy_type(self) -> StrategyType:
        """Type of strategy (options, stocks, mixed, etc.)."""
        ...
    
    @property
    def enabled(self) -> bool:
        """Whether the strategy is enabled."""
        ...
    
    @property
    def timeframes(self) -> List[TimeFrame]:
        """Timeframes used by this strategy."""
        ...
    
    @property
    def symbols(self) -> List[str]:
        """Symbols traded by this strategy."""
        ...
    
    def validate(self) -> bool:
        """Validate the configuration."""
        ...


@runtime_checkable
class IMarketData(Protocol):
    """Interface for market data providers."""
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get historical market data."""
        ...
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        ...
    
    def get_option_chain(self, symbol: str) -> List[Contract]:
        """Get option chain for a symbol."""
        ...
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        ...


@runtime_checkable
class IIndicator(Protocol):
    """Interface for technical indicators."""
    
    @property
    def name(self) -> str:
        """Name of the indicator."""
        ...
    
    @property
    def required_periods(self) -> int:
        """Minimum number of periods needed for calculation."""
        ...
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values."""
        ...
    
    def is_ready(self, data: pd.DataFrame) -> bool:
        """Check if enough data is available for calculation."""
        ...


@runtime_checkable
class IOrderManager(Protocol):
    """Interface for order management."""
    
    def place_order(self, contract: Contract, order: Order) -> str:
        """Place an order and return order ID."""
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        ...
    
    def get_order_status(self, order_id: str) -> str:
        """Get status of an order."""
        ...
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get list of open orders."""
        ...


@runtime_checkable
class IPositionManager(Protocol):
    """Interface for position management."""
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions."""
        ...
    
    def get_buying_power(self) -> float:
        """Get available buying power."""
        ...
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_percentage: float
    ) -> int:
        """Calculate appropriate position size."""
        ...


@runtime_checkable
class IRiskManager(Protocol):
    """Interface for risk management."""
    
    def check_position_risk(
        self,
        symbol: str,
        quantity: int,
        price: float
    ) -> bool:
        """Check if position meets risk criteria."""
        ...
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics."""
        ...
    
    def should_exit_position(
        self,
        symbol: str,
        current_price: float,
        entry_price: float
    ) -> bool:
        """Determine if position should be exited."""
        ...


class IStrategyRegistry(Protocol):
    """Interface for strategy registry."""
    
    def register_strategy(self, strategy_class: type, name: str) -> None:
        """Register a strategy class."""
        ...
    
    def get_strategy(self, name: str) -> Optional[type]:
        """Get a strategy class by name."""
        ...
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        ...
    
    def is_registered(self, name: str) -> bool:
        """Check if strategy is registered."""
        ... 
