"""
Base strategy class for the MoneyTrailz strategy framework.

This module provides the abstract base class that all strategies must inherit from,
defining the standard interface and common functionality for strategy execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from ib_async import Contract, Order

from .enums import StrategySignal, StrategyStatus, StrategyType, TimeFrame
from .exceptions import StrategyConfigError, StrategyExecutionError, StrategyValidationError
from .interfaces import IMarketData, IOrderManager, IPositionManager, IRiskManager


@dataclass
class StrategyResult:
    """Result of strategy analysis/execution."""
    
    strategy_name: str
    symbol: str
    signal: StrategySignal
    confidence: float = 0.0
    price: Optional[float] = None
    quantity: Optional[int] = None
    contracts: Optional[List[Contract]] = field(default_factory=list)
    orders: Optional[List[Order]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class StrategyContext:
    """Context information passed to strategies during execution."""
    
    market_data: IMarketData
    order_manager: IOrderManager
    position_manager: IPositionManager
    risk_manager: IRiskManager
    account_summary: Dict[str, Any]
    portfolio_positions: Dict[str, List[Any]]
    current_time: datetime = field(default_factory=datetime.now)
    dry_run: bool = False


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement,
    providing common functionality and ensuring consistent behavior across
    different strategy implementations.
    """
    
    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        config: Dict[str, Any],
        symbols: List[str],
        timeframes: List[TimeFrame]
    ) -> None:
        """
        Initialize the strategy.
        
        Args:
            name: Unique name for this strategy instance
            strategy_type: Type of strategy (options, stocks, etc.)
            config: Strategy-specific configuration
            symbols: List of symbols this strategy trades
            timeframes: List of timeframes this strategy uses
        """
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
        self.symbols = set(symbols)
        self.timeframes = set(timeframes)
        self.status = StrategyStatus.INITIALIZED
        self.last_execution: Optional[datetime] = None
        self.execution_count = 0
        self.errors: List[Exception] = []
        
        # Validate configuration on initialization
        try:
            self.validate_config()
        except Exception as e:
            raise StrategyConfigError(
                f"Strategy configuration validation failed: {e}",
                strategy_name=self.name
            ) from e
    
    @abstractmethod
    async def analyze(
        self,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        context: StrategyContext
    ) -> StrategyResult:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Symbol to analyze
            data: Market data for different timeframes
            context: Execution context with market interfaces
            
        Returns:
            StrategyResult containing signal and execution details
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate strategy configuration.
        
        Raises:
            StrategyConfigError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """
        Get the timeframes required by this strategy.
        
        Returns:
            Set of required timeframes
        """
        pass
    
    @abstractmethod
    def get_required_symbols(self) -> Set[str]:
        """
        Get the symbols required by this strategy.
        
        Returns:
            Set of required symbols
        """
        pass
    
    def get_required_data_fields(self) -> Set[str]:
        """
        Get the data fields required by this strategy.
        
        Returns:
            Set of required field names (e.g., 'open', 'high', 'low', 'close', 'volume')
        """
        return {"open", "high", "low", "close", "volume"}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this strategy.
        
        Returns:
            Dictionary containing strategy metadata
        """
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "status": self.status.name,
            "symbols": list(self.symbols),
            "timeframes": [tf.value for tf in self.timeframes],
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "execution_count": self.execution_count,
            "error_count": len(self.errors),
            "config": self.config.copy()
        }
    
    async def execute(
        self,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        context: StrategyContext
    ) -> Optional[StrategyResult]:
        """
        Execute the strategy for a given symbol.
        
        This is the main entry point for strategy execution. It handles
        common tasks like status management, error handling, and logging.
        
        Args:
            symbol: Symbol to execute strategy for
            data: Market data for different timeframes
            context: Execution context
            
        Returns:
            StrategyResult if execution successful, None otherwise
        """
        if self.status not in (StrategyStatus.INITIALIZED, StrategyStatus.RUNNING):
            return None
        
        if symbol not in self.symbols:
            return None
        
        self.status = StrategyStatus.RUNNING
        
        try:
            # Validate we have all required data
            self._validate_data_availability(data)
            
            # Execute strategy analysis
            result = await self.analyze(symbol, data, context)
            
            # Update execution tracking
            self.last_execution = datetime.now()
            self.execution_count += 1
            
            return result
            
        except Exception as e:
            self.status = StrategyStatus.ERROR
            self.errors.append(e)
            
            raise StrategyExecutionError(
                f"Strategy execution failed: {e}",
                strategy_name=self.name,
                execution_phase="analyze",
                original_exception=e
            ) from e
        
        finally:
            if self.status == StrategyStatus.RUNNING:
                self.status = StrategyStatus.INITIALIZED
    
    def _validate_data_availability(self, data: Dict[TimeFrame, pd.DataFrame]) -> None:
        """
        Validate that all required data is available.
        
        Args:
            data: Market data for different timeframes
            
        Raises:
            StrategyValidationError: If required data is missing
        """
        required_timeframes = self.get_required_timeframes()
        missing_timeframes = required_timeframes - set(data.keys())
        
        if missing_timeframes:
            raise StrategyValidationError(
                f"Missing required timeframes: {[tf.value for tf in missing_timeframes]}",
                strategy_name=self.name
            )
        
        required_fields = self.get_required_data_fields()
        
        for timeframe, df in data.items():
            if timeframe in required_timeframes:
                missing_fields = required_fields - set(df.columns)
                if missing_fields:
                    raise StrategyValidationError(
                        f"Missing required fields for {timeframe.value}: {list(missing_fields)}",
                        strategy_name=self.name
                    )
                
                if df.empty:
                    raise StrategyValidationError(
                        f"No data available for timeframe {timeframe.value}",
                        strategy_name=self.name
                    )
    
    def reset_errors(self) -> None:
        """Reset error tracking."""
        self.errors.clear()
        if self.status == StrategyStatus.ERROR:
            self.status = StrategyStatus.INITIALIZED
    
    def pause(self) -> None:
        """Pause strategy execution."""
        self.status = StrategyStatus.PAUSED
    
    def resume(self) -> None:
        """Resume strategy execution."""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.INITIALIZED
    
    def stop(self) -> None:
        """Stop strategy execution."""
        self.status = StrategyStatus.STOPPED
    
    def is_active(self) -> bool:
        """Check if strategy is in an active state."""
        return self.status in (StrategyStatus.INITIALIZED, StrategyStatus.RUNNING)
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', type={self.strategy_type.value}, status={self.status.name})" 
