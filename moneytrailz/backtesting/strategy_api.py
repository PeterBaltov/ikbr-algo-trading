"""
Backtesting Strategy API

Enhanced strategy base class for backtesting with comprehensive lifecycle hooks,
state management, and performance tracking. Extends the Phase 1 strategy framework
with backtesting-specific capabilities.

Features:
- Lifecycle hooks (on_start, on_data, on_end)
- State persistence and restoration
- Performance tracking per strategy
- Risk management integration
- Parameter validation and optimization support
- Backtesting context awareness

Integration:
- Extends Phase 1 BaseStrategy class
- Works with backtesting engine event system
- Supports Phase 2 technical analysis integration
- Compatible with Phase 3 multi-timeframe architecture
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from ..strategies import BaseStrategy, StrategyResult, StrategyContext
from ..strategies.enums import StrategySignal, StrategyType, TimeFrame, StrategyStatus


class StrategyLifecycle(Enum):
    """Strategy lifecycle states in backtesting"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class BacktestContext(StrategyContext):
    """Enhanced context for backtesting strategies"""
    
    # Backtesting specific fields
    is_backtesting: bool = True
    current_bar_index: int = 0
    total_bars: int = 0
    
    # Historical performance
    strategy_pnl: float = 0.0
    strategy_trades: int = 0
    win_rate: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Data availability
    available_symbols: List[str] = field(default_factory=list)
    available_timeframes: List[TimeFrame] = field(default_factory=list)
    
    # Execution context
    can_trade: bool = True
    order_queue_size: int = 0
    pending_orders: int = 0


@dataclass
class StrategyState:
    """Persistent state for backtesting strategies"""
    
    strategy_name: str
    lifecycle: StrategyLifecycle = StrategyLifecycle.CREATED
    
    # Execution state
    last_execution_time: Optional[datetime] = None
    execution_count: int = 0
    signal_count: int = 0
    
    # Performance state
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    total_trades: int = 0
    
    # Risk state
    current_positions: Dict[str, float] = field(default_factory=dict)
    peak_portfolio_value: float = 0.0
    current_drawdown: float = 0.0
    
    # Custom state (for strategy-specific data)
    custom_state: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class BacktestStrategy(BaseStrategy):
    """Enhanced base class for backtesting strategies"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Backtesting state
        self.state = StrategyState(strategy_name=self.__class__.__name__)
        self.is_initialized = False
        
        # Performance tracking
        self.equity_curve: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.signal_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.parameters = kwargs
        self.validation_errors: List[str] = []
        
        # Hooks for custom behavior
        self._on_start_hooks: List[Any] = []
        self._on_data_hooks: List[Any] = []
        self._on_signal_hooks: List[Any] = []
        self._on_end_hooks: List[Any] = []
    
    async def initialize_backtest(self, initial_context: BacktestContext) -> None:
        """Initialize strategy for backtesting"""
        
        if self.is_initialized:
            return
        
        try:
            self.logger.info(f"Initializing strategy {self.state.strategy_name} for backtesting")
            
            # Validate configuration
            self.validate_config()
            if self.validation_errors:
                raise ValueError(f"Configuration validation failed: {self.validation_errors}")
            
            # Initialize state
            self.state.lifecycle = StrategyLifecycle.INITIALIZED
            
            # Call strategy-specific initialization
            await self.on_start(initial_context)
            
            # Mark as initialized
            self.is_initialized = True
            
            self.logger.info(f"Strategy {self.state.strategy_name} initialized successfully")
            
        except Exception as e:
            self.state.lifecycle = StrategyLifecycle.ERROR
            self.logger.error(f"Failed to initialize strategy {self.state.strategy_name}: {e}")
            raise
    
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> StrategyResult:
        """Main analysis method - enhanced for backtesting"""
        
        # Convert to backtesting context if needed
        if not isinstance(context, BacktestContext):
            backtest_context = BacktestContext(
                market_data=context.market_data,
                order_manager=context.order_manager,
                position_manager=context.position_manager,
                risk_manager=context.risk_manager,
                account_summary=context.account_summary,
                portfolio_positions=context.portfolio_positions,
                current_time=context.current_time,
                dry_run=context.dry_run,
                available_symbols=[symbol],
                available_timeframes=list(data.keys())
            )
        else:
            backtest_context = context
        
        try:
            # Update execution state
            self.state.last_execution_time = backtest_context.current_time
            self.state.execution_count += 1
            self.state.lifecycle = StrategyLifecycle.RUNNING
            
            # Call data hooks
            await self._call_data_hooks(symbol, data, backtest_context)
            
            # Perform main analysis
            result = await self.on_data(symbol, data, backtest_context)
            
            # Process result
            if result.signal != StrategySignal.HOLD:
                self.state.signal_count += 1
                
                # Record signal
                signal_record = {
                    'timestamp': backtest_context.current_time,
                    'symbol': symbol,
                    'signal': result.signal,
                    'confidence': result.confidence,
                    'price': data[TimeFrame.DAY_1].iloc[-1]['close'] if TimeFrame.DAY_1 in data and not data[TimeFrame.DAY_1].empty else None,
                    'metadata': result.metadata
                }
                self.signal_history.append(signal_record)
                
                # Call signal hooks
                await self._call_signal_hooks(result, symbol, backtest_context)
            
            # Update state
            self.state.last_updated = datetime.now()
            
            return result
            
        except Exception as e:
            self.state.lifecycle = StrategyLifecycle.ERROR
            self.logger.error(f"Strategy analysis failed for {symbol}: {e}")
            raise
    
    async def finalize_backtest(self, final_context: BacktestContext) -> None:
        """Finalize strategy after backtesting"""
        
        try:
            self.logger.info(f"Finalizing strategy {self.state.strategy_name}")
            
            # Update final state
            self.state.lifecycle = StrategyLifecycle.COMPLETED
            
            # Call strategy-specific finalization
            await self.on_end(final_context)
            
            # Generate performance summary
            self._generate_performance_summary(final_context)
            
            self.logger.info(f"Strategy {self.state.strategy_name} finalized successfully")
            
        except Exception as e:
            self.state.lifecycle = StrategyLifecycle.ERROR
            self.logger.error(f"Failed to finalize strategy {self.state.strategy_name}: {e}")
            raise
    
    # Abstract methods for strategy implementation
    
    async def on_start(self, context: BacktestContext) -> None:
        """Called once at the beginning of backtesting"""
        # Override in subclasses for initialization logic
        pass
    
    @abstractmethod
    async def on_data(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: BacktestContext) -> StrategyResult:
        """Called for each data point during backtesting"""
        # Must be implemented by subclasses
        pass
    
    async def on_end(self, context: BacktestContext) -> None:
        """Called once at the end of backtesting"""
        # Override in subclasses for cleanup logic
        pass
    
    # Hook management
    
    def add_on_start_hook(self, hook: callable) -> None:
        """Add a hook to be called on strategy start"""
        self._on_start_hooks.append(hook)
    
    def add_on_data_hook(self, hook: callable) -> None:
        """Add a hook to be called on each data point"""
        self._on_data_hooks.append(hook)
    
    def add_on_signal_hook(self, hook: callable) -> None:
        """Add a hook to be called when signals are generated"""
        self._on_signal_hooks.append(hook)
    
    def add_on_end_hook(self, hook: callable) -> None:
        """Add a hook to be called on strategy end"""
        self._on_end_hooks.append(hook)
    
    # Hook execution
    
    async def _call_data_hooks(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: BacktestContext) -> None:
        """Call all registered data hooks"""
        for hook in self._on_data_hooks:
            try:
                if callable(hook):
                    await hook(symbol, data, context) if asyncio.iscoroutinefunction(hook) else hook(symbol, data, context)
            except Exception as e:
                self.logger.warning(f"Data hook failed: {e}")
    
    async def _call_signal_hooks(self, result: StrategyResult, symbol: str, context: BacktestContext) -> None:
        """Call all registered signal hooks"""
        for hook in self._on_signal_hooks:
            try:
                if callable(hook):
                    await hook(result, symbol, context) if asyncio.iscoroutinefunction(hook) else hook(result, symbol, context)
            except Exception as e:
                self.logger.warning(f"Signal hook failed: {e}")
    
    # State management
    
    def get_state(self) -> StrategyState:
        """Get current strategy state"""
        return self.state
    
    def set_state(self, state: StrategyState) -> None:
        """Set strategy state (for restoration)"""
        self.state = state
        self.state.last_updated = datetime.now()
    
    def save_custom_state(self, key: str, value: Any) -> None:
        """Save custom state data"""
        self.state.custom_state[key] = value
        self.state.last_updated = datetime.now()
    
    def get_custom_state(self, key: str, default: Any = None) -> Any:
        """Get custom state data"""
        return self.state.custom_state.get(key, default)
    
    # Performance tracking
    
    def record_trade(self, trade_info: Dict[str, Any]) -> None:
        """Record a trade for performance tracking"""
        trade_info['timestamp'] = datetime.now()
        trade_info['strategy'] = self.state.strategy_name
        self.trade_history.append(trade_info)
        
        # Update trade statistics
        if 'pnl' in trade_info:
            pnl = trade_info['pnl']
            self.state.total_pnl += pnl
            self.state.total_trades += 1
            
            if pnl > 0:
                self.state.winning_trades += 1
            else:
                self.state.losing_trades += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        
        total_trades = self.state.total_trades
        win_rate = self.state.winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'strategy_name': self.state.strategy_name,
            'total_pnl': self.state.total_pnl,
            'total_trades': total_trades,
            'winning_trades': self.state.winning_trades,
            'losing_trades': self.state.losing_trades,
            'win_rate': win_rate,
            'signal_count': self.state.signal_count,
            'execution_count': self.state.execution_count,
            'current_drawdown': self.state.current_drawdown,
            'lifecycle': self.state.lifecycle.value,
            'last_execution': self.state.last_execution_time
        }
    
    def _generate_performance_summary(self, context: BacktestContext) -> None:
        """Generate final performance summary"""
        
        metrics = self.get_performance_metrics()
        
        self.logger.info(f"=== Strategy Performance Summary: {self.state.strategy_name} ===")
        self.logger.info(f"Total P&L: ${metrics['total_pnl']:.2f}")
        self.logger.info(f"Total Trades: {metrics['total_trades']}")
        self.logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        self.logger.info(f"Signals Generated: {metrics['signal_count']}")
        self.logger.info(f"Executions: {metrics['execution_count']}")
        self.logger.info(f"Current Drawdown: {metrics['current_drawdown']:.2%}")
        self.logger.info(f"Final Status: {metrics['lifecycle']}")
    
    # Utility methods
    
    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during market hours"""
        # Simple implementation - can be enhanced
        weekday = timestamp.weekday()
        hour = timestamp.hour
        
        # Monday = 0, Friday = 4
        if weekday > 4:  # Weekend
            return False
        
        # Market hours 9:30 AM to 4:00 PM ET
        if hour < 9 or hour >= 16:
            return False
        
        if hour == 9 and timestamp.minute < 30:
            return False
        
        return True
    
    def get_position_size(self, symbol: str, context: BacktestContext) -> float:
        """Get current position size for a symbol"""
        return context.positions.get(symbol, 0.0)
    
    def calculate_position_value(self, symbol: str, price: float, context: BacktestContext) -> float:
        """Calculate current position value"""
        position = self.get_position_size(symbol, context)
        return position * price
    
    def __str__(self) -> str:
        return f"BacktestStrategy(name={self.state.strategy_name}, lifecycle={self.state.lifecycle.value})"
    
    def __repr__(self) -> str:
        return self.__str__() 
