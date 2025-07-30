"""
Core Backtesting Engine

Event-driven backtesting engine inspired by backtrader architecture with deep
integration into our multi-strategy platform. Provides realistic historical
simulation with comprehensive state management and event handling.

Features:
- Event-driven simulation loop
- Multiple strategy execution coordination
- Realistic broker simulation
- Performance tracking and analytics
- Risk management integration
- Memory-efficient data handling
- Extensible plugin architecture

Integration:
- Uses Phase 1 strategy framework and registry
- Leverages Phase 2 technical analysis engine  
- Coordinates with Phase 3 multi-timeframe architecture
- Provides foundation for live trading bridge
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable, AsyncIterator
import asyncio
import logging
from collections import defaultdict, deque
import warnings

import pandas as pd
import numpy as np

from ..strategies import BaseStrategy, StrategyResult, StrategyContext, get_registry
from ..strategies.enums import StrategySignal, StrategyType, TimeFrame, StrategyStatus, OrderSide
from ..strategies.exceptions import StrategyError
from ..timeframes import TimeFrameManager, get_timeframe_manager
from ..analysis import TechnicalAnalysisEngine

from .data import DataManager, DataConfig


class BacktestState(Enum):
    """Backtesting engine states"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class EventType(Enum):
    """Types of events in the backtesting system"""
    DATA = "data"  # New market data available
    SIGNAL = "signal"  # Strategy signal generated
    ORDER = "order"  # Order placement/execution
    POSITION = "position"  # Position update
    PORTFOLIO = "portfolio"  # Portfolio update
    TIMER = "timer"  # Scheduled timer event
    CUSTOM = "custom"  # Custom user event


@dataclass
class BacktestEvent:
    """Event in the backtesting system"""
    
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Event metadata
    source: Optional[str] = None
    priority: int = 0  # Higher priority = processed first
    processed: bool = False
    
    # Performance tracking
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine"""
    
    # Time range
    start_date: datetime
    end_date: datetime
    
    # Data configuration
    data_config: DataConfig
    
    # Strategy settings
    strategies: List[str] = field(default_factory=list)  # Strategy names from registry
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Capital and portfolio
    initial_capital: float = 100000.0
    max_positions: int = 20
    position_sizing: str = "equal_weight"  # equal_weight, risk_parity, custom
    
    # Execution settings
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    market_impact: bool = True
    
    # Risk management
    max_drawdown: float = 0.20  # 20% max drawdown
    position_size_limit: float = 0.10  # 10% max position size
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    
    # Performance settings
    benchmark: Optional[str] = "SPY"  # Benchmark symbol
    risk_free_rate: float = 0.02  # 2% risk-free rate
    
    # Simulation settings
    enable_lookahead_bias_check: bool = True
    enable_survivorship_bias_check: bool = True
    enable_data_validation: bool = True
    
    # Event system
    max_events_per_iteration: int = 1000
    enable_custom_events: bool = True
    
    # Reporting
    save_trades: bool = True
    save_portfolio_snapshots: bool = True
    snapshot_frequency: str = "daily"  # daily, weekly, monthly
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a trading position"""
    
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    last_update: datetime
    
    # Position details
    side: OrderSide = OrderSide.BUY  # BUY (long) or SELL (short)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk metrics
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    
    # Metadata
    strategy_source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class Portfolio:
    """Represents the portfolio state"""
    
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Performance metrics
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk metrics
    leverage: float = 0.0
    exposure: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results of a backtest run"""
    
    # Basic info
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Performance summary
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio evolution
    portfolio_history: List[Portfolio] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    
    # Detailed analytics
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    yearly_returns: pd.Series = field(default_factory=pd.Series)
    
    # Strategy-specific results
    strategy_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    config_used: Optional[BacktestConfig] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """Main backtesting engine with event-driven simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.state = BacktestState.INITIALIZED
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.data_manager = DataManager(config.data_config)
        self.timeframe_manager = get_timeframe_manager()
        self.technical_engine = TechnicalAnalysisEngine()
        
        # Event system
        self.event_queue: deque = deque()
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.current_time: Optional[datetime] = None
        
        # Portfolio and positions
        self.portfolio = Portfolio(
            timestamp=config.start_date,
            total_value=config.initial_capital,
            cash=config.initial_capital
        )
        
        # Strategy management
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.strategy_registry = get_registry()
        
        # Data storage
        self.market_data: Dict[str, Dict[TimeFrame, pd.DataFrame]] = {}
        self.data_cursors: Dict[TimeFrame, int] = defaultdict(int)  # Current position in data
        
        # Performance tracking
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_snapshots: List[Portfolio] = []
        self.performance_cache: Dict[str, Any] = {}
        
        # Event tracking
        self.events_processed = 0
        self.events_total = 0
        
        # Register default event handlers
        self._register_default_handlers()
    
    async def initialize(self) -> None:
        """Initialize the backtesting engine"""
        
        if self.state != BacktestState.INITIALIZED:
            raise ValueError(f"Engine must be in INITIALIZED state, currently {self.state}")
        
        self.logger.info("Initializing backtesting engine...")
        
        try:
            # Load market data
            await self._load_market_data()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Validate configuration
            self._validate_configuration()
            
            # Prepare data cursors
            self._prepare_data_iteration()
            
            self.logger.info("Backtesting engine initialized successfully")
            
        except Exception as e:
            self.state = BacktestState.ERROR
            self.logger.error(f"Failed to initialize backtesting engine: {e}")
            raise
    
    async def run(self) -> BacktestResult:
        """Run the backtest simulation"""
        
        if self.state != BacktestState.INITIALIZED:
            raise ValueError(f"Engine must be initialized before running")
        
        self.logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        start_time = datetime.now()
        
        try:
            self.state = BacktestState.RUNNING
            
            # Main simulation loop
            async for current_time in self._iterate_time():
                self.current_time = current_time
                
                # Generate data events
                await self._generate_data_events(current_time)
                
                # Process all events for this timestamp
                await self._process_events()
                
                # Update portfolio
                await self._update_portfolio(current_time)
                
                # Take snapshot if needed
                if self._should_take_snapshot(current_time):
                    self._take_portfolio_snapshot()
                
                # Check risk limits
                if not self._check_risk_limits():
                    self.logger.warning("Risk limits breached, stopping backtest")
                    break
            
            # Finalize backtest
            result = await self._finalize_backtest(start_time)
            self.state = BacktestState.COMPLETED
            
            self.logger.info(f"Backtest completed successfully in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.state = BacktestState.ERROR
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    async def _load_market_data(self) -> None:
        """Load market data for backtesting"""
        
        self.logger.info("Loading market data...")
        
        # Load data through data manager
        self.market_data = await self.data_manager.load_data()
        
        if not self.market_data:
            raise ValueError("No market data loaded")
        
        # Validate data coverage
        total_symbols = len(self.market_data)
        total_records = sum(
            len(timeframe_data) 
            for symbol_data in self.market_data.values() 
            for timeframe_data in symbol_data.values()
        )
        
        self.logger.info(f"Loaded data for {total_symbols} symbols, {total_records} total records")
    
    async def _initialize_strategies(self) -> None:
        """Initialize trading strategies"""
        
        self.logger.info("Initializing strategies...")
        
        for strategy_name in self.config.strategies:
            try:
                # Get strategy class from registry
                strategy_class = self.strategy_registry.get_strategy(strategy_name)
                
                # Get strategy configuration
                strategy_config = self.config.strategy_configs.get(strategy_name, {})
                
                # Create strategy instance
                strategy = strategy_class(**strategy_config)
                
                # Validate strategy
                strategy.validate_config()
                
                self.active_strategies[strategy_name] = strategy
                self.logger.info(f"Initialized strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
                if self.config.data_config.strict_validation:
                    raise
        
        if not self.active_strategies:
            raise ValueError("No strategies successfully initialized")
    
    def _validate_configuration(self) -> None:
        """Validate backtesting configuration"""
        
        # Validate date range
        if self.config.start_date >= self.config.end_date:
            raise ValueError("start_date must be before end_date")
        
        # Validate capital
        if self.config.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        
        # Validate risk parameters
        if not 0 < self.config.max_drawdown <= 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        
        if not 0 < self.config.position_size_limit <= 1:
            raise ValueError("position_size_limit must be between 0 and 1")
    
    def _prepare_data_iteration(self) -> None:
        """Prepare data structures for time-based iteration"""
        
        # Find all unique timestamps across all data
        all_timestamps = set()
        
        for symbol_data in self.market_data.values():
            for timeframe_data in symbol_data.values():
                all_timestamps.update(timeframe_data.index)
        
        # Sort timestamps for iteration
        self.iteration_timestamps = sorted(all_timestamps)
        
        # Filter to backtest date range
        self.iteration_timestamps = [
            ts for ts in self.iteration_timestamps 
            if self.config.start_date <= ts <= self.config.end_date
        ]
        
        self.logger.info(f"Prepared {len(self.iteration_timestamps)} timestamps for iteration")
    
    async def _iterate_time(self) -> AsyncIterator[datetime]:
        """Iterate through time during backtesting"""
        
        for timestamp in self.iteration_timestamps:
            yield timestamp
    
    async def _generate_data_events(self, current_time: datetime) -> None:
        """Generate data events for the current timestamp"""
        
        # Check each symbol and timeframe for new data
        for symbol, symbol_data in self.market_data.items():
            for timeframe, data in symbol_data.items():
                
                # Check if we have data for this timestamp
                if current_time in data.index:
                    current_bar = data.loc[current_time]
                    
                    # Create data event
                    event = BacktestEvent(
                        event_type=EventType.DATA,
                        timestamp=current_time,
                        data={
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'bar': current_bar.to_dict(),
                            'ohlcv': {
                                'open': current_bar.get('open'),
                                'high': current_bar.get('high'),
                                'low': current_bar.get('low'),
                                'close': current_bar.get('close'),
                                'volume': current_bar.get('volume')
                            }
                        },
                        source='market_data'
                    )
                    
                    await self._add_event(event)
    
    async def _process_events(self) -> None:
        """Process all events in the queue"""
        
        events_this_iteration = 0
        
        while self.event_queue and events_this_iteration < self.config.max_events_per_iteration:
            # Get next event (prioritized)
            event = self._get_next_event()
            
            # Process event based on type
            await self._handle_event(event)
            
            # Mark as processed
            event.processed = True
            event.processed_at = datetime.now()
            
            events_this_iteration += 1
            self.events_processed += 1
    
    async def _handle_event(self, event: BacktestEvent) -> None:
        """Handle a specific event"""
        
        # Call registered handlers for this event type
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {e}")
    
    async def _add_event(self, event: BacktestEvent) -> None:
        """Add event to the queue with proper prioritization"""
        
        # Insert based on priority (higher priority first)
        inserted = False
        for i, existing_event in enumerate(self.event_queue):
            if event.priority > existing_event.priority:
                self.event_queue.insert(i, event)
                inserted = True
                break
        
        if not inserted:
            self.event_queue.append(event)
        
        self.events_total += 1
    
    def _get_next_event(self) -> BacktestEvent:
        """Get the next event from the queue"""
        
        if not self.event_queue:
            raise RuntimeError("No events in queue")
        
        return self.event_queue.popleft()
    
    async def _update_portfolio(self, current_time: datetime) -> None:
        """Update portfolio valuation"""
        
        total_value = self.portfolio.cash
        unrealized_pnl = 0.0
        
        # Update position values
        for symbol, position in self.portfolio.positions.items():
            if symbol in self.market_data and TimeFrame.DAY_1 in self.market_data[symbol]:
                data = self.market_data[symbol][TimeFrame.DAY_1]
                
                if current_time in data.index:
                    current_price = data.loc[current_time, 'close']
                    position.current_price = current_price
                    position.last_update = current_time
                    
                    # Calculate P&L
                    if position.side == OrderSide.BUY:
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:  # Short position
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    # Update MFE/MAE
                    if position.side == OrderSide.BUY:
                        favorable_move = current_price - position.entry_price
                        adverse_move = position.entry_price - current_price
                    else:
                        favorable_move = position.entry_price - current_price
                        adverse_move = current_price - position.entry_price
                    
                    position.max_favorable_excursion = max(position.max_favorable_excursion, favorable_move)
                    position.max_adverse_excursion = max(position.max_adverse_excursion, adverse_move)
                    
                    total_value += abs(position.quantity) * current_price
                    unrealized_pnl += position.unrealized_pnl
        
        # Update portfolio
        self.portfolio.timestamp = current_time
        self.portfolio.total_value = total_value
        self.portfolio.unrealized_pnl = unrealized_pnl
        self.portfolio.total_pnl = self.portfolio.realized_pnl + unrealized_pnl
        
        # Calculate leverage and exposure
        total_position_value = sum(
            abs(pos.quantity) * pos.current_price 
            for pos in self.portfolio.positions.values()
        )
        
        self.portfolio.leverage = total_position_value / total_value if total_value > 0 else 0
        self.portfolio.exposure = total_position_value / self.config.initial_capital
    
    def _should_take_snapshot(self, current_time: datetime) -> bool:
        """Check if we should take a portfolio snapshot"""
        
        if not self.config.save_portfolio_snapshots:
            return False
        
        if self.config.snapshot_frequency == "daily":
            return True  # Take snapshot every day
        elif self.config.snapshot_frequency == "weekly":
            return current_time.weekday() == 0  # Monday
        elif self.config.snapshot_frequency == "monthly":
            return current_time.day == 1  # First day of month
        
        return False
    
    def _take_portfolio_snapshot(self) -> None:
        """Take a snapshot of the current portfolio"""
        
        # Create a deep copy of the portfolio
        import copy
        snapshot = copy.deepcopy(self.portfolio)
        self.portfolio_snapshots.append(snapshot)
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are being respected"""
        
        # Check max drawdown
        if len(self.portfolio_snapshots) > 0:
            peak_value = max(snapshot.total_value for snapshot in self.portfolio_snapshots)
            current_drawdown = (peak_value - self.portfolio.total_value) / peak_value
            
            if current_drawdown > self.config.max_drawdown:
                self.logger.warning(f"Max drawdown exceeded: {current_drawdown:.2%}")
                return False
        
        # Check position size limits
        for position in self.portfolio.positions.values():
            position_value = abs(position.quantity) * position.current_price
            position_pct = position_value / self.portfolio.total_value
            
            if position_pct > self.config.position_size_limit:
                self.logger.warning(f"Position size limit exceeded for {position.symbol}: {position_pct:.2%}")
                return False
        
        return True
    
    async def _finalize_backtest(self, start_time: datetime) -> BacktestResult:
        """Finalize backtest and create results"""
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        result.execution_time = execution_time
        result.config_used = self.config
        
        return result
    
    def _calculate_performance_metrics(self) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        # Build equity curve from snapshots
        if self.portfolio_snapshots:
            equity_curve = pd.Series(
                [snapshot.total_value for snapshot in self.portfolio_snapshots],
                index=[snapshot.timestamp for snapshot in self.portfolio_snapshots]
            )
        else:
            equity_curve = pd.Series([self.config.initial_capital, self.portfolio.total_value],
                                   index=[self.config.start_date, self.config.end_date])
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (self.portfolio.total_value - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized return
        days = (self.config.end_date - self.config.start_date).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        winning_pnls = [trade['pnl'] for trade in self.trades if trade.get('pnl', 0) > 0]
        losing_pnls = [trade['pnl'] for trade in self.trades if trade.get('pnl', 0) < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls else float('inf')
        
        return BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration_days=days,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            portfolio_history=self.portfolio_snapshots
        )
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers"""
        
        # Data event handler - triggers strategy analysis
        self.event_handlers[EventType.DATA].append(self._handle_data_event)
        
        # Signal event handler - processes strategy signals
        self.event_handlers[EventType.SIGNAL].append(self._handle_signal_event)
    
    async def _handle_data_event(self, event: BacktestEvent) -> None:
        """Handle market data events"""
        
        symbol = event.data['symbol']
        timeframe = event.data['timeframe']
        
        # Run strategies that require this timeframe
        for strategy_name, strategy in self.active_strategies.items():
            if timeframe in strategy.get_required_timeframes():
                try:
                    # Prepare strategy context
                    context = StrategyContext(
                        current_time=event.timestamp,
                        portfolio_value=self.portfolio.total_value,
                        cash=self.portfolio.cash,
                        positions={sym: pos.quantity for sym, pos in self.portfolio.positions.items()},
                        metadata={}
                    )
                    
                    # Get historical data for analysis
                    strategy_data = self._get_strategy_data(strategy, symbol, event.timestamp)
                    
                    # Run strategy analysis
                    strategy_result = await strategy.analyze(symbol, strategy_data, context)
                    
                    # Generate signal event if strategy has signal
                    if strategy_result.signal != StrategySignal.HOLD:
                        signal_event = BacktestEvent(
                            event_type=EventType.SIGNAL,
                            timestamp=event.timestamp,
                            data={
                                'strategy': strategy_name,
                                'symbol': symbol,
                                'signal': strategy_result.signal,
                                'confidence': strategy_result.confidence,
                                'metadata': strategy_result.metadata
                            },
                            source=strategy_name,
                            priority=strategy_result.confidence * 100  # Higher confidence = higher priority
                        )
                        
                        await self._add_event(signal_event)
                
                except Exception as e:
                    self.logger.error(f"Strategy {strategy_name} failed on {symbol}: {e}")
    
    async def _handle_signal_event(self, event: BacktestEvent) -> None:
        """Handle strategy signal events"""
        
        # This would be implemented to place orders based on signals
        # For now, just log the signal
        self.logger.info(f"Signal from {event.data['strategy']}: {event.data['signal']} on {event.data['symbol']}")
    
    def _get_strategy_data(self, strategy: BaseStrategy, symbol: str, current_time: datetime) -> Dict[TimeFrame, pd.DataFrame]:
        """Get historical data for strategy analysis"""
        
        strategy_data = {}
        
        for timeframe in strategy.get_required_timeframes():
            if symbol in self.market_data and timeframe in self.market_data[symbol]:
                data = self.market_data[symbol][timeframe]
                
                # Get data up to current time (no lookahead bias)
                historical_data = data[data.index <= current_time]
                strategy_data[timeframe] = historical_data
        
        return strategy_data 
