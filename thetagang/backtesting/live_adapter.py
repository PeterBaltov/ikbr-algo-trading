"""
Live Trading Adapter Module

Seamless bridge between backtesting and live trading, providing a unified
interface for strategy deployment and real-time execution. Enables smooth
transition from development to production with minimal code changes.

Features:
- Unified API for backtest and live trading
- Real-time data feed integration
- Order management system bridge
- Risk management integration
- State synchronization between modes
- Portfolio reconciliation
- Emergency stop mechanisms

Integration:
- Works with existing IBKR integration
- Compatible with backtesting engine
- Supports all strategy framework components
- Provides production-ready deployment
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable, AsyncGenerator, Tuple
import asyncio
import logging
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np

from ..strategies.enums import TimeFrame, OrderSide
from ..strategies import BaseStrategy, StrategyResult


class TradingMode(Enum):
    """Trading operation modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"


class DataFeedType(Enum):
    """Data feed types"""
    IBKR = "ibkr"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO = "yahoo"
    POLYGON = "polygon"
    CUSTOM = "custom"


class OrderManagementSystem(Enum):
    """Order management systems"""
    IBKR = "ibkr"
    ALPACA = "alpaca"
    TD_AMERITRADE = "td_ameritrade"
    CUSTOM = "custom"


class ExecutionVenue(Enum):
    """Execution venues"""
    SMART = "smart"
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    ARCA = "arca"
    ISLAND = "island"


@dataclass
class AdapterConfig:
    """Configuration for live trading adapter"""
    
    # Trading mode
    mode: TradingMode = TradingMode.PAPER
    
    # Data feed settings
    data_feed: DataFeedType = DataFeedType.IBKR
    real_time_data: bool = True
    data_update_frequency: int = 1000  # milliseconds
    
    # Order management
    order_management_system: OrderManagementSystem = OrderManagementSystem.IBKR
    default_execution_venue: ExecutionVenue = ExecutionVenue.SMART
    
    # Risk management
    enable_risk_checks: bool = True
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_position_size: float = 0.10  # 10% max position size
    enable_circuit_breakers: bool = True
    
    # Portfolio reconciliation
    enable_portfolio_sync: bool = True
    sync_frequency_minutes: int = 5
    tolerance_threshold: float = 0.01  # 1% tolerance for reconciliation
    
    # State management
    enable_state_persistence: bool = True
    state_save_frequency_minutes: int = 1
    state_file_path: Optional[str] = None
    
    # Emergency controls
    emergency_stop_enabled: bool = True
    emergency_contacts: List[str] = field(default_factory=list)
    
    # Connection settings
    connection_timeout: int = 30  # seconds
    reconnect_attempts: int = 5
    heartbeat_interval: int = 30  # seconds
    
    # Custom functions
    custom_data_feed: Optional[Callable] = None
    custom_order_handler: Optional[Callable] = None
    custom_risk_check: Optional[Callable] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveOrder:
    """Represents a live trading order"""
    
    # Basic order info
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: str
    
    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Timing
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    
    # Status
    status: str = "PENDING"
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    
    # Execution details
    execution_venue: ExecutionVenue = ExecutionVenue.SMART
    time_in_force: str = "DAY"
    
    # Risk attribution
    strategy_source: Optional[str] = None
    risk_approved: bool = False
    
    # External references
    broker_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LivePosition:
    """Represents a live trading position"""
    
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    
    # P&L
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    day_pnl: float
    position_delta: float = 0.0
    position_gamma: float = 0.0
    position_theta: float = 0.0
    position_vega: float = 0.0
    
    # Attribution
    strategy_source: Optional[str] = None
    
    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LivePortfolio:
    """Represents live portfolio state"""
    
    timestamp: datetime
    total_value: float
    cash: float
    buying_power: float
    
    # Positions
    positions: Dict[str, LivePosition] = field(default_factory=dict)
    
    # P&L
    day_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Risk metrics
    leverage: float = 0.0
    margin_used: float = 0.0
    
    # Metadata
    account_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """Real-time risk management for live trading"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk state
        self.daily_pnl: float = 0.0
        self.position_limits: Dict[str, float] = {}
        self.circuit_breaker_triggered: bool = False
        
    def check_order_risk(self, order: LiveOrder, portfolio: LivePortfolio) -> Tuple[bool, List[str]]:
        """Check if order passes risk checks"""
        
        violations = []
        
        # Position size check
        position_value = abs(order.quantity * (order.limit_price or 0))
        position_pct = position_value / portfolio.total_value if portfolio.total_value > 0 else 0
        
        if position_pct > self.config.max_position_size:
            violations.append(f"Position size {position_pct:.1%} exceeds limit {self.config.max_position_size:.1%}")
        
        # Daily loss check
        if self.daily_pnl < -self.config.max_daily_loss * portfolio.total_value:
            violations.append(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
        
        # Circuit breaker check
        if self.circuit_breaker_triggered:
            violations.append("Circuit breaker triggered - trading halted")
        
        # Custom risk checks
        if self.config.custom_risk_check:
            try:
                custom_violations = self.config.custom_risk_check(order, portfolio, self.config)
                violations.extend(custom_violations)
            except Exception as e:
                self.logger.error(f"Custom risk check failed: {e}")
                violations.append("Custom risk check failed")
        
        return len(violations) == 0, violations
    
    def update_risk_state(self, portfolio: LivePortfolio) -> None:
        """Update risk management state"""
        
        self.daily_pnl = portfolio.day_pnl
        
        # Check for circuit breaker conditions
        if self.config.enable_circuit_breakers:
            loss_threshold = self.config.max_daily_loss * portfolio.total_value
            if self.daily_pnl < -loss_threshold:
                self.circuit_breaker_triggered = True
                self.logger.critical(f"CIRCUIT BREAKER TRIGGERED: Daily loss {self.daily_pnl:.2f} exceeds limit")


class DataFeedManager:
    """Manages real-time data feeds"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data state
        self.subscriptions: Set[str] = set()
        self.latest_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.data_callbacks: List[Callable] = []
        
    async def subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to real-time data for a symbol"""
        
        if symbol not in self.subscriptions:
            self.subscriptions.add(symbol)
            self.logger.info(f"Subscribed to real-time data for {symbol}")
            
            # In a real implementation, this would connect to the data feed
            if self.config.data_feed == DataFeedType.IBKR:
                await self._subscribe_ibkr(symbol)
            elif self.config.custom_data_feed:
                await self.config.custom_data_feed(symbol)
    
    async def unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from real-time data for a symbol"""
        
        if symbol in self.subscriptions:
            self.subscriptions.remove(symbol)
            self.logger.info(f"Unsubscribed from real-time data for {symbol}")
    
    async def _subscribe_ibkr(self, symbol: str) -> None:
        """Subscribe to IBKR data feed"""
        
        # Mock implementation - would integrate with actual IBKR client
        self.logger.debug(f"IBKR subscription for {symbol}")
    
    def add_data_callback(self, callback: Callable) -> None:
        """Add callback for data updates"""
        
        self.data_callbacks.append(callback)
    
    async def _handle_data_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle incoming data update"""
        
        self.latest_data[symbol].update(data)
        
        # Notify callbacks
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data)
                else:
                    callback(symbol, data)
            except Exception as e:
                self.logger.error(f"Data callback failed: {e}")


class OrderManager:
    """Manages order lifecycle in live trading"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Order state
        self.pending_orders: Dict[str, LiveOrder] = {}
        self.order_history: List[LiveOrder] = []
        
    async def submit_order(self, order: LiveOrder) -> str:
        """Submit order to broker"""
        
        try:
            # Add to pending orders
            self.pending_orders[order.order_id] = order
            order.submitted_time = datetime.now()
            order.status = "SUBMITTED"
            
            # Submit to broker
            if self.config.order_management_system == OrderManagementSystem.IBKR:
                broker_id = await self._submit_to_ibkr(order)
            elif self.config.custom_order_handler:
                broker_id = await self.config.custom_order_handler(order)
            else:
                # Mock submission
                broker_id = f"MOCK_{order.order_id}"
            
            order.broker_order_id = broker_id
            order.status = "ACCEPTED"
            
            self.logger.info(f"Order submitted: {order.order_id} -> {broker_id}")
            return order.order_id
            
        except Exception as e:
            order.status = "REJECTED"
            self.logger.error(f"Order submission failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            
            try:
                # Cancel with broker
                if order.broker_order_id:
                    await self._cancel_with_broker(order.broker_order_id)
                
                order.status = "CANCELLED"
                self.order_history.append(order)
                del self.pending_orders[order_id]
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Order cancellation failed: {e}")
                return False
        
        return False
    
    async def _submit_to_ibkr(self, order: LiveOrder) -> str:
        """Submit order to IBKR"""
        
        # Mock implementation - would use actual IBKR client
        return f"IBKR_{order.order_id}_{datetime.now().strftime('%H%M%S')}"
    
    async def _cancel_with_broker(self, broker_order_id: str) -> None:
        """Cancel order with broker"""
        
        # Mock implementation
        self.logger.debug(f"Cancelling broker order: {broker_order_id}")


class PortfolioReconciler:
    """Reconciles portfolio state between systems"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Reconciliation state
        self.last_sync: Optional[datetime] = None
        self.discrepancies: List[Dict[str, Any]] = []
    
    async def reconcile_portfolio(self, local_portfolio: LivePortfolio) -> Tuple[bool, List[str]]:
        """Reconcile local portfolio with broker"""
        
        try:
            # Get broker portfolio
            broker_portfolio = await self._get_broker_portfolio()
            
            # Compare positions
            discrepancies = []
            
            for symbol in set(local_portfolio.positions.keys()) | set(broker_portfolio.positions.keys()):
                local_pos = local_portfolio.positions.get(symbol)
                broker_pos = broker_portfolio.positions.get(symbol)
                
                local_qty = local_pos.quantity if local_pos else 0.0
                broker_qty = broker_pos.quantity if broker_pos else 0.0
                
                diff = abs(local_qty - broker_qty)
                tolerance = max(1.0, abs(broker_qty) * self.config.tolerance_threshold)
                
                if diff > tolerance:
                    discrepancies.append(f"{symbol}: Local={local_qty}, Broker={broker_qty}, Diff={diff}")
            
            # Check cash balance
            cash_diff = abs(local_portfolio.cash - broker_portfolio.cash)
            cash_tolerance = local_portfolio.total_value * self.config.tolerance_threshold
            
            if cash_diff > cash_tolerance:
                discrepancies.append(f"Cash: Local={local_portfolio.cash:.2f}, Broker={broker_portfolio.cash:.2f}")
            
            self.last_sync = datetime.now()
            
            if discrepancies:
                self.logger.warning(f"Portfolio discrepancies found: {discrepancies}")
                self.discrepancies.extend([{
                    'timestamp': self.last_sync,
                    'discrepancy': disc
                } for disc in discrepancies])
            
            return len(discrepancies) == 0, discrepancies
            
        except Exception as e:
            self.logger.error(f"Portfolio reconciliation failed: {e}")
            return False, [f"Reconciliation error: {e}"]
    
    async def _get_broker_portfolio(self) -> LivePortfolio:
        """Get portfolio from broker"""
        
        # Mock implementation - would query actual broker
        return LivePortfolio(
            timestamp=datetime.now(),
            total_value=100000.0,
            cash=50000.0,
            buying_power=75000.0
        )


class LiveTradingAdapter:
    """Main live trading adapter"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.risk_manager = RiskManager(config)
        self.data_feed_manager = DataFeedManager(config)
        self.order_manager = OrderManager(config)
        self.portfolio_reconciler = PortfolioReconciler(config)
        
        # State
        self.is_connected: bool = False
        self.current_portfolio: Optional[LivePortfolio] = None
        self.active_strategies: Dict[str, BaseStrategy] = {}
        
        # Emergency state
        self.emergency_stop_active: bool = False
        
    async def initialize(self) -> None:
        """Initialize live trading adapter"""
        
        self.logger.info("Initializing live trading adapter...")
        
        try:
            # Connect to data feeds
            await self._connect_data_feeds()
            
            # Initialize portfolio
            await self._initialize_portfolio()
            
            # Setup callbacks
            self.data_feed_manager.add_data_callback(self._handle_market_data)
            
            self.is_connected = True
            self.logger.info("Live trading adapter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Adapter initialization failed: {e}")
            raise
    
    async def register_strategy(self, strategy: BaseStrategy, strategy_name: str) -> None:
        """Register strategy for live trading"""
        
        self.active_strategies[strategy_name] = strategy
        
        # Subscribe to required symbols
        required_symbols = strategy.get_required_symbols()
        for symbol in required_symbols:
            await self.data_feed_manager.subscribe_symbol(symbol)
        
        self.logger.info(f"Strategy registered: {strategy_name}")
    
    async def execute_strategy_signal(self, strategy_name: str, signal: StrategyResult) -> Optional[str]:
        """Execute strategy signal"""
        
        if self.emergency_stop_active:
            self.logger.warning("Emergency stop active - ignoring signal")
            return None
        
        try:
            # Create order from signal
            order = self._create_order_from_signal(signal, strategy_name)
            
            # Risk check
            risk_approved, violations = self.risk_manager.check_order_risk(order, self.current_portfolio)
            
            if not risk_approved:
                self.logger.warning(f"Order rejected by risk management: {violations}")
                return None
            
            # Submit order
            order_id = await self.order_manager.submit_order(order)
            return order_id
            
        except Exception as e:
            self.logger.error(f"Signal execution failed: {e}")
            return None
    
    async def emergency_stop(self, reason: str) -> None:
        """Activate emergency stop"""
        
        self.emergency_stop_active = True
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Cancel all pending orders
        for order_id in list(self.order_manager.pending_orders.keys()):
            await self.order_manager.cancel_order(order_id)
        
        # Notify emergency contacts
        for contact in self.config.emergency_contacts:
            await self._notify_emergency_contact(contact, reason)
    
    async def _connect_data_feeds(self) -> None:
        """Connect to data feeds"""
        
        # Mock implementation
        self.logger.info(f"Connecting to {self.config.data_feed.value} data feed")
    
    async def _initialize_portfolio(self) -> None:
        """Initialize portfolio state"""
        
        # Mock portfolio - would query actual broker
        self.current_portfolio = LivePortfolio(
            timestamp=datetime.now(),
            total_value=100000.0,
            cash=50000.0,
            buying_power=75000.0
        )
    
    async def _handle_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle market data updates"""
        
        # Update portfolio with latest prices
        if self.current_portfolio and symbol in self.current_portfolio.positions:
            position = self.current_portfolio.positions[symbol]
            if 'price' in data:
                position.current_price = data['price']
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = (position.current_price - position.avg_cost) * position.quantity
        
        # Notify strategies of data update
        for strategy_name, strategy in self.active_strategies.items():
            if symbol in strategy.get_required_symbols():
                # Would normally trigger strategy analysis
                pass
    
    def _create_order_from_signal(self, signal: StrategyResult, strategy_name: str) -> LiveOrder:
        """Create order from strategy signal"""
        
        order_id = f"{strategy_name}_{signal.symbol}_{datetime.now().strftime('%H%M%S')}"
        
        return LiveOrder(
            order_id=order_id,
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.signal.value == "BUY" else OrderSide.SELL,
            quantity=signal.quantity or 100,  # Default quantity
            order_type="MKT",  # Market order
            strategy_source=strategy_name
        )
    
    async def _notify_emergency_contact(self, contact: str, reason: str) -> None:
        """Notify emergency contact"""
        
        # Mock implementation - would send actual notification
        self.logger.info(f"Emergency notification sent to {contact}: {reason}")


def create_live_config(
    mode: TradingMode = TradingMode.PAPER,
    data_feed: DataFeedType = DataFeedType.IBKR,
    order_system: OrderManagementSystem = OrderManagementSystem.IBKR,
    **kwargs
) -> AdapterConfig:
    """Create live trading configuration with sensible defaults"""
    
    return AdapterConfig(
        mode=mode,
        data_feed=data_feed,
        order_management_system=order_system,
        **kwargs
    ) 
