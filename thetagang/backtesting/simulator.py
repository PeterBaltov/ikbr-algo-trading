"""
Trade Execution Simulator

Realistic trade execution simulation for backtesting with slippage modeling,
market impact, and sophisticated order handling. Inspired by backtrader's
broker simulation with enhancements for modern trading conditions.

Features:
- Multiple order types (Market, Limit, Stop, Stop-Limit)
- Realistic slippage modeling based on volume and volatility
- Market impact simulation with price improvement/degradation
- Liquidity constraints and partial fills
- Commission and fee modeling
- Order execution timing simulation
- Fill probability modeling

Integration:
- Works with core backtesting engine
- Supports Phase 1 strategy framework order types
- Leverages Phase 3 multi-timeframe data
- Provides foundation for live trading bridge
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable
import logging
from collections import defaultdict, deque
import random

import pandas as pd
import numpy as np

from ..strategies.enums import OrderSide, TimeFrame


class OrderType(Enum):
    """Types of orders supported"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price


class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SlippageModel(Enum):
    """Slippage modeling approaches"""
    FIXED = "fixed"  # Fixed percentage slippage
    VOLUME_BASED = "volume_based"  # Based on order size vs average volume
    VOLATILITY_BASED = "volatility_based"  # Based on price volatility
    SQRT = "sqrt"  # Square root model (common in academic literature)
    CUSTOM = "custom"  # Custom function provided


class MarketImpactModel(Enum):
    """Market impact modeling approaches"""
    NONE = "none"  # No market impact
    LINEAR = "linear"  # Linear impact based on order size
    SQRT = "sqrt"  # Square root impact model
    ALMGREN_CHRISS = "almgren_chriss"  # Almgren-Chriss model
    CUSTOM = "custom"  # Custom function provided


@dataclass
class Order:
    """Represents a trading order"""
    
    # Basic order info
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    
    # Timing
    created_time: datetime
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    
    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None
    
    # Status and fills
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    
    # Execution details
    fills: List[Dict[str, Any]] = field(default_factory=list)
    commission_paid: float = 0.0
    slippage_paid: float = 0.0
    
    # Order specifications
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    all_or_none: bool = False
    good_till_date: Optional[datetime] = None
    
    # Strategy context
    strategy_source: Optional[str] = None
    parent_order_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity


@dataclass
class ExecutionConfig:
    """Configuration for trade execution simulation"""
    
    # Slippage settings
    slippage_model: SlippageModel = SlippageModel.FIXED
    base_slippage: float = 0.0005  # 0.05% base slippage
    max_slippage: float = 0.01  # 1% maximum slippage
    
    # Market impact settings
    market_impact_model: MarketImpactModel = MarketImpactModel.SQRT
    impact_coefficient: float = 0.1  # Market impact coefficient
    temporary_impact_decay: float = 0.5  # How quickly temporary impact decays
    
    # Liquidity settings
    enable_liquidity_constraints: bool = True
    min_daily_volume_multiple: float = 0.1  # Can't trade more than 10% of daily volume
    max_order_size_pct: float = 0.05  # Maximum order size as % of average volume
    
    # Commission and fees
    commission_per_share: float = 0.005  # $0.005 per share
    commission_percentage: float = 0.0  # 0% of trade value
    min_commission: float = 1.0  # Minimum $1 commission
    max_commission: float = 100.0  # Maximum $100 commission
    
    # Order execution timing
    market_order_delay_ms: int = 100  # Market order execution delay
    limit_order_delay_ms: int = 500  # Limit order processing delay
    enable_execution_uncertainty: bool = True  # Add randomness to execution
    
    # Fill probability modeling
    enable_partial_fills: bool = True
    partial_fill_probability: float = 0.1  # 10% chance of partial fill
    min_fill_ratio: float = 0.3  # Minimum 30% fill on partial fills
    
    # Custom functions
    custom_slippage_func: Optional[Callable[[Order, Dict[str, float], 'ExecutionConfig'], float]] = None
    custom_impact_func: Optional[Callable[[Order, Dict[str, float], 'ExecutionConfig'], float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Represents a trade fill"""
    
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    
    # Execution details
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    
    # Market context
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    last_price: Optional[float] = None
    volume_at_fill: Optional[float] = None
    
    # Metadata
    execution_venue: str = "simulated"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeSimulator:
    """Main trade execution simulator"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Order management
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fill_history: List[Fill] = []
        
        # Market data cache
        self.current_prices: Dict[str, float] = {}
        self.volume_profiles: Dict[str, Dict[TimeFrame, pd.Series]] = defaultdict(dict)
        self.volatility_cache: Dict[str, float] = {}
        
        # Performance tracking
        self.total_commission_paid: float = 0.0
        self.total_slippage_paid: float = 0.0
        self.total_market_impact: float = 0.0
        self.orders_processed: int = 0
        self.fills_executed: int = 0
        
        # Random number generator for simulation
        self.rng = random.Random(42)  # Fixed seed for reproducibility
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution"""
        
        # Validate order
        self._validate_order(order)
        
        # Set submission time
        order.submitted_time = datetime.now()
        order.status = OrderStatus.SUBMITTED
        
        # Add to pending orders
        self.pending_orders[order.order_id] = order
        
        self.logger.debug(f"Order submitted: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            self.order_history.append(order)
            
            self.logger.debug(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    def process_market_data(self, symbol: str, timestamp: datetime, 
                          ohlcv: Dict[str, float], volume_profile: Optional[pd.Series] = None) -> None:
        """Process new market data for execution simulation"""
        
        # Update current prices
        self.current_prices[symbol] = ohlcv['close']
        
        # Update volume profile
        if volume_profile is not None:
            self.volume_profiles[symbol][TimeFrame.DAY_1] = volume_profile
        
        # Calculate volatility (rolling 20-day)
        if symbol in self.volume_profiles and TimeFrame.DAY_1 in self.volume_profiles[symbol]:
            price_data = self.volume_profiles[symbol][TimeFrame.DAY_1]
            if len(price_data) >= 20:
                returns = price_data.pct_change().dropna()
                self.volatility_cache[symbol] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Process pending orders for this symbol
        self._process_pending_orders(symbol, timestamp, ohlcv)
    
    def _validate_order(self, order: Order) -> None:
        """Validate order parameters"""
        
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("Limit orders must have limit_price")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            raise ValueError("Stop orders must have stop_price")
        
        if order.order_type == OrderType.STOP_LIMIT and order.limit_price is None:
            raise ValueError("Stop-limit orders must have both stop_price and limit_price")
    
    def _process_pending_orders(self, symbol: str, timestamp: datetime, ohlcv: Dict[str, float]) -> None:
        """Process pending orders for a symbol"""
        
        orders_to_remove = []
        
        for order_id, order in self.pending_orders.items():
            if order.symbol != symbol:
                continue
            
            # Check if order should be filled
            if self._should_fill_order(order, timestamp, ohlcv):
                fill_price = self._calculate_fill_price(order, ohlcv)
                fill_quantity = self._calculate_fill_quantity(order, ohlcv)
                
                # Execute the fill
                self._execute_fill(order, fill_price, fill_quantity, timestamp, ohlcv)
                
                # Check if order is completely filled
                if order.remaining_quantity <= 0:
                    order.status = OrderStatus.FILLED
                    order.filled_time = timestamp
                    orders_to_remove.append(order_id)
                else:
                    order.status = OrderStatus.PARTIAL
            
            # Check for order expiration
            elif self._is_order_expired(order, timestamp):
                order.status = OrderStatus.EXPIRED
                orders_to_remove.append(order_id)
        
        # Remove completed/expired orders
        for order_id in orders_to_remove:
            order = self.pending_orders.pop(order_id)
            self.order_history.append(order)
    
    def _should_fill_order(self, order: Order, timestamp: datetime, ohlcv: Dict[str, float]) -> bool:
        """Determine if an order should be filled"""
        
        current_price = ohlcv['close']
        high_price = ohlcv['high']
        low_price = ohlcv['low']
        
        if order.order_type == OrderType.MARKET:
            return True  # Market orders always fill (in simulation)
        
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return False
            if order.side == OrderSide.BUY:
                # Buy limit order fills if price dropped to or below limit
                return low_price <= order.limit_price
            else:
                # Sell limit order fills if price rose to or above limit
                return high_price >= order.limit_price
        
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                return False
            if order.side == OrderSide.BUY:
                # Buy stop order triggers if price rises above stop price
                return high_price >= order.stop_price
            else:
                # Sell stop order triggers if price falls below stop price
                return low_price <= order.stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                return False
            # First check if stop is triggered
            stop_triggered = False
            if order.side == OrderSide.BUY:
                stop_triggered = high_price >= order.stop_price
            else:
                stop_triggered = low_price <= order.stop_price
            
            if not stop_triggered:
                return False
            
            # Then check if limit can be filled
            if order.side == OrderSide.BUY:
                return low_price <= order.limit_price
            else:
                return high_price >= order.limit_price
        
        return False
    
    def _calculate_fill_price(self, order: Order, ohlcv: Dict[str, float]) -> float:
        """Calculate the fill price for an order"""
        
        base_price = ohlcv['close']
        
        if order.order_type == OrderType.MARKET:
            # Market orders use current price plus slippage
            slippage = self._calculate_slippage(order, ohlcv)
            market_impact = self._calculate_market_impact(order, ohlcv)
            
            if order.side == OrderSide.BUY:
                return base_price * (1 + slippage + market_impact)
            else:
                return base_price * (1 - slippage - market_impact)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill at limit price (or better with probability)
            if self.config.enable_execution_uncertainty and self.rng.random() < 0.1:
                # 10% chance of price improvement
                improvement = self.rng.uniform(0.0001, 0.001)  # 0.01% to 0.1%
                if order.side == OrderSide.BUY:
                    return order.limit_price * (1 - improvement)
                else:
                    return order.limit_price * (1 + improvement)
            else:
                return order.limit_price
        
        else:
            # Stop orders convert to market orders
            return self._calculate_fill_price(
                Order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.remaining_quantity,
                    order_type=OrderType.MARKET,
                    created_time=order.created_time
                ),
                ohlcv
            )
    
    def _calculate_fill_quantity(self, order: Order, ohlcv: Dict[str, float]) -> float:
        """Calculate how much of the order gets filled"""
        
        if not self.config.enable_partial_fills:
            return order.remaining_quantity
        
        # Check liquidity constraints
        if self.config.enable_liquidity_constraints:
            daily_volume = ohlcv.get('volume', 0)
            max_fillable = daily_volume * self.config.max_order_size_pct
            
            if order.remaining_quantity > max_fillable:
                # Partial fill due to liquidity
                return max_fillable
        
        # Random partial fills (simulate market conditions)
        if self.rng.random() < self.config.partial_fill_probability:
            fill_ratio = self.rng.uniform(self.config.min_fill_ratio, 1.0)
            return order.remaining_quantity * fill_ratio
        
        return order.remaining_quantity
    
    def _calculate_slippage(self, order: Order, ohlcv: Dict[str, float]) -> float:
        """Calculate slippage for an order"""
        
        if self.config.slippage_model == SlippageModel.FIXED:
            return self.config.base_slippage
        
        elif self.config.slippage_model == SlippageModel.VOLUME_BASED:
            daily_volume = ohlcv.get('volume', 1)
            volume_ratio = order.remaining_quantity / daily_volume
            return min(self.config.base_slippage * (1 + volume_ratio * 10), self.config.max_slippage)
        
        elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            volatility = self.volatility_cache.get(order.symbol, 0.2)  # Default 20% volatility
            return min(self.config.base_slippage * (1 + volatility), self.config.max_slippage)
        
        elif self.config.slippage_model == SlippageModel.SQRT:
            daily_volume = ohlcv.get('volume', 1)
            volume_ratio = order.remaining_quantity / daily_volume
            return min(self.config.base_slippage * np.sqrt(1 + volume_ratio), self.config.max_slippage)
        
        elif self.config.slippage_model == SlippageModel.CUSTOM and self.config.custom_slippage_func:
            return self.config.custom_slippage_func(order, ohlcv, self.config)
        
        return self.config.base_slippage
    
    def _calculate_market_impact(self, order: Order, ohlcv: Dict[str, float]) -> float:
        """Calculate market impact for an order"""
        
        if self.config.market_impact_model == MarketImpactModel.NONE:
            return 0.0
        
        daily_volume = ohlcv.get('volume', 1)
        volume_ratio = order.remaining_quantity / daily_volume
        
        if self.config.market_impact_model == MarketImpactModel.LINEAR:
            return self.config.impact_coefficient * volume_ratio
        
        elif self.config.market_impact_model == MarketImpactModel.SQRT:
            return self.config.impact_coefficient * np.sqrt(volume_ratio)
        
        elif self.config.market_impact_model == MarketImpactModel.ALMGREN_CHRISS:
            # Simplified Almgren-Chriss model
            volatility = self.volatility_cache.get(order.symbol, 0.2)
            return self.config.impact_coefficient * volatility * np.sqrt(volume_ratio)
        
        elif self.config.market_impact_model == MarketImpactModel.CUSTOM and self.config.custom_impact_func:
            return self.config.custom_impact_func(order, ohlcv, self.config)
        
        return 0.0
    
    def _execute_fill(self, order: Order, fill_price: float, fill_quantity: float, 
                     timestamp: datetime, ohlcv: Dict[str, float]) -> None:
        """Execute a fill for an order"""
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, fill_price)
        
        # Calculate slippage cost
        reference_price = ohlcv['close']
        slippage_cost = abs(fill_price - reference_price) / reference_price
        
        # Create fill record
        fill = Fill(
            fill_id=f"{order.order_id}_{len(order.fills) + 1}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage_cost,
            last_price=reference_price,
            volume_at_fill=ohlcv.get('volume')
        )
        
        # Update order
        order.fills.append(fill.__dict__)
        order.filled_quantity += fill_quantity
        order.remaining_quantity -= fill_quantity
        order.commission_paid += commission
        order.slippage_paid += slippage_cost * fill_quantity * fill_price
        
        # Update average fill price
        total_filled_value = sum(f['quantity'] * f['price'] for f in order.fills)
        order.avg_fill_price = total_filled_value / order.filled_quantity
        
        # Add to fill history
        self.fill_history.append(fill)
        
        # Update performance tracking
        self.total_commission_paid += commission
        self.total_slippage_paid += slippage_cost * fill_quantity * fill_price
        self.fills_executed += 1
        
        self.logger.debug(f"Fill executed: {fill.fill_id} - {fill_quantity} @ {fill_price:.4f}")
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade"""
        
        per_share_commission = quantity * self.config.commission_per_share
        percentage_commission = quantity * price * self.config.commission_percentage
        
        total_commission = per_share_commission + percentage_commission
        
        # Apply min/max limits
        total_commission = max(total_commission, self.config.min_commission)
        total_commission = min(total_commission, self.config.max_commission)
        
        return total_commission
    
    def _is_order_expired(self, order: Order, current_time: datetime) -> bool:
        """Check if an order has expired"""
        
        if order.time_in_force == "DAY":
            # Assume market closes at 4 PM
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            return current_time >= market_close
        
        elif order.good_till_date:
            return current_time >= order.good_till_date
        
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        
        return {
            "orders_processed": self.orders_processed,
            "fills_executed": self.fills_executed,
            "total_commission_paid": self.total_commission_paid,
            "total_slippage_paid": self.total_slippage_paid,
            "total_market_impact": self.total_market_impact,
            "pending_orders": len(self.pending_orders),
            "avg_commission_per_fill": self.total_commission_paid / max(self.fills_executed, 1),
            "avg_slippage_per_fill": self.total_slippage_paid / max(self.fills_executed, 1)
        }
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get the status of an order"""
        
        # Check pending orders
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        
        # Check order history
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol"""
        
        orders = list(self.pending_orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    def get_fill_history(self, symbol: Optional[str] = None, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Fill]:
        """Get fill history with optional filters"""
        
        fills = self.fill_history
        
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        
        if start_date:
            fills = [f for f in fills if f.timestamp >= start_date]
        
        if end_date:
            fills = [f for f in fills if f.timestamp <= end_date]
        
        return fills 
