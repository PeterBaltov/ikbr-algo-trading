"""
Portfolio Management Module

Advanced portfolio management system with position sizing, capital allocation,
risk budgeting, and sophisticated rebalancing strategies for backtesting.

Features:
- Multiple position sizing methods (equal weight, risk parity, kelly criterion)
- Dynamic capital allocation with rebalancing schedules
- Risk budgeting and exposure management
- Leverage and margin rules enforcement
- Performance attribution analysis
- Portfolio optimization integration

Integration:
- Works with backtesting engine for real-time management
- Supports multi-strategy coordination
- Compatible with risk management framework
- Provides foundation for live trading deployment
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
import logging
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np

from ..strategies.enums import TimeFrame


class PositionSizingMethod(Enum):
    """Position sizing methodologies"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    FIXED_DOLLAR = "fixed_dollar"
    PERCENT_OF_PORTFOLIO = "percent_of_portfolio"
    CUSTOM = "custom"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies"""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    THRESHOLD_BASED = "threshold_based"
    CUSTOM = "custom"


class AllocationConstraint(Enum):
    """Types of allocation constraints"""
    MIN_WEIGHT = "min_weight"
    MAX_WEIGHT = "max_weight"
    SECTOR_LIMIT = "sector_limit"
    CORRELATION_LIMIT = "correlation_limit"
    LIQUIDITY_REQUIREMENT = "liquidity_requirement"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    
    method: PositionSizingMethod = PositionSizingMethod.EQUAL_WEIGHT
    target_volatility: float = 0.15  # 15% target volatility
    max_position_size: float = 0.10  # 10% max per position
    min_position_size: float = 0.01  # 1% min per position
    
    # Kelly criterion parameters
    kelly_lookback_days: int = 252
    kelly_max_leverage: float = 2.0
    
    # Risk parity parameters
    risk_lookback_days: int = 252
    risk_adjustment_factor: float = 1.0
    
    # Custom sizing function
    custom_sizing_func: Optional[Callable] = None
    
    # Constraints
    constraints: Dict[AllocationConstraint, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RebalanceConfig:
    """Configuration for portfolio rebalancing"""
    
    frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    threshold: float = 0.05  # 5% drift threshold for threshold-based rebalancing
    
    # Timing
    rebalance_time: str = "market_close"  # market_open, market_close, specific_time
    specific_time: Optional[datetime] = None
    
    # Constraints
    min_trade_size: float = 100.0  # Minimum dollar amount to trade
    max_turnover: float = 0.20  # Maximum 20% portfolio turnover per rebalance
    
    # Transaction costs
    include_transaction_costs: bool = True
    cost_threshold: float = 0.001  # Don't rebalance if costs > 0.1%
    
    # Custom logic
    custom_rebalance_func: Optional[Callable] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioAllocation:
    """Represents target portfolio allocation"""
    
    timestamp: datetime
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    
    # Allocation metrics
    total_value: float
    cash_weight: float
    leverage: float
    
    # Risk metrics
    estimated_volatility: float
    estimated_return: float
    max_drawdown_estimate: float
    
    # Constraints
    constraint_violations: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSize:
    """Calculated position size for a symbol"""
    
    symbol: str
    target_weight: float
    target_dollar_amount: float
    target_shares: int
    
    # Current state
    current_weight: float
    current_dollar_amount: float
    current_shares: int
    
    # Sizing rationale
    sizing_method: PositionSizingMethod
    confidence: float
    risk_contribution: float
    
    # Trade details
    trade_required: bool
    trade_amount: float
    trade_shares: int
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionSizer:
    """Calculates position sizes using various methodologies"""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Historical data cache
        self.price_history: Dict[str, pd.Series] = {}
        self.return_history: Dict[str, pd.Series] = {}
        self.volatility_cache: Dict[str, float] = {}
        
    def calculate_position_sizes(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, PositionSize]:
        """Calculate optimal position sizes for all symbols"""
        
        if self.config.method == PositionSizingMethod.EQUAL_WEIGHT:
            return self._equal_weight_sizing(symbols, portfolio_value, current_positions, market_data)
        elif self.config.method == PositionSizingMethod.RISK_PARITY:
            return self._risk_parity_sizing(symbols, portfolio_value, current_positions, market_data)
        elif self.config.method == PositionSizingMethod.KELLY_CRITERION:
            return self._kelly_criterion_sizing(symbols, portfolio_value, current_positions, market_data)
        elif self.config.method == PositionSizingMethod.VOLATILITY_TARGET:
            return self._volatility_target_sizing(symbols, portfolio_value, current_positions, market_data)
        elif self.config.method == PositionSizingMethod.CUSTOM and self.config.custom_sizing_func:
            return self.config.custom_sizing_func(symbols, portfolio_value, current_positions, market_data, self.config)
        else:
            raise ValueError(f"Unsupported position sizing method: {self.config.method}")
    
    def _equal_weight_sizing(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, PositionSize]:
        """Equal weight position sizing"""
        
        position_sizes = {}
        equal_weight = 1.0 / len(symbols) if symbols else 0.0
        
        for symbol in symbols:
            current_value = current_positions.get(symbol, 0.0)
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Apply constraints
            target_weight = max(self.config.min_position_size, 
                              min(self.config.max_position_size, equal_weight))
            
            target_dollar = target_weight * portfolio_value
            
            # Get current price
            current_price = market_data.get(symbol, {}).get('close', 1.0)
            target_shares = int(target_dollar / current_price) if current_price > 0 else 0
            
            # Calculate trade requirements
            trade_dollar = target_dollar - current_value
            trade_shares = target_shares - int(current_value / current_price) if current_price > 0 else 0
            trade_required = abs(trade_dollar) > 100.0  # $100 minimum trade
            
            position_sizes[symbol] = PositionSize(
                symbol=symbol,
                target_weight=target_weight,
                target_dollar_amount=target_dollar,
                target_shares=target_shares,
                current_weight=current_weight,
                current_dollar_amount=current_value,
                current_shares=int(current_value / current_price) if current_price > 0 else 0,
                sizing_method=PositionSizingMethod.EQUAL_WEIGHT,
                confidence=1.0,
                risk_contribution=target_weight,
                trade_required=trade_required,
                trade_amount=trade_dollar,
                trade_shares=trade_shares
            )
        
        return position_sizes
    
    def _risk_parity_sizing(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, PositionSize]:
        """Risk parity position sizing"""
        
        # Calculate volatilities
        volatilities = {}
        for symbol in symbols:
            vol = self._calculate_volatility(symbol)
            volatilities[symbol] = vol if vol > 0 else 0.2  # Default 20% vol
        
        # Calculate inverse volatility weights
        inv_vol_weights = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol_weights.values())
        
        # Normalize weights
        risk_parity_weights = {symbol: weight / total_inv_vol for symbol, weight in inv_vol_weights.items()}
        
        position_sizes = {}
        
        for symbol in symbols:
            current_value = current_positions.get(symbol, 0.0)
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Apply constraints
            target_weight = max(self.config.min_position_size, 
                              min(self.config.max_position_size, risk_parity_weights[symbol]))
            
            target_dollar = target_weight * portfolio_value
            
            # Get current price
            current_price = market_data.get(symbol, {}).get('close', 1.0)
            target_shares = int(target_dollar / current_price) if current_price > 0 else 0
            
            # Calculate trade requirements
            trade_dollar = target_dollar - current_value
            trade_shares = target_shares - int(current_value / current_price) if current_price > 0 else 0
            trade_required = abs(trade_dollar) > 100.0
            
            position_sizes[symbol] = PositionSize(
                symbol=symbol,
                target_weight=target_weight,
                target_dollar_amount=target_dollar,
                target_shares=target_shares,
                current_weight=current_weight,
                current_dollar_amount=current_value,
                current_shares=int(current_value / current_price) if current_price > 0 else 0,
                sizing_method=PositionSizingMethod.RISK_PARITY,
                confidence=0.8,  # Risk parity confidence
                risk_contribution=target_weight * volatilities[symbol],
                trade_required=trade_required,
                trade_amount=trade_dollar,
                trade_shares=trade_shares
            )
        
        return position_sizes
    
    def _kelly_criterion_sizing(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, PositionSize]:
        """Kelly criterion position sizing"""
        
        kelly_weights = {}
        
        for symbol in symbols:
            # Calculate Kelly weight
            returns = self._get_return_history(symbol)
            if len(returns) < 30:  # Need minimum history
                kelly_weights[symbol] = 1.0 / len(symbols)  # Fall back to equal weight
                continue
                
            mean_return = returns.mean()
            variance = returns.var()
            
            if variance > 0:
                kelly_weight = mean_return / variance
                # Apply leverage limit
                kelly_weight = min(kelly_weight, self.config.kelly_max_leverage / len(symbols))
                kelly_weight = max(kelly_weight, 0.0)  # No shorts
            else:
                kelly_weight = 1.0 / len(symbols)
                
            kelly_weights[symbol] = kelly_weight
        
        # Normalize weights
        total_weight = sum(kelly_weights.values())
        if total_weight > 0:
            kelly_weights = {symbol: weight / total_weight for symbol, weight in kelly_weights.items()}
        
        position_sizes = {}
        
        for symbol in symbols:
            current_value = current_positions.get(symbol, 0.0)
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Apply constraints
            target_weight = max(self.config.min_position_size, 
                              min(self.config.max_position_size, kelly_weights[symbol]))
            
            target_dollar = target_weight * portfolio_value
            
            # Get current price
            current_price = market_data.get(symbol, {}).get('close', 1.0)
            target_shares = int(target_dollar / current_price) if current_price > 0 else 0
            
            # Calculate trade requirements
            trade_dollar = target_dollar - current_value
            trade_shares = target_shares - int(current_value / current_price) if current_price > 0 else 0
            trade_required = abs(trade_dollar) > 100.0
            
            position_sizes[symbol] = PositionSize(
                symbol=symbol,
                target_weight=target_weight,
                target_dollar_amount=target_dollar,
                target_shares=target_shares,
                current_weight=current_weight,
                current_dollar_amount=current_value,
                current_shares=int(current_value / current_price) if current_price > 0 else 0,
                sizing_method=PositionSizingMethod.KELLY_CRITERION,
                confidence=0.7,  # Kelly confidence
                risk_contribution=target_weight,
                trade_required=trade_required,
                trade_amount=trade_dollar,
                trade_shares=trade_shares
            )
        
        return position_sizes
    
    def _volatility_target_sizing(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, PositionSize]:
        """Volatility target position sizing"""
        
        target_vol = self.config.target_volatility
        vol_weights = {}
        
        for symbol in symbols:
            vol = self._calculate_volatility(symbol)
            if vol > 0:
                # Weight inversely proportional to volatility to achieve target
                vol_weight = target_vol / vol
                vol_weight = min(vol_weight, self.config.max_position_size)
            else:
                vol_weight = 1.0 / len(symbols)
                
            vol_weights[symbol] = vol_weight
        
        # Normalize weights
        total_weight = sum(vol_weights.values())
        if total_weight > 0:
            vol_weights = {symbol: weight / total_weight for symbol, weight in vol_weights.items()}
        
        position_sizes = {}
        
        for symbol in symbols:
            current_value = current_positions.get(symbol, 0.0)
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Apply constraints
            target_weight = max(self.config.min_position_size, 
                              min(self.config.max_position_size, vol_weights[symbol]))
            
            target_dollar = target_weight * portfolio_value
            
            # Get current price
            current_price = market_data.get(symbol, {}).get('close', 1.0)
            target_shares = int(target_dollar / current_price) if current_price > 0 else 0
            
            # Calculate trade requirements
            trade_dollar = target_dollar - current_value
            trade_shares = target_shares - int(current_value / current_price) if current_price > 0 else 0
            trade_required = abs(trade_dollar) > 100.0
            
            position_sizes[symbol] = PositionSize(
                symbol=symbol,
                target_weight=target_weight,
                target_dollar_amount=target_dollar,
                target_shares=target_shares,
                current_weight=current_weight,
                current_dollar_amount=current_value,
                current_shares=int(current_value / current_price) if current_price > 0 else 0,
                sizing_method=PositionSizingMethod.VOLATILITY_TARGET,
                confidence=0.8,
                risk_contribution=target_weight * self._calculate_volatility(symbol),
                trade_required=trade_required,
                trade_amount=trade_dollar,
                trade_shares=trade_shares
            )
        
        return position_sizes
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate historical volatility for a symbol"""
        
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        returns = self._get_return_history(symbol)
        if len(returns) < 30:
            vol = 0.2  # Default 20% volatility
        else:
            vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        self.volatility_cache[symbol] = vol
        return vol
    
    def _get_return_history(self, symbol: str) -> pd.Series:
        """Get historical returns for a symbol"""
        
        if symbol in self.return_history:
            return self.return_history[symbol]
        
        # This would normally come from market data
        # For now, return empty series
        returns = pd.Series(dtype=float)
        self.return_history[symbol] = returns
        return returns


class CapitalAllocator:
    """Manages capital allocation across strategies and assets"""
    
    def __init__(self, position_config: PositionSizingConfig, rebalance_config: RebalanceConfig):
        self.position_config = position_config
        self.rebalance_config = rebalance_config
        self.position_sizer = PositionSizer(position_config)
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.last_rebalance: Optional[datetime] = None
        self.allocation_history: List[PortfolioAllocation] = []
        
    def allocate_capital(
        self,
        current_time: datetime,
        portfolio_value: float,
        current_positions: Dict[str, float],
        available_symbols: List[str],
        market_data: Dict[str, Dict[str, float]]
    ) -> Optional[PortfolioAllocation]:
        """Determine optimal capital allocation"""
        
        # Check if rebalancing is needed
        if not self._should_rebalance(current_time):
            return None
        
        self.logger.info(f"Rebalancing portfolio at {current_time}")
        
        # Calculate position sizes
        position_sizes = self.position_sizer.calculate_position_sizes(
            available_symbols, portfolio_value, current_positions, market_data
        )
        
        # Build allocation
        target_weights = {symbol: pos.target_weight for symbol, pos in position_sizes.items()}
        current_weights = {symbol: pos.current_weight for symbol, pos in position_sizes.items()}
        
        # Calculate portfolio metrics
        total_target_weight = sum(target_weights.values())
        cash_weight = max(0.0, 1.0 - total_target_weight)
        leverage = total_target_weight
        
        # Estimate risk metrics (simplified)
        estimated_volatility = self._estimate_portfolio_volatility(target_weights, market_data)
        estimated_return = self._estimate_portfolio_return(target_weights, market_data)
        
        allocation = PortfolioAllocation(
            timestamp=current_time,
            target_weights=target_weights,
            current_weights=current_weights,
            total_value=portfolio_value,
            cash_weight=cash_weight,
            leverage=leverage,
            estimated_volatility=estimated_volatility,
            estimated_return=estimated_return,
            max_drawdown_estimate=estimated_volatility * 2.0  # Simple estimate
        )
        
        # Validate constraints
        self._validate_allocation_constraints(allocation)
        
        # Update state
        self.last_rebalance = current_time
        self.allocation_history.append(allocation)
        
        return allocation
    
    def _should_rebalance(self, current_time: datetime) -> bool:
        """Determine if portfolio should be rebalanced"""
        
        if self.last_rebalance is None:
            return True  # First allocation
        
        if self.rebalance_config.frequency == RebalanceFrequency.NEVER:
            return False
        elif self.rebalance_config.frequency == RebalanceFrequency.DAILY:
            return True
        elif self.rebalance_config.frequency == RebalanceFrequency.WEEKLY:
            return (current_time - self.last_rebalance).days >= 7
        elif self.rebalance_config.frequency == RebalanceFrequency.MONTHLY:
            return current_time.month != self.last_rebalance.month or current_time.year != self.last_rebalance.year
        elif self.rebalance_config.frequency == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_time.month - 1) // 3
            last_quarter = (self.last_rebalance.month - 1) // 3
            return current_quarter != last_quarter or current_time.year != self.last_rebalance.year
        elif self.rebalance_config.frequency == RebalanceFrequency.ANNUALLY:
            return current_time.year != self.last_rebalance.year
        
        return False
    
    def _estimate_portfolio_volatility(self, weights: Dict[str, float], market_data: Dict[str, Dict[str, float]]) -> float:
        """Estimate portfolio volatility (simplified)"""
        
        # Simple weighted average volatility
        total_vol = 0.0
        for symbol, weight in weights.items():
            symbol_vol = self.position_sizer._calculate_volatility(symbol)
            total_vol += weight * symbol_vol
        
        return total_vol
    
    def _estimate_portfolio_return(self, weights: Dict[str, float], market_data: Dict[str, Dict[str, float]]) -> float:
        """Estimate portfolio return (simplified)"""
        
        # Simple assumption of 8% annual return
        return 0.08
    
    def _validate_allocation_constraints(self, allocation: PortfolioAllocation) -> None:
        """Validate allocation against constraints"""
        
        violations = []
        
        # Check maximum position sizes
        for symbol, weight in allocation.target_weights.items():
            if weight > self.position_config.max_position_size:
                violations.append(f"{symbol} weight {weight:.1%} exceeds max {self.position_config.max_position_size:.1%}")
        
        # Check leverage
        total_weight = sum(allocation.target_weights.values())
        if total_weight > 1.2:  # 120% max leverage
            violations.append(f"Total leverage {total_weight:.1%} exceeds 120%")
        
        allocation.constraint_violations = violations


class PortfolioManager:
    """Main portfolio management class"""
    
    def __init__(self, position_config: PositionSizingConfig, rebalance_config: RebalanceConfig):
        self.capital_allocator = CapitalAllocator(position_config, rebalance_config)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.allocation_performance: List[Dict[str, Any]] = []
    
    def manage_portfolio(
        self,
        current_time: datetime,
        portfolio_value: float,
        current_positions: Dict[str, float],
        available_symbols: List[str],
        market_data: Dict[str, Dict[str, float]]
    ) -> Optional[PortfolioAllocation]:
        """Main portfolio management function"""
        
        return self.capital_allocator.allocate_capital(
            current_time, portfolio_value, current_positions, available_symbols, market_data
        )
    
    def get_allocation_history(self) -> List[PortfolioAllocation]:
        """Get history of portfolio allocations"""
        return self.capital_allocator.allocation_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get portfolio management performance metrics"""
        
        history = self.get_allocation_history()
        
        if not history:
            return {}
        
        return {
            "total_rebalances": len(history),
            "average_leverage": np.mean([alloc.leverage for alloc in history]),
            "average_cash_weight": np.mean([alloc.cash_weight for alloc in history]),
            "constraint_violations": sum(len(alloc.constraint_violations) for alloc in history)
        }


def create_portfolio_config(
    sizing_method: PositionSizingMethod = PositionSizingMethod.EQUAL_WEIGHT,
    max_position_size: float = 0.10,
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
    **kwargs
) -> Tuple[PositionSizingConfig, RebalanceConfig]:
    """Create portfolio configuration with sensible defaults"""
    
    position_config = PositionSizingConfig(
        method=sizing_method,
        max_position_size=max_position_size,
        **{k: v for k, v in kwargs.items() if k in PositionSizingConfig.__annotations__}
    )
    
    rebalance_config = RebalanceConfig(
        frequency=rebalance_frequency,
        **{k: v for k, v in kwargs.items() if k in RebalanceConfig.__annotations__}
    )
    
    return position_config, rebalance_config 
