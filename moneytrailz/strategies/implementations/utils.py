"""
🎯 STRATEGY UTILITIES IMPLEMENTATION
===================================

Utility classes and functions for strategy implementations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

import pandas as pd
import numpy as np


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    size: float
    risk_level: RiskLevel
    explanation: str
    max_size: float
    recommended_size: float


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    value_at_risk: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_return: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    trades_count: int = 0


class PositionSizer:
    """Advanced position sizing utility"""
    
    def __init__(self, max_position_size: float = 0.10, risk_per_trade: float = 0.02):
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, account_value: float, entry_price: float,
                              stop_loss_price: float, volatility: Optional[float] = None) -> PositionSizeResult:
        """Calculate optimal position size"""
        # Risk-based sizing
        risk_per_share = abs(entry_price - stop_loss_price)
        max_risk_amount = account_value * self.risk_per_trade
        
        if risk_per_share > 0:
            risk_based_size = max_risk_amount / risk_per_share / entry_price
        else:
            risk_based_size = self.max_position_size
        
        # Volatility adjustment
        if volatility:
            vol_adjustment = min(1.0, 0.02 / volatility)  # Reduce size for high volatility
            risk_based_size *= vol_adjustment
        
        # Apply maximum constraints
        final_size = min(risk_based_size, self.max_position_size)
        
        # Determine risk level
        risk_level = self._classify_risk_level(final_size, volatility or 0.02)
        
        return PositionSizeResult(
            size=final_size,
            risk_level=risk_level,
            explanation=f"Risk-based sizing with {self.risk_per_trade:.1%} risk per trade",
            max_size=self.max_position_size,
            recommended_size=final_size
        )
    
    def _classify_risk_level(self, position_size: float, volatility: float) -> RiskLevel:
        """Classify risk level based on position size and volatility"""
        risk_score = position_size * volatility * 100
        
        if risk_score < 0.5:
            return RiskLevel.LOW
        elif risk_score < 1.0:
            return RiskLevel.MEDIUM
        elif risk_score < 2.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME


class RiskManager:
    """Comprehensive risk management utility"""
    
    def __init__(self, max_portfolio_risk: float = 0.20):
        self.max_portfolio_risk = max_portfolio_risk
        self.logger = logging.getLogger(__name__)
    
    def calculate_portfolio_risk(self, positions: Dict[str, Any]) -> RiskMetrics:
        """Calculate portfolio-level risk metrics"""
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        
        if total_value == 0:
            return RiskMetrics()
        
        # Calculate weighted volatility
        weighted_vol = 0.0
        for symbol, position in positions.items():
            weight = position.get('value', 0) / total_value
            vol = position.get('volatility', 0.02)
            weighted_vol += weight * vol
        
        # Estimate portfolio VaR (simplified)
        var_95 = weighted_vol * 1.65 * np.sqrt(total_value)
        
        return RiskMetrics(
            value_at_risk=var_95,
            volatility=weighted_vol,
            risk_level=self._classify_portfolio_risk(weighted_vol)
        )
    
    def _classify_portfolio_risk(self, volatility: float) -> RiskLevel:
        """Classify portfolio risk level"""
        if volatility < 0.01:
            return RiskLevel.LOW
        elif volatility < 0.02:
            return RiskLevel.MEDIUM
        elif volatility < 0.04:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def check_risk_limits(self, position_size: float, current_exposure: float) -> bool:
        """Check if position passes risk limits"""
        total_exposure = current_exposure + position_size
        return total_exposure <= self.max_portfolio_risk


class SignalFilter:
    """Signal filtering and validation utility"""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
    
    def filter_signal(self, signal_strength: float, market_conditions: Dict[str, Any]) -> bool:
        """Filter signals based on confidence and market conditions"""
        # Basic confidence filter
        if signal_strength < self.min_confidence:
            return False
        
        # Market condition filters
        volatility = market_conditions.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility filter
            return signal_strength > self.min_confidence * 1.2
        
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        if volume_ratio < 0.5:  # Low volume filter
            return False
        
        return True
    
    def combine_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple signals into a consensus"""
        if not signals:
            return {'confidence': 0.0, 'signal': 'HOLD'}
        
        # Calculate weighted confidence
        total_weight = sum(s.get('weight', 1.0) for s in signals)
        weighted_confidence = sum(
            s.get('confidence', 0.0) * s.get('weight', 1.0) 
            for s in signals
        ) / total_weight
        
        # Determine consensus signal
        buy_signals = sum(1 for s in signals if s.get('signal') == 'BUY')
        sell_signals = sum(1 for s in signals if s.get('signal') == 'SELL')
        
        if buy_signals > sell_signals:
            consensus_signal = 'BUY'
        elif sell_signals > buy_signals:
            consensus_signal = 'SELL'
        else:
            consensus_signal = 'HOLD'
        
        return {
            'confidence': weighted_confidence,
            'signal': consensus_signal,
            'signal_count': len(signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }


class PerformanceTracker:
    """Performance tracking and analysis utility"""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_trade(self, symbol: str, entry_price: float, exit_price: float,
                  quantity: float, entry_time: datetime, exit_time: datetime):
        """Add a completed trade"""
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price
        
        trade = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_period': (exit_time - entry_time).days,
            'is_winner': pnl > 0
        }
        
        self.trades.append(trade)
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return PerformanceMetrics()
        
        # Basic metrics
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = sum(t['pnl_pct'] for t in self.trades)
        
        winners = [t for t in self.trades if t['is_winner']]
        losers = [t for t in self.trades if not t['is_winner']]
        
        win_rate = len(winners) / len(self.trades) if self.trades else 0
        avg_win = float(np.mean([t['pnl_pct'] for t in winners])) if winners else 0.0
        avg_loss = float(np.mean([t['pnl_pct'] for t in losers])) if losers else 0.0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consecutive losses
        max_consecutive_losses = self._calculate_max_consecutive_losses()
        
        return PerformanceMetrics(
            total_return=total_return,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses,
            trades_count=len(self.trades)
        )
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if not trade['is_winner']:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


class StrategyUtils:
    """General strategy utilities"""
    
    @staticmethod
    def calculate_correlation(prices1: pd.Series, prices2: pd.Series) -> float:
        """Calculate correlation between two price series"""
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()
        
        if len(returns1) < 2 or len(returns2) < 2:
            return 0.0
        
        # Align series
        aligned = pd.concat([returns1, returns2], axis=1, join='inner')
        if len(aligned) < 2:
            return 0.0
        
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        aligned = pd.concat([asset_returns, market_returns], axis=1, join='inner')
        if len(aligned) < 10:
            return 1.0
        
        asset_ret = aligned.iloc[:, 0]
        market_ret = aligned.iloc[:, 1]
        
        covariance = asset_ret.cov(market_ret)
        market_variance = market_ret.var()
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    @staticmethod
    def normalize_signal(value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range"""
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    @staticmethod
    def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        if avg_loss == 0:
            return 0.0
        
        win_prob = win_rate
        loss_prob = 1 - win_rate
        win_loss_ratio = avg_win / abs(avg_loss)
        
        kelly = win_prob - (loss_prob / win_loss_ratio)
        return max(0.0, min(0.25, kelly))  # Cap at 25% 
