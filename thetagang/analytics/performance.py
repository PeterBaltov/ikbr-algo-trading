"""
Core Performance Metrics Module

Provides essential performance calculations for trading strategies including
return metrics, risk-adjusted performance measures, and trade statistics.

Features:
- Return metrics (Total, Annualized, Sharpe, Sortino, Calmar)
- Risk-adjusted performance measures
- Trade-level statistics and analysis  
- Time-based performance breakdown
- Rolling performance metrics

Integration:
- Used by backtesting engine for strategy evaluation
- Supports benchmark comparison workflows
- Compatible with risk analysis modules
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple
import logging

import pandas as pd
import numpy as np


class ReturnMetric(Enum):
    """Types of return metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    TREYNOR_RATIO = "treynor_ratio"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    
    # Basic return metrics
    total_return: float
    annualized_return: float
    volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    downside_deviation: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Time-based metrics
    best_month: float
    worst_month: float
    best_year: float
    worst_year: float
    
    # Additional metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    # Metadata
    start_date: datetime
    end_date: datetime
    total_days: int
    trading_days: int
    
    # Optional benchmark comparison
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None


class PerformanceCalculator:
    """Core performance calculation engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_performance_metrics(
        self,
        equity_curve: pd.Series,
        trades: Optional[List[Dict[str, Any]]] = None,
        benchmark: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if equity_curve.empty:
            raise ValueError("Equity curve cannot be empty")
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        trading_days = len(returns)
        
        # Handle datetime index properly
        if hasattr(equity_curve.index[0], 'to_pydatetime'):
            start_date = equity_curve.index[0].to_pydatetime()
            end_date = equity_curve.index[-1].to_pydatetime()
            calendar_days = (end_date - start_date).days
        else:
            start_date = datetime.now()
            end_date = datetime.now()
            calendar_days = trading_days
        
        # Annualized return
        if trading_days > 0:
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        else:
            annualized_return = 0.0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Risk-adjusted metrics
        excess_returns = returns - (self.risk_free_rate / 252)
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = float('inf') if excess_returns.mean() > 0 else 0.0
            downside_deviation = 0.0
        
        # Max drawdown and Calmar ratio
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Risk metrics
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0.0
        cvar_95_value = returns[returns <= var_95].mean() if len(returns) > 0 and var_95 < 0 else 0.0
        if isinstance(cvar_95_value, pd.Series):
            cvar_95 = cvar_95_value.iloc[0] if len(cvar_95_value) > 0 else 0.0
        else:
            cvar_95 = float(cvar_95_value)
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades) if trades else {}
        
        # Time-based performance
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        yearly_returns = self._calculate_yearly_returns(equity_curve)
        
        # Distribution metrics
        skewness_value = returns.skew() if len(returns) > 2 else 0.0
        kurtosis_value = returns.kurtosis() if len(returns) > 3 else 0.0
        
        # Convert pandas scalar to float if needed
        if isinstance(skewness_value, pd.Series):
            skewness = float(skewness_value.iloc[0]) if len(skewness_value) > 0 else 0.0
        else:
            skewness = float(skewness_value)
            
        if isinstance(kurtosis_value, pd.Series):
            kurtosis = float(kurtosis_value.iloc[0]) if len(kurtosis_value) > 0 else 0.0
        else:
            kurtosis = float(kurtosis_value)
        
        # Tail ratio
        if len(returns) > 0:
            tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 1.0
        else:
            tail_ratio = 1.0
        
        # Benchmark comparison if provided
        benchmark_metrics = {}
        if benchmark is not None:
            benchmark_metrics = self._calculate_benchmark_comparison(equity_curve, benchmark)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            downside_deviation=downside_deviation,
            total_trades=int(trade_stats.get('total_trades', 0)),
            winning_trades=int(trade_stats.get('winning_trades', 0)),
            losing_trades=int(trade_stats.get('losing_trades', 0)),
            win_rate=float(trade_stats.get('win_rate', 0.0)),
            profit_factor=float(trade_stats.get('profit_factor', 0.0)),
            avg_win=float(trade_stats.get('avg_win', 0.0)),
            avg_loss=float(trade_stats.get('avg_loss', 0.0)),
            largest_win=float(trade_stats.get('largest_win', 0.0)),
            largest_loss=float(trade_stats.get('largest_loss', 0.0)),
            best_month=monthly_returns.max() if len(monthly_returns) > 0 else 0.0,
            worst_month=monthly_returns.min() if len(monthly_returns) > 0 else 0.0,
            best_year=yearly_returns.max() if len(yearly_returns) > 0 else 0.0,
            worst_year=yearly_returns.min() if len(yearly_returns) > 0 else 0.0,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            start_date=start_date,
            end_date=end_date,
            total_days=calendar_days,
            trading_days=trading_days,
            **benchmark_metrics
        )
    
    def _calculate_trade_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trade-level statistics"""
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Extract P&L from trades
        pnls = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
        
        if not pnls:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        largest_win = float(max(pnls)) if pnls else 0.0
        largest_loss = float(min(pnls)) if pnls else 0.0
        
        return {
            'total_trades': float(total_trades),
            'winning_trades': float(winning_trades),
            'losing_trades': float(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns"""
        
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        return monthly_returns
    
    def _calculate_yearly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate yearly returns"""
        
        yearly_equity = equity_curve.resample('Y').last()
        yearly_returns = yearly_equity.pct_change().dropna()
        return yearly_returns
    
    def _calculate_benchmark_comparison(self, equity_curve: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
        """Calculate metrics relative to benchmark"""
        
        # Align dates
        common_dates = equity_curve.index.intersection(benchmark.index)
        if len(common_dates) == 0:
            return {}
        
        strategy_aligned = equity_curve.loc[common_dates]
        benchmark_aligned = benchmark.loc[common_dates]
        
        # Calculate returns
        strategy_returns = strategy_aligned.pct_change().dropna()
        benchmark_returns = benchmark_aligned.pct_change().dropna()
        
        # Align returns
        common_return_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_return_dates) == 0:
            return {}
        
        strategy_returns = strategy_returns.loc[common_return_dates]
        benchmark_returns = benchmark_returns.loc[common_return_dates]
        
        # Benchmark return
        benchmark_total_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
        
        # Beta calculation
        if len(strategy_returns) > 1 and benchmark_returns.std() > 0:
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 0.0
        
        # Alpha calculation (annualized)
        strategy_annualized = (1 + strategy_returns.mean()) ** 252 - 1
        benchmark_annualized = (1 + benchmark_returns.mean()) ** 252 - 1
        alpha = strategy_annualized - (self.risk_free_rate + beta * (benchmark_annualized - self.risk_free_rate))
        
        # Tracking error
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
        
        return {
            'benchmark_return': float(benchmark_total_return),
            'alpha': float(alpha),
            'beta': float(beta),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio)
        } 
