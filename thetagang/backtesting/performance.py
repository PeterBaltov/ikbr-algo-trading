"""
Performance Calculation Module

Comprehensive performance metrics and risk analytics for backtesting results.
Provides industry-standard metrics and custom analytics for strategy evaluation.

Features:
- Return metrics (Total, Annualized, Sharpe, Sortino, Calmar)
- Risk metrics (VaR, CVaR, Max Drawdown, Volatility)
- Trade statistics (Win rate, Profit factor, Average win/loss)
- Benchmark comparison and relative performance
- Rolling performance analysis
- Monte Carlo simulation support

Integration:
- Works with backtesting engine results
- Supports multi-strategy analysis
- Compatible with external benchmarks
- Provides visualization-ready outputs
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple
import logging
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats


class RiskMetric(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_deviation"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"


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


@dataclass
class RiskMetrics:
    """Detailed risk analysis metrics"""
    
    # Value at Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Drawdown analysis
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    recovery_time: int
    
    # Volatility metrics
    volatility: float
    downside_volatility: float
    upside_volatility: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Tail risk
    tail_ratio: float  # 95th percentile / 5th percentile
    gain_pain_ratio: float
    
    # Rolling metrics
    rolling_sharpe: pd.Series = field(default_factory=pd.Series)
    rolling_volatility: pd.Series = field(default_factory=pd.Series)
    rolling_drawdown: pd.Series = field(default_factory=pd.Series)


class PerformanceCalculator:
    """Main performance calculation engine"""
    
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
        calendar_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        
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
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 and var_95 < 0 else 0.0
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades) if trades else {}
        
        # Time-based performance
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        yearly_returns = self._calculate_yearly_returns(equity_curve)
        
        # Distribution metrics
        skewness = returns.skew() if len(returns) > 2 else 0.0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0.0
        
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
            total_trades=trade_stats.get('total_trades', 0),
            winning_trades=trade_stats.get('winning_trades', 0),
            losing_trades=trade_stats.get('losing_trades', 0),
            win_rate=trade_stats.get('win_rate', 0.0),
            profit_factor=trade_stats.get('profit_factor', 0.0),
            avg_win=trade_stats.get('avg_win', 0.0),
            avg_loss=trade_stats.get('avg_loss', 0.0),
            largest_win=trade_stats.get('largest_win', 0.0),
            largest_loss=trade_stats.get('largest_loss', 0.0),
            best_month=monthly_returns.max() if len(monthly_returns) > 0 else 0.0,
            worst_month=monthly_returns.min() if len(monthly_returns) > 0 else 0.0,
            best_year=yearly_returns.max() if len(yearly_returns) > 0 else 0.0,
            worst_year=yearly_returns.min() if len(yearly_returns) > 0 else 0.0,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            start_date=equity_curve.index[0],
            end_date=equity_curve.index[-1],
            total_days=calendar_days,
            trading_days=trading_days,
            **benchmark_metrics
        )
    
    def calculate_risk_metrics(self, equity_curve: pd.Series, window: int = 252) -> RiskMetrics:
        """Calculate detailed risk metrics"""
        
        returns = equity_curve.pct_change().dropna()
        
        # VaR calculations
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # CVaR calculations
        cvar_95 = returns[returns <= var_95].mean() if var_95 < 0 else 0.0
        cvar_99 = returns[returns <= var_99].mean() if var_99 < 0 else 0.0
        
        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
        
        # Drawdown duration analysis
        dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        upside_returns = returns[returns > 0]
        
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        upside_volatility = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0.0
        
        # Distribution analysis
        skewness = returns.skew() if len(returns) > 2 else 0.0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0.0
        
        # Jarque-Bera test for normality
        if len(returns) >= 8:  # Minimum sample size for JB test
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
        else:
            jb_stat, jb_pvalue = 0.0, 1.0
        
        # Tail risk metrics
        tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 1.0
        
        # Gain-Pain ratio
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        gain_pain_ratio = gains / losses if losses > 0 else float('inf') if gains > 0 else 0.0
        
        # Rolling metrics
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window)
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252)
        rolling_drawdown = self._calculate_rolling_drawdown(equity_curve, window)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=dd_duration['max_duration'],
            recovery_time=dd_duration['avg_recovery'],
            volatility=volatility,
            downside_volatility=downside_volatility,
            upside_volatility=upside_volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            tail_ratio=tail_ratio,
            gain_pain_ratio=gain_pain_ratio,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_volatility,
            rolling_drawdown=rolling_drawdown
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
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        largest_win = max(pnls) if pnls else 0.0
        largest_loss = min(pnls) if pnls else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
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
            'benchmark_return': benchmark_total_return,
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict[str, int]:
        """Calculate drawdown duration statistics"""
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # Handle case where drawdown continues to end
        if start is not None:
            drawdown_periods.append(len(drawdown) - start)
        
        if not drawdown_periods:
            return {'max_duration': 0, 'avg_recovery': 0}
        
        max_duration = max(drawdown_periods)
        avg_recovery = int(np.mean(drawdown_periods))
        
        return {'max_duration': max_duration, 'avg_recovery': avg_recovery}
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        
        excess_returns = returns - (self.risk_free_rate / 252)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        return rolling_sharpe.fillna(0)
    
    def _calculate_rolling_drawdown(self, equity_curve: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        
        rolling_max = equity_curve.rolling(window).max()
        rolling_drawdown = (equity_curve - rolling_max) / rolling_max
        return rolling_drawdown.fillna(0)
    
    def monte_carlo_analysis(
        self,
        returns: pd.Series,
        initial_capital: float = 100000,
        num_simulations: int = 1000,
        num_periods: int = 252
    ) -> Dict[str, Any]:
        """Perform Monte Carlo analysis on returns"""
        
        if len(returns) < 2:
            return {}
        
        # Calculate return statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, num_periods))
        
        # Calculate portfolio values
        portfolio_values = np.zeros((num_simulations, num_periods + 1))
        portfolio_values[:, 0] = initial_capital
        
        for i in range(num_periods):
            portfolio_values[:, i + 1] = portfolio_values[:, i] * (1 + simulated_returns[:, i])
        
        # Calculate final returns
        final_returns = (portfolio_values[:, -1] / initial_capital) - 1
        
        # Statistics
        percentiles = [5, 25, 50, 75, 95]
        return_percentiles = {f"p{p}": np.percentile(final_returns, p) for p in percentiles}
        
        # Probability of loss
        prob_loss = (final_returns < 0).mean()
        
        # Expected value
        expected_return = final_returns.mean()
        
        return {
            'expected_return': expected_return,
            'probability_of_loss': prob_loss,
            'return_percentiles': return_percentiles,
            'simulation_params': {
                'num_simulations': num_simulations,
                'num_periods': num_periods,
                'mean_return': mean_return,
                'std_return': std_return
            }
        } 
