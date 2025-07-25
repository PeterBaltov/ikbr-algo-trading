"""
Risk Analysis Module

Comprehensive risk analytics including Value at Risk (VaR), Conditional VaR,
drawdown analysis, volatility metrics, and tail risk measures.

Features:
- Value at Risk (VaR) and Conditional VaR calculations
- Comprehensive drawdown analysis
- Volatility decomposition and analysis
- Distribution analysis and normality testing
- Tail risk and extreme event analysis
- Rolling risk metrics

Integration:
- Used by performance calculator for risk-adjusted metrics
- Provides inputs for risk management systems
- Compatible with visualization modules
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple
import logging

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


class VaRCalculator:
    """Specialized Value at Risk calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        method: str = "historical"
    ) -> Dict[str, float]:
        """Calculate Value at Risk at different confidence levels"""
        
        var_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            if method == "historical":
                var_value = returns.quantile(alpha)
            elif method == "parametric":
                var_value = returns.mean() + returns.std() * stats.norm.ppf(alpha)
            elif method == "cornish_fisher":
                # Cornish-Fisher expansion for non-normal distributions
                z_score = stats.norm.ppf(alpha)
                skew = returns.skew()
                kurt = returns.kurtosis()
                
                # Cornish-Fisher adjustment
                cf_adjustment = (skew / 6) * (z_score**2 - 1) + (kurt / 24) * z_score * (z_score**2 - 3)
                adjusted_z = z_score + cf_adjustment
                
                var_value = returns.mean() + returns.std() * adjusted_z
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            var_results[f"var_{int(confidence*100)}"] = float(var_value)
        
        return var_results
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        
        cvar_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            var_value = returns.quantile(alpha)
            
            # CVaR is the mean of returns below VaR
            tail_returns = returns[returns <= var_value]
            if len(tail_returns) > 0:
                cvar_value = tail_returns.mean()
            else:
                cvar_value = var_value
            
            cvar_results[f"cvar_{int(confidence*100)}"] = float(cvar_value)
        
        return cvar_results


class RiskCalculator:
    """Main risk analysis engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.var_calculator = VaRCalculator()
        self.logger = logging.getLogger(__name__)
    
    def calculate_risk_metrics(self, equity_curve: pd.Series, window: int = 252) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        returns = equity_curve.pct_change().dropna()
        
        # VaR calculations
        var_results = self.var_calculator.calculate_var(returns, [0.95, 0.99])
        var_95 = var_results.get('var_95', 0.0)
        var_99 = var_results.get('var_99', 0.0)
        
        # CVaR calculations
        cvar_results = self.var_calculator.calculate_cvar(returns, [0.95, 0.99])
        cvar_95 = cvar_results.get('cvar_95', 0.0)
        cvar_99 = cvar_results.get('cvar_99', 0.0)
        
        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = float(drawdown.min())
        avg_drawdown = float(drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0)
        
        # Drawdown duration analysis
        dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # Volatility metrics
        volatility = float(returns.std() * np.sqrt(252))
        downside_returns = returns[returns < 0]
        upside_returns = returns[returns > 0]
        
        downside_volatility = float(downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0)
        upside_volatility = float(upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0.0)
        
        # Distribution analysis
        skewness_val = returns.skew() if len(returns) > 2 else 0.0
        kurtosis_val = returns.kurtosis() if len(returns) > 3 else 0.0
        
        # Handle pandas scalar conversion
        skewness = float(skewness_val.iloc[0] if isinstance(skewness_val, pd.Series) else skewness_val)
        kurtosis = float(kurtosis_val.iloc[0] if isinstance(kurtosis_val, pd.Series) else kurtosis_val)
        
        # Jarque-Bera test for normality
        if len(returns) >= 8:  # Minimum sample size for JB test
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            jb_stat = float(jb_stat)
            jb_pvalue = float(jb_pvalue)
        else:
            jb_stat, jb_pvalue = 0.0, 1.0
        
        # Tail risk metrics
        if len(returns) > 0:
            q95 = returns.quantile(0.95)
            q05 = returns.quantile(0.05)
            tail_ratio = float(q95 / abs(q05) if q05 != 0 else 1.0)
        else:
            tail_ratio = 1.0
        
        # Gain-Pain ratio
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses > 0:
            gain_pain_ratio = float(gains / losses)
        else:
            gain_pain_ratio = float('inf') if gains > 0 else 0.0
        
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
    
    def calculate_stress_scenarios(
        self,
        returns: pd.Series,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance under stress scenarios"""
        
        if scenarios is None:
            # Default stress scenarios
            scenarios = {
                "market_crash": {"mean_shift": -0.05, "vol_mult": 2.0},
                "volatility_spike": {"mean_shift": 0.0, "vol_mult": 3.0},
                "persistent_bear": {"mean_shift": -0.02, "vol_mult": 1.5},
                "flash_crash": {"mean_shift": -0.10, "vol_mult": 5.0}
            }
        
        results = {}
        base_mean = returns.mean()
        base_std = returns.std()
        
        for scenario_name, params in scenarios.items():
            # Apply scenario parameters
            stressed_mean = base_mean + params.get("mean_shift", 0.0)
            stressed_std = base_std * params.get("vol_mult", 1.0)
            
            # Generate stressed returns
            num_periods = len(returns)
            stressed_returns = np.random.normal(stressed_mean, stressed_std, num_periods)
            stressed_series = pd.Series(stressed_returns, index=returns.index)
            
            # Calculate metrics for stressed scenario
            var_95 = float(stressed_series.quantile(0.05))
            max_loss = float(stressed_series.min())
            volatility = float(stressed_series.std() * np.sqrt(252))
            
            results[scenario_name] = {
                "var_95": var_95,
                "max_single_day_loss": max_loss,
                "annualized_volatility": volatility,
                "probability_of_loss": float((stressed_series < 0).mean())
            }
        
        return results
    
    def calculate_correlation_risk(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        rolling_window: int = 252
    ) -> Dict[str, Any]:
        """Calculate correlation-based risk metrics"""
        
        # Align series
        common_dates = strategy_returns.index.intersection(market_returns.index)
        if len(common_dates) == 0:
            return {}
        
        strategy_aligned = strategy_returns.loc[common_dates]
        market_aligned = market_returns.loc[common_dates]
        
        # Static correlation
        correlation = float(strategy_aligned.corr(market_aligned))
        
        # Rolling correlation
        rolling_corr = strategy_aligned.rolling(rolling_window).corr(market_aligned)
        
        # Beta calculation
        if market_aligned.std() > 0:
            beta = float(np.cov(strategy_aligned, market_aligned)[0, 1] / np.var(market_aligned))
        else:
            beta = 0.0
        
        # Downside beta (correlation during market stress)
        market_down = market_aligned < market_aligned.quantile(0.25)
        if market_down.any():
            downside_corr = float(strategy_aligned[market_down].corr(market_aligned[market_down]))
            if market_aligned[market_down].std() > 0:
                downside_beta = float(np.cov(strategy_aligned[market_down], market_aligned[market_down])[0, 1] / 
                                    np.var(market_aligned[market_down]))
            else:
                downside_beta = 0.0
        else:
            downside_corr = correlation
            downside_beta = beta
        
        return {
            "correlation": correlation,
            "beta": beta,
            "downside_correlation": downside_corr,
            "downside_beta": downside_beta,
            "rolling_correlation": rolling_corr,
            "correlation_stability": float(rolling_corr.std()) if len(rolling_corr.dropna()) > 0 else 0.0
        } 
