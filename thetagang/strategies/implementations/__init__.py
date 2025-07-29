"""
🎯 PHASE 6: CONCRETE STRATEGY IMPLEMENTATIONS
============================================

This package contains production-ready trading strategy implementations
that leverage the complete ThetaGang framework infrastructure:

- Phase 1: Strategy Framework (BaseStrategy, Registry)
- Phase 2: Technical Analysis Engine (Indicators, Signals)
- Phase 3: Multi-Timeframe Architecture (Synchronization, Coordination)
- Phase 4: Backtesting Framework (Simulation, Analytics)
- Phase 5: Configuration System (Multi-strategy, Parameters)

Strategy Categories:
- Options Strategies: Enhanced wheel, covered calls, volatility plays
- Momentum Strategies: RSI/MACD scalping, trend momentum
- Mean Reversion: Bollinger bands, oversold/overbought reversals
- Trend Following: Moving average crossovers, trend identification
- Volatility Strategies: VIX-based hedging, volatility breakouts
- Hybrid Strategies: Multi-timeframe combinations, complex logic
"""

# Enhanced Wheel Strategy
from .wheel_strategy import (
    EnhancedWheelStrategy,
    WheelConfig,
    WheelState,
    WheelPosition,
    DeltaNeutralAdjuster,
    VolatilityTimer
)

# Momentum-Based Strategies
from .momentum_strategies import (
    RSIMomentumStrategy,
    MACDMomentumStrategy,
    MomentumScalperStrategy,
    DualMomentumStrategy,
    MomentumConfig
)

# Mean Reversion Strategies
from .mean_reversion import (
    BollingerBandStrategy,
    RSIMeanReversionStrategy,
    CombinedMeanReversionStrategy,
    MeanReversionConfig,
    OverboughtOversoldDetector
)

# Trend Following Strategies
from .trend_following import (
    MovingAverageCrossoverStrategy,
    TrendFollowingStrategy,
    MultiTimeframeTrendStrategy,
    TrendConfig,
    TrendDetector
)

# Volatility-Based Strategies
from .volatility_strategies import (
    VIXHedgeStrategy,
    VolatilityBreakoutStrategy,
    StraddleStrategy,
    VolatilityConfig,
    VIXSignalGenerator
)

# Hybrid Multi-Strategy Implementations
from .hybrid_strategies import (
    MultiTimeframeStrategy,
    AdaptiveStrategy,
    PortfolioStrategy,
    HybridConfig,
    StrategyOrchestrator
)

# Strategy Registry and Factory
from .factory import (
    StrategyFactory,
    create_strategy_from_config,
    register_all_strategies,
    get_available_strategies
)

# Utility Classes
from .utils import (
    PositionSizer,
    RiskManager,
    SignalFilter,
    PerformanceTracker,
    StrategyUtils
)

__all__ = [
    # Enhanced Wheel Strategy
    "EnhancedWheelStrategy",
    "WheelConfig", 
    "WheelState",
    "WheelPosition",
    "DeltaNeutralAdjuster",
    "VolatilityTimer",
    
    # Momentum Strategies
    "RSIMomentumStrategy",
    "MACDMomentumStrategy", 
    "MomentumScalperStrategy",
    "DualMomentumStrategy",
    "MomentumConfig",
    
    # Mean Reversion Strategies
    "BollingerBandStrategy",
    "RSIMeanReversionStrategy",
    "CombinedMeanReversionStrategy",
    "MeanReversionConfig",
    "OverboughtOversoldDetector",
    
    # Trend Following Strategies
    "MovingAverageCrossoverStrategy",
    "TrendFollowingStrategy",
    "MultiTimeframeTrendStrategy",
    "TrendConfig",
    "TrendDetector",
    
    # Volatility Strategies
    "VIXHedgeStrategy",
    "VolatilityBreakoutStrategy",
    "StradleStrategy",
    "VolatilityConfig",
    "VIXSignalGenerator",
    
    # Hybrid Strategies
    "MultiTimeframeStrategy",
    "AdaptiveStrategy",
    "PortfolioStrategy",
    "HybridConfig",
    "StrategyOrchestrator",
    
    # Factory and Registry
    "StrategyFactory",
    "create_strategy_from_config",
    "register_all_strategies",
    "get_available_strategies",
    
    # Utilities
    "PositionSizer",
    "RiskManager",
    "SignalFilter",
    "PerformanceTracker",
    "StrategyUtils",
]

# Version and metadata
__version__ = "6.0.0"
__author__ = "ThetaGang Phase 6 Implementation"
__description__ = "Production-ready algorithmic trading strategy implementations"

def get_strategy_info():
    """Get information about all available strategies"""
    return {
        "enhanced_wheel": {
            "class": "EnhancedWheelStrategy",
            "type": "options",
            "timeframes": ["1D"],
            "description": "Enhanced wheel strategy with delta-neutral adjustments"
        },
        "rsi_momentum": {
            "class": "RSIMomentumStrategy", 
            "type": "stocks",
            "timeframes": ["5M", "15M", "1H"],
            "description": "RSI-based momentum strategy for short-term trades"
        },
        "macd_momentum": {
            "class": "MACDMomentumStrategy",
            "type": "stocks", 
            "timeframes": ["15M", "1H", "4H"],
            "description": "MACD-based momentum strategy"
        },
        "momentum_scalper": {
            "class": "MomentumScalperStrategy",
            "type": "stocks",
            "timeframes": ["5M", "15M"],
            "description": "High-frequency momentum scalping"
        },
        "bollinger_mean_reversion": {
            "class": "BollingerBandStrategy",
            "type": "stocks",
            "timeframes": ["1H", "4H"],
            "description": "Bollinger Band mean reversion strategy"
        },
        "rsi_mean_reversion": {
            "class": "RSIMeanReversionStrategy",
            "type": "stocks",
            "timeframes": ["1H", "4H", "1D"],
            "description": "RSI-based mean reversion strategy"
        },
        "ma_crossover": {
            "class": "MovingAverageCrossoverStrategy",
            "type": "stocks",
            "timeframes": ["1H", "4H", "1D"],
            "description": "Moving average crossover trend following"
        },
        "trend_following": {
            "class": "TrendFollowingStrategy",
            "type": "stocks",
            "timeframes": ["4H", "1D"],
            "description": "Comprehensive trend following strategy"
        },
        "vix_hedge": {
            "class": "VIXHedgeStrategy",
            "type": "mixed",
            "timeframes": ["1D"],
            "description": "VIX-based portfolio hedging strategy"
        },
        "volatility_breakout": {
            "class": "VolatilityBreakoutStrategy",
            "type": "stocks",
            "timeframes": ["1H", "4H"],
            "description": "Volatility breakout strategy"
        },
        "multi_timeframe": {
            "class": "MultiTimeframeStrategy",
            "type": "hybrid",
            "timeframes": ["5M", "1H", "1D"],
            "description": "Multi-timeframe hybrid strategy"
        },
        "adaptive": {
            "class": "AdaptiveStrategy",
            "type": "hybrid",
            "timeframes": ["15M", "1H", "4H"],
            "description": "Adaptive market condition strategy"
        }
    }

def list_strategies_by_type(strategy_type: str = None):
    """List strategies filtered by type"""
    strategies = get_strategy_info()
    if strategy_type:
        return {k: v for k, v in strategies.items() if v["type"] == strategy_type}
    return strategies

def list_strategies_by_timeframe(timeframe: str):
    """List strategies that support a specific timeframe"""
    strategies = get_strategy_info()
    return {k: v for k, v in strategies.items() if timeframe in v["timeframes"]} 
