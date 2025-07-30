"""
🎯 STRATEGY FACTORY IMPLEMENTATION
=================================

Factory for creating strategy instances from configuration.
"""

import logging
from typing import Dict, List, Optional, Any, Type

from moneytrailz.strategies.base import BaseStrategy
from moneytrailz.strategies.registry import get_registry


class StrategyFactory:
    """Factory for creating strategy instances"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default strategy implementations"""
        try:
            # Import and register all strategies
            from .wheel_strategy import EnhancedWheelStrategy
            from .momentum_strategies import (
                RSIMomentumStrategy, MACDMomentumStrategy, 
                MomentumScalperStrategy, DualMomentumStrategy
            )
            from .mean_reversion import (
                BollingerBandStrategy, RSIMeanReversionStrategy, 
                CombinedMeanReversionStrategy
            )
            from .trend_following import (
                MovingAverageCrossoverStrategy, TrendFollowingStrategy,
                MultiTimeframeTrendStrategy
            )
            from .volatility_strategies import (
                VIXHedgeStrategy, VolatilityBreakoutStrategy, StraddleStrategy
            )
            from .hybrid_strategies import (
                MultiTimeframeStrategy, AdaptiveStrategy, PortfolioStrategy
            )
            
            # Register strategies
            self.strategy_classes.update({
                'enhanced_wheel': EnhancedWheelStrategy,
                'rsi_momentum': RSIMomentumStrategy,
                'macd_momentum': MACDMomentumStrategy,
                'momentum_scalper': MomentumScalperStrategy,
                'dual_momentum': DualMomentumStrategy,
                'bollinger_mean_reversion': BollingerBandStrategy,
                'rsi_mean_reversion': RSIMeanReversionStrategy,
                'combined_mean_reversion': CombinedMeanReversionStrategy,
                'ma_crossover': MovingAverageCrossoverStrategy,
                'trend_following': TrendFollowingStrategy,
                'multi_timeframe_trend': MultiTimeframeTrendStrategy,
                'vix_hedge': VIXHedgeStrategy,
                'volatility_breakout': VolatilityBreakoutStrategy,
                'straddle': StraddleStrategy,
                'multi_timeframe': MultiTimeframeStrategy,
                'adaptive': AdaptiveStrategy,
                'portfolio': PortfolioStrategy,
            })
            
        except ImportError as e:
            self.logger.warning(f"Could not import some strategies: {e}")
    
    def create_strategy(self, strategy_name: str, name: str, symbols: List[str], 
                       timeframes: List[str], config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """Create a strategy instance"""
        if strategy_name not in self.strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = self.strategy_classes[strategy_name]
        # Convert timeframes to the expected format for the strategy constructor
        # Note: This may need adjustment based on actual BaseStrategy signature
        return strategy_class(name, symbols, timeframes, config)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategy_classes.keys())
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]):
        """Register a custom strategy"""
        self.strategy_classes[name] = strategy_class


def create_strategy_from_config(strategy_config: Dict[str, Any]) -> BaseStrategy:
    """Create strategy from configuration dictionary"""
    factory = StrategyFactory()
    
    strategy_type = strategy_config.get('type', 'unknown')
    name = strategy_config.get('name', f'strategy_{strategy_type}')
    symbols = strategy_config.get('symbols', [])
    timeframes = strategy_config.get('timeframes', ['1D'])
    config = strategy_config.get('config', {})
    
    return factory.create_strategy(strategy_type, name, symbols, timeframes, config)


def register_all_strategies():
    """Register all strategies with the global registry"""
    factory = StrategyFactory()
    registry = get_registry()
    
    for name, strategy_class in factory.strategy_classes.items():
        registry.register_strategy(strategy_class, name)


def get_available_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Get all available strategy classes"""
    factory = StrategyFactory()
    return factory.strategy_classes.copy() 
