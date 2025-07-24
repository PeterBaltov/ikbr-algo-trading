"""
Technical Analysis Engine

Main engine that orchestrates technical analysis across multiple indicators,
timeframes, and signal processing capabilities.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from .indicators import (
    BaseIndicator, SMA, EMA, RSI, MACD, Stochastic,
    BollingerBands, ATR, VWAP
)
from .signals import SignalAggregator, CombinedSignal, IndicatorResult
from .timeframes import TimeFrameManager
from .performance import IndicatorCache, AsyncIndicatorEngine
from ..strategies.enums import TimeFrame


class TechnicalAnalysisEngine:
    """
    Main technical analysis engine for ThetaGang strategies
    
    Provides comprehensive technical analysis capabilities including:
    - Multiple technical indicators
    - Signal aggregation and confidence scoring
    - Multi-timeframe analysis
    - Performance optimization
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the technical analysis engine
        
        Args:
            cache_enabled: Whether to enable indicator result caching
        """
        self.indicators: Dict[str, BaseIndicator] = {}
        self.signal_aggregator = SignalAggregator()
        self.timeframe_manager = TimeFrameManager()
        self.cache = IndicatorCache() if cache_enabled else None
        self.async_engine = AsyncIndicatorEngine()
        
        # Performance tracking
        self._calculation_count = 0
        self._last_analysis_time: Optional[datetime] = None
    
    def add_indicator(self, indicator: BaseIndicator, name: Optional[str] = None) -> str:
        """
        Add a technical indicator to the engine
        
        Args:
            indicator: The indicator instance to add
            name: Optional custom name (defaults to indicator.name)
            
        Returns:
            The name used to register the indicator
        """
        indicator_name = name or indicator.name
        self.indicators[indicator_name] = indicator
        return indicator_name
    
    def remove_indicator(self, name: str) -> bool:
        """
        Remove an indicator from the engine
        
        Args:
            name: Name of the indicator to remove
            
        Returns:
            True if indicator was removed, False if not found
        """
        return self.indicators.pop(name, None) is not None
    
    def create_default_indicators(self, timeframe: TimeFrame) -> None:
        """
        Create a default set of indicators for a given timeframe
        
        Args:
            timeframe: The timeframe to create indicators for
        """
        # Trend indicators
        self.add_indicator(SMA(timeframe, period=20), "SMA_20")
        self.add_indicator(SMA(timeframe, period=50), "SMA_50")
        self.add_indicator(EMA(timeframe, period=20), "EMA_20")
        
        # Momentum indicators
        self.add_indicator(RSI(timeframe, period=14), "RSI_14")
        self.add_indicator(MACD(timeframe), "MACD")
        self.add_indicator(Stochastic(timeframe), "STOCH")
        
        # Volatility indicators
        self.add_indicator(BollingerBands(timeframe), "BB")
        self.add_indicator(ATR(timeframe), "ATR")
        
        # Volume indicators
        self.add_indicator(VWAP(timeframe), "VWAP")
    
    def analyze(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        timestamp: Optional[datetime] = None,
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis
        
        Args:
            data: OHLCV price data
            symbol: Symbol being analyzed
            timestamp: Analysis timestamp (defaults to last data point)
            indicators: List of indicator names to use (defaults to all)
            
        Returns:
            Dict containing analysis results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Determine which indicators to run
        indicators_to_run = indicators or list(self.indicators.keys())
        
        # Calculate all indicators
        indicator_results = []
        successful_calculations = 0
        failed_calculations = 0
        
        for indicator_name in indicators_to_run:
            if indicator_name not in self.indicators:
                continue
                
            indicator = self.indicators[indicator_name]
            
            try:
                # Check if indicator can calculate with available data
                if not indicator.can_calculate(data):
                    continue
                
                # Calculate indicator
                result = indicator.calculate(data, symbol, timestamp)
                
                if result and result.is_valid:
                    indicator_results.append(result)
                    successful_calculations += 1
                else:
                    failed_calculations += 1
                    
            except Exception as e:
                failed_calculations += 1
                # Log error in production
                continue
        
        # Aggregate signals
        combined_signal = self.signal_aggregator.aggregate_signals(
            indicator_results, symbol, timestamp
        )
        
        # Update performance tracking
        self._calculation_count += 1
        self._last_analysis_time = timestamp
        
        # Compile analysis results
        analysis_results = {
            "timestamp": timestamp,
            "symbol": symbol,
            "data_points": len(data),
            
            # Individual indicator results
            "indicators": {
                result.indicator_name: {
                    "value": result.value,
                    "signal_strength": result.signal_strength,
                    "signal_direction": result.signal_direction,
                    "confidence": result.confidence,
                    "metadata": result.metadata
                }
                for result in indicator_results
            },
            
            # Combined analysis
            "combined_signal": {
                "overall_strength": combined_signal.overall_strength,
                "overall_direction": combined_signal.overall_direction.value,
                "confidence": combined_signal.confidence,
                "contributing_indicators": len(combined_signal.contributing_signals)
            },
            
            # Performance metrics
            "performance": {
                "successful_calculations": successful_calculations,
                "failed_calculations": failed_calculations,
                "total_indicators_attempted": len(indicators_to_run),
                "calculation_success_rate": (
                    successful_calculations / len(indicators_to_run) 
                    if indicators_to_run else 0
                )
            }
        }
        
        return analysis_results
    
    async def analyze_async(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        timestamp: Optional[datetime] = None,
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous version of analysis for better performance
        
        Args:
            data: OHLCV price data
            symbol: Symbol being analyzed
            timestamp: Analysis timestamp
            indicators: List of indicator names to use
            
        Returns:
            Analysis results dictionary
        """
        # For now, delegate to synchronous version
        # In production, this would use async indicator calculations
        return self.analyze(data, symbol, timestamp, indicators)
    
    def get_indicator_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific indicator"""
        if name in self.indicators:
            return self.indicators[name].get_info()
        return None
    
    def list_indicators(self) -> List[str]:
        """Get list of all registered indicator names"""
        return list(self.indicators.keys())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return {
            "total_calculations": self._calculation_count,
            "last_analysis_time": self._last_analysis_time,
            "registered_indicators": len(self.indicators),
            "cache_enabled": self.cache is not None
        }
    
    def clear_cache(self) -> None:
        """Clear the indicator cache"""
        if self.cache:
            # Cache clear implementation would go here
            pass
    
    def __str__(self) -> str:
        return f"TechnicalAnalysisEngine(indicators={len(self.indicators)}, calculations={self._calculation_count})"
    
    def __repr__(self) -> str:
        return f"TechnicalAnalysisEngine(indicators={list(self.indicators.keys())})" 
