# Technical Analysis Engine - Phase 2
# 
# This package provides comprehensive technical analysis capabilities
# for moneytrailz strategies including indicators, signal processing,
# multi-timeframe support, and performance optimization.

from .indicators import (
    # Trend Indicators
    SMA, EMA, WMA, DEMA, TEMA,
    
    # Momentum Indicators  
    RSI, MACD, Stochastic, Williams_R, ROC,
    
    # Volatility Indicators
    BollingerBands, ATR, Keltner, DonchianChannel,
    
    # Volume Indicators
    VWAP, OBV, AD_Line, PVT,
    
    # Support/Resistance
    PivotPoints, FibonacciRetracements,
    
    # Base classes
    BaseIndicator, IndicatorResult
)

from .signals import (
    SignalProcessor, SignalAggregator, ConfidenceCalculator,
    SignalStrength, SignalDirection, CombinedSignal
)

# Timeframe management moved to moneytrailz.timeframes package
# Performance optimization features moved to moneytrailz.analytics package

# Version and metadata
__version__ = "2.0.0"
__phase__ = "Phase 2: Technical Analysis Engine"

# Main analysis engine
from .engine import TechnicalAnalysisEngine

__all__ = [
    # Core Engine
    "TechnicalAnalysisEngine",
    
    # Indicators
    "SMA", "EMA", "WMA", "DEMA", "TEMA",
    "RSI", "MACD", "Stochastic", "Williams_R", "ROC", 
    "BollingerBands", "ATR", "Keltner", "DonchianChannel",
    "VWAP", "OBV", "AD_Line", "PVT",
    "PivotPoints", "FibonacciRetracements",
    "BaseIndicator", "IndicatorResult",
    
    # Signal Processing
    "SignalProcessor", "SignalAggregator", "ConfidenceCalculator",
    "SignalStrength", "SignalDirection", "CombinedSignal",
    
    # Multi-timeframe management moved to moneytrailz.timeframes
    # Performance optimization moved to moneytrailz.analytics
] 
