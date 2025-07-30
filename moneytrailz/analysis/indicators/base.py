"""
Base classes and interfaces for technical indicators

This module provides the foundation for all technical analysis indicators,
including common functionality, data validation, and result structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from ...strategies.enums import TimeFrame


class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = "trend"
    MOMENTUM = "momentum" 
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    OSCILLATOR = "oscillator"
    OVERLAY = "overlay"


@dataclass
class IndicatorResult:
    """Result from a technical indicator calculation"""
    
    # Core result data
    value: Union[float, Dict[str, float], None]
    timestamp: datetime
    symbol: str
    timeframe: TimeFrame
    
    # Additional metadata
    indicator_name: str
    indicator_type: IndicatorType
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    # Signal interpretation
    signal_strength: Optional[float] = None  # -1.0 to 1.0 (bearish to bullish)
    signal_direction: Optional[str] = None   # "bullish", "bearish", "neutral"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_valid(self) -> bool:
        """Check if the indicator result is valid"""
        if self.value is None:
            return False
        if isinstance(self.value, float) and (np.isnan(self.value) or np.isinf(self.value)):
            return False
        if isinstance(self.value, dict):
            return all(
                v is not None and not (np.isnan(v) if isinstance(v, float) else False) 
                for v in self.value.values()
            )
        return True
    
    @property
    def is_bullish(self) -> bool:
        """Check if result indicates bullish signal"""
        return self.signal_strength is not None and self.signal_strength > 0.1
    
    @property
    def is_bearish(self) -> bool:
        """Check if result indicates bearish signal"""
        return self.signal_strength is not None and self.signal_strength < -0.1
    
    @property
    def is_neutral(self) -> bool:
        """Check if result indicates neutral signal"""
        return (self.signal_strength is not None and 
                -0.1 <= self.signal_strength <= 0.1)


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators
    
    Provides common functionality including data validation,
    parameter management, and result formatting.
    """
    
    def __init__(
        self,
        name: str,
        indicator_type: IndicatorType,
        timeframe: TimeFrame,
        **parameters
    ):
        self.name = name
        self.indicator_type = indicator_type
        self.timeframe = timeframe
        self.parameters = parameters
        
        # Validation
        self._validate_parameters()
        
        # State tracking
        self._last_calculation_time: Optional[datetime] = None
        self._last_result: Optional[IndicatorResult] = None
        self._calculation_count = 0
    
    @abstractmethod
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        """
        Calculate the indicator value
        
        Args:
            data: OHLCV data with DatetimeIndex
            symbol: Symbol being analyzed
            timestamp: Timestamp for the calculation (defaults to last data point)
            
        Returns:
            IndicatorResult with calculated values
        """
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """
        Get minimum number of data periods required for calculation
        
        Returns:
            Number of periods needed
        """
        pass
    
    def _validate_parameters(self) -> None:
        """Validate indicator parameters"""
        # Override in subclasses for specific validation
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and completeness"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < self.get_required_periods():
            raise ValueError(
                f"Insufficient data: need {self.get_required_periods()} periods, "
                f"got {len(data)}"
            )
        
        # Check for NaN values in recent data
        recent_data = data.tail(self.get_required_periods())
        has_nan = recent_data[required_columns].isnull().sum().sum() > 0
        if has_nan:
            raise ValueError("NaN values found in recent data")
    
    def _create_result(
        self,
        value: Union[float, Dict[str, float], None],
        symbol: str,
        timestamp: datetime,
        signal_strength: Optional[float] = None,
        signal_direction: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IndicatorResult:
        """Create a standardized indicator result"""
        
        result = IndicatorResult(
            value=value,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=self.timeframe,
            indicator_name=self.name,
            indicator_type=self.indicator_type,
            confidence=confidence,
            metadata=metadata or {},
            signal_strength=signal_strength,
            signal_direction=signal_direction
        )
        
        # Update state
        self._last_calculation_time = timestamp
        self._last_result = result
        self._calculation_count += 1
        
        return result
    
    def can_calculate(self, data: pd.DataFrame) -> bool:
        """Check if indicator can be calculated with given data"""
        try:
            self._validate_data(data)
            return True
        except ValueError:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get indicator information and current state"""
        return {
            "name": self.name,
            "type": self.indicator_type.value,
            "timeframe": self.timeframe.value,
            "parameters": self.parameters,
            "required_periods": self.get_required_periods(),
            "calculation_count": self._calculation_count,
            "last_calculation": self._last_calculation_time,
            "last_result_valid": self._last_result.is_valid if self._last_result else None
        }
    
    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}({params_str})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', timeframe={self.timeframe})"


class TrendIndicator(BaseIndicator):
    """Base class for trend-following indicators"""
    
    def __init__(self, name: str, timeframe: TimeFrame, **parameters):
        super().__init__(name, IndicatorType.TREND, timeframe, **parameters)


class MomentumIndicator(BaseIndicator):
    """Base class for momentum indicators"""
    
    def __init__(self, name: str, timeframe: TimeFrame, **parameters):
        super().__init__(name, IndicatorType.MOMENTUM, timeframe, **parameters)


class VolatilityIndicator(BaseIndicator):
    """Base class for volatility indicators"""
    
    def __init__(self, name: str, timeframe: TimeFrame, **parameters):
        super().__init__(name, IndicatorType.VOLATILITY, timeframe, **parameters)


class VolumeIndicator(BaseIndicator):
    """Base class for volume indicators"""
    
    def __init__(self, name: str, timeframe: TimeFrame, **parameters):
        super().__init__(name, IndicatorType.VOLUME, timeframe, **parameters)


# Utility functions for common calculations
def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Typical Price (HLC/3)"""
    return (high + low + close) / 3


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range calculation"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close) 
    tr3 = np.abs(low - prev_close)
    tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    return tr_df.max(axis=1)


def standardize_signal(value: float, min_val: float, max_val: float) -> float:
    """Standardize a signal to -1.0 to 1.0 range"""
    if max_val == min_val:
        return 0.0
    normalized = 2 * (value - min_val) / (max_val - min_val) - 1
    return np.clip(normalized, -1.0, 1.0) 
