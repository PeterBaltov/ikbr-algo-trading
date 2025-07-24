"""
Trend-following technical indicators

This module implements popular trend indicators including various 
moving averages and trend-strength measurements.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from .base import TrendIndicator, IndicatorResult, standardize_signal, sma, ema
from ...strategies.enums import TimeFrame


class SMA(TrendIndicator):
    """Simple Moving Average (SMA)"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 20):
        self.period = period
        super().__init__(f"SMA_{period}", timeframe, period=period)
    
    def get_required_periods(self) -> int:
        return self.period
    
    def _validate_parameters(self) -> None:
        if self.period < 1:
            raise ValueError("Period must be positive")
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        if timestamp is None:
            timestamp = data.index[-1]
        
        # Calculate SMA
        sma_values = sma(data['close'], self.period)
        current_sma = sma_values.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate signal strength
        price_vs_sma = (current_price - current_sma) / current_sma
        signal_strength = standardize_signal(price_vs_sma, -0.05, 0.05)  # ±5% range
        
        # Determine direction
        if signal_strength > 0.1:
            signal_direction = "bullish"
        elif signal_strength < -0.1:
            signal_direction = "bearish"
        else:
            signal_direction = "neutral"
        
        metadata = {
            "current_price": current_price,
            "sma_value": current_sma,
            "price_vs_sma_pct": price_vs_sma * 100,
            "period": self.period
        }
        
        return self._create_result(
            value=current_sma,
            symbol=symbol,
            timestamp=timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


class EMA(TrendIndicator):
    """Exponential Moving Average (EMA)"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 20):
        self.period = period
        super().__init__(f"EMA_{period}", timeframe, period=period)
    
    def get_required_periods(self) -> int:
        return self.period * 2  # EMA needs more data for stability
    
    def _validate_parameters(self) -> None:
        if self.period < 1:
            raise ValueError("Period must be positive")
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        if timestamp is None:
            timestamp = data.index[-1]
        
        # Calculate EMA
        ema_values = ema(data['close'], self.period)
        current_ema = ema_values.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate signal strength
        price_vs_ema = (current_price - current_ema) / current_ema
        signal_strength = standardize_signal(price_vs_ema, -0.05, 0.05)
        
        # Determine direction
        if signal_strength > 0.1:
            signal_direction = "bullish"
        elif signal_strength < -0.1:
            signal_direction = "bearish"
        else:
            signal_direction = "neutral"
        
        metadata = {
            "current_price": current_price,
            "ema_value": current_ema,
            "price_vs_ema_pct": price_vs_ema * 100,
            "period": self.period
        }
        
        return self._create_result(
            value=current_ema,
            symbol=symbol,
            timestamp=timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


class WMA(TrendIndicator):
    """Weighted Moving Average (WMA)"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 20):
        super().__init__(f"WMA_{period}", timeframe, period=period)
        self.period = period
    
    def get_required_periods(self) -> int:
        return self.period
    
    def _validate_parameters(self) -> None:
        if self.period < 1:
            raise ValueError("Period must be positive")
    
    def _calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()
        
        def wma_calc(x):
            return np.dot(x, weights)
        
        return series.rolling(window=period).apply(wma_calc, raw=True)
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        if timestamp is None:
            timestamp = data.index[-1]
        
        # Calculate WMA
        wma_values = self._calculate_wma(data['close'], self.period)
        current_wma = wma_values.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate signal strength
        price_vs_wma = (current_price - current_wma) / current_wma
        signal_strength = standardize_signal(price_vs_wma, -0.05, 0.05)
        
        # Determine direction
        if signal_strength > 0.1:
            signal_direction = "bullish"
        elif signal_strength < -0.1:
            signal_direction = "bearish"
        else:
            signal_direction = "neutral"
        
        metadata = {
            "current_price": current_price,
            "wma_value": current_wma,
            "price_vs_wma_pct": price_vs_wma * 100,
            "period": self.period
        }
        
        return self._create_result(
            value=current_wma,
            symbol=symbol,
            timestamp=timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


class DEMA(TrendIndicator):
    """Double Exponential Moving Average (DEMA)"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 20):
        super().__init__(f"DEMA_{period}", timeframe, period=period)
        self.period = period
    
    def get_required_periods(self) -> int:
        return self.period * 3  # DEMA needs more data
    
    def _validate_parameters(self) -> None:
        if self.period < 1:
            raise ValueError("Period must be positive")
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        if timestamp is None:
            timestamp = data.index[-1]
        
        # Calculate DEMA: 2*EMA(period) - EMA(EMA(period))
        ema1 = ema(data['close'], self.period)
        ema2 = ema(ema1, self.period)
        dema_values = 2 * ema1 - ema2
        
        current_dema = dema_values.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate signal strength
        price_vs_dema = (current_price - current_dema) / current_dema
        signal_strength = standardize_signal(price_vs_dema, -0.05, 0.05)
        
        # Determine direction
        if signal_strength > 0.1:
            signal_direction = "bullish"
        elif signal_strength < -0.1:
            signal_direction = "bearish"
        else:
            signal_direction = "neutral"
        
        metadata = {
            "current_price": current_price,
            "dema_value": current_dema,
            "price_vs_dema_pct": price_vs_dema * 100,
            "period": self.period
        }
        
        return self._create_result(
            value=current_dema,
            symbol=symbol,
            timestamp=timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


class TEMA(TrendIndicator):
    """Triple Exponential Moving Average (TEMA)"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 20):
        super().__init__(f"TEMA_{period}", timeframe, period=period)
        self.period = period
    
    def get_required_periods(self) -> int:
        return self.period * 4  # TEMA needs even more data
    
    def _validate_parameters(self) -> None:
        if self.period < 1:
            raise ValueError("Period must be positive")
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        if timestamp is None:
            timestamp = data.index[-1]
        
        # Calculate TEMA: 3*EMA(period) - 3*EMA(EMA(period)) + EMA(EMA(EMA(period)))
        ema1 = ema(data['close'], self.period)
        ema2 = ema(ema1, self.period)
        ema3 = ema(ema2, self.period)
        tema_values = 3 * ema1 - 3 * ema2 + ema3
        
        current_tema = tema_values.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate signal strength
        price_vs_tema = (current_price - current_tema) / current_tema
        signal_strength = standardize_signal(price_vs_tema, -0.05, 0.05)
        
        # Determine direction
        if signal_strength > 0.1:
            signal_direction = "bullish"
        elif signal_strength < -0.1:
            signal_direction = "bearish"
        else:
            signal_direction = "neutral"
        
        metadata = {
            "current_price": current_price,
            "tema_value": current_tema,
            "price_vs_tema_pct": price_vs_tema * 100,
            "period": self.period
        }
        
        return self._create_result(
            value=current_tema,
            symbol=symbol,
            timestamp=timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        ) 
