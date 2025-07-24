"""
Momentum technical indicators

Popular momentum oscillators for measuring price momentum and overbought/oversold conditions.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import MomentumIndicator, IndicatorResult, standardize_signal, ema
from ...strategies.enums import TimeFrame


class RSI(MomentumIndicator):
    """Relative Strength Index (RSI)"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 14):
        self.period = period
        super().__init__(f"RSI_{period}", timeframe, period=period)
    
    def get_required_periods(self) -> int:
        return self.period + 1
    
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
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = float(rsi.iloc[-1])
        
        # Signal interpretation
        if current_rsi > 70:
            signal_strength = -0.8  # Overbought (bearish)
            signal_direction = "bearish"
        elif current_rsi < 30:
            signal_strength = 0.8   # Oversold (bullish)
            signal_direction = "bullish"
        else:
            signal_strength = standardize_signal(current_rsi, 30, 70)
            signal_direction = "neutral"
        
        metadata = {
            "rsi_value": current_rsi,
            "period": self.period,
            "overbought_threshold": 70,
            "oversold_threshold": 30
        }
        
        return self._create_result(
            value=current_rsi,
            symbol=symbol,
            timestamp=datetime.now() if timestamp is None else timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


class MACD(MomentumIndicator):
    """Moving Average Convergence Divergence (MACD)"""
    
    def __init__(self, timeframe: TimeFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}", timeframe, 
                        fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def get_required_periods(self) -> int:
        return self.slow_period + self.signal_period
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        # Calculate MACD components
        ema_fast = ema(data['close'], self.fast_period)
        ema_slow = ema(data['close'], self.slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, self.signal_period)
        histogram = macd_line - signal_line
        
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])
        
        # Signal interpretation
        if current_macd > current_signal and current_histogram > 0:
            signal_strength = 0.7
            signal_direction = "bullish"
        elif current_macd < current_signal and current_histogram < 0:
            signal_strength = -0.7
            signal_direction = "bearish"
        else:
            signal_strength = 0.0
            signal_direction = "neutral"
        
        result_value = {
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_histogram
        }
        
        metadata = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period
        }
        
        return self._create_result(
            value=result_value,
            symbol=symbol,
            timestamp=datetime.now() if timestamp is None else timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


class Stochastic(MomentumIndicator):
    """Stochastic Oscillator"""
    
    def __init__(self, timeframe: TimeFrame, k_period: int = 14, d_period: int = 3):
        super().__init__(f"STOCH_{k_period}_{d_period}", timeframe, k_period=k_period, d_period=d_period)
        self.k_period = k_period
        self.d_period = d_period
    
    def get_required_periods(self) -> int:
        return self.k_period + self.d_period
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        self._validate_data(data)
        
        # Calculate Stochastic
        lowest_low = data['low'].rolling(window=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        current_k = float(k_percent.iloc[-1])
        current_d = float(d_percent.iloc[-1])
        
        # Signal interpretation
        if current_k > 80 and current_d > 80:
            signal_strength = -0.8
            signal_direction = "bearish"
        elif current_k < 20 and current_d < 20:
            signal_strength = 0.8
            signal_direction = "bullish"
        else:
            signal_strength = 0.0
            signal_direction = "neutral"
        
        result_value = {
            "k_percent": current_k,
            "d_percent": current_d
        }
        
        metadata = {
            "k_period": self.k_period,
            "d_period": self.d_period,
            "overbought_threshold": 80,
            "oversold_threshold": 20
        }
        
        return self._create_result(
            value=result_value,
            symbol=symbol,
            timestamp=datetime.now() if timestamp is None else timestamp,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            metadata=metadata
        )


# Placeholder for additional momentum indicators
class Williams_R(MomentumIndicator):
    """Williams %R"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 14):
        super().__init__(f"WILLIAMS_R_{period}", timeframe, period=period)
        self.period = period
    
    def get_required_periods(self) -> int:
        return self.period
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        # Simplified implementation
        return self._create_result(
            value=0.0,
            symbol=symbol,
            timestamp=datetime.now() if timestamp is None else timestamp,
            metadata={"period": self.period}
        )


class ROC(MomentumIndicator):
    """Rate of Change"""
    
    def __init__(self, timeframe: TimeFrame, period: int = 12):
        super().__init__(f"ROC_{period}", timeframe, period=period)
        self.period = period
    
    def get_required_periods(self) -> int:
        return self.period + 1
    
    def calculate(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> IndicatorResult:
        # Simplified implementation
        return self._create_result(
            value=0.0,
            symbol=symbol,
            timestamp=datetime.now() if timestamp is None else timestamp,
            metadata={"period": self.period}
        ) 
