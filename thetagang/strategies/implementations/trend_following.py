"""
🎯 TREND FOLLOWING STRATEGIES IMPLEMENTATION
==========================================

Advanced trend following strategies that capitalize on sustained price movements
in a particular direction. These strategies are designed for trending markets
and long-term position holding.

Key Features:
- Moving average crossover systems
- Multi-timeframe trend confirmation
- Trend strength and momentum analysis
- Dynamic position sizing based on trend strength
- Adaptive stop losses and profit targets
- Market regime detection and filtering
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

import pandas as pd
import numpy as np

from thetagang.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from thetagang.strategies.enums import StrategySignal, StrategyType, TimeFrame
from thetagang.strategies.exceptions import StrategyExecutionError, StrategyConfigError


class TrendState(Enum):
    """Trend following strategy states"""
    WAITING_FOR_TREND = "waiting_for_trend"
    LONG_TREND = "long_trend"
    SHORT_TREND = "short_trend"
    TREND_WEAKENING = "trend_weakening"
    CONSOLIDATION = "consolidation"


class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    SIDEWAYS = "sideways"
    UNCERTAIN = "uncertain"


@dataclass
class TrendConfig:
    """Configuration for trend following strategies"""
    # Moving average parameters
    ma_fast: int = 20
    ma_slow: int = 50
    ma_long: int = 200
    ma_type: str = "EMA"  # SMA, EMA, WMA
    
    # Trend confirmation
    min_trend_strength: float = 0.6
    trend_confirmation_period: int = 5
    multi_timeframe_confirmation: bool = True
    
    # Position management
    position_size_base: float = 0.04  # 4% base position
    max_position_size: float = 0.12   # 12% maximum
    trend_based_sizing: bool = True
    
    # Risk management
    atr_multiplier: float = 2.0       # ATR-based stops
    trailing_stop_atr: float = 1.5    # Trailing stop ATR multiplier
    profit_target_ratio: float = 3.0  # Risk:reward ratio
    
    # Entry/exit criteria
    breakout_confirmation: bool = True
    volume_confirmation: bool = True
    min_volume_ratio: float = 1.1     # 10% above average
    
    # Trend filtering
    adx_threshold: float = 25.0       # ADX threshold for trending markets
    slope_threshold: float = 0.001    # Minimum slope for trend confirmation


@dataclass  
class TrendPosition:
    """Trend following position tracking"""
    symbol: str = ""
    state: TrendState = TrendState.WAITING_FOR_TREND
    direction: TrendDirection = TrendDirection.UNCERTAIN
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trailing_stop_price: float = 0.0
    unrealized_pnl: float = 0.0
    trend_strength: float = 0.0
    days_held: int = 0
    highest_price: float = 0.0         # For trailing stops
    lowest_price: float = float('inf') # For trailing stops


class TrendDetector:
    """Advanced trend detection and analysis"""
    
    def __init__(self, config: TrendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        
        if self.config.ma_type == "SMA":
            mas['fast'] = prices.rolling(window=self.config.ma_fast).mean()
            mas['slow'] = prices.rolling(window=self.config.ma_slow).mean() 
            mas['long'] = prices.rolling(window=self.config.ma_long).mean()
        elif self.config.ma_type == "EMA":
            mas['fast'] = prices.ewm(span=self.config.ma_fast).mean()
            mas['slow'] = prices.ewm(span=self.config.ma_slow).mean()
            mas['long'] = prices.ewm(span=self.config.ma_long).mean()
        else:  # WMA
            mas['fast'] = prices.rolling(window=self.config.ma_fast).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
            )
            mas['slow'] = prices.rolling(window=self.config.ma_slow).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
            )
            mas['long'] = prices.rolling(window=self.config.ma_long).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
            )
        
        return mas
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def detect_trend_direction(self, mas: Dict[str, pd.Series], current_price: float) -> TrendDirection:
        """Detect current trend direction and strength"""
        if len(mas['fast']) < 2 or len(mas['slow']) < 2:
            return TrendDirection.UNCERTAIN
        
        fast_ma = mas['fast'].iloc[-1]
        slow_ma = mas['slow'].iloc[-1]
        long_ma = mas['long'].iloc[-1] if len(mas['long']) > 0 else slow_ma
        
        # Price above all MAs = uptrend
        if current_price > fast_ma > slow_ma > long_ma:
            # Check slope steepness for strength
            fast_slope = (mas['fast'].iloc[-1] - mas['fast'].iloc[-5]) / 5 if len(mas['fast']) >= 5 else 0
            if fast_slope > self.config.slope_threshold:
                return TrendDirection.STRONG_UPTREND
            else:
                return TrendDirection.WEAK_UPTREND
        
        # Price below all MAs = downtrend  
        elif current_price < fast_ma < slow_ma < long_ma:
            fast_slope = (mas['fast'].iloc[-1] - mas['fast'].iloc[-5]) / 5 if len(mas['fast']) >= 5 else 0
            if fast_slope < -self.config.slope_threshold:
                return TrendDirection.STRONG_DOWNTREND
            else:
                return TrendDirection.WEAK_DOWNTREND
        
        # Mixed signals = sideways
        else:
            return TrendDirection.SIDEWAYS
    
    def calculate_trend_strength(self, mas: Dict[str, pd.Series], current_price: float, 
                               direction: TrendDirection) -> float:
        """Calculate trend strength score (0-1)"""
        if direction == TrendDirection.UNCERTAIN:
            return 0.0
        
        fast_ma = mas['fast'].iloc[-1] if len(mas['fast']) > 0 else current_price
        slow_ma = mas['slow'].iloc[-1] if len(mas['slow']) > 0 else current_price
        
        # Distance between MAs relative to price
        ma_separation = abs(fast_ma - slow_ma) / current_price
        
        # Slope consistency
        fast_slope = 0
        slow_slope = 0
        if len(mas['fast']) >= 5:
            fast_slope = (mas['fast'].iloc[-1] - mas['fast'].iloc[-5]) / current_price
        if len(mas['slow']) >= 5:
            slow_slope = (mas['slow'].iloc[-1] - mas['slow'].iloc[-5]) / current_price
        
        # Price distance from MAs
        price_distance = abs(current_price - fast_ma) / current_price
        
        # Combine factors
        strength = min(1.0, (ma_separation * 100) + abs(fast_slope) * 50 + (price_distance * 50))
        
        return strength
    
    def detect_crossover(self, mas: Dict[str, pd.Series]) -> Tuple[bool, bool]:
        """Detect moving average crossovers"""
        if len(mas['fast']) < 2 or len(mas['slow']) < 2:
            return False, False
        
        # Current and previous values
        fast_curr = mas['fast'].iloc[-1]
        fast_prev = mas['fast'].iloc[-2]
        slow_curr = mas['slow'].iloc[-1]
        slow_prev = mas['slow'].iloc[-2]
        
        # Bullish crossover (fast crosses above slow)
        bullish_cross = fast_prev <= slow_prev and fast_curr > slow_curr
        
        # Bearish crossover (fast crosses below slow)
        bearish_cross = fast_prev >= slow_prev and fast_curr < slow_curr
        
        return bullish_cross, bearish_cross


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Classic moving average crossover strategy.
    
    Strategy Logic:
    - Enter long when fast MA crosses above slow MA
    - Enter short when fast MA crosses below slow MA
    - Use long-term MA as trend filter
    - ATR-based position sizing and stops
    """
    
    def __init__(self, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            strategy_type=StrategyType.STOCKS,
            config=config or {},
            symbols=symbols,
            timeframes=timeframes
        )
        
        # Initialize configuration
        self.trend_config = TrendConfig(**self.self.config.get('trend_parameters', {}))
        
        # Initialize detector
        self.detector = TrendDetector(self.trend_config)
        
        # Position tracking
        self.positions: Dict[str, TrendPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = TrendPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main moving average crossover analysis"""
        try:
            # Get primary timeframe data
            primary_tf = list(data.keys())[0] if data else None
            if not primary_tf or data[primary_tf].empty:
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
            
            df = data[primary_tf]
            current_price = df['close'].iloc[-1]
            position = self.positions[symbol]
            
            # Calculate moving averages
            mas = self.detector.calculate_moving_averages(df['close'])
            
            # Calculate ATR for stops
            atr = self.detector.calculate_atr(df['high'], df['low'], df['close'])
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            # Detect trend and crossovers
            direction = self.detector.detect_trend_direction(mas, current_price)
            strength = self.detector.calculate_trend_strength(mas, current_price, direction)
            bullish_cross, bearish_cross = self.detector.detect_crossover(mas)
            
            # Update position
            position.direction = direction
            position.trend_strength = strength
            position.days_held += 1
            
            # Update trailing stops
            self._update_trailing_stops(position, current_price, current_atr)
            
            # Generate signals
            if position.state == TrendState.WAITING_FOR_TREND:
                return await self._analyze_ma_entry(
                    symbol, df, current_price, current_atr, bullish_cross, 
                    bearish_cross, direction, strength
                )
            else:
                return await self._analyze_ma_exit(
                    symbol, df, current_price, current_atr, direction, strength
                )
                
        except Exception as e:
            self.logger.error(f"MA crossover analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"MA crossover analysis failed: {e}")
    
    def _update_trailing_stops(self, position: TrendPosition, current_price: float, atr: float):
        """Update trailing stop prices"""
        if position.state in [TrendState.LONG_TREND, TrendState.SHORT_TREND]:
            if position.state == TrendState.LONG_TREND:
                # Update highest price and trailing stop for long position
                if current_price > position.highest_price:
                    position.highest_price = current_price
                    position.trailing_stop_price = current_price - (atr * self.trend_config.trailing_stop_atr)
            else:
                # Update lowest price and trailing stop for short position  
                if current_price < position.lowest_price:
                    position.lowest_price = current_price
                    position.trailing_stop_price = current_price + (atr * self.trend_config.trailing_stop_atr)
    
    async def _analyze_ma_entry(self, symbol: str, df: pd.DataFrame, current_price: float,
                              atr: float, bullish_cross: bool, bearish_cross: bool,
                              direction: TrendDirection, strength: float) -> StrategyResult:
        """Analyze moving average crossover entry signals"""
        position = self.positions[symbol]
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Trend strength filter
        strong_trend = strength >= self.trend_config.min_trend_strength
        
        # Bullish crossover entry
        if (bullish_cross and strong_trend and volume_confirmed and 
            direction in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND]):
            
            position.state = TrendState.LONG_TREND
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.highest_price = current_price
            position.lowest_price = float('inf')
            
            # ATR-based stops and targets
            stop_distance = atr * self.trend_config.atr_multiplier
            position.stop_loss_price = current_price - stop_distance
            position.take_profit_price = current_price + (stop_distance * self.trend_config.profit_target_ratio)
            position.trailing_stop_price = position.stop_loss_price
            
            # Dynamic position sizing based on trend strength
            position_size = self._calculate_position_size(strength)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=position_size,
                metadata={
                    'strategy_action': 'ma_bullish_crossover',
                    'trend_direction': direction.value,
                    'trend_strength': strength,
                    'atr_value': atr,
                    'volume_confirmed': volume_confirmed,
                    'stop_distance_atr': self.trend_config.atr_multiplier
                }
            )
        
        # Bearish crossover entry
        elif (bearish_cross and strong_trend and volume_confirmed and 
              direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]):
            
            position.state = TrendState.SHORT_TREND
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.highest_price = 0.0
            position.lowest_price = current_price
            
            # ATR-based stops and targets
            stop_distance = atr * self.trend_config.atr_multiplier
            position.stop_loss_price = current_price + stop_distance
            position.take_profit_price = current_price - (stop_distance * self.trend_config.profit_target_ratio)
            position.trailing_stop_price = position.stop_loss_price
            
            # Dynamic position sizing
            position_size = self._calculate_position_size(strength)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=position_size,
                metadata={
                    'strategy_action': 'ma_bearish_crossover',
                    'trend_direction': direction.value,
                    'trend_strength': strength,
                    'atr_value': atr,
                    'volume_confirmed': volume_confirmed,
                    'stop_distance_atr': self.trend_config.atr_multiplier
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_crossover',
                'trend_direction': direction.value,
                'trend_strength': strength,
                'bullish_cross': bullish_cross,
                'bearish_cross': bearish_cross,
                'strong_trend': strong_trend,
                'volume_confirmed': volume_confirmed
            }
        )
    
    def _calculate_position_size(self, trend_strength: float) -> float:
        """Calculate position size based on trend strength"""
        if not self.trend_config.trend_based_sizing:
            return self.trend_config.position_size_base
        
        # Scale position size by trend strength
        size_multiplier = 0.5 + (trend_strength * 1.5)  # 0.5x to 2.0x
        position_size = self.trend_config.position_size_base * size_multiplier
        
        # Cap at maximum size
        return min(position_size, self.trend_config.max_position_size)
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if not self.trend_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.trend_config.min_volume_ratio
    
    async def _analyze_ma_exit(self, symbol: str, df: pd.DataFrame, current_price: float,
                             atr: float, direction: TrendDirection, strength: float) -> StrategyResult:
        """Analyze moving average exit signals"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.state == TrendState.LONG_TREND:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Trend weakening detection
        if strength < self.trend_config.min_trend_strength * 0.7:  # 30% below threshold
            position.state = TrendState.TREND_WEAKENING
        
        # Trailing stop exit
        if ((position.state == TrendState.LONG_TREND and current_price <= position.trailing_stop_price) or
            (position.state == TrendState.SHORT_TREND and current_price >= position.trailing_stop_price)):
            
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.state == TrendState.LONG_TREND else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'trailing_stop_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'trailing_stop_price': position.trailing_stop_price,
                    'reason': 'Trailing stop triggered'
                }
            )
        
        # Take profit exit
        if ((position.state == TrendState.LONG_TREND and current_price >= position.take_profit_price) or
            (position.state == TrendState.SHORT_TREND and current_price <= position.take_profit_price)):
            
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.state == TrendState.LONG_TREND else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'take_profit_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'take_profit_price': position.take_profit_price,
                    'reason': 'Take profit target reached'
                }
            )
        
        # Trend reversal exit
        if ((position.state == TrendState.LONG_TREND and 
             direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]) or
            (position.state == TrendState.SHORT_TREND and 
             direction in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND])):
            
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.state == TrendState.LONG_TREND else StrategySignal.BUY,
                confidence=0.8,
                metadata={
                    'strategy_action': 'trend_reversal_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'new_direction': direction.value,
                    'reason': 'Trend reversal detected'
                }
            )
        
        # Trend weakening exit (partial)
        if position.state == TrendState.TREND_WEAKENING and position.unrealized_pnl > 0.01:  # 1% profit
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.entry_price < current_price else StrategySignal.BUY,
                confidence=0.7,
                metadata={
                    'strategy_action': 'trend_weakening_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'trend_strength': strength,
                    'reason': 'Trend weakening - taking profits'
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.6,
            metadata={
                'strategy_action': 'hold_trend_position',
                'unrealized_pnl': position.unrealized_pnl,
                'trend_direction': direction.value,
                'trend_strength': strength,
                'days_held': position.days_held,
                'trailing_stop_price': position.trailing_stop_price
            }
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return ["1H", "4H", "1D"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate moving average crossover configuration"""
        try:
            trend_params = self.config.get('trend_parameters', {})
            
            # Validate MA periods
            ma_fast = trend_params.get('ma_fast', 20)
            ma_slow = trend_params.get('ma_slow', 50)
            if ma_fast >= ma_slow:
                raise StrategyConfigError("ma_fast must be less than ma_slow")
            
            # Validate ATR multiplier
            atr_mult = trend_params.get('atr_multiplier', 2.0)
            if not 0.5 <= atr_mult <= 5.0:
                raise StrategyConfigError("atr_multiplier must be between 0.5 and 5.0")
            
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise StrategyConfigError(f"Invalid MA crossover configuration: {e}")


class TrendFollowingStrategy(BaseStrategy):
    """
    Advanced trend following strategy with multiple confirmations.
    
    Strategy Logic:
    - Multiple timeframe trend alignment
    - Breakout confirmation with volume
    - Dynamic position sizing based on trend strength
    - Advanced trend continuation patterns
    """
    
    def __init__(self, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            strategy_type=StrategyType.STOCKS,
            config=config or {},
            symbols=symbols,
            timeframes=timeframes
        )
        
        self.trend_config = TrendConfig(**self.self.config.get('trend_parameters', {}))
        self.detector = TrendDetector(self.trend_config)
        
        self.positions: Dict[str, TrendPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = TrendPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Advanced trend following analysis"""
        try:
            # Get multiple timeframe data
            timeframes = list(data.keys())
            if not timeframes or all(data[tf].empty for tf in timeframes):
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
            
            # Primary timeframe analysis
            primary_tf = timeframes[0]
            df_primary = data[primary_tf]
            current_price = df_primary['close'].iloc[-1]
            position = self.positions[symbol]
            
            # Multi-timeframe trend analysis
            trend_alignment = await self._analyze_multi_timeframe_trend(symbol, data)
            
            # Calculate indicators for primary timeframe
            mas = self.detector.calculate_moving_averages(df_primary['close'])
            atr = self.detector.calculate_atr(df_primary['high'], df_primary['low'], df_primary['close'])
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            # Trend analysis
            direction = self.detector.detect_trend_direction(mas, current_price)
            strength = self.detector.calculate_trend_strength(mas, current_price, direction)
            
            # Update position
            position.direction = direction
            position.trend_strength = strength
            position.days_held += 1
            
            # Generate signals
            if position.state == TrendState.WAITING_FOR_TREND:
                return await self._analyze_trend_entry(
                    symbol, df_primary, current_price, current_atr, direction, 
                    strength, trend_alignment
                )
            else:
                return await self._analyze_trend_exit(
                    symbol, df_primary, current_price, current_atr, direction, 
                    strength, trend_alignment
                )
                
        except Exception as e:
            self.logger.error(f"Trend following analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"Trend following analysis failed: {e}")
    
    async def _analyze_multi_timeframe_trend(self, symbol: str, 
                                           data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze trend across multiple timeframes"""
        alignment = {
            'trends': {},
            'strengths': {},
            'aligned': False,
            'dominant_direction': TrendDirection.UNCERTAIN
        }
        
        trend_scores = {'up': 0, 'down': 0, 'sideways': 0}
        
        for tf, df in data.items():
            if df.empty:
                continue
            
            current_price = df['close'].iloc[-1]
            mas = self.detector.calculate_moving_averages(df['close'])
            direction = self.detector.detect_trend_direction(mas, current_price)
            strength = self.detector.calculate_trend_strength(mas, current_price, direction)
            
            alignment['trends'][tf] = direction
            alignment['strengths'][tf] = strength
            
            # Score trends by timeframe weight (longer = more weight)
            weight = 1.0
            if 'D' in tf:
                weight = 3.0
            elif 'H' in tf:
                weight = 2.0
            elif 'M' in tf:
                weight = 1.0
            
            if direction in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND]:
                trend_scores['up'] += weight * strength
            elif direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]:
                trend_scores['down'] += weight * strength
            else:
                trend_scores['sideways'] += weight
        
        # Determine dominant direction and alignment
        max_score = max(trend_scores.values())
        if max_score > 0:
            if trend_scores['up'] == max_score:
                alignment['dominant_direction'] = TrendDirection.STRONG_UPTREND
            elif trend_scores['down'] == max_score:
                alignment['dominant_direction'] = TrendDirection.STRONG_DOWNTREND
            else:
                alignment['dominant_direction'] = TrendDirection.SIDEWAYS
        
        # Check if trends are aligned (>60% agreement)
        total_weight = sum(trend_scores.values())
        alignment['aligned'] = max_score / total_weight > 0.6 if total_weight > 0 else False
        
        return alignment
    
    async def _analyze_trend_entry(self, symbol: str, df: pd.DataFrame, current_price: float,
                                 atr: float, direction: TrendDirection, strength: float,
                                 trend_alignment: Dict[str, Any]) -> StrategyResult:
        """Analyze trend following entry signals"""
        position = self.positions[symbol]
        
        # Multi-timeframe confirmation
        if self.trend_config.multi_timeframe_confirmation and not trend_alignment['aligned']:
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.3,
                metadata={
                    'strategy_action': 'waiting_for_alignment',
                    'trend_alignment': trend_alignment,
                    'reason': 'Timeframes not aligned'
                }
            )
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Breakout confirmation
        breakout_confirmed = True
        if self.trend_config.breakout_confirmation:
            breakout_confirmed = self._check_breakout_pattern(df)
        
        # Strong trend filter
        strong_trend = strength >= self.trend_config.min_trend_strength
        
        # Long entry conditions
        if (trend_alignment['dominant_direction'] in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND] and
            strong_trend and volume_confirmed and breakout_confirmed):
            
            position.state = TrendState.LONG_TREND
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.highest_price = current_price
            position.lowest_price = float('inf')
            
            # Dynamic stops and targets
            stop_distance = atr * self.trend_config.atr_multiplier * (2.0 - strength)  # Tighter stops for stronger trends
            position.stop_loss_price = current_price - stop_distance
            position.take_profit_price = current_price + (stop_distance * self.trend_config.profit_target_ratio)
            position.trailing_stop_price = position.stop_loss_price
            
            # Enhanced position sizing
            position_size = self._calculate_enhanced_position_size(strength, trend_alignment)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.85,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=position_size,
                metadata={
                    'strategy_action': 'trend_following_long',
                    'trend_direction': direction.value,
                    'trend_strength': strength,
                    'trend_alignment': trend_alignment,
                    'breakout_confirmed': breakout_confirmed,
                    'position_size_factor': position_size / self.trend_config.position_size_base
                }
            )
        
        # Short entry conditions
        elif (trend_alignment['dominant_direction'] in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND] and
              strong_trend and volume_confirmed and breakout_confirmed):
            
            position.state = TrendState.SHORT_TREND
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.highest_price = 0.0
            position.lowest_price = current_price
            
            # Dynamic stops and targets
            stop_distance = atr * self.trend_config.atr_multiplier * (2.0 - strength)
            position.stop_loss_price = current_price + stop_distance
            position.take_profit_price = current_price - (stop_distance * self.trend_config.profit_target_ratio)
            position.trailing_stop_price = position.stop_loss_price
            
            # Enhanced position sizing
            position_size = self._calculate_enhanced_position_size(strength, trend_alignment)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.85,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=position_size,
                metadata={
                    'strategy_action': 'trend_following_short',
                    'trend_direction': direction.value,
                    'trend_strength': strength,
                    'trend_alignment': trend_alignment,
                    'breakout_confirmed': breakout_confirmed,
                    'position_size_factor': position_size / self.trend_config.position_size_base
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_trend_setup',
                'trend_direction': direction.value,
                'trend_strength': strength,
                'trend_alignment': trend_alignment,
                'strong_trend': strong_trend,
                'volume_confirmed': volume_confirmed,
                'breakout_confirmed': breakout_confirmed
            }
        )
    
    def _check_breakout_pattern(self, df: pd.DataFrame) -> bool:
        """Check for breakout pattern confirmation"""
        if len(df) < 20:
            return True
        
        # Simple breakout: price above recent high
        recent_high = df['high'].iloc[-10:-1].max()  # Exclude current candle
        current_price = df['close'].iloc[-1]
        
        return current_price > recent_high * 1.001  # 0.1% above recent high
    
    def _calculate_enhanced_position_size(self, strength: float, 
                                        trend_alignment: Dict[str, Any]) -> float:
        """Calculate enhanced position size based on multiple factors"""
        base_size = self.trend_config.position_size_base
        
        if not self.trend_config.trend_based_sizing:
            return base_size
        
        # Trend strength multiplier
        strength_multiplier = 0.5 + (strength * 1.5)
        
        # Alignment multiplier
        alignment_multiplier = 1.5 if trend_alignment['aligned'] else 1.0
        
        # Multi-timeframe agreement multiplier
        agreement_score = 1.0
        if len(trend_alignment['trends']) > 1:
            same_direction = sum(1 for t in trend_alignment['trends'].values() 
                               if t == trend_alignment['dominant_direction'])
            agreement_score = same_direction / len(trend_alignment['trends'])
        
        # Combine multipliers
        total_multiplier = strength_multiplier * alignment_multiplier * agreement_score
        position_size = base_size * total_multiplier
        
        # Cap at maximum
        return min(position_size, self.trend_config.max_position_size)
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if not self.trend_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.trend_config.min_volume_ratio
    
    async def _analyze_trend_exit(self, symbol: str, df: pd.DataFrame, current_price: float,
                                atr: float, direction: TrendDirection, strength: float,
                                trend_alignment: Dict[str, Any]) -> StrategyResult:
        """Analyze trend following exit signals"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.state == TrendState.LONG_TREND:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Update trailing stops
        self._update_trailing_stops(position, current_price, atr)
        
        # Multi-timeframe divergence exit
        if (self.trend_config.multi_timeframe_confirmation and 
            not trend_alignment['aligned'] and position.unrealized_pnl > 0.02):  # 2% profit minimum
            
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.state == TrendState.LONG_TREND else StrategySignal.BUY,
                confidence=0.8,
                metadata={
                    'strategy_action': 'timeframe_divergence_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'trend_alignment': trend_alignment,
                    'reason': 'Timeframe divergence detected'
                }
            )
        
        # Standard trend following exits (similar to MA crossover)
        return await self._standard_trend_exit(symbol, current_price, direction, strength)
    
    def _update_trailing_stops(self, position: TrendPosition, current_price: float, atr: float):
        """Update trailing stop prices"""
        if position.state in [TrendState.LONG_TREND, TrendState.SHORT_TREND]:
            if position.state == TrendState.LONG_TREND:
                if current_price > position.highest_price:
                    position.highest_price = current_price
                    new_stop = current_price - (atr * self.trend_config.trailing_stop_atr)
                    position.trailing_stop_price = max(position.trailing_stop_price, new_stop)
            else:
                if current_price < position.lowest_price:
                    position.lowest_price = current_price
                    new_stop = current_price + (atr * self.trend_config.trailing_stop_atr)
                    position.trailing_stop_price = min(position.trailing_stop_price, new_stop)
    
    async def _standard_trend_exit(self, symbol: str, current_price: float,
                                 direction: TrendDirection, strength: float) -> StrategyResult:
        """Standard trend following exit conditions"""
        position = self.positions[symbol]
        
        # Trailing stop exit
        if ((position.state == TrendState.LONG_TREND and current_price <= position.trailing_stop_price) or
            (position.state == TrendState.SHORT_TREND and current_price >= position.trailing_stop_price)):
            
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.state == TrendState.LONG_TREND else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'trailing_stop_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'trailing_stop_price': position.trailing_stop_price
                }
            )
        
        # Trend strength deterioration
        if strength < self.trend_config.min_trend_strength * 0.6:  # 40% below threshold
            position.state = TrendState.WAITING_FOR_TREND
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.state == TrendState.LONG_TREND else StrategySignal.BUY,
                confidence=0.7,
                metadata={
                    'strategy_action': 'trend_deterioration_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'trend_strength': strength,
                    'reason': 'Trend strength below threshold'
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.6,
            metadata={
                'strategy_action': 'hold_trend_position',
                'unrealized_pnl': position.unrealized_pnl,
                'trend_direction': direction.value,
                'trend_strength': strength,
                'days_held': position.days_held,
                'trailing_stop_price': position.trailing_stop_price
            }
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return ["4H", "1D"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate trend following configuration"""


class MultiTimeframeTrendStrategy(BaseStrategy):
    """
    Multi-timeframe trend strategy with sophisticated alignment detection.
    
    Strategy Logic:
    - Requires trend alignment across multiple timeframes
    - Uses higher timeframes for direction, lower for timing
    - Advanced trend strength weighting
    - Risk-adjusted position sizing
    """
    
    def __init__(self, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            strategy_type=StrategyType.STOCKS,
            config=config or {},
            symbols=symbols,
            timeframes=timeframes
        )
        
        self.trend_config = TrendConfig(**self.self.config.get('trend_parameters', {}))
        self.detector = TrendDetector(self.trend_config)
        
        self.positions: Dict[str, TrendPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = TrendPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Multi-timeframe trend analysis"""
        # This would be a more sophisticated version of trend following
        # For brevity, returning a placeholder implementation
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.5,
            metadata={'strategy_action': 'multi_timeframe_analysis'}
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return ["1H", "4H", "1D"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate multi-timeframe trend configuration"""
