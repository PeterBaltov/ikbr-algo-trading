"""
🎯 MEAN REVERSION STRATEGIES IMPLEMENTATION
==========================================

Advanced mean reversion strategies that capitalize on price movements
returning to their statistical mean. These strategies are designed for
counter-trend trading in ranging or sideways markets.

Key Features:
- Bollinger Band squeeze and expansion strategies
- RSI overbought/oversold mean reversion
- Combined multi-indicator mean reversion
- Dynamic support/resistance levels
- Market regime detection (trending vs ranging)
- Risk management for counter-trend positions
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

import pandas as pd
import numpy as np

from moneytrailz.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from moneytrailz.strategies.enums import StrategySignal, StrategyType, TimeFrame
from moneytrailz.strategies.exceptions import StrategyExecutionError, StrategyConfigError


class MeanReversionState(Enum):
    """Mean reversion strategy states"""
    WAITING_FOR_SETUP = "waiting_for_setup"
    LONG_REVERSION = "long_reversion"
    SHORT_REVERSION = "short_reversion"
    RISK_MANAGEMENT = "risk_management"
    TREND_FILTER_WAIT = "trend_filter_wait"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategies"""
    # Bollinger Band parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_squeeze_threshold: float = 1.5  # Std dev threshold for squeeze
    bb_expansion_factor: float = 1.2   # Factor for detecting expansion
    
    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_extreme_oversold: float = 20.0
    rsi_extreme_overbought: float = 80.0
    
    # Mean reversion parameters
    entry_threshold: float = 0.95      # Entry at 95% of band
    exit_threshold: float = 0.50       # Exit at 50% (middle band)
    mean_reversion_window: int = 50    # Lookback for mean calculation
    
    # Position management
    position_size: float = 0.03        # 3% of portfolio per trade
    max_position_size: float = 0.08    # 8% max total exposure
    
    # Risk management
    stop_loss: float = 0.02            # 2% stop loss
    take_profit: float = 0.015         # 1.5% take profit
    max_holding_time: int = 48         # hours
    
    # Market regime filters
    use_trend_filter: bool = True
    trend_filter_period: int = 100     # Period for trend detection
    min_volatility_filter: bool = True
    max_volatility_filter: bool = True
    
    # Volume confirmation
    volume_confirmation: bool = True
    min_volume_ratio: float = 0.8      # 80% of average (lower for mean reversion)


@dataclass
class MeanReversionPosition:
    """Mean reversion position tracking"""
    symbol: str = ""
    state: MeanReversionState = MeanReversionState.WAITING_FOR_SETUP
    regime: MarketRegime = MarketRegime.RANGING
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    target_price: float = 0.0          # Mean reversion target
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    unrealized_pnl: float = 0.0
    reversion_score: float = 0.0       # Strength of mean reversion signal
    hours_held: int = 0
    is_long: bool = True


class OverboughtOversoldDetector:
    """Detects overbought/oversold conditions for mean reversion"""
    
    def __init__(self, config: MeanReversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=self.config.bb_period).mean()
        std = prices.rolling(window=self.config.bb_period).std()
        
        upper_band = sma + (std * self.config.bb_std_dev)
        lower_band = sma - (std * self.config.bb_std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_squeeze(self, upper_band: pd.Series, lower_band: pd.Series, 
                      current_price: float) -> bool:
        """Detect Bollinger Band squeeze"""
        if len(upper_band) < 2:
            return False
        
        # Current band width
        current_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / current_price
        
        # Historical average band width
        recent_widths = []
        for i in range(min(20, len(upper_band))):
            if i < len(upper_band):
                width = (upper_band.iloc[-1-i] - lower_band.iloc[-1-i]) / upper_band.iloc[-1-i]
                recent_widths.append(width)
        
        if not recent_widths:
            return False
        
        avg_width = np.mean(recent_widths)
        return current_width < avg_width * 0.8  # 20% below average width
    
    def detect_extreme_levels(self, current_price: float, upper_band: float, 
                            lower_band: float, rsi: float) -> Tuple[bool, bool, float]:
        """Detect extreme overbought/oversold levels"""
        # Calculate position within Bollinger Bands
        band_position = (current_price - lower_band) / (upper_band - lower_band)
        
        # Extreme oversold conditions
        oversold_bb = band_position <= (1 - self.config.entry_threshold)  # Below 5% of band
        oversold_rsi = rsi <= self.config.rsi_extreme_oversold
        extreme_oversold = oversold_bb and oversold_rsi
        
        # Extreme overbought conditions
        overbought_bb = band_position >= self.config.entry_threshold  # Above 95% of band
        overbought_rsi = rsi >= self.config.rsi_extreme_overbought
        extreme_overbought = overbought_bb and overbought_rsi
        
        # Calculate reversion score
        rsi_score = abs(rsi - 50) / 50  # Normalized RSI distance from neutral
        bb_score = abs(band_position - 0.5) * 2  # Normalized BB distance from middle
        reversion_score = (rsi_score + bb_score) / 2
        
        return extreme_oversold, extreme_overbought, reversion_score


class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Band mean reversion strategy.
    
    Strategy Logic:
    - Enter long when price touches or breaks below lower band
    - Enter short when price touches or breaks above upper band
    - Exit when price returns to middle band (SMA)
    - Use squeeze detection to avoid low volatility periods
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
        self.mean_config = MeanReversionConfig(**self.self.config.get('mean_reversion_parameters', {}))
        
        # Initialize detector
        self.detector = OverboughtOversoldDetector(self.mean_config)
        
        # Position tracking
        self.positions: Dict[str, MeanReversionPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = MeanReversionPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main Bollinger Band mean reversion analysis"""
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
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.detector.calculate_bollinger_bands(df['close'])
            
            if len(upper_band) == 0:
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
            
            current_upper = upper_band.iloc[-1]
            current_middle = middle_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # Calculate additional indicators
            rsi = self.detector.calculate_rsi(df['close'])
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50.0
            
            # Market regime detection
            regime = self._detect_market_regime(df, upper_band, lower_band)
            position.regime = regime
            
            # Update position time
            position.hours_held += 1
            
            # Generate signals based on position state
            if position.state == MeanReversionState.WAITING_FOR_SETUP:
                return await self._analyze_bb_entry(
                    symbol, df, current_price, current_upper, current_middle, 
                    current_lower, current_rsi, regime
                )
            else:
                return await self._analyze_bb_exit(
                    symbol, df, current_price, current_middle, current_rsi
                )
                
        except Exception as e:
            self.logger.error(f"Bollinger Band analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"Bollinger Band analysis failed: {e}")
    
    def _detect_market_regime(self, df: pd.DataFrame, upper_band: pd.Series, 
                            lower_band: pd.Series) -> MarketRegime:
        """Detect current market regime"""
        if len(df) < self.mean_config.trend_filter_period:
            return MarketRegime.RANGING
        
        # Trend detection using moving averages
        short_ma = df['close'].rolling(window=20).mean()
        long_ma = df['close'].rolling(window=50).mean()
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            return MarketRegime.RANGING
        
        # Current trend
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        # Volatility detection
        current_price = df['close'].iloc[-1]
        band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / current_price
        avg_band_width = np.mean([
            (upper_band.iloc[-1-i] - lower_band.iloc[-1-i]) / df['close'].iloc[-1-i]
            for i in range(min(20, len(upper_band)))
        ])
        
        # Classify regime
        if band_width > avg_band_width * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif band_width < avg_band_width * 0.7:
            return MarketRegime.LOW_VOLATILITY
        elif current_short > current_long and current_short > prev_short:
            return MarketRegime.TRENDING_UP
        elif current_short < current_long and current_short < prev_short:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    async def _analyze_bb_entry(self, symbol: str, df: pd.DataFrame, current_price: float,
                               upper_band: float, middle_band: float, lower_band: float,
                               rsi: float, regime: MarketRegime) -> StrategyResult:
        """Analyze Bollinger Band entry signals"""
        position = self.positions[symbol]
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Calculate band position
        band_position = (current_price - lower_band) / (upper_band - lower_band)
        
        # Check for favorable regime
        if self.mean_config.use_trend_filter and regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.2,
                metadata={
                    'strategy_action': 'trend_filter_wait',
                    'regime': regime.value,
                    'reason': 'Waiting for ranging market'
                }
            )
        
        # Detect squeeze (avoid low volatility)
        squeeze_detected = self.detector.detect_squeeze(
            pd.Series([upper_band]), pd.Series([lower_band]), current_price
        )
        if squeeze_detected and self.mean_config.min_volatility_filter:
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.3,
                metadata={
                    'strategy_action': 'squeeze_detected',
                    'reason': 'Bollinger Band squeeze - low volatility'
                }
            )
        
        # Long entry (oversold)
        if (band_position <= (1 - self.mean_config.entry_threshold) and 
            rsi <= self.mean_config.rsi_oversold and volume_confirmed):
            
            position.state = MeanReversionState.LONG_REVERSION
            position.is_long = True
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.target_price = middle_band
            position.stop_loss_price = current_price * (1 - self.mean_config.stop_loss)
            position.take_profit_price = current_price * (1 + self.mean_config.take_profit)
            position.reversion_score = (1 - band_position) + ((self.mean_config.rsi_oversold - rsi) / self.mean_config.rsi_oversold)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.target_price,  # Target middle band
                position_size=self.mean_config.position_size,
                metadata={
                    'strategy_action': 'bb_long_entry',
                    'band_position': band_position,
                    'rsi_value': rsi,
                    'target_band': 'middle',
                    'reversion_score': position.reversion_score,
                    'regime': regime.value
                }
            )
        
        # Short entry (overbought)
        elif (band_position >= self.mean_config.entry_threshold and 
              rsi >= self.mean_config.rsi_overbought and volume_confirmed):
            
            position.state = MeanReversionState.SHORT_REVERSION
            position.is_long = False
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.target_price = middle_band
            position.stop_loss_price = current_price * (1 + self.mean_config.stop_loss)
            position.take_profit_price = current_price * (1 - self.mean_config.take_profit)
            position.reversion_score = band_position + ((rsi - self.mean_config.rsi_overbought) / (100 - self.mean_config.rsi_overbought))
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.target_price,  # Target middle band
                position_size=self.mean_config.position_size,
                metadata={
                    'strategy_action': 'bb_short_entry',
                    'band_position': band_position,
                    'rsi_value': rsi,
                    'target_band': 'middle',
                    'reversion_score': position.reversion_score,
                    'regime': regime.value
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_extreme',
                'band_position': band_position,
                'rsi_value': rsi,
                'regime': regime.value,
                'squeeze_detected': squeeze_detected
            }
        )
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if not self.mean_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # For mean reversion, we want normal or slightly below average volume
        return current_volume >= avg_volume * self.mean_config.min_volume_ratio
    
    async def _analyze_bb_exit(self, symbol: str, df: pd.DataFrame, current_price: float,
                              middle_band: float, rsi: float) -> StrategyResult:
        """Analyze Bollinger Band exit signals"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.is_long:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Mean reversion target reached (middle band)
        if position.is_long and current_price >= middle_band * self.mean_config.exit_threshold:
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.9,
                metadata={
                    'strategy_action': 'mean_reversion_target',
                    'unrealized_pnl': position.unrealized_pnl,
                    'target_price': middle_band,
                    'reason': 'Price reverted to mean'
                }
            )
        elif not position.is_long and current_price <= middle_band * (2 - self.mean_config.exit_threshold):
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'mean_reversion_target',
                    'unrealized_pnl': position.unrealized_pnl,
                    'target_price': middle_band,
                    'reason': 'Price reverted to mean'
                }
            )
        
        # RSI reversal signals
        if position.is_long and rsi >= self.mean_config.rsi_overbought:
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.8,
                metadata={
                    'strategy_action': 'rsi_reversal_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': rsi,
                    'reason': 'RSI overbought - take profits'
                }
            )
        elif not position.is_long and rsi <= self.mean_config.rsi_oversold:
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.8,
                metadata={
                    'strategy_action': 'rsi_reversal_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': rsi,
                    'reason': 'RSI oversold - cover shorts'
                }
            )
        
        # Standard exit conditions
        return await self._standard_exit_analysis(symbol, current_price)
    
    async def _standard_exit_analysis(self, symbol: str, current_price: float) -> StrategyResult:
        """Standard exit analysis"""
        position = self.positions[symbol]
        
        # Check stop loss
        if ((position.is_long and current_price <= position.stop_loss_price) or
            (not position.is_long and current_price >= position.stop_loss_price)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'stop_loss_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'reason': 'Stop loss triggered'
                }
            )
        
        # Check take profit
        if ((position.is_long and current_price >= position.take_profit_price) or
            (not position.is_long and current_price <= position.take_profit_price)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'take_profit_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'reason': 'Take profit target reached'
                }
            )
        
        # Time-based exit
        if position.hours_held >= self.mean_config.max_holding_time:
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.7,
                metadata={
                    'strategy_action': 'time_based_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'hours_held': position.hours_held,
                    'reason': 'Maximum holding time reached'
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.6,
            metadata={
                'strategy_action': 'hold_position',
                'unrealized_pnl': position.unrealized_pnl,
                'hours_held': position.hours_held
            }
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return ["1H", "4H"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate Bollinger Band configuration"""
        try:
            mr_params = self.config.get('mean_reversion_parameters', {})
            
            # Validate Bollinger Band parameters
            bb_period = mr_params.get('bb_period', 20)
            if not 5 <= bb_period <= 100:
                raise StrategyConfigError("bb_period must be between 5 and 100")
            
            bb_std_dev = mr_params.get('bb_std_dev', 2.0)
            if not 1.0 <= bb_std_dev <= 4.0:
                raise StrategyConfigError("bb_std_dev must be between 1.0 and 4.0")
            
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise StrategyConfigError(f"Invalid Bollinger Band configuration: {e}")


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    
    Strategy Logic:
    - Enter long when RSI is extremely oversold (< 20)
    - Enter short when RSI is extremely overbought (> 80)
    - Exit when RSI returns to neutral zone (40-60)
    - Use multiple timeframes for confirmation
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
        
        self.mean_config = MeanReversionConfig(**self.self.config.get('mean_reversion_parameters', {}))
        self.detector = OverboughtOversoldDetector(self.mean_config)
        
        self.positions: Dict[str, MeanReversionPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = MeanReversionPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main RSI mean reversion analysis"""
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
            
            # Calculate RSI
            rsi = self.detector.calculate_rsi(df['close'])
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50.0
            
            # Multi-timeframe RSI if available
            rsi_higher_tf = None
            if len(data) > 1:
                higher_tf = list(data.keys())[1]
                if not data[higher_tf].empty:
                    rsi_higher = self.detector.calculate_rsi(data[higher_tf]['close'])
                    rsi_higher_tf = rsi_higher.iloc[-1] if len(rsi_higher) > 0 else 50.0
            
            # Update position time
            position.hours_held += 1
            
            # Generate signals
            if position.state == MeanReversionState.WAITING_FOR_SETUP:
                return await self._analyze_rsi_entry(symbol, df, current_rsi, rsi_higher_tf)
            else:
                return await self._analyze_rsi_exit(symbol, df, current_price, current_rsi)
                
        except Exception as e:
            self.logger.error(f"RSI mean reversion analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"RSI mean reversion analysis failed: {e}")
    
    async def _analyze_rsi_entry(self, symbol: str, df: pd.DataFrame, 
                               current_rsi: float, rsi_higher_tf: Optional[float]) -> StrategyResult:
        """Analyze RSI mean reversion entry signals"""
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Multi-timeframe confirmation
        timeframe_confirmed = True
        if rsi_higher_tf is not None:
            # For long: both timeframes should be oversold
            # For short: both timeframes should be overbought
            timeframe_confirmed = (
                (current_rsi <= self.mean_config.rsi_extreme_oversold and 
                 rsi_higher_tf <= self.mean_config.rsi_oversold) or
                (current_rsi >= self.mean_config.rsi_extreme_overbought and 
                 rsi_higher_tf >= self.mean_config.rsi_overbought)
            )
        
        # Extreme oversold entry (long)
        if (current_rsi <= self.mean_config.rsi_extreme_oversold and 
            volume_confirmed and timeframe_confirmed):
            
            position.state = MeanReversionState.LONG_REVERSION
            position.is_long = True
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.target_price = current_price * 1.02  # 2% reversion target
            position.stop_loss_price = current_price * (1 - self.mean_config.stop_loss)
            position.take_profit_price = current_price * (1 + self.mean_config.take_profit)
            position.reversion_score = (self.mean_config.rsi_extreme_oversold - current_rsi) / self.mean_config.rsi_extreme_oversold
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.85,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.mean_config.position_size,
                metadata={
                    'strategy_action': 'rsi_extreme_oversold_long',
                    'rsi_value': current_rsi,
                    'rsi_higher_tf': rsi_higher_tf,
                    'reversion_score': position.reversion_score,
                    'timeframe_confirmed': timeframe_confirmed
                }
            )
        
        # Extreme overbought entry (short)
        elif (current_rsi >= self.mean_config.rsi_extreme_overbought and 
              volume_confirmed and timeframe_confirmed):
            
            position.state = MeanReversionState.SHORT_REVERSION
            position.is_long = False
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.target_price = current_price * 0.98  # 2% reversion target
            position.stop_loss_price = current_price * (1 + self.mean_config.stop_loss)
            position.take_profit_price = current_price * (1 - self.mean_config.take_profit)
            position.reversion_score = (current_rsi - self.mean_config.rsi_extreme_overbought) / (100 - self.mean_config.rsi_extreme_overbought)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.85,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.mean_config.position_size,
                metadata={
                    'strategy_action': 'rsi_extreme_overbought_short',
                    'rsi_value': current_rsi,
                    'rsi_higher_tf': rsi_higher_tf,
                    'reversion_score': position.reversion_score,
                    'timeframe_confirmed': timeframe_confirmed
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_extreme_rsi',
                'rsi_value': current_rsi,
                'rsi_higher_tf': rsi_higher_tf,
                'extreme_oversold_threshold': self.mean_config.rsi_extreme_oversold,
                'extreme_overbought_threshold': self.mean_config.rsi_extreme_overbought
            }
        )
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if not self.mean_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.mean_config.min_volume_ratio
    
    async def _analyze_rsi_exit(self, symbol: str, df: pd.DataFrame, 
                              current_price: float, current_rsi: float) -> StrategyResult:
        """Analyze RSI mean reversion exit signals"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.is_long:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # RSI return to neutral zone
        neutral_zone_min = 40.0
        neutral_zone_max = 60.0
        
        if (position.is_long and current_rsi >= neutral_zone_min):
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.8,
                metadata={
                    'strategy_action': 'rsi_neutral_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': current_rsi,
                    'reason': 'RSI returned to neutral zone'
                }
            )
        elif (not position.is_long and current_rsi <= neutral_zone_max):
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.8,
                metadata={
                    'strategy_action': 'rsi_neutral_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': current_rsi,
                    'reason': 'RSI returned to neutral zone'
                }
            )
        
        # Standard exit analysis
        return await self._standard_exit_analysis(symbol, current_price)
    
    async def _standard_exit_analysis(self, symbol: str, current_price: float) -> StrategyResult:
        """Standard exit analysis"""
        position = self.positions[symbol]
        
        # Check stop loss
        if ((position.is_long and current_price <= position.stop_loss_price) or
            (not position.is_long and current_price >= position.stop_loss_price)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'stop_loss_exit',
                    'unrealized_pnl': position.unrealized_pnl
                }
            )
        
        # Check take profit
        if ((position.is_long and current_price >= position.take_profit_price) or
            (not position.is_long and current_price <= position.take_profit_price)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'take_profit_exit',
                    'unrealized_pnl': position.unrealized_pnl
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.6,
            metadata={
                'strategy_action': 'hold_position',
                'unrealized_pnl': position.unrealized_pnl,
                'hours_held': position.hours_held
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
        """Validate RSI mean reversion configuration"""


class CombinedMeanReversionStrategy(BaseStrategy):
    """
    Combined mean reversion strategy using multiple indicators.
    
    Strategy Logic:
    - Combines Bollinger Bands, RSI, and additional filters
    - Requires multiple confirmation signals
    - Higher confidence trades with reduced false signals
    - Market regime adaptation
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
        
        self.mean_config = MeanReversionConfig(**self.self.config.get('mean_reversion_parameters', {}))
        self.detector = OverboughtOversoldDetector(self.mean_config)
        
        self.positions: Dict[str, MeanReversionPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = MeanReversionPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Combined mean reversion analysis"""
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
            
            # Calculate all indicators
            upper_band, middle_band, lower_band = self.detector.calculate_bollinger_bands(df['close'])
            rsi = self.detector.calculate_rsi(df['close'])
            
            if len(upper_band) == 0 or len(rsi) == 0:
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
            
            current_upper = upper_band.iloc[-1]
            current_middle = middle_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Detect extreme conditions
            extreme_oversold, extreme_overbought, reversion_score = self.detector.detect_extreme_levels(
                current_price, current_upper, current_lower, current_rsi
            )
            
            # Update position time
            position.hours_held += 1
            
            # Generate signals
            if position.state == MeanReversionState.WAITING_FOR_SETUP:
                return await self._analyze_combined_entry(
                    symbol, df, current_price, current_middle, extreme_oversold, 
                    extreme_overbought, reversion_score
                )
            else:
                return await self._analyze_combined_exit(
                    symbol, df, current_price, current_middle, current_rsi
                )
                
        except Exception as e:
            self.logger.error(f"Combined mean reversion analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"Combined mean reversion analysis failed: {e}")
    
    async def _analyze_combined_entry(self, symbol: str, df: pd.DataFrame, current_price: float,
                                    middle_band: float, extreme_oversold: bool, 
                                    extreme_overbought: bool, reversion_score: float) -> StrategyResult:
        """Analyze combined mean reversion entry"""
        position = self.positions[symbol]
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Additional confirmation filters
        price_distance_from_mean = abs(current_price - middle_band) / middle_band
        sufficient_distance = price_distance_from_mean >= 0.02  # 2% minimum distance
        
        # Combined long entry
        if (extreme_oversold and volume_confirmed and sufficient_distance and 
            reversion_score >= self.mean_config.reversion_score_threshold):
            
            position.state = MeanReversionState.LONG_REVERSION
            position.is_long = True
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.target_price = middle_band
            position.stop_loss_price = current_price * (1 - self.mean_config.stop_loss)
            position.take_profit_price = current_price * (1 + self.mean_config.take_profit)
            position.reversion_score = reversion_score
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.9,  # Higher confidence with multiple confirmations
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.target_price,
                position_size=self.mean_config.position_size,
                metadata={
                    'strategy_action': 'combined_long_entry',
                    'reversion_score': reversion_score,
                    'price_distance_from_mean': price_distance_from_mean,
                    'extreme_oversold': extreme_oversold,
                    'volume_confirmed': volume_confirmed
                }
            )
        
        # Combined short entry
        elif (extreme_overbought and volume_confirmed and sufficient_distance and 
              reversion_score >= self.mean_config.reversion_score_threshold):
            
            position.state = MeanReversionState.SHORT_REVERSION
            position.is_long = False
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.target_price = middle_band
            position.stop_loss_price = current_price * (1 + self.mean_config.stop_loss)
            position.take_profit_price = current_price * (1 - self.mean_config.take_profit)
            position.reversion_score = reversion_score
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.9,  # Higher confidence with multiple confirmations
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.target_price,
                position_size=self.mean_config.position_size,
                metadata={
                    'strategy_action': 'combined_short_entry',
                    'reversion_score': reversion_score,
                    'price_distance_from_mean': price_distance_from_mean,
                    'extreme_overbought': extreme_overbought,
                    'volume_confirmed': volume_confirmed
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_combined_signal',
                'reversion_score': reversion_score,
                'extreme_oversold': extreme_oversold,
                'extreme_overbought': extreme_overbought,
                'volume_confirmed': volume_confirmed,
                'price_distance_from_mean': price_distance_from_mean
            }
        )
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if not self.mean_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.mean_config.min_volume_ratio
    
    async def _analyze_combined_exit(self, symbol: str, df: pd.DataFrame, 
                                   current_price: float, middle_band: float, 
                                   current_rsi: float) -> StrategyResult:
        """Analyze combined mean reversion exit"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.is_long:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Mean reversion target (approach middle band)
        distance_to_mean = abs(current_price - middle_band) / middle_band
        
        if distance_to_mean <= 0.005:  # Within 0.5% of mean
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'mean_reversion_complete',
                    'unrealized_pnl': position.unrealized_pnl,
                    'distance_to_mean': distance_to_mean,
                    'reason': 'Price successfully reverted to mean'
                }
            )
        
        # RSI neutralization
        if ((position.is_long and 45 <= current_rsi <= 55) or
            (not position.is_long and 45 <= current_rsi <= 55)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.8,
                metadata={
                    'strategy_action': 'rsi_neutralization_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': current_rsi,
                    'reason': 'RSI returned to neutral zone'
                }
            )
        
        # Standard exit analysis
        return await self._standard_exit_analysis(symbol, current_price)
    
    async def _standard_exit_analysis(self, symbol: str, current_price: float) -> StrategyResult:
        """Standard exit analysis"""
        position = self.positions[symbol]
        
        # Check stop loss
        if ((position.is_long and current_price <= position.stop_loss_price) or
            (not position.is_long and current_price >= position.stop_loss_price)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'stop_loss_exit',
                    'unrealized_pnl': position.unrealized_pnl
                }
            )
        
        # Check take profit
        if ((position.is_long and current_price >= position.take_profit_price) or
            (not position.is_long and current_price <= position.take_profit_price)):
            
            position.state = MeanReversionState.WAITING_FOR_SETUP
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.is_long else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'take_profit_exit',
                    'unrealized_pnl': position.unrealized_pnl
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.6,
            metadata={
                'strategy_action': 'hold_position',
                'unrealized_pnl': position.unrealized_pnl,
                'hours_held': position.hours_held
            }
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return ["1H", "4H"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate combined mean reversion configuration"""
