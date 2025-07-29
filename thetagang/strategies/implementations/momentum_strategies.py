"""
🎯 MOMENTUM STRATEGIES IMPLEMENTATION
===================================

Advanced momentum-based trading strategies leveraging RSI, MACD, and other 
momentum indicators for short to medium-term trading opportunities.

Key Features:
- RSI-based momentum detection
- MACD signal line crossovers
- High-frequency scalping strategies
- Dual momentum confirmation systems
- Multi-timeframe momentum analysis
- Dynamic position sizing
- Risk management with momentum-based stops
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


class MomentumState(Enum):
    """Momentum strategy states"""
    LOOKING_FOR_ENTRY = "looking_for_entry"
    LONG_POSITION = "long_position"
    SHORT_POSITION = "short_position"
    RISK_MANAGEMENT = "risk_management"
    PROFIT_TAKING = "profit_taking"


class MomentumDirection(Enum):
    """Momentum direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    CHOPPY = "choppy"


@dataclass
class MomentumConfig:
    """Configuration for momentum strategies"""
    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_entry_threshold: float = 50.0
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Position management
    position_size: float = 0.02  # 2% of portfolio
    max_position_size: float = 0.05  # 5% max per trade
    
    # Risk management
    stop_loss: float = 0.015  # 1.5%
    take_profit: float = 0.025  # 2.5%
    trailing_stop: float = 0.01  # 1%
    max_holding_time: int = 24  # hours
    
    # Entry/exit criteria
    momentum_threshold: float = 0.5
    volume_confirmation: bool = True
    min_volume_ratio: float = 1.2  # 20% above average
    
    # Scalping specific
    scalp_profit_target: float = 0.005  # 0.5%
    scalp_stop_loss: float = 0.003  # 0.3%
    max_scalp_positions: int = 3


@dataclass
class MomentumPosition:
    """Momentum position tracking"""
    symbol: str = ""
    state: MomentumState = MomentumState.LOOKING_FOR_ENTRY
    direction: MomentumDirection = MomentumDirection.NEUTRAL
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    quantity: int = 0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    unrealized_pnl: float = 0.0
    momentum_score: float = 0.0
    hours_held: int = 0


class RSIMomentumStrategy(BaseStrategy):
    """
    RSI-based momentum strategy for stocks.
    
    Strategy Logic:
    - Enter long when RSI crosses above entry threshold with volume confirmation
    - Enter short when RSI crosses below (100 - entry threshold) with volume confirmation
    - Use dynamic stop losses based on recent volatility
    - Take profits at overbought/oversold levels
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
        self.momentum_config = MomentumConfig(**self.self.config.get('momentum_parameters', {}))
        
        # Position tracking
        self.positions: Dict[str, MomentumPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = MomentumPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main RSI momentum analysis"""
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
            rsi_values = self._calculate_rsi(df['close'], self.momentum_config.rsi_period)
            current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50.0
            
            # Calculate volume confirmation
            volume_confirmed = self._check_volume_confirmation(df)
            
            # Update position state
            position.hours_held += 1
            
            # Generate signals based on position state
            if position.state == MomentumState.LOOKING_FOR_ENTRY:
                return await self._analyze_entry_signals(symbol, df, current_rsi, volume_confirmed)
            elif position.state in [MomentumState.LONG_POSITION, MomentumState.SHORT_POSITION]:
                return await self._analyze_exit_signals(symbol, df, current_rsi, current_price)
            else:
                return await self._analyze_risk_management(symbol, current_price)
                
        except Exception as e:
            self.logger.error(f"RSI momentum analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"RSI momentum analysis failed: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms the momentum"""
        if not self.momentum_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.momentum_config.min_volume_ratio
    
    async def _analyze_entry_signals(self, symbol: str, df: pd.DataFrame, 
                                   current_rsi: float, volume_confirmed: bool) -> StrategyResult:
        """Analyze entry signals"""
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Long entry signal
        if (current_rsi > self.momentum_config.rsi_entry_threshold and 
            current_rsi < self.momentum_config.rsi_overbought and 
            volume_confirmed):
            
            # Setup long position
            position.state = MomentumState.LONG_POSITION
            position.direction = MomentumDirection.BULLISH
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.stop_loss_price = current_price * (1 - self.momentum_config.stop_loss)
            position.take_profit_price = current_price * (1 + self.momentum_config.take_profit)
            position.momentum_score = (current_rsi - 50) / 50  # Normalized momentum
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size,
                metadata={
                    'strategy_action': 'enter_long',
                    'rsi_value': current_rsi,
                    'volume_confirmed': volume_confirmed,
                    'momentum_score': position.momentum_score
                }
            )
        
        # Short entry signal
        elif (current_rsi < (100 - self.momentum_config.rsi_entry_threshold) and 
              current_rsi > self.momentum_config.rsi_oversold and 
              volume_confirmed):
            
            # Setup short position
            position.state = MomentumState.SHORT_POSITION
            position.direction = MomentumDirection.BEARISH
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.stop_loss_price = current_price * (1 + self.momentum_config.stop_loss)
            position.take_profit_price = current_price * (1 - self.momentum_config.take_profit)
            position.momentum_score = (50 - current_rsi) / 50  # Normalized momentum
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size,
                metadata={
                    'strategy_action': 'enter_short',
                    'rsi_value': current_rsi,
                    'volume_confirmed': volume_confirmed,
                    'momentum_score': position.momentum_score
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_entry',
                'rsi_value': current_rsi,
                'volume_confirmed': volume_confirmed
            }
        )
    
    async def _analyze_exit_signals(self, symbol: str, df: pd.DataFrame,
                                  current_rsi: float, current_price: float) -> StrategyResult:
        """Analyze exit signals"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.direction == MomentumDirection.BULLISH:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Check stop loss
        if ((position.direction == MomentumDirection.BULLISH and current_price <= position.stop_loss_price) or
            (position.direction == MomentumDirection.BEARISH and current_price >= position.stop_loss_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'stop_loss_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'reason': 'Stop loss triggered'
                }
            )
        
        # Check take profit
        if ((position.direction == MomentumDirection.BULLISH and current_price >= position.take_profit_price) or
            (position.direction == MomentumDirection.BEARISH and current_price <= position.take_profit_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'take_profit_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'reason': 'Take profit target reached'
                }
            )
        
        # Check momentum reversal
        if ((position.direction == MomentumDirection.BULLISH and current_rsi <= self.momentum_config.rsi_oversold) or
            (position.direction == MomentumDirection.BEARISH and current_rsi >= self.momentum_config.rsi_overbought)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.8,
                metadata={
                    'strategy_action': 'momentum_reversal_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': current_rsi,
                    'reason': 'Momentum reversal detected'
                }
            )
        
        # Check maximum holding time
        if position.hours_held >= self.momentum_config.max_holding_time:
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
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
                'rsi_value': current_rsi,
                'hours_held': position.hours_held
            }
        )
    
    async def _analyze_risk_management(self, symbol: str, current_price: float) -> StrategyResult:
        """Risk management analysis"""
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.SELL,
            confidence=0.95,
            metadata={
                'strategy_action': 'risk_management_exit',
                'reason': 'Risk management triggered'
            }
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return {TimeFrame.MIN_5, TimeFrame.MIN_15, TimeFrame.HOUR_1}
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate strategy configuration"""
        try:
            momentum_params = self.self.config.get('momentum_parameters', {})
            
            # Validate RSI parameters
            rsi_period = momentum_params.get('rsi_period', 14)
            if not 2 <= rsi_period <= 50:
                raise StrategyConfigError("rsi_period must be between 2 and 50")
            
            # Validate thresholds
            rsi_oversold = momentum_params.get('rsi_oversold', 30.0)
            rsi_overbought = momentum_params.get('rsi_overbought', 70.0)
            if rsi_oversold >= rsi_overbought:
                raise StrategyConfigError("rsi_oversold must be less than rsi_overbought")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise StrategyConfigError(f"Invalid RSI momentum configuration: {e}")


class MACDMomentumStrategy(BaseStrategy):
    """
    MACD-based momentum strategy.
    
    Strategy Logic:
    - Enter long on MACD bullish crossover above zero line
    - Enter short on MACD bearish crossover below zero line
    - Use histogram for momentum confirmation
    - Exit on signal line crossover or momentum divergence
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
        
        self.momentum_config = MomentumConfig(**self.self.config.get('momentum_parameters', {}))
        self.positions: Dict[str, MomentumPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = MomentumPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main MACD momentum analysis"""
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
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(
                df['close'], 
                self.momentum_config.macd_fast,
                self.momentum_config.macd_slow,
                self.momentum_config.macd_signal
            )
            
            current_macd = macd_line.iloc[-1] if len(macd_line) > 0 else 0.0
            current_signal = signal_line.iloc[-1] if len(signal_line) > 0 else 0.0
            current_histogram = histogram.iloc[-1] if len(histogram) > 0 else 0.0
            
            # Detect crossovers
            bullish_crossover = self._detect_bullish_crossover(macd_line, signal_line)
            bearish_crossover = self._detect_bearish_crossover(macd_line, signal_line)
            
            # Update position state
            position.hours_held += 1
            
            # Generate signals based on MACD analysis
            if position.state == MomentumState.LOOKING_FOR_ENTRY:
                return await self._analyze_macd_entry(
                    symbol, df, current_macd, current_signal, current_histogram,
                    bullish_crossover, bearish_crossover
                )
            else:
                return await self._analyze_macd_exit(
                    symbol, df, current_macd, current_signal, current_histogram,
                    bullish_crossover, bearish_crossover, current_price
                )
                
        except Exception as e:
            self.logger.error(f"MACD momentum analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"MACD momentum analysis failed: {e}")
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _detect_bullish_crossover(self, macd: pd.Series, signal: pd.Series) -> bool:
        """Detect bullish MACD crossover"""
        if len(macd) < 2:
            return False
        return macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
    
    def _detect_bearish_crossover(self, macd: pd.Series, signal: pd.Series) -> bool:
        """Detect bearish MACD crossover"""
        if len(macd) < 2:
            return False
        return macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]
    
    async def _analyze_macd_entry(self, symbol: str, df: pd.DataFrame,
                                 macd: float, signal: float, histogram: float,
                                 bullish_cross: bool, bearish_cross: bool) -> StrategyResult:
        """Analyze MACD entry signals"""
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Bullish entry
        if bullish_cross and macd > 0 and histogram > 0 and volume_confirmed:
            position.state = MomentumState.LONG_POSITION
            position.direction = MomentumDirection.BULLISH
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.stop_loss_price = current_price * (1 - self.momentum_config.stop_loss)
            position.take_profit_price = current_price * (1 + self.momentum_config.take_profit)
            position.momentum_score = abs(macd - signal) / current_price  # Momentum strength
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=0.85,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size,
                metadata={
                    'strategy_action': 'macd_bullish_entry',
                    'macd_value': macd,
                    'signal_value': signal,
                    'histogram': histogram,
                    'momentum_score': position.momentum_score
                }
            )
        
        # Bearish entry
        elif bearish_cross and macd < 0 and histogram < 0 and volume_confirmed:
            position.state = MomentumState.SHORT_POSITION
            position.direction = MomentumDirection.BEARISH
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.stop_loss_price = current_price * (1 + self.momentum_config.stop_loss)
            position.take_profit_price = current_price * (1 - self.momentum_config.take_profit)
            position.momentum_score = abs(macd - signal) / current_price  # Momentum strength
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=0.85,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size,
                metadata={
                    'strategy_action': 'macd_bearish_entry',
                    'macd_value': macd,
                    'signal_value': signal,
                    'histogram': histogram,
                    'momentum_score': position.momentum_score
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_macd_signal',
                'macd_value': macd,
                'signal_value': signal,
                'histogram': histogram,
                'bullish_cross': bullish_cross,
                'bearish_cross': bearish_cross
            }
        )
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation for MACD strategy"""
        if not self.momentum_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.momentum_config.min_volume_ratio
    
    async def _analyze_macd_exit(self, symbol: str, df: pd.DataFrame,
                               macd: float, signal: float, histogram: float,
                               bullish_cross: bool, bearish_cross: bool,
                               current_price: float) -> StrategyResult:
        """Analyze MACD exit signals"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.direction == MomentumDirection.BULLISH:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Exit on opposite crossover
        if ((position.direction == MomentumDirection.BULLISH and bearish_cross) or
            (position.direction == MomentumDirection.BEARISH and bullish_cross)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'macd_crossover_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'reason': 'MACD signal crossover'
                }
            )
        
        # Standard exit conditions (stop loss, take profit, time)
        return await self._standard_exit_analysis(symbol, current_price)
    
    async def _standard_exit_analysis(self, symbol: str, current_price: float) -> StrategyResult:
        """Standard exit analysis for stop loss, take profit, time"""
        position = self.positions[symbol]
        
        # Check stop loss
        if ((position.direction == MomentumDirection.BULLISH and current_price <= position.stop_loss_price) or
            (position.direction == MomentumDirection.BEARISH and current_price >= position.stop_loss_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'stop_loss_exit',
                    'unrealized_pnl': position.unrealized_pnl
                }
            )
        
        # Check take profit
        if ((position.direction == MomentumDirection.BULLISH and current_price >= position.take_profit_price) or
            (position.direction == MomentumDirection.BEARISH and current_price <= position.take_profit_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
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
        return ["15M", "1H", "4H"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate MACD strategy configuration"""
        try:
            momentum_params = self.config.get('momentum_parameters', {})
            
            # Validate MACD parameters
            macd_fast = momentum_params.get('macd_fast', 12)
            macd_slow = momentum_params.get('macd_slow', 26)
            if macd_fast >= macd_slow:
                raise StrategyConfigError("macd_fast must be less than macd_slow")
            
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise StrategyConfigError(f"Invalid MACD momentum configuration: {e}")


class MomentumScalperStrategy(BaseStrategy):
    """
    High-frequency momentum scalping strategy for 5M/15M timeframes.
    
    Strategy Logic:
    - Quick entry/exit on short-term momentum bursts
    - Small profit targets with tight stops
    - Multiple concurrent positions
    - Volume and volatility filters
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
        
        self.momentum_config = MomentumConfig(**self.self.config.get('momentum_parameters', {}))
        self.positions: Dict[str, List[MomentumPosition]] = {}  # Multiple positions per symbol
        for symbol in symbols:
            self.positions[symbol] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main scalping analysis"""
        try:
            # Get 5M timeframe data
            tf_5m = "5M"
            if tf_5m not in data or data[tf_5m].empty:
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
            
            df = data[tf_5m]
            current_price = df['close'].iloc[-1]
            
            # Clean up expired positions
            self._cleanup_positions(symbol)
            
            # Check if we can open new positions
            active_positions = len([p for p in self.positions[symbol] 
                                  if p.state in [MomentumState.LONG_POSITION, MomentumState.SHORT_POSITION]])
            
            if active_positions < self.momentum_config.max_scalp_positions:
                # Look for new scalping opportunities
                scalp_signal = await self._analyze_scalp_entry(symbol, df)
                if scalp_signal.signal != StrategySignal.HOLD:
                    return scalp_signal
            
            # Manage existing positions
            for position in self.positions[symbol]:
                if position.state in [MomentumState.LONG_POSITION, MomentumState.SHORT_POSITION]:
                    exit_signal = await self._analyze_scalp_exit(position, current_price)
                    if exit_signal.signal != StrategySignal.HOLD:
                        return exit_signal
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.5,
                metadata={
                    'strategy_action': 'monitoring_scalp_positions',
                    'active_positions': active_positions,
                    'max_positions': self.momentum_config.max_scalp_positions
                }
            )
            
        except Exception as e:
            self.logger.error(f"Momentum scalper analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"Momentum scalper analysis failed: {e}")
    
    def _cleanup_positions(self, symbol: str):
        """Remove completed positions"""
        self.positions[symbol] = [p for p in self.positions[symbol] 
                                 if p.state != MomentumState.LOOKING_FOR_ENTRY]
    
    async def _analyze_scalp_entry(self, symbol: str, df: pd.DataFrame) -> StrategyResult:
        """Analyze scalping entry opportunities"""
        current_price = df['close'].iloc[-1]
        
        # Calculate short-term momentum
        if len(df) < 10:
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
        
        # Price momentum (last 5 candles)
        price_change = (current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]
        
        # Volume spike
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-10:].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volatility check
        recent_range = (df['high'].iloc[-5:].max() - df['low'].iloc[-5:].min()) / current_price
        
        # Entry conditions
        strong_momentum = abs(price_change) > 0.003  # 0.3% move
        volume_confirmation = volume_ratio > 1.5
        sufficient_volatility = recent_range > 0.005  # 0.5% range
        
        if strong_momentum and volume_confirmation and sufficient_volatility:
            # Create new position
            position = MomentumPosition(symbol=symbol)
            position.entry_price = current_price
            position.entry_time = datetime.now()
            
            if price_change > 0:  # Bullish momentum
                position.state = MomentumState.LONG_POSITION
                position.direction = MomentumDirection.BULLISH
                position.stop_loss_price = current_price * (1 - self.momentum_config.scalp_stop_loss)
                position.take_profit_price = current_price * (1 + self.momentum_config.scalp_profit_target)
                signal = StrategySignal.BUY
            else:  # Bearish momentum
                position.state = MomentumState.SHORT_POSITION
                position.direction = MomentumDirection.BEARISH
                position.stop_loss_price = current_price * (1 + self.momentum_config.scalp_stop_loss)
                position.take_profit_price = current_price * (1 - self.momentum_config.scalp_profit_target)
                signal = StrategySignal.SELL
            
            position.momentum_score = abs(price_change) * volume_ratio
            self.positions[symbol].append(position)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=signal,
                confidence=0.8,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size * 0.5,  # Smaller size for scalping
                metadata={
                    'strategy_action': 'scalp_entry',
                    'direction': position.direction.value,
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'momentum_score': position.momentum_score
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
    
    async def _analyze_scalp_exit(self, position: MomentumPosition, current_price: float) -> StrategyResult:
        """Analyze scalping exit conditions"""
        # Calculate P&L
        if position.direction == MomentumDirection.BULLISH:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Quick exits for scalping
        time_held = (datetime.now() - position.entry_time).total_seconds() / 60  # minutes
        
        # Stop loss
        if ((position.direction == MomentumDirection.BULLISH and current_price <= position.stop_loss_price) or
            (position.direction == MomentumDirection.BEARISH and current_price >= position.stop_loss_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'scalp_stop_loss',
                    'unrealized_pnl': position.unrealized_pnl,
                    'time_held_minutes': time_held
                }
            )
        
        # Take profit
        if ((position.direction == MomentumDirection.BULLISH and current_price >= position.take_profit_price) or
            (position.direction == MomentumDirection.BEARISH and current_price <= position.take_profit_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'scalp_take_profit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'time_held_minutes': time_held
                }
            )
        
        # Time-based exit (max 30 minutes for scalps)
        if time_held > 30:
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.7,
                metadata={
                    'strategy_action': 'scalp_time_exit',
                    'unrealized_pnl': position.unrealized_pnl,
                    'time_held_minutes': time_held
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,signal=StrategySignal.HOLD, confidence=0.0)
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes"""
        return ["5M", "15M"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate scalper configuration"""
        return True  # Basic validation for now


class DualMomentumStrategy(BaseStrategy):
    """
    Dual momentum strategy combining RSI and MACD for confirmation.
    
    Strategy Logic:
    - Requires both RSI and MACD to confirm momentum direction
    - Higher confidence signals with dual confirmation
    - Reduced false signals compared to single indicator strategies
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
        
        self.momentum_config = MomentumConfig(**self.self.config.get('momentum_parameters', {}))
        self.positions: Dict[str, MomentumPosition] = {}
        for symbol in symbols:
            self.positions[symbol] = MomentumPosition(symbol=symbol)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Dual momentum analysis with RSI and MACD confirmation"""
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
            
            # Calculate both indicators
            rsi_values = self._calculate_rsi(df['close'], self.momentum_config.rsi_period)
            current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50.0
            
            macd_line, signal_line, histogram = self._calculate_macd(
                df['close'], 
                self.momentum_config.macd_fast,
                self.momentum_config.macd_slow,
                self.momentum_config.macd_signal
            )
            
            current_macd = macd_line.iloc[-1] if len(macd_line) > 0 else 0.0
            current_signal = signal_line.iloc[-1] if len(signal_line) > 0 else 0.0
            
            # Analyze dual momentum signals
            if position.state == MomentumState.LOOKING_FOR_ENTRY:
                return await self._analyze_dual_entry(symbol, df, current_rsi, current_macd, current_signal)
            else:
                return await self._analyze_dual_exit(symbol, df, current_rsi, current_macd, current_price)
                
        except Exception as e:
            self.logger.error(f"Dual momentum analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"Dual momentum analysis failed: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    async def _analyze_dual_entry(self, symbol: str, df: pd.DataFrame,
                                 rsi: float, macd: float, signal: float) -> StrategyResult:
        """Analyze entry with dual momentum confirmation"""
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # RSI momentum signals
        rsi_bullish = rsi > self.momentum_config.rsi_entry_threshold and rsi < self.momentum_config.rsi_overbought
        rsi_bearish = rsi < (100 - self.momentum_config.rsi_entry_threshold) and rsi > self.momentum_config.rsi_oversold
        
        # MACD momentum signals
        macd_bullish = macd > signal and macd > 0
        macd_bearish = macd < signal and macd < 0
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # Dual confirmation for long entry
        if rsi_bullish and macd_bullish and volume_confirmed:
            position.state = MomentumState.LONG_POSITION
            position.direction = MomentumDirection.BULLISH
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.stop_loss_price = current_price * (1 - self.momentum_config.stop_loss)
            position.take_profit_price = current_price * (1 + self.momentum_config.take_profit)
            
            # Higher confidence with dual confirmation
            confidence = 0.9
            position.momentum_score = ((rsi - 50) / 50) * ((macd - signal) / current_price)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,
                confidence=confidence,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size,
                metadata={
                    'strategy_action': 'dual_momentum_long',
                    'rsi_value': rsi,
                    'macd_value': macd,
                    'signal_value': signal,
                    'rsi_bullish': rsi_bullish,
                    'macd_bullish': macd_bullish,
                    'momentum_score': position.momentum_score
                }
            )
        
        # Dual confirmation for short entry
        elif rsi_bearish and macd_bearish and volume_confirmed:
            position.state = MomentumState.SHORT_POSITION
            position.direction = MomentumDirection.BEARISH
            position.entry_price = current_price
            position.entry_time = datetime.now()
            position.stop_loss_price = current_price * (1 + self.momentum_config.stop_loss)
            position.take_profit_price = current_price * (1 - self.momentum_config.take_profit)
            
            # Higher confidence with dual confirmation
            confidence = 0.9
            position.momentum_score = ((50 - rsi) / 50) * ((signal - macd) / current_price)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL,
                confidence=confidence,
                target_price=current_price,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_size=self.momentum_config.position_size,
                metadata={
                    'strategy_action': 'dual_momentum_short',
                    'rsi_value': rsi,
                    'macd_value': macd,
                    'signal_value': signal,
                    'rsi_bearish': rsi_bearish,
                    'macd_bearish': macd_bearish,
                    'momentum_score': position.momentum_score
                }
            )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            metadata={
                'strategy_action': 'waiting_for_dual_confirmation',
                'rsi_value': rsi,
                'macd_value': macd,
                'rsi_bullish': rsi_bullish,
                'rsi_bearish': rsi_bearish,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish
            }
        )
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        if not self.momentum_config.volume_confirmation:
            return True
        
        if len(df) < 20:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        return current_volume >= avg_volume * self.momentum_config.min_volume_ratio
    
    async def _analyze_dual_exit(self, symbol: str, df: pd.DataFrame,
                               rsi: float, macd: float, current_price: float) -> StrategyResult:
        """Analyze exit with dual momentum indicators"""
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        if position.direction == MomentumDirection.BULLISH:
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
        
        # Check for momentum reversal in both indicators
        rsi_reversal_long = position.direction == MomentumDirection.BULLISH and rsi >= self.momentum_config.rsi_overbought
        rsi_reversal_short = position.direction == MomentumDirection.BEARISH and rsi <= self.momentum_config.rsi_oversold
        
        macd_reversal_long = position.direction == MomentumDirection.BULLISH and macd < 0
        macd_reversal_short = position.direction == MomentumDirection.BEARISH and macd > 0
        
        # Exit on dual reversal signal
        if (rsi_reversal_long and macd_reversal_long) or (rsi_reversal_short and macd_reversal_short):
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.9,
                metadata={
                    'strategy_action': 'dual_momentum_reversal',
                    'unrealized_pnl': position.unrealized_pnl,
                    'rsi_value': rsi,
                    'macd_value': macd
                }
            )
        
        # Standard exit conditions
        return await self._standard_exit_analysis(symbol, current_price)
    
    async def _standard_exit_analysis(self, symbol: str, current_price: float) -> StrategyResult:
        """Standard exit analysis"""
        position = self.positions[symbol]
        
        # Check stop loss
        if ((position.direction == MomentumDirection.BULLISH and current_price <= position.stop_loss_price) or
            (position.direction == MomentumDirection.BEARISH and current_price >= position.stop_loss_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
                confidence=0.95,
                metadata={
                    'strategy_action': 'stop_loss_exit',
                    'unrealized_pnl': position.unrealized_pnl
                }
            )
        
        # Check take profit
        if ((position.direction == MomentumDirection.BULLISH and current_price >= position.take_profit_price) or
            (position.direction == MomentumDirection.BEARISH and current_price <= position.take_profit_price)):
            
            position.state = MomentumState.LOOKING_FOR_ENTRY
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.SELL if position.direction == MomentumDirection.BULLISH else StrategySignal.BUY,
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
        return ["15M", "1H", "4H"]
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Validate dual momentum configuration"""
        return True  # Basic validation for now 
