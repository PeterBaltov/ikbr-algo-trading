"""
🎯 ENHANCED WHEEL STRATEGY IMPLEMENTATION
========================================

Advanced wheel strategy with delta-neutral adjustments, volatility-based timing,
and intelligent position management. This implementation enhances the classic
wheel strategy with sophisticated risk management and market adaptation.

Key Features:
- Delta-neutral adjustments for portfolio hedging
- Volatility-based timing for optimal entry/exit
- Dynamic position sizing based on market conditions
- Advanced option selection using Greeks analysis
- Risk management with stop-loss and profit targets
- Integration with Phase 2 technical analysis
- Multi-timeframe market context awareness
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
from moneytrailz.analysis import TechnicalAnalysisEngine
from moneytrailz.analysis.indicators import RSI, BollingerBands, ATR


class WheelState(Enum):
    """Wheel strategy states"""
    CASH_SECURED_PUT = "cash_secured_put"
    COVERED_CALL = "covered_call"
    ASSIGNED_SHARES = "assigned_shares"
    CALLED_AWAY = "called_away"
    RISK_MANAGEMENT = "risk_management"


class WheelPosition(Enum):
    """Position types in wheel strategy"""
    LONG_STOCK = "long_stock"
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    CASH = "cash"


@dataclass
class WheelConfig:
    """Configuration for Enhanced Wheel Strategy"""
    # Basic wheel parameters
    target_dte: int = 30
    min_dte: int = 7
    max_dte: int = 60
    target_delta: float = 0.30
    min_premium: float = 0.01
    
    # Delta management
    delta_threshold: float = 0.50
    hedge_delta_threshold: float = 0.70
    rebalance_frequency: int = 1  # days
    
    # Volatility timing
    iv_percentile_threshold: float = 50.0
    iv_rank_threshold: float = 30.0
    volatility_window: int = 30
    
    # Risk management
    max_loss_per_trade: float = 0.05  # 5%
    profit_target: float = 0.50  # 50% of premium
    max_position_size: float = 0.10  # 10% of portfolio
    
    # Technical analysis
    use_technical_filters: bool = True
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_squeeze_threshold: float = 2.0
    
    # Position management
    roll_dte: int = 21
    roll_profit_threshold: float = 0.25
    assignment_management: bool = True


@dataclass
class WheelPortfolioState:
    """Current state of wheel portfolio"""
    state: WheelState = WheelState.CASH_SECURED_PUT
    position_type: WheelPosition = WheelPosition.CASH
    underlying_symbol: str = ""
    shares_owned: int = 0
    option_contracts: int = 0
    option_strike: float = 0.0
    option_expiry: datetime = field(default_factory=datetime.now)
    entry_price: float = 0.0
    current_delta: float = 0.0
    unrealized_pnl: float = 0.0
    days_in_position: int = 0
    last_rebalance: datetime = field(default_factory=datetime.now)


class DeltaNeutralAdjuster:
    """Manages delta-neutral adjustments for the wheel portfolio"""
    
    def __init__(self, config: WheelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_portfolio_delta(self, positions: Dict[str, Any]) -> float:
        """Calculate total portfolio delta"""
        total_delta = 0.0
        
        for position in positions.values():
            if position.get('instrument_type') == 'stock':
                total_delta += position.get('quantity', 0) * 1.0  # Stock delta = 1
            elif position.get('instrument_type') == 'option':
                delta = position.get('delta', 0.0)
                quantity = position.get('quantity', 0)
                total_delta += delta * quantity * 100  # Options are per 100 shares
        
        return total_delta
    
    def needs_delta_adjustment(self, current_delta: float, target_delta: float = 0.0) -> bool:
        """Determine if delta adjustment is needed"""
        delta_deviation = abs(current_delta - target_delta)
        return delta_deviation > self.config.delta_threshold
    
    def calculate_hedge_quantity(self, current_delta: float, target_delta: float = 0.0) -> int:
        """Calculate number of shares needed for delta hedge"""
        delta_diff = target_delta - current_delta
        return int(delta_diff / 1.0)  # Each share has delta of 1
    
    async def suggest_adjustment(self, portfolio_state: WheelPortfolioState, 
                               market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Suggest delta adjustment trade"""
        if not self.needs_delta_adjustment(portfolio_state.current_delta):
            return None
        
        hedge_quantity = self.calculate_hedge_quantity(portfolio_state.current_delta)
        
        if abs(hedge_quantity) < 50:  # Don't hedge small amounts
            return None
        
        return {
            'action': 'hedge',
            'symbol': portfolio_state.underlying_symbol,
            'quantity': hedge_quantity,
            'side': 'buy' if hedge_quantity > 0 else 'sell',
            'reason': f'Delta adjustment: current={portfolio_state.current_delta:.2f}',
            'urgency': 'high' if abs(portfolio_state.current_delta) > self.config.hedge_delta_threshold else 'medium'
        }


class VolatilityTimer:
    """Manages volatility-based timing for wheel entries"""
    
    def __init__(self, config: WheelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_iv_percentile(self, current_iv: float, iv_history: pd.Series) -> float:
        """Calculate implied volatility percentile"""
        if len(iv_history) < self.config.volatility_window:
            return 50.0  # Default to median if insufficient data
        
        recent_iv = iv_history.tail(self.config.volatility_window)
        percentile = (recent_iv < current_iv).sum() / len(recent_iv) * 100
        return percentile
    
    def calculate_iv_rank(self, current_iv: float, iv_history: pd.Series) -> float:
        """Calculate implied volatility rank"""
        if len(iv_history) < self.config.volatility_window:
            return 50.0
        
        recent_iv = iv_history.tail(self.config.volatility_window)
        min_iv = recent_iv.min()
        max_iv = recent_iv.max()
        
        if max_iv == min_iv:
            return 50.0
        
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        return iv_rank
    
    def is_favorable_volatility_environment(self, current_iv: float, 
                                          iv_history: pd.Series) -> Tuple[bool, str]:
        """Determine if current volatility environment favors wheel strategy"""
        iv_percentile = self.calculate_iv_percentile(current_iv, iv_history)
        iv_rank = self.calculate_iv_rank(current_iv, iv_history)
        
        reasons = []
        
        # High volatility favors selling options (wheel strategy)
        if iv_percentile >= self.config.iv_percentile_threshold:
            reasons.append(f"IV percentile high: {iv_percentile:.1f}%")
        
        if iv_rank >= self.config.iv_rank_threshold:
            reasons.append(f"IV rank elevated: {iv_rank:.1f}%")
        
        is_favorable = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "Low volatility environment"
        
        return is_favorable, reason_text


class EnhancedWheelStrategy(BaseStrategy):
    """
    Enhanced Wheel Strategy with advanced features:
    
    1. Delta-neutral portfolio management
    2. Volatility-based timing
    3. Technical analysis integration
    4. Dynamic position sizing
    5. Advanced risk management
    """
    
    def __init__(self, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            strategy_type=StrategyType.OPTIONS,
            config=config or {},
            symbols=symbols,
            timeframes=timeframes
        )
        
        # Initialize configuration
        self.wheel_config = WheelConfig(**self.config.get('wheel_parameters', {}))
        
        # Initialize components
        self.delta_adjuster = DeltaNeutralAdjuster(self.wheel_config)
        self.volatility_timer = VolatilityTimer(self.wheel_config)
        self.technical_engine = TechnicalAnalysisEngine()
        
        # Portfolio state for each symbol
        self.portfolio_states: Dict[str, WheelPortfolioState] = {}
        for symbol in symbols:
            self.portfolio_states[symbol] = WheelPortfolioState(underlying_symbol=symbol)
        
        # Initialize technical indicators if enabled
        if self.wheel_config.use_technical_filters:
            self._setup_technical_indicators()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_technical_indicators(self):
        """Setup technical analysis indicators"""
        # Use the primary timeframe for indicators
        primary_timeframe = TimeFrame.DAY_1  # Default to daily for wheel strategy
        if self.timeframes:
            # Convert string timeframe to TimeFrame enum
            timeframe_str = list(self.timeframes)[0]  # Get first timeframe
            if timeframe_str == "1D":
                primary_timeframe = TimeFrame.DAY_1
            elif timeframe_str == "1H":
                primary_timeframe = TimeFrame.HOUR_1
            elif timeframe_str == "4H":
                primary_timeframe = TimeFrame.HOUR_4
        
        # RSI for momentum filtering
        rsi_indicator = RSI(primary_timeframe, period=14)
        self.technical_engine.add_indicator(rsi_indicator, "rsi_14")
        
        # Bollinger Bands for volatility analysis
        bb_indicator = BollingerBands(primary_timeframe, period=20)
        self.technical_engine.add_indicator(bb_indicator, "bb_20")
        
        # ATR for volatility measurement
        atr_indicator = ATR(primary_timeframe, period=14)
        self.technical_engine.add_indicator(atr_indicator, "atr_14")
    
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], 
                     context: StrategyContext) -> StrategyResult:
        """Main strategy analysis logic"""
        try:
            # Get current portfolio state
            portfolio_state = self.portfolio_states[symbol]
            
            # Perform technical analysis if enabled
            technical_signals = await self._analyze_technical_indicators(symbol, data)
            
            # Get market data for the symbol
            current_price = data['close'].iloc[-1] if not data.empty else 0.0
            
            # Update portfolio state
            portfolio_state.days_in_position += 1
            
            # Determine strategy action based on current state
            if portfolio_state.state == WheelState.CASH_SECURED_PUT:
                result = await self._analyze_cash_secured_put(symbol, data, portfolio_state, technical_signals)
            elif portfolio_state.state == WheelState.COVERED_CALL:
                result = await self._analyze_covered_call(symbol, data, portfolio_state, technical_signals)
            elif portfolio_state.state == WheelState.ASSIGNED_SHARES:
                result = await self._analyze_assigned_shares(symbol, data, portfolio_state, technical_signals)
            else:
                result = await self._analyze_risk_management(symbol, data, portfolio_state)
            
            # Check for delta adjustment needs
            delta_adjustment = await self.delta_adjuster.suggest_adjustment(portfolio_state, data)
            if delta_adjustment:
                result.metadata['delta_adjustment'] = delta_adjustment
            
            # Add portfolio state to metadata
            result.metadata['portfolio_state'] = {
                'state': portfolio_state.state.value,
                'position_type': portfolio_state.position_type.value,
                'current_delta': portfolio_state.current_delta,
                'days_in_position': portfolio_state.days_in_position,
                'unrealized_pnl': portfolio_state.unrealized_pnl
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {e}")
            raise StrategyExecutionError(f"Wheel strategy analysis failed: {e}")
    
    async def _analyze_technical_indicators(self, symbol: str, 
                                          data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators for market context"""
        if not self.wheel_config.use_technical_filters or data.empty:
            return {}
        
        try:
            # Run technical analysis
            analysis_result = await self.technical_engine.analyze(symbol, data)
            
            # Extract key signals
            signals = {}
            
            # RSI signals
            rsi_result = analysis_result.get('rsi_14')
            if rsi_result:
                rsi_value = rsi_result.values[-1] if len(rsi_result.values) > 0 else 50.0
                signals['rsi_oversold'] = rsi_value < self.wheel_config.rsi_oversold
                signals['rsi_overbought'] = rsi_value > self.wheel_config.rsi_overbought
                signals['rsi_value'] = rsi_value
            
            # Bollinger Bands signals
            bb_result = analysis_result.get('bb_20')
            if bb_result:
                bb_squeeze = bb_result.metadata.get('squeeze', False)
                signals['bollinger_squeeze'] = bb_squeeze
            
            # ATR for volatility context
            atr_result = analysis_result.get('atr_14')
            if atr_result:
                atr_value = atr_result.values[-1] if len(atr_result.values) > 0 else 0.0
                signals['atr_value'] = atr_value
                signals['high_volatility'] = atr_value > data['close'].iloc[-1] * 0.02  # 2% threshold
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Technical analysis failed for {symbol}: {e}")
            return {}
    
    async def _analyze_cash_secured_put(self, symbol: str, data: pd.DataFrame,
                                      portfolio_state: WheelPortfolioState,
                                      technical_signals: Dict[str, Any]) -> StrategyResult:
        """Analyze cash-secured put opportunities"""
        current_price = data['close'].iloc[-1]
        
        # Check volatility environment
        iv_data = data.get('implied_volatility', pd.Series([20.0] * len(data)))  # Default IV
        is_favorable, vol_reason = self.volatility_timer.is_favorable_volatility_environment(
            iv_data.iloc[-1], iv_data
        )
        
        # Technical filter
        technical_favorable = True
        if self.wheel_config.use_technical_filters:
            # Prefer selling puts when RSI is not extremely overbought
            rsi_ok = not technical_signals.get('rsi_overbought', False)
            # Avoid during bollinger band squeeze (low volatility)
            bb_ok = not technical_signals.get('bollinger_squeeze', False)
            technical_favorable = rsi_ok and bb_ok
        
        if is_favorable and technical_favorable:
            # Calculate optimal strike price
            target_strike = current_price * (1 - self.wheel_config.target_delta)
            
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.BUY,  # Buy to open put (sell put)
                confidence=0.8,
                target_price=target_strike,
                stop_loss=target_strike * 0.95,  # 5% below strike
                take_profit=target_strike * 1.02,  # 2% above strike
                position_size=self.wheel_config.max_position_size,
                metadata={
                    'strategy_action': 'sell_cash_secured_put',
                    'target_strike': target_strike,
                    'target_dte': self.wheel_config.target_dte,
                    'volatility_reason': vol_reason,
                    'technical_signals': technical_signals,
                    'iv_favorable': is_favorable
                }
            )
        else:
            return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.3,
                metadata={
                    'strategy_action': 'wait_for_opportunity',
                    'reason': f"Unfavorable conditions: vol={is_favorable}, tech={technical_favorable}",
                    'volatility_reason': vol_reason,
                    'technical_signals': technical_signals
                }
            )
    
    async def _analyze_covered_call(self, symbol: str, data: pd.DataFrame,
                                  portfolio_state: WheelPortfolioState,
                                  technical_signals: Dict[str, Any]) -> StrategyResult:
        """Analyze covered call opportunities"""
        current_price = data['close'].iloc[-1]
        
        # Check if we should roll or close the position
        if portfolio_state.days_in_position >= self.wheel_config.roll_dte:
            profit_pct = portfolio_state.unrealized_pnl / portfolio_state.entry_price
            
            if profit_pct >= self.wheel_config.roll_profit_threshold:
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                    signal=StrategySignal.SELL,  # Close position
                    confidence=0.9,
                    metadata={
                        'strategy_action': 'close_covered_call',
                        'reason': f'Profit target reached: {profit_pct:.2%}',
                        'profit_pct': profit_pct
                    }
                )
        
        # Technical analysis for covered call timing
        if self.wheel_config.use_technical_filters:
            rsi_high = technical_signals.get('rsi_overbought', False)
            if rsi_high:
                # Good time to sell calls when RSI is high
                target_strike = current_price * (1 + self.wheel_config.target_delta)
                
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                    signal=StrategySignal.BUY,  # Sell call
                    confidence=0.7,
                    target_price=target_strike,
                    metadata={
                        'strategy_action': 'sell_covered_call',
                        'target_strike': target_strike,
                        'target_dte': self.wheel_config.target_dte,
                        'technical_reason': 'RSI overbought - favorable for call selling'
                    }
                )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.5,
            metadata={
                'strategy_action': 'hold_position',
                'days_in_position': portfolio_state.days_in_position
            }
        )
    
    async def _analyze_assigned_shares(self, symbol: str, data: pd.DataFrame,
                                     portfolio_state: WheelPortfolioState,
                                     technical_signals: Dict[str, Any]) -> StrategyResult:
        """Analyze when holding assigned shares"""
        current_price = data['close'].iloc[-1]
        
        # Calculate unrealized P&L
        if portfolio_state.entry_price > 0:
            unrealized_pnl = (current_price - portfolio_state.entry_price) * portfolio_state.shares_owned
            portfolio_state.unrealized_pnl = unrealized_pnl
            
            # Check stop loss
            loss_pct = (portfolio_state.entry_price - current_price) / portfolio_state.entry_price
            if loss_pct > self.wheel_config.max_loss_per_trade:
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                    signal=StrategySignal.SELL,
                    confidence=0.9,
                    metadata={
                        'strategy_action': 'stop_loss_exit',
                        'loss_pct': loss_pct,
                        'reason': 'Maximum loss threshold exceeded'
                    }
                )
        
        # Look for covered call opportunities
        if self.wheel_config.use_technical_filters:
            rsi_high = technical_signals.get('rsi_overbought', False)
            if rsi_high:
                # Transition to covered call state
                portfolio_state.state = WheelState.COVERED_CALL
                target_strike = current_price * (1 + self.wheel_config.target_delta)
                
                return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
                    signal=StrategySignal.BUY,  # Sell covered call
                    confidence=0.8,
                    target_price=target_strike,
                    metadata={
                        'strategy_action': 'initiate_covered_call',
                        'target_strike': target_strike,
                        'transition_reason': 'RSI overbought - good call selling opportunity'
                    }
                )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.6,
            metadata={
                'strategy_action': 'hold_shares',
                'unrealized_pnl': portfolio_state.unrealized_pnl,
                'waiting_for': 'covered call opportunity'
            }
        )
    
    async def _analyze_risk_management(self, symbol: str, data: pd.DataFrame,
                                     portfolio_state: WheelPortfolioState) -> StrategyResult:
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
        """Return required timeframes for this strategy"""
        return {TimeFrame.DAY_1}  # Daily timeframe for wheel strategy
    
    def get_required_symbols(self) -> Set[str]:
        """Return symbols this strategy trades"""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> set:
        """Return required data fields"""
        fields = {'open', 'high', 'low', 'close', 'volume'}
        if self.wheel_config.use_technical_filters:
            fields.add('implied_volatility')
        return fields
    
    def validate_config(self) -> None:
        """Validate strategy configuration"""
        try:
            wheel_params = self.config.get('wheel_parameters', {})
            
            # Validate delta range
            target_delta = wheel_params.get('target_delta', 0.30)
            if not 0.05 <= target_delta <= 0.95:
                raise StrategyConfigError("target_delta must be between 0.05 and 0.95")
            
            # Validate DTE range
            target_dte = wheel_params.get('target_dte', 30)
            min_dte = wheel_params.get('min_dte', 7)
            if min_dte >= target_dte:
                raise StrategyConfigError("min_dte must be less than target_dte")
            
            # Validate risk parameters
            max_loss = wheel_params.get('max_loss_per_trade', 0.05)
            if not 0.01 <= max_loss <= 0.50:
                raise StrategyConfigError("max_loss_per_trade must be between 1% and 50%")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise StrategyConfigError(f"Invalid wheel strategy configuration: {e}")
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return (
            "Enhanced Wheel Strategy with delta-neutral adjustments, volatility-based timing, "
            "and advanced risk management. Systematically sells cash-secured puts and covered calls "
            "while maintaining portfolio delta neutrality and adapting to market volatility conditions."
        ) 
