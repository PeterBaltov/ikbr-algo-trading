"""
Signal Processing and Aggregation

This module provides tools for processing, combining, and analyzing
signals from multiple technical indicators.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from .indicators.base import IndicatorResult


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalDirection(Enum):
    """Signal direction"""
    BULLISH = "bullish"
    BEARISH = "bearish" 
    NEUTRAL = "neutral"


@dataclass
class CombinedSignal:
    """Combined signal from multiple indicators"""
    
    overall_strength: float  # -1.0 to 1.0
    overall_direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    
    contributing_signals: List[IndicatorResult]
    timestamp: datetime
    symbol: str
    
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalProcessor:
    """Process individual indicator signals"""
    
    @staticmethod
    def normalize_signal_strength(signal_strength: Optional[float]) -> float:
        """Normalize signal strength to -1.0 to 1.0 range"""
        if signal_strength is None:
            return 0.0
        return max(-1.0, min(1.0, signal_strength))
    
    @staticmethod
    def categorize_strength(signal_strength: float) -> SignalStrength:
        """Categorize signal strength"""
        abs_strength = abs(signal_strength)
        
        if abs_strength >= 0.8:
            return SignalStrength.VERY_STRONG
        elif abs_strength >= 0.6:
            return SignalStrength.STRONG
        elif abs_strength >= 0.4:
            return SignalStrength.MODERATE
        elif abs_strength >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    @staticmethod
    def determine_direction(signal_strength: float) -> SignalDirection:
        """Determine signal direction from strength"""
        if signal_strength > 0.1:
            return SignalDirection.BULLISH
        elif signal_strength < -0.1:
            return SignalDirection.BEARISH
        else:
            return SignalDirection.NEUTRAL


class SignalAggregator:
    """Aggregate signals from multiple indicators"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize aggregator with optional indicator weights
        
        Args:
            weights: Dict mapping indicator names to weights (0.0 to 1.0)
        """
        self.weights = weights or {}
    
    def aggregate_signals(
        self, 
        signals: List[IndicatorResult],
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> CombinedSignal:
        """
        Aggregate multiple indicator signals into a combined signal
        
        Args:
            signals: List of indicator results
            symbol: Symbol being analyzed
            timestamp: Timestamp for the combined signal
            
        Returns:
            CombinedSignal with aggregated results
        """
        if not signals:
            return CombinedSignal(
                overall_strength=0.0,
                overall_direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                contributing_signals=[],
                timestamp=timestamp or datetime.now(),
                symbol=symbol
            )
        
        # Calculate weighted average of signal strengths
        total_weight = 0.0
        weighted_strength = 0.0
        
        valid_signals = [s for s in signals if s.is_valid]
        
        for signal in valid_signals:
            weight = self.weights.get(signal.indicator_name, 1.0)
            signal_strength = SignalProcessor.normalize_signal_strength(signal.signal_strength)
            
            weighted_strength += signal_strength * weight
            total_weight += weight
        
        # Calculate overall strength
        if total_weight > 0:
            overall_strength = weighted_strength / total_weight
        else:
            overall_strength = 0.0
        
        # Determine overall direction
        overall_direction = SignalProcessor.determine_direction(overall_strength)
        
        # Calculate confidence based on signal agreement
        confidence = self._calculate_confidence(valid_signals, overall_direction)
        
        return CombinedSignal(
            overall_strength=overall_strength,
            overall_direction=overall_direction,
            confidence=confidence,
            contributing_signals=valid_signals,
            timestamp=timestamp or datetime.now(),
            symbol=symbol,
            metadata={
                "num_signals": len(valid_signals),
                "total_weight": total_weight,
                "signal_agreement": confidence
            }
        )
    
    def _calculate_confidence(
        self, 
        signals: List[IndicatorResult], 
        overall_direction: SignalDirection
    ) -> float:
        """Calculate confidence based on signal agreement"""
        if not signals:
            return 0.0
        
        # Count signals that agree with overall direction
        agreeing_signals = 0
        
        for signal in signals:
            signal_direction = SignalProcessor.determine_direction(
                signal.signal_strength or 0.0
            )
            
            if signal_direction == overall_direction:
                agreeing_signals += 1
            elif signal_direction == SignalDirection.NEUTRAL:
                agreeing_signals += 0.5  # Neutral counts as half agreement
        
        # Calculate agreement ratio
        agreement_ratio = agreeing_signals / len(signals)
        
        # Apply confidence scaling
        if agreement_ratio >= 0.8:
            return 0.9
        elif agreement_ratio >= 0.6:
            return 0.7
        elif agreement_ratio >= 0.5:
            return 0.5
        else:
            return 0.3


class ConfidenceCalculator:
    """Calculate confidence scores for signals"""
    
    @staticmethod
    def calculate_indicator_confidence(result: IndicatorResult) -> float:
        """Calculate confidence for an individual indicator result"""
        if not result.is_valid:
            return 0.0
        
        # Base confidence from indicator's own confidence
        base_confidence = result.confidence
        
        # Adjust based on signal strength
        if result.signal_strength is not None:
            strength_factor = abs(result.signal_strength)
            base_confidence *= (0.5 + 0.5 * strength_factor)
        
        return min(1.0, base_confidence)
    
    @staticmethod
    def calculate_multi_timeframe_confidence(
        signals_by_timeframe: Dict[str, List[IndicatorResult]]
    ) -> float:
        """Calculate confidence across multiple timeframes"""
        if not signals_by_timeframe:
            return 0.0
        
        timeframe_confidences = []
        
        for timeframe, signals in signals_by_timeframe.items():
            if signals:
                avg_confidence = sum(
                    ConfidenceCalculator.calculate_indicator_confidence(s) 
                    for s in signals
                ) / len(signals)
                timeframe_confidences.append(avg_confidence)
        
        if timeframe_confidences:
            return sum(timeframe_confidences) / len(timeframe_confidences)
        else:
            return 0.0 
