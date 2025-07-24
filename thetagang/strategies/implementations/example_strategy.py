"""
Example strategy implementation demonstrating the ThetaGang strategy framework.

This is a simple example strategy that shows how to implement the BaseStrategy
interface. It can be used as a template for creating new strategies.
"""

from typing import Any, Dict, List, Set
import pandas as pd

from ..base import BaseStrategy, StrategyResult, StrategyContext
from ..enums import StrategySignal, StrategyType, TimeFrame
from ..exceptions import StrategyConfigError


class ExampleStrategy(BaseStrategy):
    """
    Example strategy implementation for demonstration purposes.
    
    This strategy generates simple signals based on price movements
    and serves as a template for creating more sophisticated strategies.
    """
    
    STRATEGY_NAME = "example"
    STRATEGY_TYPE = StrategyType.STOCKS
    
    def __init__(self, name: str, strategy_type: StrategyType, config: Dict[str, Any], symbols: List[str], timeframes: List[TimeFrame]):
        """Initialize the example strategy."""
        super().__init__(name, strategy_type, config, symbols, timeframes)
        
        # Strategy-specific configuration
        self.threshold = config.get("threshold", 0.02)  # 2% default threshold
        self.min_volume = config.get("min_volume", 100000)  # Minimum volume
        
    async def analyze(
        self,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        context: StrategyContext
    ) -> StrategyResult:
        """
        Analyze market data and generate trading signals.
        
        This example strategy:
        1. Checks if current price is above/below previous close by threshold
        2. Verifies minimum volume requirements
        3. Generates BUY/SELL signals accordingly
        """
        # Get the daily data
        daily_data = data[TimeFrame.DAY_1]
        
        if len(daily_data) < 2:
            return StrategyResult(
                strategy_name=self.name,
                symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.0,
                metadata={"reason": "Insufficient data"}
            )
        
        # Get current and previous prices
        current_price = daily_data['close'].iloc[-1]
        previous_price = daily_data['close'].iloc[-2]
        current_volume = daily_data['volume'].iloc[-1]
        
        # Calculate price change percentage
        price_change = (current_price - previous_price) / previous_price
        
        # Check volume requirement
        if current_volume < self.min_volume:
            return StrategyResult(
                strategy_name=self.name,
                symbol=symbol,
                signal=StrategySignal.HOLD,
                confidence=0.0,
                metadata={
                    "reason": "Volume too low",
                    "current_volume": current_volume,
                    "min_volume": self.min_volume
                }
            )
        
        # Generate signals based on price movement
        if price_change > self.threshold:
            signal = StrategySignal.BUY
            confidence = min(abs(price_change) / self.threshold, 1.0)
        elif price_change < -self.threshold:
            signal = StrategySignal.SELL
            confidence = min(abs(price_change) / self.threshold, 1.0)
        else:
            signal = StrategySignal.HOLD
            confidence = 0.5
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            metadata={
                "price_change": price_change,
                "threshold": self.threshold,
                "volume": current_volume,
                "previous_price": previous_price
            }
        )
    
    def validate_config(self) -> None:
        """Validate strategy-specific configuration."""
        if "threshold" in self.config:
            threshold = self.config["threshold"]
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                raise StrategyConfigError(
                    "threshold must be a positive number",
                    strategy_name=self.name,
                    config_field="threshold",
                    invalid_value=threshold
                )
        
        if "min_volume" in self.config:
            min_volume = self.config["min_volume"]
            if not isinstance(min_volume, (int, float)) or min_volume < 0:
                raise StrategyConfigError(
                    "min_volume must be a non-negative number",
                    strategy_name=self.name,
                    config_field="min_volume",
                    invalid_value=min_volume
                )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return the timeframes required by this strategy."""
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> Set[str]:
        """Return the symbols required by this strategy."""
        return self.symbols.copy()
    
    def get_required_data_fields(self) -> Set[str]:
        """Return the data fields required by this strategy."""
        return {"open", "high", "low", "close", "volume"}
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for this strategy."""
        return {
            "type": StrategyType.STOCKS.value,
            "enabled": True,
            "timeframes": [TimeFrame.DAY_1.value],
            "threshold": 0.02,
            "min_volume": 100000,
            "description": "Example strategy that generates signals based on price movements"
        } 
