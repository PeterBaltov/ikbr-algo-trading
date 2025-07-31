#!/usr/bin/env python3
"""
🧪 BASE STRATEGY FRAMEWORK TESTS
================================

Comprehensive tests for the BaseStrategy abstract class and core strategy framework.
Tests strategy lifecycle, validation, execution, and integration points.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Set, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

# Import the strategy framework components
from moneytrailz.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from moneytrailz.strategies.exceptions import (
    StrategyError, StrategyConfigError, StrategyExecutionError, StrategyValidationError
)
from moneytrailz.strategies.enums import (
    StrategySignal, StrategyType, TimeFrame, StrategyStatus, OrderSide, PositionSide
)


class ConcreteTestStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing purposes."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: Dict[str, Any]):
        super().__init__(name, StrategyType.STOCKS, config, symbols, timeframes)
        self.analysis_calls = 0
        self.validation_calls = 0
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> Optional[StrategyResult]:
        """Test implementation of analyze method."""
        self.analysis_calls += 1
        
        if not data or TimeFrame.DAY_1 not in data:
            return None
            
        df = data[TimeFrame.DAY_1]
        if df.empty:
            return None
            
        # Simple test logic: BUY if price increased, SELL if decreased, HOLD otherwise
        if len(df) < 2:
            signal = StrategySignal.HOLD
        else:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            if price_change > 0.01:  # 1% increase
                signal = StrategySignal.BUY
            elif price_change < -0.01:  # 1% decrease
                signal = StrategySignal.SELL
            else:
                signal = StrategySignal.HOLD
                
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=signal,
            confidence=0.8,
            price=float(df['close'].iloc[-1]),
            timestamp=datetime.now(),
            metadata={'price_change': price_change, 'data_points': len(df)}
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        """Return required timeframes for testing."""
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> Set[str]:
        """Return required symbols for testing."""
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        """Return required data fields for testing."""
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        """Test implementation of config validation."""
        self.validation_calls += 1
        if self.config.get('invalid_param') == 'fail':
            raise StrategyConfigError("Invalid configuration for testing")


class TestBaseStrategy:
    """Test suite for BaseStrategy abstract class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            'risk_tolerance': 0.02,
            'max_positions': 5,
            'test_param': 'value'
        }
        self.test_symbols = ['AAPL', 'GOOGL']
        self.test_timeframes = [TimeFrame.DAY_1, TimeFrame.HOUR_1]
        
        # Create test strategy instance
        self.strategy = ConcreteTestStrategy(
            name='test_strategy',
            symbols=self.test_symbols,
            timeframes=self.test_timeframes,
            config=self.test_config
        )
    
    def test_strategy_initialization(self):
        """Test strategy initialization and basic properties."""
        assert self.strategy.name == 'test_strategy'
        assert self.strategy.strategy_type == StrategyType.STOCKS
        assert self.strategy.symbols == self.test_symbols
        assert self.strategy.timeframes == self.test_timeframes
        assert self.strategy.config == self.test_config
        assert self.strategy.status == StrategyStatus.INITIALIZED
        assert self.strategy.execution_count == 0
        assert isinstance(self.strategy.creation_time, datetime)
        
    def test_strategy_properties(self):
        """Test strategy property accessors."""
        # Test timeframe properties
        required_timeframes = self.strategy.get_required_timeframes()
        assert isinstance(required_timeframes, set)
        assert TimeFrame.DAY_1 in required_timeframes
        
        # Test symbol properties
        required_symbols = self.strategy.get_required_symbols()
        assert isinstance(required_symbols, set)
        assert 'AAPL' in required_symbols
        assert 'GOOGL' in required_symbols
        
        # Test data field properties
        required_fields = self.strategy.get_required_data_fields()
        assert isinstance(required_fields, set)
        assert all(field in required_fields for field in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        self.strategy.validate_config()
        assert self.strategy.validation_calls == 1
        
        # Test invalid configuration
        invalid_strategy = ConcreteTestStrategy(
            name='invalid_test',
            symbols=['AAPL'],
            timeframes=[TimeFrame.DAY_1],
            config={'invalid_param': 'fail'}
        )
        
        with pytest.raises(StrategyConfigError):
            invalid_strategy.validate_config()
    
    def create_test_data(self, days: int = 10) -> Dict[TimeFrame, pd.DataFrame]:
        """Create test market data."""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # Create realistic OHLCV data
        prices = []
        base_price = 100.0
        for i in range(days):
            # Add some randomness but keep it realistic
            change = (i % 3 - 1) * 0.02  # -2%, 0%, +2% pattern
            price = base_price * (1 + change)
            
            high = price * 1.02
            low = price * 0.98
            open_price = price * (1 + (i % 5 - 2) * 0.005)  # Small open variation
            close_price = price
            volume = 1000000 + (i % 7) * 100000  # Varying volume
            
            prices.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
            base_price = close_price  # Next day starts from previous close
        
        df = pd.DataFrame(prices, index=dates)
        return {TimeFrame.DAY_1: df}
    
    @pytest.mark.asyncio
    async def test_strategy_analysis(self):
        """Test strategy analysis execution."""
        # Create test data and context
        test_data = self.create_test_data(5)
        test_context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(),
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        # Test analysis execution
        result = await self.strategy.analyze('AAPL', test_data, test_context)
        
        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == 'test_strategy'
        assert result.symbol == 'AAPL'
        assert result.signal in [StrategySignal.BUY, StrategySignal.SELL, StrategySignal.HOLD]
        assert 0.0 <= result.confidence <= 1.0
        assert result.price > 0
        assert isinstance(result.timestamp, datetime)
        assert 'price_change' in result.metadata
        assert 'data_points' in result.metadata
        
        # Verify execution count increased
        assert self.strategy.analysis_calls == 1
    
    @pytest.mark.asyncio
    async def test_strategy_analysis_with_insufficient_data(self):
        """Test strategy behavior with insufficient data."""
        # Create minimal data
        test_data = self.create_test_data(1)
        test_context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(), 
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        result = await self.strategy.analyze('AAPL', test_data, test_context)
        
        # Should still return result but with HOLD signal
        assert result is not None
        assert result.signal == StrategySignal.HOLD
    
    @pytest.mark.asyncio
    async def test_strategy_analysis_with_no_data(self):
        """Test strategy behavior with no data."""
        test_context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(),
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        # Test with empty data
        result = await self.strategy.analyze('AAPL', {}, test_context)
        assert result is None
        
        # Test with missing timeframe
        result = await self.strategy.analyze('AAPL', {TimeFrame.HOUR_1: pd.DataFrame()}, test_context)
        assert result is None
    
    def test_strategy_status_transitions(self):
        """Test strategy status state transitions."""
        # Initial status
        assert self.strategy.status == StrategyStatus.INITIALIZED
        
        # Test status changes (would normally be done by execution engine)
        self.strategy.status = StrategyStatus.RUNNING
        assert self.strategy.status == StrategyStatus.RUNNING
        
        self.strategy.status = StrategyStatus.PAUSED
        assert self.strategy.status == StrategyStatus.PAUSED
        
        self.strategy.status = StrategyStatus.STOPPED
        assert self.strategy.status == StrategyStatus.STOPPED
    
    def test_strategy_execution_tracking(self):
        """Test execution count tracking."""
        initial_count = self.strategy.execution_count
        
        # Simulate execution count updates (normally done by execution engine)
        self.strategy.execution_count += 1
        assert self.strategy.execution_count == initial_count + 1
        
        self.strategy.execution_count += 5
        assert self.strategy.execution_count == initial_count + 6
    
    def test_strategy_error_handling(self):
        """Test strategy error handling."""
        # Test StrategyError hierarchy
        config_error = StrategyConfigError("Config error")
        assert isinstance(config_error, StrategyError)
        
        execution_error = StrategyExecutionError("Execution error")
        assert isinstance(execution_error, StrategyError)
        
        validation_error = StrategyValidationError("Validation error")
        assert isinstance(validation_error, StrategyError)
    
    def test_strategy_result_creation(self):
        """Test StrategyResult creation and validation."""
        timestamp = datetime.now()
        metadata = {'test_key': 'test_value'}
        
        result = StrategyResult(
            strategy_name='test_strategy',
            symbol='AAPL',
            signal=StrategySignal.BUY,
            confidence=0.85,
            price=150.50,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert result.strategy_name == 'test_strategy'
        assert result.symbol == 'AAPL'
        assert result.signal == StrategySignal.BUY
        assert result.confidence == 0.85
        assert result.price == 150.50
        assert result.timestamp == timestamp
        assert result.metadata == metadata
    
    def test_strategy_context_creation(self):
        """Test StrategyContext creation and properties."""
        market_data = MagicMock()
        order_manager = MagicMock()
        portfolio_manager = MagicMock()
        timestamp = datetime.now()
        
        context = StrategyContext(
            market_data=market_data,
            order_manager=order_manager,
            portfolio_manager=portfolio_manager,
            timestamp=timestamp
        )
        
        assert context.market_data == market_data
        assert context.order_manager == order_manager
        assert context.portfolio_manager == portfolio_manager
        assert context.timestamp == timestamp


class TestStrategySignalGeneration:
    """Test suite for strategy signal generation logic."""
    
    def setup_method(self):
        """Set up test fixtures for signal generation."""
        self.strategy = ConcreteTestStrategy(
            name='signal_test',
            symbols=['TEST'],
            timeframes=[TimeFrame.DAY_1],
            config={'test': True}
        )
    
    def create_trend_data(self, trend: str, days: int = 10) -> Dict[TimeFrame, pd.DataFrame]:
        """Create data with specific trend for testing signal generation."""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        prices = []
        base_price = 100.0
        
        for i in range(days):
            if trend == 'uptrend':
                price = base_price * (1 + i * 0.02)  # 2% increase per day
            elif trend == 'downtrend':
                price = base_price * (1 - i * 0.02)  # 2% decrease per day
            else:  # sideways
                price = base_price * (1 + (i % 2 - 0.5) * 0.005)  # Minimal variation
            
            prices.append({
                'open': price * 0.999,
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price,
                'volume': 1000000
            })
        
        df = pd.DataFrame(prices, index=dates)
        return {TimeFrame.DAY_1: df}
    
    @pytest.mark.asyncio
    async def test_buy_signal_generation(self):
        """Test BUY signal generation."""
        uptrend_data = self.create_trend_data('uptrend', 5)
        context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(),
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        result = await self.strategy.analyze('TEST', uptrend_data, context)
        
        assert result is not None
        assert result.signal == StrategySignal.BUY
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_sell_signal_generation(self):
        """Test SELL signal generation."""
        downtrend_data = self.create_trend_data('downtrend', 5)
        context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(),
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        result = await self.strategy.analyze('TEST', downtrend_data, context)
        
        assert result is not None
        assert result.signal == StrategySignal.SELL
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_hold_signal_generation(self):
        """Test HOLD signal generation."""
        sideways_data = self.create_trend_data('sideways', 5)
        context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(),
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        result = await self.strategy.analyze('TEST', sideways_data, context)
        
        assert result is not None
        assert result.signal == StrategySignal.HOLD
        assert result.confidence > 0


def run_base_strategy_tests():
    """Run all base strategy tests."""
    print("🧪 RUNNING BASE STRATEGY FRAMEWORK TESTS")
    print("=" * 50)
    
    # Run tests using pytest
    test_results = []
    
    # Test categories
    test_categories = [
        ("Strategy Initialization", TestBaseStrategy().test_strategy_initialization),
        ("Strategy Properties", TestBaseStrategy().test_strategy_properties),
        ("Config Validation", TestBaseStrategy().test_config_validation),
        ("Strategy Result Creation", TestBaseStrategy().test_strategy_result_creation),
        ("Strategy Context Creation", TestBaseStrategy().test_strategy_context_creation),
        ("Strategy Status Transitions", TestBaseStrategy().test_strategy_status_transitions),
        ("Execution Tracking", TestBaseStrategy().test_strategy_execution_tracking),
        ("Error Handling", TestBaseStrategy().test_strategy_error_handling),
    ]
    
    # Async test categories
    async_test_categories = [
        ("Strategy Analysis", TestBaseStrategy().test_strategy_analysis),
        ("Insufficient Data Analysis", TestBaseStrategy().test_strategy_analysis_with_insufficient_data),
        ("No Data Analysis", TestBaseStrategy().test_strategy_analysis_with_no_data),
        ("BUY Signal Generation", TestStrategySignalGeneration().test_buy_signal_generation),
        ("SELL Signal Generation", TestStrategySignalGeneration().test_sell_signal_generation),
        ("HOLD Signal Generation", TestStrategySignalGeneration().test_hold_signal_generation),
    ]
    
    passed = 0
    total = len(test_categories) + len(async_test_categories)
    
    # Run synchronous tests
    for test_name, test_func in test_categories:
        try:
            # Set up test instance
            test_instance = TestBaseStrategy()
            test_instance.setup_method()
            
            # Run test
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Run asynchronous tests
    async def run_async_tests():
        nonlocal passed
        for test_name, test_func in async_test_categories:
            try:
                # Set up test instance
                if 'Signal Generation' in test_name:
                    test_instance = TestStrategySignalGeneration()
                    test_instance.setup_method()
                else:
                    test_instance = TestBaseStrategy()
                    test_instance.setup_method()
                
                # Run async test
                await test_func()
                print(f"✅ {test_name}: PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 50)
    print("📋 BASE STRATEGY TEST RESULTS")
    print("=" * 50)
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 ALL BASE STRATEGY TESTS PASSED!")
        print("🚀 Strategy framework core is robust and ready!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Strategy framework needs attention.")
    
    return passed == total


if __name__ == "__main__":
    run_base_strategy_tests() 
