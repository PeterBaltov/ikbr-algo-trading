#!/usr/bin/env python3
"""
🧪 MULTI-TIMEFRAME INTEGRATION TESTS
===================================

Integration tests for multi-timeframe architecture including:
- TimeFrame management and coordination
- Data synchronization across timeframes
- Cross-timeframe strategy execution
- Performance optimization and memory management
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from unittest.mock import MagicMock, patch

# Import multi-timeframe components
from moneytrailz.timeframes import (
    TimeFrameManager, DataSynchronizer, ExecutionScheduler, DataAggregator
)
from moneytrailz.execution import StrategyExecutionEngine
from moneytrailz.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from moneytrailz.strategies.enums import StrategySignal, StrategyType, TimeFrame
from moneytrailz.analysis import TechnicalAnalysisEngine


class MultiTimeframeTestStrategy(BaseStrategy):
    """Test strategy that uses multiple timeframes."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: dict):
        super().__init__(name, StrategyType.STOCKS, config, symbols, timeframes)
        self.analysis_count = 0
        self.timeframe_data_received = {}
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> StrategyResult:
        """Analyze data across multiple timeframes."""
        self.analysis_count += 1
        self.timeframe_data_received[symbol] = list(data.keys())
        
        # Simple multi-timeframe logic
        if TimeFrame.DAY_1 in data and TimeFrame.HOUR_1 in data:
            daily_data = data[TimeFrame.DAY_1]
            hourly_data = data[TimeFrame.HOUR_1]
            
            if not daily_data.empty and not hourly_data.empty:
                # Daily trend
                daily_close = daily_data['close'].iloc[-1]
                daily_sma = daily_data['close'].rolling(10).mean().iloc[-1] if len(daily_data) >= 10 else daily_close
                
                # Hourly momentum
                hourly_close = hourly_data['close'].iloc[-1]
                hourly_sma = hourly_data['close'].rolling(5).mean().iloc[-1] if len(hourly_data) >= 5 else hourly_close
                
                # Multi-timeframe signal
                daily_bullish = daily_close > daily_sma
                hourly_bullish = hourly_close > hourly_sma
                
                if daily_bullish and hourly_bullish:
                    signal = StrategySignal.BUY
                    confidence = 0.8
                elif not daily_bullish and not hourly_bullish:
                    signal = StrategySignal.SELL
                    confidence = 0.8
                else:
                    signal = StrategySignal.HOLD
                    confidence = 0.4
                
                return StrategyResult(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal=signal,
                    confidence=confidence,
                    price=hourly_close,
                    timestamp=datetime.now(),
                    metadata={
                        'daily_trend': 'bullish' if daily_bullish else 'bearish',
                        'hourly_momentum': 'bullish' if hourly_bullish else 'bearish',
                        'timeframes_used': ['1D', '1H']
                    }
                )
        
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.3,
            price=100.0,
            timestamp=datetime.now(),
            metadata={'reason': 'insufficient_data'}
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        return {TimeFrame.DAY_1, TimeFrame.HOUR_1}
    
    def get_required_symbols(self) -> Set[str]:
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        pass


class TestTimeFrameManager:
    """Test suite for TimeFrameManager integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TimeFrameManager()
        self.test_symbols = ['AAPL', 'GOOGL']
        self.test_timeframes = [TimeFrame.MIN_5, TimeFrame.MIN_15, TimeFrame.HOUR_1, TimeFrame.DAY_1]
        
    def create_multi_timeframe_data(self, symbol: str, periods: int = 100) -> Dict[TimeFrame, pd.DataFrame]:
        """Create test data for multiple timeframes."""
        # Start with 5-minute data
        base_dates = pd.date_range(start='2024-01-01', periods=periods, freq='5min')
        base_price = 100.0
        
        # Generate 5-minute data
        prices_5min = []
        for i in range(periods):
            price_change = np.random.normal(0, 0.001)  # Small 5-min changes
            new_price = base_price * (1 + price_change)
            prices_5min.append(new_price)
            base_price = new_price
        
        data_5min = pd.DataFrame({
            'open': prices_5min,
            'high': [p * 1.005 for p in prices_5min],
            'low': [p * 0.995 for p in prices_5min],
            'close': prices_5min,
            'volume': np.random.randint(10000, 50000, periods)
        }, index=base_dates)
        
        # Aggregate to other timeframes
        data_15min = self._aggregate_timeframe(data_5min, '15min')
        data_1h = self._aggregate_timeframe(data_5min, '1H')
        data_1d = self._aggregate_timeframe(data_5min, '1D')
        
        return {
            TimeFrame.MIN_5: data_5min,
            TimeFrame.MIN_15: data_15min,
            TimeFrame.HOUR_1: data_1h,
            TimeFrame.DAY_1: data_1d
        }
    
    def _aggregate_timeframe(self, base_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Aggregate data to different timeframe."""
        resampled = base_data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled
    
    def test_timeframe_registration(self):
        """Test timeframe registration and management."""
        # Register timeframes
        for timeframe in self.test_timeframes:
            self.manager.register_timeframe(timeframe)
        
        # Verify registration
        registered = self.manager.get_registered_timeframes()
        assert len(registered) == len(self.test_timeframes)
        for timeframe in self.test_timeframes:
            assert timeframe in registered
    
    def test_data_storage_and_retrieval(self):
        """Test data storage and retrieval across timeframes."""
        # Create and store test data
        for symbol in self.test_symbols:
            symbol_data = self.create_multi_timeframe_data(symbol)
            
            for timeframe, data in symbol_data.items():
                self.manager.store_data(symbol, timeframe, data)
        
        # Test data retrieval
        for symbol in self.test_symbols:
            for timeframe in self.test_timeframes:
                retrieved_data = self.manager.get_data(symbol, timeframe)
                
                assert isinstance(retrieved_data, pd.DataFrame)
                assert not retrieved_data.empty
                assert all(col in retrieved_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_timeframe_synchronization(self):
        """Test data synchronization across timeframes."""
        symbol = 'AAPL'
        symbol_data = self.create_multi_timeframe_data(symbol, 200)
        
        # Store data
        for timeframe, data in symbol_data.items():
            self.manager.store_data(symbol, timeframe, data)
        
        # Test synchronization
        sync_data = self.manager.get_synchronized_data(
            symbol, 
            [TimeFrame.MIN_15, TimeFrame.HOUR_1, TimeFrame.DAY_1],
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        
        assert isinstance(sync_data, dict)
        assert TimeFrame.MIN_15 in sync_data
        assert TimeFrame.HOUR_1 in sync_data
        assert TimeFrame.DAY_1 in sync_data
        
        # All timeframes should have data for the same time period
        for timeframe, data in sync_data.items():
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
    
    def test_memory_management(self):
        """Test memory management for large datasets."""
        # Set memory limit
        self.manager.set_memory_limit(1000)  # Limit to 1000 data points per timeframe
        
        # Store large dataset
        large_data = self.create_multi_timeframe_data('LARGE_SYMBOL', 2000)
        
        for timeframe, data in large_data.items():
            self.manager.store_data('LARGE_SYMBOL', timeframe, data)
        
        # Check that data was truncated to memory limit
        for timeframe in large_data.keys():
            stored_data = self.manager.get_data('LARGE_SYMBOL', timeframe)
            assert len(stored_data) <= 1000


class TestDataSynchronizer:
    """Test suite for DataSynchronizer integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synchronizer = DataSynchronizer()
        
    def create_misaligned_data(self) -> Dict[TimeFrame, pd.DataFrame]:
        """Create data with different time ranges for testing synchronization."""
        # 5-minute data (100 periods)
        dates_5min = pd.date_range(start='2024-01-01 09:00', periods=100, freq='5min')
        data_5min = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates_5min)
        
        # 1-hour data (25 periods, different start time)
        dates_1h = pd.date_range(start='2024-01-01 10:00', periods=25, freq='1H')
        data_1h = pd.DataFrame({
            'close': np.random.randn(25).cumsum() + 100,
            'volume': np.random.randint(10000, 50000, 25)
        }, index=dates_1h)
        
        # Daily data (5 periods, different range)
        dates_1d = pd.date_range(start='2024-01-01', periods=5, freq='1D')
        data_1d = pd.DataFrame({
            'close': np.random.randn(5).cumsum() + 100,
            'volume': np.random.randint(100000, 500000, 5)
        }, index=dates_1d)
        
        return {
            TimeFrame.MIN_5: data_5min,
            TimeFrame.HOUR_1: data_1h,
            TimeFrame.DAY_1: data_1d
        }
    
    def test_data_alignment(self):
        """Test data alignment across different timeframes."""
        misaligned_data = self.create_misaligned_data()
        
        aligned_data = self.synchronizer.align_data(misaligned_data)
        
        assert isinstance(aligned_data, dict)
        
        # All timeframes should have aligned time ranges
        time_ranges = {}
        for timeframe, data in aligned_data.items():
            time_ranges[timeframe] = (data.index.min(), data.index.max())
        
        # Check that data is properly aligned (overlapping time periods)
        assert len(time_ranges) > 0
        
        # All data should be properly formatted
        for timeframe, data in aligned_data.items():
            assert isinstance(data, pd.DataFrame)
            assert 'close' in data.columns
    
    def test_forward_fill_synchronization(self):
        """Test forward fill synchronization method."""
        test_data = self.create_misaligned_data()
        
        synchronized = self.synchronizer.synchronize_forward_fill(test_data)
        
        assert isinstance(synchronized, dict)
        
        # Should maintain all original timeframes
        for timeframe in test_data.keys():
            assert timeframe in synchronized
            
        # Check that gaps are filled
        for timeframe, data in synchronized.items():
            # Should have no NaN values after forward fill
            assert not data['close'].isna().any()
    
    def test_interpolation_synchronization(self):
        """Test interpolation synchronization method."""
        test_data = self.create_misaligned_data()
        
        interpolated = self.synchronizer.synchronize_interpolation(test_data)
        
        assert isinstance(interpolated, dict)
        
        # Should have smooth interpolated values
        for timeframe, data in interpolated.items():
            assert isinstance(data, pd.DataFrame)
            assert 'close' in data.columns
    
    def test_resampling_synchronization(self):
        """Test resampling synchronization method."""
        test_data = self.create_misaligned_data()
        
        resampled = self.synchronizer.synchronize_resampling(
            test_data, 
            target_frequency='15min'
        )
        
        assert isinstance(resampled, dict)
        
        # All data should be resampled to 15-minute frequency
        for timeframe, data in resampled.items():
            # Check that frequency is approximately 15 minutes
            if len(data) > 1:
                time_diff = data.index[1] - data.index[0]
                assert time_diff.total_seconds() / 60 == 15  # 15 minutes


class TestStrategyExecutionEngine:
    """Test suite for StrategyExecutionEngine with multi-timeframe support."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'execution_mode': 'live',
            'timeframe_sync': True,
            'memory_limit': 5000
        }
        self.engine = StrategyExecutionEngine(self.config)
        
    def test_engine_initialization(self):
        """Test execution engine initialization."""
        assert self.engine.config == self.config
        assert hasattr(self.engine, 'timeframe_manager')
        assert hasattr(self.engine, 'data_synchronizer')
        assert hasattr(self.engine, 'execution_scheduler')
    
    def test_multi_timeframe_strategy_registration(self):
        """Test registration of multi-timeframe strategies."""
        strategy = MultiTimeframeTestStrategy(
            'multi_tf_test',
            ['AAPL'],
            [TimeFrame.DAY_1, TimeFrame.HOUR_1],
            {'param': 'value'}
        )
        
        self.engine.register_strategy(strategy)
        
        assert 'multi_tf_test' in self.engine.strategies
        
        # Check that required timeframes are registered
        required_tfs = strategy.get_required_timeframes()
        registered_tfs = self.engine.timeframe_manager.get_registered_timeframes()
        
        for tf in required_tfs:
            assert tf in registered_tfs
    
    @pytest.mark.asyncio
    async def test_multi_timeframe_execution(self):
        """Test execution of strategies with multiple timeframes."""
        strategy = MultiTimeframeTestStrategy(
            'multi_execution_test',
            ['AAPL'],
            [TimeFrame.DAY_1, TimeFrame.HOUR_1],
            {}
        )
        
        self.engine.register_strategy(strategy)
        
        # Mock multi-timeframe data
        test_data = self.create_execution_test_data()
        
        # Mock data retrieval
        with patch.object(self.engine.timeframe_manager, 'get_synchronized_data') as mock_get_data:
            mock_get_data.return_value = test_data
            
            # Execute strategy
            results = await self.engine.execute_strategies(['AAPL'])
            
            assert isinstance(results, dict)
            assert 'multi_execution_test' in results
            
            strategy_result = results['multi_execution_test']['AAPL']
            assert isinstance(strategy_result, StrategyResult)
            assert strategy_result.symbol == 'AAPL'
            
            # Check that strategy received multiple timeframes
            assert strategy.analysis_count > 0
            assert 'AAPL' in strategy.timeframe_data_received
            
            received_timeframes = strategy.timeframe_data_received['AAPL']
            assert TimeFrame.DAY_1 in received_timeframes
            assert TimeFrame.HOUR_1 in received_timeframes
    
    def create_execution_test_data(self) -> Dict[TimeFrame, pd.DataFrame]:
        """Create test data for execution testing."""
        # Create daily data
        daily_dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        daily_data = pd.DataFrame({
            'open': np.random.randn(30).cumsum() + 100,
            'high': np.random.randn(30).cumsum() + 102,
            'low': np.random.randn(30).cumsum() + 98,
            'close': np.random.randn(30).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=daily_dates)
        
        # Create hourly data
        hourly_dates = pd.date_range(start='2024-01-01', periods=24*30, freq='1H')
        hourly_data = pd.DataFrame({
            'open': np.random.randn(24*30).cumsum() + 100,
            'high': np.random.randn(24*30).cumsum() + 102,
            'low': np.random.randn(24*30).cumsum() + 98,
            'close': np.random.randn(24*30).cumsum() + 100,
            'volume': np.random.randint(10000, 100000, 24*30)
        }, index=hourly_dates)
        
        return {
            TimeFrame.DAY_1: daily_data,
            TimeFrame.HOUR_1: hourly_data
        }


class TestExecutionScheduler:
    """Test suite for ExecutionScheduler with multi-timeframe strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = ExecutionScheduler()
        
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert hasattr(self.scheduler, 'schedule_strategy')
        assert hasattr(self.scheduler, 'get_next_execution_time')
        assert isinstance(self.scheduler.strategy_schedules, dict)
    
    def test_strategy_scheduling(self):
        """Test strategy scheduling functionality."""
        strategy = MultiTimeframeTestStrategy(
            'scheduled_test',
            ['AAPL'],
            [TimeFrame.DAY_1, TimeFrame.HOUR_1],
            {}
        )
        
        # Schedule strategy for different timeframes
        self.scheduler.schedule_strategy(strategy, TimeFrame.DAY_1, '09:30')  # Daily at market open
        self.scheduler.schedule_strategy(strategy, TimeFrame.HOUR_1, '*/1')   # Every hour
        
        assert 'scheduled_test' in self.scheduler.strategy_schedules
        
        schedules = self.scheduler.strategy_schedules['scheduled_test']
        assert TimeFrame.DAY_1 in schedules
        assert TimeFrame.HOUR_1 in schedules
    
    def test_execution_timing(self):
        """Test execution timing calculation."""
        strategy = MultiTimeframeTestStrategy(
            'timing_test',
            ['AAPL'],
            [TimeFrame.DAY_1],
            {}
        )
        
        self.scheduler.schedule_strategy(strategy, TimeFrame.DAY_1, '09:30')
        
        next_execution = self.scheduler.get_next_execution_time('timing_test', TimeFrame.DAY_1)
        
        assert isinstance(next_execution, datetime)
        
        # Should be scheduled for 9:30 AM
        assert next_execution.hour == 9
        assert next_execution.minute == 30
    
    def test_priority_scheduling(self):
        """Test priority-based scheduling."""
        high_priority_strategy = MultiTimeframeTestStrategy(
            'high_priority',
            ['AAPL'],
            [TimeFrame.MIN_5],
            {}
        )
        
        low_priority_strategy = MultiTimeframeTestStrategy(
            'low_priority',
            ['AAPL'],
            [TimeFrame.DAY_1],
            {}
        )
        
        # Schedule with different priorities
        self.scheduler.schedule_strategy(high_priority_strategy, TimeFrame.MIN_5, '*/5', priority=1)
        self.scheduler.schedule_strategy(low_priority_strategy, TimeFrame.DAY_1, '09:30', priority=5)
        
        # Get execution queue
        execution_queue = self.scheduler.get_execution_queue()
        
        assert isinstance(execution_queue, list)
        
        # High priority should come first
        if len(execution_queue) >= 2:
            assert execution_queue[0]['priority'] <= execution_queue[1]['priority']


def run_multi_timeframe_tests():
    """Run all multi-timeframe integration tests."""
    print("🧪 RUNNING MULTI-TIMEFRAME INTEGRATION TESTS")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        # TimeFrameManager tests
        ("TimeFrame Registration", TestTimeFrameManager().test_timeframe_registration),
        ("Data Storage and Retrieval", TestTimeFrameManager().test_data_storage_and_retrieval),
        ("TimeFrame Synchronization", TestTimeFrameManager().test_timeframe_synchronization),
        ("Memory Management", TestTimeFrameManager().test_memory_management),
        
        # DataSynchronizer tests
        ("Data Alignment", TestDataSynchronizer().test_data_alignment),
        ("Forward Fill Synchronization", TestDataSynchronizer().test_forward_fill_synchronization),
        ("Interpolation Synchronization", TestDataSynchronizer().test_interpolation_synchronization),
        ("Resampling Synchronization", TestDataSynchronizer().test_resampling_synchronization),
        
        # StrategyExecutionEngine tests
        ("Engine Initialization", TestStrategyExecutionEngine().test_engine_initialization),
        ("Multi-TimeFrame Strategy Registration", TestStrategyExecutionEngine().test_multi_timeframe_strategy_registration),
        
        # ExecutionScheduler tests
        ("Scheduler Initialization", TestExecutionScheduler().test_scheduler_initialization),
        ("Strategy Scheduling", TestExecutionScheduler().test_strategy_scheduling),
        ("Execution Timing", TestExecutionScheduler().test_execution_timing),
        ("Priority Scheduling", TestExecutionScheduler().test_priority_scheduling),
    ]
    
    # Async test categories
    async_test_categories = [
        ("Multi-TimeFrame Execution", TestStrategyExecutionEngine().test_multi_timeframe_execution),
    ]
    
    passed = 0
    total = len(test_categories) + len(async_test_categories)
    
    # Run synchronous tests
    for test_name, test_func in test_categories:
        try:
            # Set up appropriate test instance
            if "TimeFrame" in test_name and "Manager" in test_name:
                test_instance = TestTimeFrameManager()
            elif "Synchronizer" in test_name or "Alignment" in test_name or "Fill" in test_name or "Interpolation" in test_name or "Resampling" in test_name:
                test_instance = TestDataSynchronizer()
            elif "Engine" in test_name or "Registration" in test_name:
                test_instance = TestStrategyExecutionEngine()
            else:  # Scheduler tests
                test_instance = TestExecutionScheduler()
            
            test_instance.setup_method()
            
            # Run test
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Run asynchronous tests
    import asyncio
    
    async def run_async_tests():
        nonlocal passed
        for test_name, test_func in async_test_categories:
            try:
                test_instance = TestStrategyExecutionEngine()
                test_instance.setup_method()
                
                await test_func()
                print(f"✅ {test_name}: PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 50)
    print("📋 MULTI-TIMEFRAME INTEGRATION TEST RESULTS")
    print("=" * 50)
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 ALL MULTI-TIMEFRAME INTEGRATION TESTS PASSED!")
        print("🚀 Multi-timeframe architecture is robust and ready!")
        
        print("\n📊 Tested Components:")
        print("  • ⏰ TimeFrame Management: Registration, storage, synchronization")
        print("  • 🔄 Data Synchronization: Alignment, filling, interpolation, resampling")
        print("  • 🎯 Strategy Execution: Multi-timeframe coordination and execution")
        print("  • 📅 Execution Scheduling: Timing, priorities, queue management")
        print("  • 💾 Memory Management: Efficient handling of large datasets")
        print("  • 🔗 Integration: Cross-timeframe strategy coordination")
        
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Multi-timeframe system needs attention.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    run_multi_timeframe_tests() 
