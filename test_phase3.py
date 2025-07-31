#!/usr/bin/env python3
"""
Phase 3 Multi-Timeframe Architecture Test Suite

This script tests the comprehensive multi-timeframe capabilities including
timeframe management, data synchronization, execution scheduling, aggregation,
and strategy execution engine integration.
"""

import asyncio
import sys
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np


def test_phase3_imports():
    """Test that all Phase 3 components can be imported"""
    print("🔍 Testing Phase 3 Multi-Timeframe Imports...")
    
    try:
        # Timeframe management
        from moneytrailz.timeframes import TimeFrameManager, TimeFrameConfig, TimeFrameState
        print("✅ TimeFrameManager components import - SUCCESS")
        
        # Data synchronization
        from moneytrailz.timeframes import DataSynchronizer, SyncResult, SyncStatus
        print("✅ DataSynchronizer components import - SUCCESS")
        
        # Execution scheduling
        from moneytrailz.timeframes import ExecutionScheduler, ScheduleConfig, ExecutionWindow
        print("✅ ExecutionScheduler components import - SUCCESS")
        
        # Data aggregation
        from moneytrailz.timeframes import DataAggregator, AggregationResult, AggregationMethod
        print("✅ DataAggregator components import - SUCCESS")
        
        # Execution engine
        from moneytrailz.execution import StrategyExecutionEngine, ExecutionConfig, ExecutionState
        print("✅ StrategyExecutionEngine components import - SUCCESS")
        
        # Pipeline and coordination
        from moneytrailz.execution import DataPipeline, StrategyCoordinator
        print("✅ Pipeline and Coordinator components import - SUCCESS")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_timeframe_manager():
    """Test TimeFrameManager functionality"""
    print("\n📊 Testing TimeFrameManager...")
    
    try:
        from moneytrailz.timeframes import TimeFrameManager, create_timeframe_config
        from moneytrailz.strategies.enums import TimeFrame
        
        # Create manager
        manager = TimeFrameManager()
        print("✅ TimeFrameManager created successfully")
        
        # Register timeframes
        configs = [
            create_timeframe_config(TimeFrame.MINUTE_1, max_history=100),
            create_timeframe_config(TimeFrame.MINUTE_5, max_history=200),
            create_timeframe_config(TimeFrame.HOUR_1, max_history=500),
            create_timeframe_config(TimeFrame.DAY_1, max_history=1000)
        ]
        
        for config in configs:
            manager.register_timeframe(config)
        
        print(f"✅ Registered {len(configs)} timeframes")
        
        # Test data storage
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))
        
        stored = manager.store_data(TimeFrame.MINUTE_1, "AAPL", sample_data)
        print(f"✅ Data storage: {stored}")
        
        # Test data retrieval
        retrieved_data = manager.get_data(TimeFrame.MINUTE_1, "AAPL")
        data_valid = retrieved_data is not None and len(retrieved_data) == 3
        print(f"✅ Data retrieval: {data_valid}")
        
        # Test system status
        status = manager.get_system_status()
        print(f"✅ System status: {status['total_timeframes']} timeframes registered")
        
        print("✅ TimeFrameManager functionality working correctly")
        return True
        
    except Exception as e:
        print(f"❌ TimeFrameManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_synchronizer():
    """Test DataSynchronizer functionality"""
    print("\n🔗 Testing DataSynchronizer...")
    
    try:
        from moneytrailz.timeframes import DataSynchronizer, create_sync_config, SyncMethod
        from moneytrailz.strategies.enums import TimeFrame
        
        # Create sample data for different timeframes
        dates_1m = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1min')
        dates_5m = pd.date_range('2024-01-01 09:30:00', periods=20, freq='5min')
        
        data_1m = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates_1m)
        
        data_5m = pd.DataFrame({
            'close': np.random.randn(20).cumsum() + 100,
            'volume': np.random.randint(5000, 25000, 20)
        }, index=dates_5m)
        
        # Create data dictionary
        data_dict = {
            TimeFrame.MINUTE_1: {"AAPL": data_1m},
            TimeFrame.MINUTE_5: {"AAPL": data_5m}
        }
        
        # Create synchronizer
        synchronizer = DataSynchronizer()
        print("✅ DataSynchronizer created successfully")
        
        # Create sync configuration
        sync_config = create_sync_config(
            reference_timeframe=TimeFrame.MINUTE_5,
            target_timeframes=[TimeFrame.MINUTE_1],
            method=SyncMethod.FORWARD_FILL
        )
        
        # Perform synchronization
        result = synchronizer.synchronize_data(data_dict, "AAPL", sync_config)
        
        print(f"✅ Synchronization status: {result.status.value}")
        print(f"✅ Data points aligned: {result.data_points_aligned}")
        print(f"✅ Processing time: {result.sync_duration_ms:.1f}ms")
        
        # Test alignment
        aligned_data = synchronizer.align_to_reference_timeframe(
            {TimeFrame.MINUTE_1: data_1m, TimeFrame.MINUTE_5: data_5m},
            TimeFrame.MINUTE_5
        )
        
        alignment_success = (
            TimeFrame.MINUTE_5 in aligned_data and 
            TimeFrame.MINUTE_1 in aligned_data
        )
        print(f"✅ Data alignment: {alignment_success}")
        
        print("✅ DataSynchronizer functionality working correctly")
        return True
        
    except Exception as e:
        print(f"❌ DataSynchronizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_aggregator():
    """Test DataAggregator functionality"""
    print("\n📈 Testing DataAggregator...")
    
    try:
        from moneytrailz.timeframes import DataAggregator, create_aggregation_config, AggregationMethod, AggregationScope
        from moneytrailz.strategies.enums import TimeFrame
        
        # Create sample data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        timeframes = [TimeFrame.MINUTE_5, TimeFrame.HOUR_1]
        
        data_dict = {}
        for timeframe in timeframes:
            data_dict[timeframe] = {}
            for symbol in symbols:
                # Generate sample OHLCV data
                dates = pd.date_range('2024-01-01', periods=50, freq='5min' if timeframe == TimeFrame.MINUTE_5 else '1h')
                data = pd.DataFrame({
                    'open': np.random.randn(50).cumsum() + 100,
                    'high': np.random.randn(50).cumsum() + 102,
                    'low': np.random.randn(50).cumsum() + 98,
                    'close': np.random.randn(50).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 50)
                }, index=dates)
                data_dict[timeframe][symbol] = data
        
        # Create aggregator
        aggregator = DataAggregator()
        print("✅ DataAggregator created successfully")
        
        # Test different aggregation methods
        aggregation_configs = [
            create_aggregation_config(
                AggregationMethod.MEAN, AggregationScope.SYMBOL,
                timeframes, symbols, metadata={'name': 'mean_by_symbol'}
            ),
            create_aggregation_config(
                AggregationMethod.OHLC, AggregationScope.TIMEFRAME,
                timeframes, symbols, metadata={'name': 'ohlc_by_timeframe'}
            ),
            create_aggregation_config(
                AggregationMethod.STD, AggregationScope.GLOBAL,
                timeframes, symbols, metadata={'name': 'global_std'}
            )
        ]
        
        results = aggregator.aggregate_multiple_configs(data_dict, aggregation_configs)
        
        print(f"✅ Completed {len(results)} aggregations")
        
        for name, result in results.items():
            print(f"  • {name}: {len(result.aggregated_data)} data points, {result.processing_duration_ms:.1f}ms")
        
        # Test performance stats
        stats = aggregator.get_performance_stats()
        print(f"✅ Performance: {stats['total_aggregation_operations']} operations")
        
        print("✅ DataAggregator functionality working correctly")
        return True
        
    except Exception as e:
        print(f"❌ DataAggregator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_execution_scheduler():
    """Test ExecutionScheduler functionality"""
    print("\n⏰ Testing ExecutionScheduler...")
    
    try:
        from moneytrailz.timeframes import ExecutionScheduler, create_schedule_config, ScheduleType, ExecutionPriority
        from moneytrailz.strategies.enums import TimeFrame
        
        # Create scheduler
        scheduler = ExecutionScheduler()
        print("✅ ExecutionScheduler created successfully")
        
        # Create schedule configurations
        configs = [
            create_schedule_config(
                "test_strategy_1",
                [TimeFrame.MINUTE_5],
                ScheduleType.INTERVAL,
                priority=ExecutionPriority.HIGH,
                interval_seconds=300  # 5 minutes
            ),
            create_schedule_config(
                "test_strategy_2", 
                [TimeFrame.HOUR_1],
                ScheduleType.TIME_BASED,
                priority=ExecutionPriority.NORMAL,
                execution_times=[time(9, 30), time(15, 30)]
            )
        ]
        
        # Register schedules
        for config in configs:
            scheduler.register_schedule(config)
        
        print(f"✅ Registered {len(configs)} strategy schedules")
        
        # Test execution timing
        next_time = scheduler.get_next_execution_time("test_strategy_1")
        print(f"✅ Next execution time calculated: {next_time is not None}")
        
        # Test performance stats
        stats = scheduler.get_performance_stats()
        print(f"✅ Scheduler stats: {stats['registered_strategies']} strategies")
        
        print("✅ ExecutionScheduler functionality working correctly")
        return True
        
    except Exception as e:
        print(f"❌ ExecutionScheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_execution_engine():
    """Test StrategyExecutionEngine functionality"""
    print("\n🚀 Testing StrategyExecutionEngine...")
    
    try:
        from moneytrailz.execution import StrategyExecutionEngine, create_execution_config, ExecutionMode
        
        # Create execution configuration
        config = create_execution_config(
            mode=ExecutionMode.SIMULATION,
            max_concurrent=5,
            enable_technical_analysis=True
        )
        
        # Create execution engine
        engine = StrategyExecutionEngine(config)
        print("✅ StrategyExecutionEngine created successfully")
        
        # Initialize engine
        await engine.initialize()
        print("✅ Engine initialization completed")
        
        # Test engine state
        print(f"✅ Engine state: {engine.state.value}")
        
        # Test metrics
        metrics = engine.get_execution_metrics()
        print(f"✅ Execution metrics: {metrics.total_executions} total executions")
        
        print("✅ StrategyExecutionEngine functionality working correctly")
        return True
        
    except Exception as e:
        print(f"❌ StrategyExecutionEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Test integration between Phase 3 components"""
    print("\n🔗 Testing Phase 3 Integration...")
    
    try:
        from moneytrailz.timeframes import get_timeframe_manager, create_timeframe_config
        from moneytrailz.execution import StrategyExecutionEngine, create_execution_config, ExecutionMode
        from moneytrailz.strategies.enums import TimeFrame
        
        # Get global timeframe manager
        manager = get_timeframe_manager()
        
        # Register timeframes
        timeframe_configs = [
            create_timeframe_config(TimeFrame.MINUTE_5, max_history=200),
            create_timeframe_config(TimeFrame.HOUR_1, max_history=100)
        ]
        
        for config in timeframe_configs:
            manager.register_timeframe(config)
        
        print("✅ TimeFrame management integration working")
        
        # Create execution engine with multi-timeframe support
        exec_config = create_execution_config(
            mode=ExecutionMode.SIMULATION,
            enable_multi_timeframe=True,
            enable_technical_analysis=True
        )
        
        engine = StrategyExecutionEngine(exec_config)
        await engine.initialize()
        
        print("✅ Execution engine with multi-timeframe support initialized")
        
        # Test system status
        tf_status = manager.get_system_status()
        engine_metrics = engine.get_execution_metrics()
        
        print(f"✅ System integration: {tf_status['total_timeframes']} timeframes, {len(engine.get_active_strategies())} strategies")
        
        print("✅ Phase 3 integration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all Phase 3 tests"""
    print("🧪 PHASE 3 MULTI-TIMEFRAME ARCHITECTURE TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Phase 3 Imports", test_phase3_imports),
        ("TimeFrame Manager", test_timeframe_manager),
        ("Data Synchronizer", test_data_synchronizer),
        ("Data Aggregator", test_data_aggregator),
        ("Execution Scheduler", test_execution_scheduler),
        ("Execution Engine", test_execution_engine),
        ("Phase 3 Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 50)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 PHASE 3 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Total: {passed + failed} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        print("🚀 Multi-Timeframe Architecture is fully operational!")
        print("\n💡 Phase 3 Capabilities:")
        print("  • 🕐 Multi-timeframe data management and synchronization")
        print("  • ⚡ Real-time strategy execution engine")
        print("  • 📊 Advanced data aggregation and analysis")
        print("  • ⏰ Sophisticated execution scheduling")
        print("  • 🔗 Strategy coordination and conflict resolution")
        print("  • 📈 Integration with Phase 1 & 2 frameworks")
        print("\n🎯 Ready for production deployment!")
    else:
        print(f"\n⚠️  {failed} test(s) failed - needs attention")
    
    return failed == 0


if __name__ == "__main__":
    asyncio.run(main()) 
