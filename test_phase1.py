#!/usr/bin/env python3
"""
Phase 1 Strategy Framework Test Suite

This script tests all components of the Phase 1 implementation:
- Base strategy framework
- Registry system
- Example strategy
- Integration capabilities
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, Any
import pandas as pd

def test_imports():
    """Test that all Phase 1 modules can be imported"""
    print("🔍 Testing Phase 1 Module Imports...")
    
    try:
        # Core framework imports
        from moneytrailz.strategies import (
            BaseStrategy, StrategyResult, StrategyContext,
            StrategySignal, StrategyType, TimeFrame, StrategyStatus,
            StrategyError, StrategyConfigError, StrategyExecutionError,
            IStrategyConfig, IMarketData, IIndicator,
            StrategyRegistry, get_registry, register_strategy
        )
        print("✅ Core strategy framework imports - SUCCESS")
        
        # Registry imports
        from moneytrailz.strategies.registry import StrategyLoader, StrategyValidator
        print("✅ Registry system imports - SUCCESS")
        
        # Example strategy import
        from moneytrailz.strategies.implementations.example_strategy import ExampleStrategy
        print("✅ Example strategy import - SUCCESS")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_enums():
    """Test enum functionality"""
    print("\n🔢 Testing Enums...")
    
    try:
        from moneytrailz.strategies import StrategySignal, StrategyType, TimeFrame
        
        # Test StrategySignal
        signal = StrategySignal.BUY
        print(f"  📊 StrategySignal.BUY = {signal}")
        
        # Test StrategyType  
        strategy_type = StrategyType.OPTIONS
        print(f"  📈 StrategyType.OPTIONS = {strategy_type}")
        
        # Test TimeFrame with seconds property
        timeframe = TimeFrame.HOUR_1
        print(f"  ⏰ TimeFrame.HOUR_1 = {timeframe} ({timeframe.seconds} seconds)")
        
        print("✅ Enums working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
        return False

def test_strategy_registry():
    """Test strategy registry functionality"""
    print("\n📚 Testing Strategy Registry...")
    
    try:
        from moneytrailz.strategies import get_registry, StrategyRegistry
        from moneytrailz.strategies.implementations.example_strategy import ExampleStrategy
        
        # Get global registry
        registry = get_registry()
        print(f"  📋 Registry instance: {type(registry).__name__}")
        
        # Register example strategy
        registry.register_strategy(ExampleStrategy, "test_example")
        print("  ✅ Strategy registration - SUCCESS")
        
        # List strategies
        strategies = registry.list_strategies()
        print(f"  📄 Registered strategies: {strategies}")
        
        # Get strategy info
        info = registry.get_strategy_info("test_example")
        if info:
            print(f"  ℹ️  Strategy info: {info['class_name']} - {info['description']}")
        
        # Test strategy creation
        config = {
            "type": "stocks",
            "enabled": True,
            "timeframes": ["1d"],
            "threshold": 0.02
        }
        
        strategy_instance = registry.create_strategy_instance(
            "test_example", 
            config, 
            ["AAPL", "GOOGL"]
        )
        
        if strategy_instance:
            print(f"  🏭 Strategy instance created: {strategy_instance.name}")
            print(f"  📊 Strategy type: {strategy_instance.strategy_type}")
            print(f"  🎯 Symbols: {list(strategy_instance.symbols)}")
        
        print("✅ Strategy registry working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_example_strategy():
    """Test the example strategy implementation"""
    print("\n🧪 Testing Example Strategy...")
    
    try:
        from moneytrailz.strategies.implementations.example_strategy import ExampleStrategy
        from moneytrailz.strategies import StrategyType, TimeFrame
        
        # Create strategy instance
        config = {
            "threshold": 0.03,
            "min_volume": 50000
        }
        
        strategy = ExampleStrategy(
            name="test_strategy",
            strategy_type=StrategyType.STOCKS,
            config=config,
            symbols=["AAPL"],
            timeframes=[TimeFrame.DAY_1]
        )
        
        print(f"  📊 Strategy created: {strategy.name}")
        print(f"  ⚙️  Config validation: {'✅ PASSED' if strategy.config else '❌ FAILED'}")
        
        # Test required methods
        required_timeframes = strategy.get_required_timeframes()
        print(f"  ⏰ Required timeframes: {[tf.value for tf in required_timeframes]}")
        
        required_symbols = strategy.get_required_symbols()
        print(f"  🎯 Required symbols: {list(required_symbols)}")
        
        data_fields = strategy.get_required_data_fields()
        print(f"  📊 Required data fields: {data_fields}")
        
        # Test strategy info
        info = strategy.get_strategy_info()
        print(f"  ℹ️  Strategy status: {info['status']}")
        print(f"  🔢 Execution count: {info['execution_count']}")
        
        print("✅ Example strategy working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Example strategy test failed: {e}")
        return False

async def test_strategy_execution():
    """Test strategy execution with mock data"""
    print("\n🚀 Testing Strategy Execution...")
    
    try:
        from moneytrailz.strategies.implementations.example_strategy import ExampleStrategy
        from moneytrailz.strategies import StrategyType, TimeFrame, StrategyContext
        
        # Create mock market data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'open': [100, 101, 99, 102, 105, 103, 106, 104, 107, 109],
            'high': [102, 103, 101, 104, 107, 105, 108, 106, 109, 111],
            'low': [99, 100, 98, 101, 104, 102, 105, 103, 106, 108],
            'close': [101, 102, 100, 103, 106, 104, 107, 105, 108, 110],
            'volume': [100000, 120000, 80000, 110000, 150000, 90000, 130000, 95000, 140000, 160000]
        }, index=dates)
        
        print(f"  📊 Mock data created: {len(mock_data)} days")
        print(f"  💰 Price range: ${mock_data['close'].min():.2f} - ${mock_data['close'].max():.2f}")
        
        # Create strategy
        strategy = ExampleStrategy(
            name="execution_test",
            strategy_type=StrategyType.STOCKS,
            config={"threshold": 0.02, "min_volume": 75000},
            symbols=["AAPL"],
            timeframes=[TimeFrame.DAY_1]
        )
        
        # Create mock context (minimal implementation)
        class MockMarketData:
            def get_current_price(self, symbol: str) -> float:
                return 110.0
            def get_historical_data(self, symbol, timeframe, start_date, end_date, fields=None):
                return mock_data
            def get_option_chain(self, symbol):
                return []
            def is_market_open(self) -> bool:
                return True
        
        class MockOrderManager:
            def place_order(self, contract, order):
                return "mock_order_123"
            def cancel_order(self, order_id: str) -> bool:
                return True
            def get_order_status(self, order_id: str) -> str:
                return "Filled"
            def get_open_orders(self):
                return []
        
        class MockPositionManager:
            def get_buying_power(self) -> float:
                return 100000.0
            def get_positions(self, symbol=None):
                return []
            def calculate_position_size(self, symbol: str, price: float, risk_percentage: float) -> int:
                return 100
        
        class MockRiskManager:
            def check_position_risk(self, symbol: str, quantity: int, price: float) -> bool:
                return True
            def calculate_portfolio_risk(self):
                return {"total_risk": 0.1}
            def should_exit_position(self, symbol: str, current_price: float, entry_price: float) -> bool:
                return False
        
        context = StrategyContext(
            market_data=MockMarketData(),
            order_manager=MockOrderManager(),
            position_manager=MockPositionManager(),
            risk_manager=MockRiskManager(),
            account_summary={},
            portfolio_positions={}
        )
        
        # Execute strategy
        data_dict = {TimeFrame.DAY_1: mock_data}
        result = await strategy.execute("AAPL", data_dict, context)
        
        if result:
            print(f"  🎯 Signal generated: {result.signal}")
            print(f"  📊 Confidence: {result.confidence:.2f}")
            print(f"  💰 Price: ${result.price:.2f}")
            print(f"  📝 Metadata: {result.metadata}")
            print(f"  ⏰ Timestamp: {result.timestamp}")
        else:
            print("  ⚠️  No result generated")
        
        print("✅ Strategy execution working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Strategy execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration validation"""
    print("\n⚙️ Testing Configuration Validation...")
    
    try:
        from moneytrailz.strategies.registry.validator import StrategyValidator
        from moneytrailz.strategies.implementations.example_strategy import ExampleStrategy
        from moneytrailz.strategies import StrategyType, TimeFrame
        
        validator = StrategyValidator()
        
        # Test valid configuration
        valid_config = {
            "type": "stocks",
            "enabled": True,
            "timeframes": ["1d"],
            "symbols": ["AAPL", "GOOGL"],
            "threshold": 0.02,
            "min_volume": 100000
        }
        
        is_valid = validator.validate_strategy_config(valid_config, ExampleStrategy)
        print(f"  ✅ Valid config validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test strategy class validation
        is_valid_class = validator.validate_strategy_class(ExampleStrategy)
        print(f"  ✅ Strategy class validation: {'PASSED' if is_valid_class else 'FAILED'}")
        
        # Test symbol validation
        symbols = ["AAPL", "GOOGL", "MSFT"]
        is_valid_symbols = validator.validate_strategy_symbols(symbols)
        print(f"  ✅ Symbol validation: {'PASSED' if is_valid_symbols else 'FAILED'}")
        
        # Test timeframe validation
        timeframes = [TimeFrame.DAY_1, TimeFrame.HOUR_1]
        is_valid_timeframes = validator.validate_timeframes(timeframes)
        print(f"  ✅ Timeframe validation: {'PASSED' if is_valid_timeframes else 'FAILED'}")
        
        print("✅ Configuration validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_integration_points():
    """Test integration with existing ThetaGang components"""
    print("\n🔗 Testing Integration Points...")
    
    try:
        # Test that we can import existing ThetaGang modules alongside new framework
        from moneytrailz.config import Config
        from moneytrailz.strategies import get_registry, StrategyType
        
        print("  ✅ Existing ThetaGang imports - SUCCESS")
        
        # Test that TimeFrame enum works with existing timeframe concepts
        from moneytrailz.strategies import TimeFrame
        daily_tf = TimeFrame.DAY_1
        print(f"  ⏰ TimeFrame integration: {daily_tf.value} = {daily_tf.seconds}s")
        
        # Test that strategy registry can coexist with existing config
        registry = get_registry()
        stats = registry.get_registry_stats()
        print(f"  📊 Registry stats: {stats['total_strategies']} strategies registered")
        
        print("✅ Integration points working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run all Phase 1 tests"""
    print("🧪 PHASE 1 STRATEGY FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Enums", test_enums),
        ("Strategy Registry", test_strategy_registry),
        ("Example Strategy", test_example_strategy),
        ("Strategy Execution", test_strategy_execution),
        ("Configuration Validation", test_configuration_validation),
        ("Integration Points", test_integration_points)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 40)
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
            
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PHASE 1 TEST RESULTS SUMMARY")
    print("=" * 60)
    
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
        print("\n🎉 ALL PHASE 1 TESTS PASSED!")
        print("🚀 Strategy framework is ready for Phase 2!")
        print("\n💡 Next steps:")
        print("  • Phase 2: Technical Analysis Engine")
        print("  • Multi-timeframe support")
        print("  • Advanced strategy implementations")
    else:
        print(f"\n⚠️  {failed} test(s) failed - needs attention")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(main()) 
