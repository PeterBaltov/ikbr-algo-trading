#!/usr/bin/env python3
"""
🧪 STRATEGY REGISTRY TESTS
==========================

Comprehensive tests for the strategy registry system including:
- Registry operations (register, get, list, validate)
- Strategy discovery and metadata
- Registry persistence and state management
- Error handling and validation
"""

import pytest
from typing import Dict, List, Any, Set
from unittest.mock import MagicMock, patch

# Import registry components
from moneytrailz.strategies.registry import (
    StrategyRegistry, StrategyLoader, StrategyValidator, get_registry
)
from moneytrailz.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from moneytrailz.strategies.enums import StrategySignal, StrategyType, TimeFrame, StrategyStatus
from moneytrailz.strategies.exceptions import StrategyRegistrationError, StrategyNotFoundError


class MockStrategy(BaseStrategy):
    """Mock strategy for testing registry operations."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: dict):
        super().__init__(name, StrategyType.STOCKS, config, symbols, timeframes)
        self.mock_data = {'test': True}
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext):
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.HOLD,
            confidence=0.5,
            price=100.0,
            timestamp=datetime.now()
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> Set[str]:
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        pass


class MockOptionsStrategy(BaseStrategy):
    """Mock options strategy for testing different strategy types."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: dict):
        super().__init__(name, StrategyType.OPTIONS, config, symbols, timeframes)
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext):
        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=StrategySignal.BUY,
            confidence=0.8,
            price=150.0,
            timestamp=datetime.now()
        )
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        return {TimeFrame.DAY_1, TimeFrame.HOUR_1}
    
    def get_required_symbols(self) -> Set[str]:
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        return {'open', 'high', 'low', 'close', 'volume', 'options_data'}
    
    def validate_config(self) -> None:
        if not self.config.get('options_enabled'):
            raise ValueError("Options must be enabled for this strategy")


class TestStrategyRegistry:
    """Test suite for StrategyRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create fresh registry for each test
        self.registry = StrategyRegistry()
        
        # Test strategy configurations
        self.mock_strategy_class = MockStrategy
        self.mock_options_strategy_class = MockOptionsStrategy
        
        self.test_metadata = {
            'name': 'mock_strategy',
            'description': 'Mock strategy for testing',
            'version': '1.0.0',
            'author': 'Test Suite',
            'category': 'test',
            'risk_level': 'low'
        }
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert isinstance(self.registry.strategies, dict)
        assert len(self.registry.strategies) == 0
        assert isinstance(self.registry.metadata, dict)
        assert len(self.registry.metadata) == 0
    
    def test_strategy_registration(self):
        """Test strategy registration."""
        # Register strategy
        self.registry.register_strategy(
            'mock_strategy',
            self.mock_strategy_class,
            self.test_metadata
        )
        
        # Verify registration
        assert 'mock_strategy' in self.registry.strategies
        assert self.registry.strategies['mock_strategy'] == self.mock_strategy_class
        assert 'mock_strategy' in self.registry.metadata
        assert self.registry.metadata['mock_strategy'] == self.test_metadata
    
    def test_strategy_registration_duplicate(self):
        """Test duplicate strategy registration handling."""
        # Register strategy twice
        self.registry.register_strategy('test_strategy', self.mock_strategy_class, self.test_metadata)
        
        # Should not raise error by default (overwrites)
        self.registry.register_strategy('test_strategy', self.mock_strategy_class, self.test_metadata)
        
        # Verify still only one entry
        assert len(self.registry.strategies) == 1
    
    def test_strategy_retrieval(self):
        """Test strategy retrieval."""
        # Register strategy
        self.registry.register_strategy('test_strategy', self.mock_strategy_class, self.test_metadata)
        
        # Test get_strategy
        strategy_class = self.registry.get_strategy('test_strategy')
        assert strategy_class == self.mock_strategy_class
        
        # Test get_strategy_info
        info = self.registry.get_strategy_info('test_strategy')
        assert info == self.test_metadata
        
        # Test non-existent strategy
        with pytest.raises(StrategyNotFoundError):
            self.registry.get_strategy('non_existent')
    
    def test_strategy_listing(self):
        """Test strategy listing operations."""
        # Register multiple strategies
        self.registry.register_strategy('stock_strategy', self.mock_strategy_class, 
                                       {**self.test_metadata, 'type': 'stocks'})
        self.registry.register_strategy('options_strategy', self.mock_options_strategy_class,
                                       {**self.test_metadata, 'type': 'options'})
        
        # Test list_strategies
        all_strategies = self.registry.list_strategies()
        assert 'stock_strategy' in all_strategies
        assert 'options_strategy' in all_strategies
        assert len(all_strategies) == 2
        
        # Test get_all_metadata
        all_metadata = self.registry.get_all_metadata()
        assert 'stock_strategy' in all_metadata
        assert 'options_strategy' in all_metadata
        assert len(all_metadata) == 2
    
    def test_strategy_filtering(self):
        """Test strategy filtering by type and attributes."""
        # Register strategies with different metadata
        stock_metadata = {**self.test_metadata, 'strategy_type': 'stocks', 'risk_level': 'low'}
        options_metadata = {**self.test_metadata, 'strategy_type': 'options', 'risk_level': 'high'}
        
        self.registry.register_strategy('low_risk_stock', self.mock_strategy_class, stock_metadata)
        self.registry.register_strategy('high_risk_options', self.mock_options_strategy_class, options_metadata)
        
        # Test filtering by strategy type
        stock_strategies = [name for name, meta in self.registry.get_all_metadata().items() 
                           if meta.get('strategy_type') == 'stocks']
        assert 'low_risk_stock' in stock_strategies
        assert 'high_risk_options' not in stock_strategies
        
        # Test filtering by risk level
        high_risk_strategies = [name for name, meta in self.registry.get_all_metadata().items()
                               if meta.get('risk_level') == 'high']
        assert 'high_risk_options' in high_risk_strategies
        assert 'low_risk_stock' not in high_risk_strategies
    
    def test_strategy_unregistration(self):
        """Test strategy unregistration."""
        # Register strategy
        self.registry.register_strategy('temp_strategy', self.mock_strategy_class, self.test_metadata)
        assert 'temp_strategy' in self.registry.strategies
        
        # Unregister strategy
        success = self.registry.unregister_strategy('temp_strategy')
        assert success
        assert 'temp_strategy' not in self.registry.strategies
        assert 'temp_strategy' not in self.registry.metadata
        
        # Try to unregister non-existent strategy
        success = self.registry.unregister_strategy('non_existent')
        assert not success
    
    def test_registry_validation(self):
        """Test registry validation operations."""
        # Register valid strategy
        self.registry.register_strategy('valid_strategy', self.mock_strategy_class, self.test_metadata)
        
        # Test validation
        is_valid = self.registry.validate_strategy('valid_strategy')
        assert is_valid
        
        # Test validation of non-existent strategy
        is_valid = self.registry.validate_strategy('non_existent')
        assert not is_valid
    
    def test_registry_persistence_state(self):
        """Test registry state management."""
        # Register strategies
        self.registry.register_strategy('strategy1', self.mock_strategy_class, self.test_metadata)
        self.registry.register_strategy('strategy2', self.mock_options_strategy_class, self.test_metadata)
        
        # Test state retrieval
        state = self.registry.get_registry_state()
        assert 'strategies' in state
        assert 'metadata' in state
        assert 'strategy1' in state['strategies']
        assert 'strategy2' in state['strategies']
        
        # Test registry statistics
        stats = self.registry.get_registry_stats()
        assert stats['total_strategies'] == 2
        assert 'strategies_by_type' in stats


class TestStrategyLoader:
    """Test suite for StrategyLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = StrategyLoader()
        self.registry = StrategyRegistry()
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        assert hasattr(self.loader, 'load_strategy')
        assert hasattr(self.loader, 'load_from_module')
        assert hasattr(self.loader, 'load_from_config')
    
    def test_strategy_loading_from_class(self):
        """Test loading strategy from class."""
        # Test basic loading
        strategy_info = {
            'class': MockStrategy,
            'name': 'test_strategy',
            'symbols': ['AAPL'],
            'timeframes': [TimeFrame.DAY_1],
            'config': {'test': True}
        }
        
        loaded_strategy = self.loader.load_strategy(strategy_info)
        
        assert isinstance(loaded_strategy, MockStrategy)
        assert loaded_strategy.name == 'test_strategy'
        assert loaded_strategy.symbols == ['AAPL']
        assert loaded_strategy.config == {'test': True}
    
    def test_strategy_loading_validation(self):
        """Test strategy loading validation."""
        # Test invalid strategy info
        invalid_info = {
            'name': 'invalid_strategy',
            'symbols': ['AAPL'],
            'timeframes': [TimeFrame.DAY_1],
            'config': {}
            # Missing 'class' key
        }
        
        with pytest.raises(ValueError):
            self.loader.load_strategy(invalid_info)
    
    def test_config_based_loading(self):
        """Test loading strategy from configuration."""
        config = {
            'strategy_type': 'mock_strategy',
            'name': 'config_strategy',
            'symbols': ['GOOGL', 'AAPL'],
            'timeframes': ['1D'],
            'parameters': {
                'risk_tolerance': 0.02,
                'max_positions': 3
            }
        }
        
        # Mock the registry to return our test strategy
        with patch.object(self.loader, '_get_strategy_class') as mock_get_class:
            mock_get_class.return_value = MockStrategy
            
            loaded_strategy = self.loader.load_from_config(config)
            
            assert isinstance(loaded_strategy, MockStrategy)
            assert loaded_strategy.name == 'config_strategy'
            assert loaded_strategy.symbols == ['GOOGL', 'AAPL']


class TestStrategyValidator:
    """Test suite for StrategyValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = StrategyValidator()
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert hasattr(self.validator, 'validate_strategy_class')
        assert hasattr(self.validator, 'validate_strategy_config')
        assert hasattr(self.validator, 'validate_strategy_metadata')
    
    def test_strategy_class_validation(self):
        """Test strategy class validation."""
        # Test valid strategy class
        is_valid = self.validator.validate_strategy_class(MockStrategy)
        assert is_valid
        
        # Test invalid strategy class (not inheriting from BaseStrategy)
        class InvalidStrategy:
            pass
        
        is_valid = self.validator.validate_strategy_class(InvalidStrategy)
        assert not is_valid
    
    def test_strategy_config_validation(self):
        """Test strategy configuration validation."""
        # Test valid config
        valid_config = {
            'name': 'test_strategy',
            'symbols': ['AAPL'],
            'timeframes': [TimeFrame.DAY_1],
            'config': {'risk_tolerance': 0.02}
        }
        
        is_valid = self.validator.validate_strategy_config(valid_config)
        assert is_valid
        
        # Test invalid config (missing required fields)
        invalid_config = {
            'name': 'test_strategy',
            # Missing symbols, timeframes, config
        }
        
        is_valid = self.validator.validate_strategy_config(invalid_config)
        assert not is_valid
    
    def test_strategy_metadata_validation(self):
        """Test strategy metadata validation."""
        # Test valid metadata
        valid_metadata = {
            'name': 'test_strategy',
            'description': 'Test strategy',
            'version': '1.0.0',
            'author': 'Test Author'
        }
        
        is_valid = self.validator.validate_strategy_metadata(valid_metadata)
        assert is_valid
        
        # Test metadata with missing recommended fields
        minimal_metadata = {
            'name': 'test_strategy'
            # Missing description, version, author
        }
        
        # Should still be valid but might warn
        is_valid = self.validator.validate_strategy_metadata(minimal_metadata)
        assert is_valid  # Minimal metadata should still pass


class TestRegistryIntegration:
    """Integration tests for registry components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.registry = StrategyRegistry()
        self.loader = StrategyLoader()
        self.validator = StrategyValidator()
    
    def test_full_strategy_lifecycle(self):
        """Test complete strategy lifecycle through registry."""
        # 1. Register strategy
        metadata = {
            'name': 'lifecycle_test',
            'description': 'Strategy for testing full lifecycle',
            'version': '1.0.0',
            'category': 'test'
        }
        
        self.registry.register_strategy('lifecycle_test', MockStrategy, metadata)
        
        # 2. Validate registration
        assert 'lifecycle_test' in self.registry.list_strategies()
        assert self.registry.validate_strategy('lifecycle_test')
        
        # 3. Load strategy instance
        strategy_class = self.registry.get_strategy('lifecycle_test')
        strategy_config = {
            'name': 'lifecycle_instance',
            'symbols': ['TEST'],
            'timeframes': [TimeFrame.DAY_1],
            'config': {}
        }
        
        strategy_info = {
            'class': strategy_class,
            **strategy_config
        }
        
        loaded_strategy = self.loader.load_strategy(strategy_info)
        
        # 4. Validate loaded strategy
        assert isinstance(loaded_strategy, MockStrategy)
        assert loaded_strategy.name == 'lifecycle_instance'
        
        # 5. Unregister strategy
        success = self.registry.unregister_strategy('lifecycle_test')
        assert success
        assert 'lifecycle_test' not in self.registry.list_strategies()
    
    def test_global_registry_singleton(self):
        """Test global registry singleton behavior."""
        # Get registry instance
        registry1 = get_registry()
        registry2 = get_registry()
        
        # Should be the same instance
        assert registry1 is registry2
        
        # Register strategy in one instance
        registry1.register_strategy('singleton_test', MockStrategy, {'name': 'singleton_test'})
        
        # Should be visible in other instance
        assert 'singleton_test' in registry2.list_strategies()


def run_registry_tests():
    """Run all registry tests."""
    print("🧪 RUNNING STRATEGY REGISTRY TESTS")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        # StrategyRegistry tests
        ("Registry Initialization", TestStrategyRegistry().test_registry_initialization),
        ("Strategy Registration", TestStrategyRegistry().test_strategy_registration),
        ("Duplicate Registration", TestStrategyRegistry().test_strategy_registration_duplicate),
        ("Strategy Retrieval", TestStrategyRegistry().test_strategy_retrieval),
        ("Strategy Listing", TestStrategyRegistry().test_strategy_listing),
        ("Strategy Filtering", TestStrategyRegistry().test_strategy_filtering),
        ("Strategy Unregistration", TestStrategyRegistry().test_strategy_unregistration),
        ("Registry Validation", TestStrategyRegistry().test_registry_validation),
        ("Registry State", TestStrategyRegistry().test_registry_persistence_state),
        
        # StrategyLoader tests
        ("Loader Initialization", TestStrategyLoader().test_loader_initialization),
        ("Strategy Loading from Class", TestStrategyLoader().test_strategy_loading_from_class),
        ("Strategy Loading Validation", TestStrategyLoader().test_strategy_loading_validation),
        ("Config-Based Loading", TestStrategyLoader().test_config_based_loading),
        
        # StrategyValidator tests
        ("Validator Initialization", TestStrategyValidator().test_validator_initialization),
        ("Strategy Class Validation", TestStrategyValidator().test_strategy_class_validation),
        ("Strategy Config Validation", TestStrategyValidator().test_strategy_config_validation),
        ("Strategy Metadata Validation", TestStrategyValidator().test_strategy_metadata_validation),
        
        # Integration tests
        ("Full Strategy Lifecycle", TestRegistryIntegration().test_full_strategy_lifecycle),
        ("Global Registry Singleton", TestRegistryIntegration().test_global_registry_singleton),
    ]
    
    passed = 0
    total = len(test_categories)
    
    for test_name, test_func in test_categories:
        try:
            # Set up appropriate test instance
            if "Registry" in test_name and "Integration" not in test_name:
                test_instance = TestStrategyRegistry()
            elif "Loader" in test_name:
                test_instance = TestStrategyLoader()
            elif "Validator" in test_name:
                test_instance = TestStrategyValidator()
            else:
                test_instance = TestRegistryIntegration()
            
            test_instance.setup_method()
            
            # Run test
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print("📋 STRATEGY REGISTRY TEST RESULTS")
    print("=" * 50)
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 ALL REGISTRY TESTS PASSED!")
        print("🚀 Strategy registry system is robust and ready!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Strategy registry needs attention.")
    
    return passed == total


if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime
    run_registry_tests() 
