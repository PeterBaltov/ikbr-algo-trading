#!/usr/bin/env python3
"""
🧪 BACKTESTING FRAMEWORK TESTS
==============================

Comprehensive tests for the backtesting framework including:
- Data management and validation
- Trade simulation and execution
- Performance calculation and metrics
- Strategy backtesting API
- Portfolio management and risk analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Import backtesting components
from thetagang.backtesting import (
    DataManager, BacktestEngine, TradeSimulator, BacktestStrategy,
    PortfolioManager, StrategyOptimizer, ReportGenerator, LiveTradingAdapter
)
from thetagang.analytics import (
    PerformanceCalculator, RiskCalculator, AttributionAnalyzer, ChartGenerator
)
from thetagang.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from thetagang.strategies.enums import StrategySignal, StrategyType, TimeFrame


class MockBacktestStrategy(BacktestStrategy):
    """Mock strategy for backtesting tests."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, StrategyType.STOCKS, config, ['TEST'], [TimeFrame.DAY_1])
        self.trade_count = 0
        self.current_position = 0
        
    async def on_start(self, context: StrategyContext) -> None:
        """Strategy initialization."""
        self.initial_capital = context.portfolio_manager.get_total_value()
        
    async def on_data(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> Optional[StrategyResult]:
        """Process new data and generate signals."""
        if TimeFrame.DAY_1 not in data or data[TimeFrame.DAY_1].empty:
            return None
            
        df = data[TimeFrame.DAY_1]
        current_price = df['close'].iloc[-1]
        
        # Simple mean reversion strategy
        if len(df) >= 20:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            
            if current_price < sma_20 * 0.95 and self.current_position == 0:
                # Buy signal
                self.current_position = 1
                return StrategyResult(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal=StrategySignal.BUY,
                    confidence=0.7,
                    price=current_price,
                    timestamp=df.index[-1],
                    metadata={'sma_20': sma_20, 'deviation': (current_price - sma_20) / sma_20}
                )
            elif current_price > sma_20 * 1.05 and self.current_position == 1:
                # Sell signal
                self.current_position = 0
                return StrategyResult(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal=StrategySignal.SELL,
                    confidence=0.7,
                    price=current_price,
                    timestamp=df.index[-1],
                    metadata={'sma_20': sma_20, 'deviation': (current_price - sma_20) / sma_20}
                )
        
        return None
    
    async def on_end(self, context: StrategyContext) -> None:
        """Strategy cleanup."""
        self.final_capital = context.portfolio_manager.get_total_value()
        
    def get_required_timeframes(self) -> set:
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> set:
        return {'TEST'}
    
    def get_required_data_fields(self) -> set:
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        pass


class TestDataManager:
    """Test suite for DataManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_config = {
            'source': 'csv',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'timeframes': [TimeFrame.DAY_1],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        self.data_manager = DataManager(self.data_config)
        
    def create_test_data(self, symbol: str, periods: int = 252) -> pd.DataFrame:
        """Create test market data."""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, periods)  # 0.1% daily return, 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(500000, 2000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def test_data_manager_initialization(self):
        """Test DataManager initialization."""
        assert self.data_manager.config == self.data_config
        assert self.data_manager.data_source == 'csv'
        assert set(self.data_manager.symbols) == set(['AAPL', 'GOOGL', 'MSFT'])
        assert self.data_manager.timeframes == [TimeFrame.DAY_1]
    
    def test_data_loading(self):
        """Test data loading functionality."""
        # Mock data loading
        test_data = self.create_test_data('AAPL')
        
        with patch.object(self.data_manager, '_load_csv_data') as mock_load:
            mock_load.return_value = test_data
            
            loaded_data = self.data_manager.load_data('AAPL', TimeFrame.DAY_1)
            
            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == len(test_data)
            assert all(col in loaded_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_data_validation(self):
        """Test data validation."""
        # Create valid data
        valid_data = self.create_test_data('TEST')
        
        is_valid = self.data_manager.validate_data(valid_data)
        assert is_valid
        
        # Create invalid data (missing columns)
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103]
            # Missing high, low, volume
        })
        
        is_valid = self.data_manager.validate_data(invalid_data)
        assert not is_valid
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        raw_data = self.create_test_data('TEST')
        
        # Add some NaN values to test preprocessing
        raw_data.loc[raw_data.index[10], 'close'] = np.nan
        
        processed_data = self.data_manager.preprocess_data(raw_data)
        
        # Should have filled NaN values
        assert not processed_data['close'].isna().any()


class TestTradeSimulator:
    """Test suite for TradeSimulator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'commission': 0.001,  # 0.1% commission
            'slippage': 0.0005,   # 0.05% slippage
            'market_impact': True,
            'initial_capital': 100000
        }
        self.simulator = TradeSimulator(self.config)
        
    def test_simulator_initialization(self):
        """Test TradeSimulator initialization."""
        assert self.simulator.commission == 0.001
        assert self.simulator.slippage == 0.0005
        assert self.simulator.market_impact == True
        assert self.simulator.initial_capital == 100000
    
    def test_order_execution(self):
        """Test order execution simulation."""
        # Create test order
        order = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.00,
            'timestamp': datetime.now()
        }
        
        execution = self.simulator.execute_order(order)
        
        assert isinstance(execution, dict)
        assert 'executed_price' in execution
        assert 'executed_quantity' in execution
        assert 'commission' in execution
        assert 'slippage_cost' in execution
        
        # Check commission calculation
        expected_commission = 150.00 * 100 * 0.001
        assert abs(execution['commission'] - expected_commission) < 0.01
    
    def test_market_impact_simulation(self):
        """Test market impact simulation."""
        # Large order should have more market impact
        large_order = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10000,  # Large quantity
            'price': 150.00,
            'timestamp': datetime.now()
        }
        
        small_order = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,   # Small quantity
            'price': 150.00,
            'timestamp': datetime.now()
        }
        
        large_execution = self.simulator.execute_order(large_order)
        small_execution = self.simulator.execute_order(small_order)
        
        # Large order should have higher execution price (more market impact)
        if self.simulator.market_impact:
            assert large_execution['executed_price'] >= small_execution['executed_price']
    
    def test_portfolio_tracking(self):
        """Test portfolio tracking during simulation."""
        # Execute some trades
        orders = [
            {'symbol': 'AAPL', 'action': 'BUY', 'quantity': 100, 'price': 150.00, 'timestamp': datetime.now()},
            {'symbol': 'GOOGL', 'action': 'BUY', 'quantity': 50, 'price': 2000.00, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'action': 'SELL', 'quantity': 50, 'price': 155.00, 'timestamp': datetime.now()}
        ]
        
        for order in orders:
            self.simulator.execute_order(order)
        
        portfolio = self.simulator.get_portfolio_state()
        
        assert isinstance(portfolio, dict)
        assert 'positions' in portfolio
        assert 'cash' in portfolio
        assert 'total_value' in portfolio
        
        # Check positions
        assert 'AAPL' in portfolio['positions']
        assert 'GOOGL' in portfolio['positions']
        assert portfolio['positions']['AAPL'] == 50  # 100 bought, 50 sold
        assert portfolio['positions']['GOOGL'] == 50


class TestBacktestEngine:
    """Test suite for BacktestEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine_config = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'start_date': '2023-01-01',
            'end_date': '2023-06-30'
        }
        self.engine = BacktestEngine(self.engine_config)
        
    def test_engine_initialization(self):
        """Test BacktestEngine initialization."""
        assert self.engine.config == self.engine_config
        assert self.engine.initial_capital == 100000
        assert isinstance(self.engine.trade_simulator, TradeSimulator)
        assert isinstance(self.engine.data_manager, DataManager)
    
    def test_strategy_registration(self):
        """Test strategy registration with engine."""
        strategy = MockBacktestStrategy('test_strategy', {'param1': 'value1'})
        
        self.engine.add_strategy(strategy)
        
        assert 'test_strategy' in self.engine.strategies
        assert self.engine.strategies['test_strategy'] == strategy
    
    def test_backtest_execution(self):
        """Test backtest execution."""
        # Create test strategy
        strategy = MockBacktestStrategy('mean_reversion', {'lookback': 20})
        self.engine.add_strategy(strategy)
        
        # Mock data for backtesting
        test_data = self.create_backtest_data()
        
        with patch.object(self.engine.data_manager, 'load_data') as mock_load:
            mock_load.return_value = test_data
            
            # Run backtest
            results = self.engine.run_backtest()
            
            assert isinstance(results, dict)
            assert 'strategy_results' in results
            assert 'portfolio_summary' in results
            assert 'performance_metrics' in results
            
            # Check strategy results
            strategy_result = results['strategy_results']['mean_reversion']
            assert 'trades' in strategy_result
            assert 'returns' in strategy_result
            assert 'final_capital' in strategy_result
    
    def create_backtest_data(self, periods: int = 126) -> pd.DataFrame:
        """Create test data for backtesting."""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        # Create mean-reverting price series for testing
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, periods):
            # Mean reversion: price tends to revert to 100
            mean_reversion = (100 - prices[-1]) * 0.05
            noise = np.random.normal(0, 1)
            change = mean_reversion + noise
            new_price = max(prices[-1] + change, 50)  # Minimum price of 50
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * 1.02
            low = price * 0.98
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(800000, 1200000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)


class TestPerformanceCalculation:
    """Test suite for performance calculation and metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.performance_calc = PerformanceCalculator()
        self.risk_calc = RiskCalculator()
        
    def create_sample_returns(self, periods: int = 252) -> pd.Series:
        """Create sample return series for testing."""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        returns = np.random.normal(0.0008, 0.02, periods)  # 0.08% daily return, 2% volatility
        return pd.Series(returns, index=dates)
    
    def test_performance_calculator_initialization(self):
        """Test PerformanceCalculator initialization."""
        assert hasattr(self.performance_calc, 'calculate_returns')
        assert hasattr(self.performance_calc, 'calculate_sharpe_ratio')
        assert hasattr(self.performance_calc, 'calculate_max_drawdown')
    
    def test_return_calculation(self):
        """Test return calculation."""
        # Create sample price series
        prices = pd.Series([100, 102, 104, 103, 105, 107])
        
        returns = self.performance_calc.calculate_returns(prices)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1  # One less return than prices
        
        # Check first return calculation
        expected_first_return = (102 - 100) / 100
        assert abs(returns.iloc[0] - expected_first_return) < 0.0001
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = self.create_sample_returns()
        
        sharpe = self.performance_calc.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        # Sharpe ratio should be reasonable for our test data
        assert -3 < sharpe < 3
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 120, 90, 85, 95, 105])  # 29.17% drawdown from 120 to 85
        
        max_dd = self.performance_calc.calculate_max_drawdown(prices)
        
        assert isinstance(max_dd, float)
        assert max_dd < 0  # Drawdown should be negative
        
        expected_dd = (85 - 120) / 120  # -29.17%
        assert abs(max_dd - expected_dd) < 0.01
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        returns = self.create_sample_returns()
        
        sortino = self.performance_calc.calculate_sortino_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sortino, float)
        # Sortino should be higher than Sharpe for same data (only downside volatility)
        sharpe = self.performance_calc.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sortino >= sharpe
    
    def test_calmar_ratio_calculation(self):
        """Test Calmar ratio calculation."""
        returns = self.create_sample_returns()
        
        calmar = self.performance_calc.calculate_calmar_ratio(returns)
        
        assert isinstance(calmar, float)
        # Calmar ratio should be reasonable
        assert -10 < calmar < 10
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        returns = self.create_sample_returns()
        
        # Test VaR calculation
        var_95 = self.risk_calc.calculate_var(returns, confidence_level=0.95)
        var_99 = self.risk_calc.calculate_var(returns, confidence_level=0.99)
        
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 < var_95  # 99% VaR should be more extreme than 95% VaR
        
        # Test CVaR calculation
        cvar_95 = self.risk_calc.calculate_cvar(returns, confidence_level=0.95)
        
        assert isinstance(cvar_95, float)
        assert cvar_95 < var_95  # CVaR should be more extreme than VaR
    
    def test_beta_calculation(self):
        """Test beta calculation."""
        returns = self.create_sample_returns()
        market_returns = self.create_sample_returns()  # Mock market returns
        
        beta = self.performance_calc.calculate_beta(returns, market_returns)
        
        assert isinstance(beta, float)
        # Beta should be reasonable
        assert -3 < beta < 3


class TestBacktestingIntegration:
    """Integration tests for complete backtesting workflow."""
    
    def test_end_to_end_backtest(self):
        """Test complete end-to-end backtesting workflow."""
        # 1. Set up backtesting configuration
        config = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'start_date': '2023-01-01',
            'end_date': '2023-06-30'
        }
        
        # 2. Create and configure backtesting engine
        engine = BacktestEngine(config)
        
        # 3. Add strategy
        strategy = MockBacktestStrategy('integration_test', {'param': 'value'})
        engine.add_strategy(strategy)
        
        # 4. Mock data
        test_data = self.create_integration_test_data()
        
        with patch.object(engine.data_manager, 'load_data') as mock_load:
            mock_load.return_value = test_data
            
            # 5. Run backtest
            results = engine.run_backtest()
            
            # 6. Validate results
            assert isinstance(results, dict)
            assert 'strategy_results' in results
            assert 'portfolio_summary' in results
            assert 'performance_metrics' in results
            
            # Check that strategy was executed
            strategy_result = results['strategy_results']['integration_test']
            assert 'final_capital' in strategy_result
            assert strategy_result['final_capital'] > 0
            
            # Check performance metrics
            metrics = results['performance_metrics']
            assert 'total_return' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
    
    def create_integration_test_data(self) -> pd.DataFrame:
        """Create test data for integration testing."""
        dates = pd.date_range(start='2023-01-01', periods=126, freq='D')
        
        # Create trending data with volatility
        base_price = 100.0
        trend = 0.001  # 0.1% daily trend
        volatility = 0.02  # 2% daily volatility
        
        prices = [base_price]
        for i in range(1, 126):
            change = trend + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 50))  # Floor at $50
        
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(1000000, 2000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)


def run_backtesting_tests():
    """Run all backtesting tests."""
    print("🧪 RUNNING BACKTESTING FRAMEWORK TESTS")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        # DataManager tests
        ("Data Manager Initialization", TestDataManager().test_data_manager_initialization),
        ("Data Loading", TestDataManager().test_data_loading),
        ("Data Validation", TestDataManager().test_data_validation),
        ("Data Preprocessing", TestDataManager().test_data_preprocessing),
        
        # TradeSimulator tests
        ("Trade Simulator Initialization", TestTradeSimulator().test_simulator_initialization),
        ("Order Execution", TestTradeSimulator().test_order_execution),
        ("Market Impact Simulation", TestTradeSimulator().test_market_impact_simulation),
        ("Portfolio Tracking", TestTradeSimulator().test_portfolio_tracking),
        
        # BacktestEngine tests
        ("Backtest Engine Initialization", TestBacktestEngine().test_engine_initialization),
        ("Strategy Registration", TestBacktestEngine().test_strategy_registration),
        ("Backtest Execution", TestBacktestEngine().test_backtest_execution),
        
        # Performance calculation tests
        ("Performance Calculator Init", TestPerformanceCalculation().test_performance_calculator_initialization),
        ("Return Calculation", TestPerformanceCalculation().test_return_calculation),
        ("Sharpe Ratio Calculation", TestPerformanceCalculation().test_sharpe_ratio_calculation),
        ("Max Drawdown Calculation", TestPerformanceCalculation().test_max_drawdown_calculation),
        ("Sortino Ratio Calculation", TestPerformanceCalculation().test_sortino_ratio_calculation),
        ("Calmar Ratio Calculation", TestPerformanceCalculation().test_calmar_ratio_calculation),
        ("Risk Metrics Calculation", TestPerformanceCalculation().test_risk_metrics_calculation),
        ("Beta Calculation", TestPerformanceCalculation().test_beta_calculation),
        
        # Integration tests
        ("End-to-End Backtest", TestBacktestingIntegration().test_end_to_end_backtest),
    ]
    
    passed = 0
    total = len(test_categories)
    
    for test_name, test_func in test_categories:
        try:
            # Set up appropriate test instance
            if "Data Manager" in test_name:
                test_instance = TestDataManager()
            elif "Trade Simulator" in test_name:
                test_instance = TestTradeSimulator()
            elif "Backtest Engine" in test_name:
                test_instance = TestBacktestEngine()
            elif "Performance" in test_name or "Risk" in test_name or "Sharpe" in test_name or "Drawdown" in test_name or "Sortino" in test_name or "Calmar" in test_name or "Beta" in test_name:
                test_instance = TestPerformanceCalculation()
            else:
                test_instance = TestBacktestingIntegration()
            
            test_instance.setup_method()
            
            # Run test
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print("📋 BACKTESTING FRAMEWORK TEST RESULTS")
    print("=" * 50)
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 ALL BACKTESTING TESTS PASSED!")
        print("🚀 Backtesting framework is robust and ready!")
        
        print("\n📊 Tested Components:")
        print("  • 📊 Data Management: Loading, validation, preprocessing")
        print("  • 💹 Trade Simulation: Order execution, slippage, market impact")
        print("  • 🎯 Backtest Engine: Strategy execution, portfolio tracking")
        print("  • 📈 Performance Analysis: Returns, Sharpe, Sortino, Calmar ratios")
        print("  • ⚠️  Risk Metrics: VaR, CVaR, maximum drawdown, beta")
        print("  • 🔗 Integration: End-to-end backtesting workflow")
        
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Backtesting framework needs attention.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    run_backtesting_tests() 
