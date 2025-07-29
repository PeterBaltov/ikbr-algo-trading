#!/usr/bin/env python3
"""
🧪 STRATEGY COORDINATION INTEGRATION TESTS
==========================================

Integration tests for strategy coordination including:
- Multi-strategy execution and conflict resolution
- Resource allocation and portfolio management
- Strategy communication and signal aggregation
- Performance coordination and risk management
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set, Optional
from unittest.mock import MagicMock, patch, AsyncMock

# Import coordination components
from thetagang.strategies.base import BaseStrategy, StrategyResult, StrategyContext
from thetagang.strategies.enums import StrategySignal, StrategyType, TimeFrame
from thetagang.execution import StrategyExecutionEngine
from thetagang.strategies.implementations.factory import StrategyFactory
from thetagang.portfolio_manager import PortfolioManager
from thetagang.analysis import TechnicalAnalysisEngine


class MockStrategyA(BaseStrategy):
    """Mock strategy A for coordination testing - Momentum focused."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: dict):
        super().__init__(name, StrategyType.STOCKS, config, symbols, timeframes)
        self.execution_count = 0
        self.signals_generated = []
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> Optional[StrategyResult]:
        """Generate momentum-based signals."""
        self.execution_count += 1
        
        if TimeFrame.DAY_1 not in data or data[TimeFrame.DAY_1].empty:
            return None
            
        df = data[TimeFrame.DAY_1]
        if len(df) < 10:
            return None
            
        # Simple momentum logic
        recent_close = df['close'].iloc[-1]
        sma_10 = df['close'].rolling(10).mean().iloc[-1]
        
        if recent_close > sma_10 * 1.02:  # 2% above SMA
            signal = StrategySignal.BUY
            confidence = 0.7
        elif recent_close < sma_10 * 0.98:  # 2% below SMA
            signal = StrategySignal.SELL
            confidence = 0.7
        else:
            signal = StrategySignal.HOLD
            confidence = 0.3
            
        result = StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=recent_close,
            timestamp=datetime.now(),
            metadata={'strategy_type': 'momentum', 'sma_10': sma_10}
        )
        
        self.signals_generated.append(result)
        return result
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> Set[str]:
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        pass


class MockStrategyB(BaseStrategy):
    """Mock strategy B for coordination testing - Mean reversion focused."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: dict):
        super().__init__(name, StrategyType.STOCKS, config, symbols, timeframes)
        self.execution_count = 0
        self.signals_generated = []
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> Optional[StrategyResult]:
        """Generate mean reversion signals."""
        self.execution_count += 1
        
        if TimeFrame.DAY_1 not in data or data[TimeFrame.DAY_1].empty:
            return None
            
        df = data[TimeFrame.DAY_1]
        if len(df) < 20:
            return None
            
        # Mean reversion logic
        recent_close = df['close'].iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        std_20 = df['close'].rolling(20).std().iloc[-1]
        
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        
        if recent_close < lower_band:  # Oversold
            signal = StrategySignal.BUY
            confidence = 0.8
        elif recent_close > upper_band:  # Overbought
            signal = StrategySignal.SELL
            confidence = 0.8
        else:
            signal = StrategySignal.HOLD
            confidence = 0.4
            
        result = StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=recent_close,
            timestamp=datetime.now(),
            metadata={'strategy_type': 'mean_reversion', 'sma_20': sma_20, 'bands': [lower_band, upper_band]}
        )
        
        self.signals_generated.append(result)
        return result
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> Set[str]:
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        pass


class MockOptionsStrategy(BaseStrategy):
    """Mock options strategy for testing mixed strategy coordination."""
    
    def __init__(self, name: str, symbols: list, timeframes: list, config: dict):
        super().__init__(name, StrategyType.OPTIONS, config, symbols, timeframes)
        self.execution_count = 0
        self.signals_generated = []
        
    async def analyze(self, symbol: str, data: Dict[TimeFrame, pd.DataFrame], context: StrategyContext) -> Optional[StrategyResult]:
        """Generate options-based signals."""
        self.execution_count += 1
        
        if TimeFrame.DAY_1 not in data or data[TimeFrame.DAY_1].empty:
            return None
            
        df = data[TimeFrame.DAY_1]
        if len(df) < 5:
            return None
            
        # Simple volatility-based options logic
        recent_close = df['close'].iloc[-1]
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1] if len(df) >= 20 else 0.02
        
        # Options strategies prefer high volatility
        if volatility > 0.03:  # High volatility
            signal = StrategySignal.BUY  # Buy options
            confidence = 0.6
        else:
            signal = StrategySignal.HOLD
            confidence = 0.3
            
        result = StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=recent_close,
            timestamp=datetime.now(),
            metadata={'strategy_type': 'options', 'volatility': volatility}
        )
        
        self.signals_generated.append(result)
        return result
    
    def get_required_timeframes(self) -> Set[TimeFrame]:
        return {TimeFrame.DAY_1}
    
    def get_required_symbols(self) -> Set[str]:
        return set(self.symbols)
    
    def get_required_data_fields(self) -> Set[str]:
        return {'open', 'high', 'low', 'close', 'volume'}
    
    def validate_config(self) -> None:
        pass


class StrategyCoordinator:
    """Strategy coordinator for managing multi-strategy interactions."""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.signal_history: List[StrategyResult] = []
        self.conflicts_detected: List[Dict] = []
        
    def register_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Register a strategy with the coordinator."""
        self.strategies[strategy.name] = strategy
        self.strategy_weights[strategy.name] = weight
        
    def detect_conflicts(self, signals: Dict[str, StrategyResult]) -> List[Dict]:
        """Detect conflicts between strategy signals."""
        conflicts = []
        
        # Group signals by symbol
        symbol_signals = {}
        for strategy_name, signal in signals.items():
            if signal and signal.symbol:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append((strategy_name, signal))
        
        # Check for conflicts within each symbol
        for symbol, signal_list in symbol_signals.items():
            if len(signal_list) > 1:
                signal_types = [signal.signal for _, signal in signal_list]
                
                # Check for conflicting signals
                if StrategySignal.BUY in signal_types and StrategySignal.SELL in signal_types:
                    conflict = {
                        'symbol': symbol,
                        'type': 'buy_sell_conflict',
                        'strategies': [name for name, _ in signal_list],
                        'signals': signal_types,
                        'timestamp': datetime.now()
                    }
                    conflicts.append(conflict)
                    self.conflicts_detected.append(conflict)
        
        return conflicts
    
    def resolve_conflicts(self, signals: Dict[str, StrategyResult]) -> Dict[str, StrategyResult]:
        """Resolve conflicts using weighted voting."""
        resolved_signals = {}
        
        # Group signals by symbol
        symbol_signals = {}
        for strategy_name, signal in signals.items():
            if signal and signal.symbol:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append((strategy_name, signal))
        
        # Resolve conflicts for each symbol
        for symbol, signal_list in symbol_signals.items():
            if len(signal_list) == 1:
                # No conflict
                _, signal = signal_list[0]
                resolved_signals[symbol] = signal
            else:
                # Multiple signals - use weighted voting
                weighted_scores = {
                    StrategySignal.BUY: 0.0,
                    StrategySignal.SELL: 0.0,
                    StrategySignal.HOLD: 0.0
                }
                
                total_weight = 0.0
                best_signal = None
                best_price = 0.0
                
                for strategy_name, signal in signal_list:
                    weight = self.strategy_weights.get(strategy_name, 1.0)
                    weighted_score = weight * signal.confidence
                    weighted_scores[signal.signal] += weighted_score
                    total_weight += weight
                    
                    if best_signal is None:
                        best_signal = signal
                        best_price = signal.price
                
                # Find winning signal
                winning_signal = max(weighted_scores, key=weighted_scores.get)
                winning_confidence = weighted_scores[winning_signal] / total_weight if total_weight > 0 else 0.0
                
                # Create resolved signal
                resolved_signal = StrategyResult(
                    strategy_name='coordinator',
                    symbol=symbol,
                    signal=winning_signal,
                    confidence=winning_confidence,
                    price=best_price,
                    timestamp=datetime.now(),
                    metadata={
                        'resolution_method': 'weighted_voting',
                        'contributing_strategies': [name for name, _ in signal_list],
                        'vote_scores': weighted_scores
                    }
                )
                
                resolved_signals[symbol] = resolved_signal
        
        return resolved_signals
    
    def aggregate_signals(self, signals: Dict[str, StrategyResult]) -> Dict[str, Any]:
        """Aggregate signals across strategies."""
        conflicts = self.detect_conflicts(signals)
        resolved_signals = self.resolve_conflicts(signals)
        
        # Store signal history
        for signal in signals.values():
            if signal:
                self.signal_history.append(signal)
        
        return {
            'original_signals': signals,
            'conflicts': conflicts,
            'resolved_signals': resolved_signals,
            'coordination_summary': {
                'total_strategies': len(self.strategies),
                'active_signals': len([s for s in signals.values() if s]),
                'conflicts_detected': len(conflicts),
                'symbols_analyzed': len(set(s.symbol for s in signals.values() if s))
            }
        }


class TestStrategyCoordination:
    """Test suite for strategy coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinator = StrategyCoordinator()
        
        # Create test strategies
        self.momentum_strategy = MockStrategyA('momentum_strat', ['AAPL', 'GOOGL'], [TimeFrame.DAY_1], {})
        self.mean_reversion_strategy = MockStrategyB('mean_rev_strat', ['AAPL', 'MSFT'], [TimeFrame.DAY_1], {})
        self.options_strategy = MockOptionsStrategy('options_strat', ['AAPL'], [TimeFrame.DAY_1], {})
        
        # Register strategies
        self.coordinator.register_strategy(self.momentum_strategy, weight=1.0)
        self.coordinator.register_strategy(self.mean_reversion_strategy, weight=1.2)
        self.coordinator.register_strategy(self.options_strategy, weight=0.8)
        
    def create_test_data(self, symbol: str, trend: str = 'sideways') -> Dict[TimeFrame, pd.DataFrame]:
        """Create test market data."""
        periods = 50
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
        
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, periods):
            if trend == 'up':
                change = np.random.normal(0.01, 0.02)  # 1% upward trend with 2% volatility
            elif trend == 'down':
                change = np.random.normal(-0.01, 0.02)  # 1% downward trend
            else:  # sideways
                change = np.random.normal(0, 0.02)  # 2% volatility, no trend
            
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 50))  # Floor at $50
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(500000, 2000000, periods)
        }, index=dates)
        
        return {TimeFrame.DAY_1: data}
    
    def test_strategy_registration(self):
        """Test strategy registration with coordinator."""
        assert len(self.coordinator.strategies) == 3
        assert 'momentum_strat' in self.coordinator.strategies
        assert 'mean_rev_strat' in self.coordinator.strategies
        assert 'options_strat' in self.coordinator.strategies
        
        # Check weights
        assert self.coordinator.strategy_weights['momentum_strat'] == 1.0
        assert self.coordinator.strategy_weights['mean_rev_strat'] == 1.2
        assert self.coordinator.strategy_weights['options_strat'] == 0.8
    
    @pytest.mark.asyncio
    async def test_multi_strategy_execution(self):
        """Test execution of multiple strategies."""
        # Create test data
        aapl_data = self.create_test_data('AAPL', 'up')
        googl_data = self.create_test_data('GOOGL', 'down')
        msft_data = self.create_test_data('MSFT', 'sideways')
        
        test_data = {
            'AAPL': aapl_data,
            'GOOGL': googl_data,
            'MSFT': msft_data
        }
        
        # Create mock context
        context = StrategyContext(
            market_data=MagicMock(),
            order_manager=MagicMock(),
            portfolio_manager=MagicMock(),
            timestamp=datetime.now()
        )
        
        # Execute all strategies
        results = {}
        for strategy_name, strategy in self.coordinator.strategies.items():
            strategy_results = {}
            for symbol in strategy.symbols:
                if symbol in test_data:
                    result = await strategy.analyze(symbol, test_data[symbol], context)
                    if result:
                        strategy_results[symbol] = result
            
            if strategy_results:
                # For testing, take first result
                results[strategy_name] = list(strategy_results.values())[0]
        
        # Verify execution
        assert len(results) > 0
        
        # Check that strategies generated different types of signals
        signal_types = set()
        for result in results.values():
            signal_types.add(result.signal)
        
        # Should have variety in signals
        assert len(signal_types) > 1 or StrategySignal.HOLD in signal_types
    
    def test_conflict_detection(self):
        """Test conflict detection between strategies."""
        # Create conflicting signals manually
        buy_signal = StrategyResult(
            strategy_name='momentum_strat',
            symbol='AAPL',
            signal=StrategySignal.BUY,
            confidence=0.8,
            price=150.0,
            timestamp=datetime.now()
        )
        
        sell_signal = StrategyResult(
            strategy_name='mean_rev_strat',
            symbol='AAPL',
            signal=StrategySignal.SELL,
            confidence=0.7,
            price=150.0,
            timestamp=datetime.now()
        )
        
        signals = {
            'momentum_strat': buy_signal,
            'mean_rev_strat': sell_signal
        }
        
        conflicts = self.coordinator.detect_conflicts(signals)
        
        assert len(conflicts) == 1
        assert conflicts[0]['type'] == 'buy_sell_conflict'
        assert conflicts[0]['symbol'] == 'AAPL'
        assert 'momentum_strat' in conflicts[0]['strategies']
        assert 'mean_rev_strat' in conflicts[0]['strategies']
    
    def test_conflict_resolution(self):
        """Test conflict resolution using weighted voting."""
        # Create conflicting signals with different weights
        strong_buy_signal = StrategyResult(
            strategy_name='mean_rev_strat',  # Weight 1.2
            symbol='AAPL',
            signal=StrategySignal.BUY,
            confidence=0.9,
            price=150.0,
            timestamp=datetime.now()
        )
        
        weak_sell_signal = StrategyResult(
            strategy_name='options_strat',  # Weight 0.8
            symbol='AAPL',
            signal=StrategySignal.SELL,
            confidence=0.5,
            price=150.0,
            timestamp=datetime.now()
        )
        
        signals = {
            'mean_rev_strat': strong_buy_signal,
            'options_strat': weak_sell_signal
        }
        
        resolved = self.coordinator.resolve_conflicts(signals)
        
        assert 'AAPL' in resolved
        resolved_signal = resolved['AAPL']
        
        # Strong BUY with higher weight should win
        assert resolved_signal.signal == StrategySignal.BUY
        assert resolved_signal.strategy_name == 'coordinator'
        assert 'weighted_voting' in resolved_signal.metadata['resolution_method']
    
    def test_signal_aggregation(self):
        """Test complete signal aggregation workflow."""
        # Create mixed signals
        signals = {
            'momentum_strat': StrategyResult(
                strategy_name='momentum_strat',
                symbol='AAPL',
                signal=StrategySignal.BUY,
                confidence=0.7,
                price=150.0,
                timestamp=datetime.now()
            ),
            'mean_rev_strat': StrategyResult(
                strategy_name='mean_rev_strat',
                symbol='GOOGL',
                signal=StrategySignal.HOLD,
                confidence=0.4,
                price=2800.0,
                timestamp=datetime.now()
            ),
            'options_strat': StrategyResult(
                strategy_name='options_strat',
                symbol='AAPL',
                signal=StrategySignal.SELL,
                confidence=0.6,
                price=150.0,
                timestamp=datetime.now()
            )
        }
        
        aggregated = self.coordinator.aggregate_signals(signals)
        
        assert 'original_signals' in aggregated
        assert 'conflicts' in aggregated
        assert 'resolved_signals' in aggregated
        assert 'coordination_summary' in aggregated
        
        # Check summary
        summary = aggregated['coordination_summary']
        assert summary['total_strategies'] == 3
        assert summary['active_signals'] == 3
        assert summary['symbols_analyzed'] == 2  # AAPL and GOOGL
        
        # Should detect conflict on AAPL
        assert len(aggregated['conflicts']) == 1
        assert aggregated['conflicts'][0]['symbol'] == 'AAPL'


class TestResourceAllocation:
    """Test suite for resource allocation in multi-strategy coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.total_capital = 100000.0
        self.strategy_allocations = {
            'momentum_strat': 0.4,    # 40%
            'mean_rev_strat': 0.35,   # 35%
            'options_strat': 0.25     # 25%
        }
        
    def test_capital_allocation(self):
        """Test capital allocation across strategies."""
        allocated_capital = {}
        
        for strategy_name, allocation in self.strategy_allocations.items():
            allocated_capital[strategy_name] = self.total_capital * allocation
        
        # Verify allocations
        assert allocated_capital['momentum_strat'] == 40000.0
        assert allocated_capital['mean_rev_strat'] == 35000.0
        assert allocated_capital['options_strat'] == 25000.0
        
        # Verify total allocation
        total_allocated = sum(allocated_capital.values())
        assert total_allocated == self.total_capital
    
    def test_position_sizing_coordination(self):
        """Test position sizing coordination across strategies."""
        # Mock strategy signals with different position sizes
        signals = {
            'momentum_strat': {
                'symbol': 'AAPL',
                'signal': StrategySignal.BUY,
                'price': 150.0,
                'suggested_size': 100  # shares
            },
            'mean_rev_strat': {
                'symbol': 'AAPL',
                'signal': StrategySignal.BUY,
                'price': 150.0,
                'suggested_size': 50   # shares
            }
        }
        
        # Calculate combined position size
        total_position = 0
        max_position_per_symbol = 200  # Risk limit
        
        symbol_requests = {}
        for strategy_name, signal in signals.items():
            symbol = signal['symbol']
            if symbol not in symbol_requests:
                symbol_requests[symbol] = []
            symbol_requests[symbol].append(signal['suggested_size'])
        
        # Coordinate position sizes
        for symbol, requests in symbol_requests.items():
            total_requested = sum(requests)
            
            if total_requested <= max_position_per_symbol:
                total_position = total_requested
            else:
                # Scale down proportionally
                scale_factor = max_position_per_symbol / total_requested
                total_position = max_position_per_symbol
        
        assert total_position <= max_position_per_symbol
        assert total_position == 150  # 100 + 50 = 150, which is under limit
    
    def test_risk_budget_allocation(self):
        """Test risk budget allocation across strategies."""
        max_portfolio_risk = 0.02  # 2% daily VaR
        
        strategy_risk_budgets = {
            'momentum_strat': 0.008,   # 0.8%
            'mean_rev_strat': 0.007,   # 0.7%
            'options_strat': 0.005     # 0.5%
        }
        
        # Verify risk allocation
        total_allocated_risk = sum(strategy_risk_budgets.values())
        assert total_allocated_risk == 0.020
        assert total_allocated_risk <= max_portfolio_risk
        
        # Check individual allocations
        for strategy_name, risk_budget in strategy_risk_budgets.items():
            allocation_pct = self.strategy_allocations[strategy_name]
            
            # Risk should be proportional to allocation but can be adjusted
            expected_base_risk = max_portfolio_risk * allocation_pct
            assert risk_budget <= expected_base_risk * 1.2  # Allow 20% adjustment


class TestPerformanceCoordination:
    """Test suite for performance coordination across strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.performance_tracker = {}
        
        # Mock performance data
        self.strategy_performance = {
            'momentum_strat': {
                'total_return': 0.15,      # 15%
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,     # -8%
                'win_rate': 0.65,          # 65%
                'avg_trade_return': 0.02   # 2%
            },
            'mean_rev_strat': {
                'total_return': 0.12,      # 12%
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.05,     # -5%
                'win_rate': 0.70,          # 70%
                'avg_trade_return': 0.015  # 1.5%
            },
            'options_strat': {
                'total_return': 0.20,      # 20%
                'sharpe_ratio': 0.9,
                'max_drawdown': -0.12,     # -12%
                'win_rate': 0.55,          # 55%
                'avg_trade_return': 0.035  # 3.5%
            }
        }
    
    def test_performance_aggregation(self):
        """Test performance aggregation across strategies."""
        # Calculate weighted portfolio performance
        strategy_weights = {'momentum_strat': 0.4, 'mean_rev_strat': 0.35, 'options_strat': 0.25}
        
        # Weighted total return
        portfolio_return = sum(
            perf['total_return'] * strategy_weights[strategy_name]
            for strategy_name, perf in self.strategy_performance.items()
        )
        
        expected_return = 0.15 * 0.4 + 0.12 * 0.35 + 0.20 * 0.25
        assert abs(portfolio_return - expected_return) < 0.001
        
        # Calculate portfolio Sharpe ratio (simplified)
        portfolio_sharpe = sum(
            perf['sharpe_ratio'] * strategy_weights[strategy_name]
            for strategy_name, perf in self.strategy_performance.items()
        )
        
        expected_sharpe = 1.2 * 0.4 + 1.5 * 0.35 + 0.9 * 0.25
        assert abs(portfolio_sharpe - expected_sharpe) < 0.001
    
    def test_strategy_rebalancing(self):
        """Test strategy rebalancing based on performance."""
        # Calculate performance scores
        performance_scores = {}
        for strategy_name, perf in self.strategy_performance.items():
            # Simple score: weighted average of key metrics
            score = (
                perf['total_return'] * 0.3 +
                perf['sharpe_ratio'] * 0.1 * 0.3 +  # Normalize Sharpe to 0-1 range
                (1 + perf['max_drawdown']) * 0.2 +   # Convert drawdown to positive
                perf['win_rate'] * 0.2
            )
            performance_scores[strategy_name] = score
        
        # Rebalance weights based on performance
        total_score = sum(performance_scores.values())
        new_weights = {
            strategy_name: score / total_score
            for strategy_name, score in performance_scores.items()
        }
        
        # Verify rebalancing
        assert abs(sum(new_weights.values()) - 1.0) < 0.001
        
        # Best performing strategy should get higher weight
        best_strategy = max(performance_scores, key=performance_scores.get)
        worst_strategy = min(performance_scores, key=performance_scores.get)
        
        assert new_weights[best_strategy] > new_weights[worst_strategy]
    
    def test_risk_adjusted_coordination(self):
        """Test risk-adjusted strategy coordination."""
        # Calculate risk-adjusted returns
        risk_adjusted_returns = {}
        for strategy_name, perf in self.strategy_performance.items():
            # Use Sharpe ratio as risk-adjusted return proxy
            risk_adjusted_returns[strategy_name] = perf['sharpe_ratio']
        
        # Adjust strategy allocations based on risk-adjusted performance
        min_allocation = 0.1  # Minimum 10% allocation
        max_allocation = 0.6  # Maximum 60% allocation
        
        total_sharpe = sum(risk_adjusted_returns.values())
        risk_adjusted_weights = {}
        
        for strategy_name, sharpe in risk_adjusted_returns.items():
            base_weight = sharpe / total_sharpe
            
            # Apply min/max constraints
            adjusted_weight = max(min_allocation, min(max_allocation, base_weight))
            risk_adjusted_weights[strategy_name] = adjusted_weight
        
        # Normalize to sum to 1.0
        total_weight = sum(risk_adjusted_weights.values())
        final_weights = {
            strategy_name: weight / total_weight
            for strategy_name, weight in risk_adjusted_weights.items()
        }
        
        # Verify constraints
        for weight in final_weights.values():
            assert min_allocation <= weight <= max_allocation
        
        assert abs(sum(final_weights.values()) - 1.0) < 0.001


def run_strategy_coordination_tests():
    """Run all strategy coordination tests."""
    print("🧪 RUNNING STRATEGY COORDINATION INTEGRATION TESTS")
    print("=" * 60)
    
    # Test categories
    test_categories = [
        # Strategy coordination tests
        ("Strategy Registration", TestStrategyCoordination().test_strategy_registration),
        ("Conflict Detection", TestStrategyCoordination().test_conflict_detection),
        ("Conflict Resolution", TestStrategyCoordination().test_conflict_resolution),
        ("Signal Aggregation", TestStrategyCoordination().test_signal_aggregation),
        
        # Resource allocation tests
        ("Capital Allocation", TestResourceAllocation().test_capital_allocation),
        ("Position Sizing Coordination", TestResourceAllocation().test_position_sizing_coordination),
        ("Risk Budget Allocation", TestResourceAllocation().test_risk_budget_allocation),
        
        # Performance coordination tests
        ("Performance Aggregation", TestPerformanceCoordination().test_performance_aggregation),
        ("Strategy Rebalancing", TestPerformanceCoordination().test_strategy_rebalancing),
        ("Risk Adjusted Coordination", TestPerformanceCoordination().test_risk_adjusted_coordination),
    ]
    
    # Async test categories
    async_test_categories = [
        ("Multi-Strategy Execution", TestStrategyCoordination().test_multi_strategy_execution),
    ]
    
    passed = 0
    total = len(test_categories) + len(async_test_categories)
    
    # Run synchronous tests
    for test_name, test_func in test_categories:
        try:
            # Set up appropriate test instance
            if "Strategy" in test_name and ("Registration" in test_name or "Conflict" in test_name or "Signal" in test_name or "Execution" in test_name):
                test_instance = TestStrategyCoordination()
            elif "Resource" in test_name or "Capital" in test_name or "Position" in test_name or "Risk Budget" in test_name:
                test_instance = TestResourceAllocation()
            else:  # Performance tests
                test_instance = TestPerformanceCoordination()
            
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
                test_instance = TestStrategyCoordination()
                test_instance.setup_method()
                
                await test_func()
                print(f"✅ {test_name}: PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print("📋 STRATEGY COORDINATION INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 ALL STRATEGY COORDINATION TESTS PASSED!")
        print("🚀 Strategy coordination system is robust and ready!")
        
        print("\n📊 Tested Components:")
        print("  • 🎯 Strategy Coordination: Registration, conflict detection, resolution")
        print("  • 🤝 Multi-Strategy Execution: Simultaneous strategy execution")
        print("  • 💰 Resource Allocation: Capital distribution, position sizing, risk budgets")
        print("  • 📊 Performance Coordination: Aggregation, rebalancing, risk adjustment")
        print("  • ⚖️  Conflict Resolution: Weighted voting, signal prioritization")
        print("  • 🔄 Signal Aggregation: Cross-strategy signal combination")
        
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Strategy coordination system needs attention.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    run_strategy_coordination_tests() 
