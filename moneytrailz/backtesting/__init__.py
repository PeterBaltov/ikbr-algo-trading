"""
Backtesting Framework Package

A comprehensive backtesting system for algorithmic trading strategies, providing
historical simulation, performance analytics, and risk assessment capabilities.
Inspired by the backtrader library architecture with deep integration into our
multi-strategy platform.

Key Features:
- Event-driven backtesting engine with realistic execution simulation
- Comprehensive performance metrics and risk analytics
- Multi-timeframe strategy testing with our Phase 3 architecture
- Monte Carlo analysis and walk-forward optimization
- Seamless integration with Phase 1 strategy framework and Phase 2 analysis
- Advanced reporting with interactive visualizations
- Live trading adapter for production deployment

Components:
- BacktestEngine: Core event-driven simulation engine
- TradeSimulator: Realistic trade execution with slippage and market impact
- DataManager: Historical data ingestion with validation and bias handling
- PerformanceCalculator: Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)
- RiskAnalyzer: VaR, CVaR, drawdown analysis, and attribution
- PortfolioManager: Capital allocation and position sizing
- ReportGenerator: Interactive dashboards and export capabilities
- BenchmarkComparator: Strategy comparison utilities
- LiveAdapter: Bridge to production trading

Integration:
- Leverages Phase 1 strategy framework and registry
- Uses Phase 2 technical analysis engine
- Coordinates with Phase 3 multi-timeframe architecture
- Provides foundation for strategy optimization and deployment
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult, BacktestState
from .data import DataManager, DataConfig, DataSource, DataValidator, create_data_config
from .simulator import TradeSimulator, ExecutionConfig, SlippageModel, MarketImpactModel
from .strategy_api import BacktestStrategy, StrategyLifecycle, BacktestContext
from .metrics import PerformanceCalculator, PerformanceMetrics, RiskMetrics
from .portfolio import PortfolioManager, PositionSizer, CapitalAllocator, create_portfolio_config
from .optimizer import StrategyOptimizer, OptimizationConfig, WalkForwardAnalysis
from .reports import ReportGenerator, ReportConfig
from .benchmark import BenchmarkComparator, BenchmarkComparison
from .live_adapter import LiveTradingAdapter, AdapterConfig, create_live_config

__all__ = [
    # Core engine
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
    "BacktestState",
    
    # Data management
    "DataManager",
    "DataConfig",
    "DataSource",
    "DataValidator",
    "create_data_config",
    
    # Trade execution
    "TradeSimulator",
    "ExecutionConfig",
    "SlippageModel", 
    "MarketImpactModel",
    
    # Strategy API
    "BacktestStrategy",
    "StrategyLifecycle",
    "BacktestContext",
    
    # Performance & Risk
    "PerformanceCalculator",
    "PerformanceMetrics",
    "RiskMetrics",
    
    # Portfolio management
    "PortfolioManager",
    "PositionSizer",
    "CapitalAllocator",
    "create_portfolio_config",
    
    # Optimization
    "StrategyOptimizer",
    "OptimizationConfig", 
    "WalkForwardAnalysis",
    
    # Reporting & Analysis
    "ReportGenerator",
    "ReportConfig",
    "BenchmarkComparator",
    "BenchmarkComparison",
    
    # Live trading
    "LiveTradingAdapter",
    "AdapterConfig",
    "create_live_config"
] 
