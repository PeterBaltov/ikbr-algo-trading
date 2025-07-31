#!/usr/bin/env python3
"""
Phase 1 Integration Demo

This demo shows how the new strategy framework integrates with
existing ThetaGang components and how you could extend the system.
"""

import asyncio
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

def demo_strategy_registration():
    """Demo: Register and manage strategies"""
    print("🏭 STRATEGY REGISTRATION DEMO")
    print("=" * 50)
    
    # Import the new framework
    from moneytrailz.strategies import get_registry, StrategyType, TimeFrame
    from moneytrailz.strategies.implementations.example_strategy import ExampleStrategy
    
    # Get the global registry
    registry = get_registry()
    
    # Register multiple strategies
    print("📝 Registering strategies...")
    
    # Register example strategy with different configs
    registry.register_strategy(ExampleStrategy, "conservative_momentum")
    registry.register_strategy(ExampleStrategy, "aggressive_momentum")
    
    # List all strategies
    strategies = registry.list_strategies()
    print(f"✅ Registered strategies: {strategies}")
    
    # Get detailed info
    for strategy_name in strategies:
        info = registry.get_strategy_info(strategy_name)
        if info:
            print(f"📊 {strategy_name}: {info['class_name']} ({info['strategy_type']})")
        else:
            print(f"📊 {strategy_name}: Info not available")
    
    return registry

def demo_strategy_configuration():
    """Demo: Configure strategies for different use cases"""
    print("\n⚙️ STRATEGY CONFIGURATION DEMO")
    print("=" * 50)
    
    # Different strategy configurations
    configs = {
        "conservative_momentum": {
            "type": "stocks",
            "enabled": True,
            "timeframes": ["1d"],
            "threshold": 0.05,  # 5% - more conservative
            "min_volume": 200000,  # Higher volume requirement
            "symbols": ["SPY", "QQQ"],
            "description": "Conservative momentum strategy for ETFs"
        },
        "aggressive_momentum": {
            "type": "stocks", 
            "enabled": True,
            "timeframes": ["1d"],
            "threshold": 0.02,  # 2% - more aggressive
            "min_volume": 50000,  # Lower volume requirement
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "description": "Aggressive momentum strategy for individual stocks"
        }
    }
    
    print("📋 Strategy configurations:")
    for name, config in configs.items():
        print(f"  {name}:")
        print(f"    Threshold: {config['threshold']*100}%")
        print(f"    Min Volume: {config['min_volume']:,}")
        print(f"    Symbols: {config['symbols']}")
    
    return configs

async def demo_strategy_execution(registry, configs):
    """Demo: Execute strategies with sample data"""
    print("\n🚀 STRATEGY EXECUTION DEMO")
    print("=" * 50)
    
    from moneytrailz.strategies import StrategyContext, TimeFrame
    
    # Create sample market data for different symbols
    symbols_data = {}
    
    for symbol in ["SPY", "QQQ", "AAPL", "GOOGL", "MSFT"]:
        # Generate sample price data
        dates = pd.date_range('2024-01-15', periods=5, freq='D')
        base_price = {"SPY": 480, "QQQ": 390, "AAPL": 180, "GOOGL": 140, "MSFT": 400}[symbol]
        
        # Simulate different market conditions
        if symbol in ["SPY", "QQQ"]:  # ETFs - stable
            price_changes = [0.002, 0.001, -0.001, 0.003, 0.001]  # Small moves
        else:  # Individual stocks - more volatile
            price_changes = [0.03, -0.02, 0.05, -0.01, 0.04]  # Larger moves
        
        prices = []
        current_price = base_price
        for change in price_changes:
            current_price *= (1 + change)
            prices.append(current_price)
        
        symbols_data[symbol] = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [150000 + i * 10000 for i in range(len(prices))]
        }, index=dates)
    
    # Create mock context
    class MockMarketData:
        def get_current_price(self, symbol: str) -> float:
            return symbols_data[symbol]['close'].iloc[-1]
        def get_historical_data(self, symbol, timeframe, start_date, end_date, fields=None):
            return symbols_data.get(symbol, pd.DataFrame())
        def get_option_chain(self, symbol):
            return []
        def is_market_open(self) -> bool:
            return True
    
    class MockOrderManager:
        def __init__(self):
            self.orders = []
        def place_order(self, contract, order):
            order_id = f"ORDER_{len(self.orders)}"
            self.orders.append({"id": order_id, "contract": contract, "order": order})
            return order_id
        def cancel_order(self, order_id: str) -> bool:
            return True
        def get_order_status(self, order_id: str) -> str:
            return "Filled"
        def get_open_orders(self):
            return self.orders
    
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
    
    order_manager = MockOrderManager()
    context = StrategyContext(
        market_data=MockMarketData(),
        order_manager=order_manager,
        position_manager=MockPositionManager(),
        risk_manager=MockRiskManager(),
        account_summary={},
        portfolio_positions={}
    )
    
    # Execute strategies for all configured symbols
    results = {}
    
    for strategy_name, config in configs.items():
        print(f"\n📊 Executing {strategy_name}...")
        
        # Create strategy instance
        strategy = registry.create_strategy_instance(
            strategy_name, config, config['symbols']
        )
        
        if not strategy:
            print(f"❌ Failed to create strategy: {strategy_name}")
            continue
            
        strategy_results = []
        
        for symbol in config['symbols']:
            if symbol in symbols_data:
                data_dict = {TimeFrame.DAY_1: symbols_data[symbol]}
                
                try:
                    result = await strategy.execute(symbol, data_dict, context)
                    if result:
                        strategy_results.append(result)
                        print(f"  {symbol}: {result.signal.value} (confidence: {result.confidence:.2f})")
                    else:
                        print(f"  {symbol}: No signal generated")
                except Exception as e:
                    print(f"  {symbol}: Error - {e}")
        
        results[strategy_name] = strategy_results
    
    return results, order_manager

def demo_future_integration():
    """Demo: Show how this could integrate with existing ThetaGang"""
    print("\n🔗 FUTURE INTEGRATION DEMO")
    print("=" * 50)
    
    print("📝 How Phase 1 framework would integrate with existing ThetaGang:")
    print()
    
    integration_code = '''
# In portfolio_manager.py - New method to execute strategies

async def execute_strategies(self, account_summary, portfolio_positions):
    """Execute registered strategies alongside existing wheel strategy"""
    
    from moneytrailz.strategies import get_registry, StrategyContext
    
    # Get strategy registry
    registry = get_registry()
    
    # Create strategy context from existing components
    context = StrategyContext(
        market_data=self.ibkr,
        order_manager=self.orders,
        position_manager=self,  # PortfolioManager can implement interfaces
        risk_manager=self,      # Risk management logic
        account_summary=account_summary,
        portfolio_positions=portfolio_positions
    )
    
    # Execute all enabled strategies
    for strategy_name in registry.list_strategies():
        strategy_config = self.config.strategies.get(strategy_name, {})
        
        if strategy_config.get('enabled', False):
            strategy = registry.create_strategy_instance(
                strategy_name, strategy_config, strategy_config.get('symbols', [])
            )
            
            for symbol in strategy.symbols:
                # Get historical data for strategy
                data = await self.get_strategy_data(symbol, strategy.get_required_timeframes())
                
                # Execute strategy
                result = await strategy.execute(symbol, data, context)
                
                if result and result.signal != StrategySignal.HOLD:
                    # Process strategy signal (create orders, etc.)
                    await self.process_strategy_signal(result)

# In moneytrailz.toml - New strategies section
[strategies]
  [strategies.conservative_momentum]
  enabled = true
  type = "stocks"
  timeframes = ["1d"]
  symbols = ["SPY", "QQQ"]
  threshold = 0.05
  
  [strategies.options_momentum]
  enabled = false
  type = "options"
  timeframes = ["1h", "1d"]
  symbols = ["SPY"]
  delta_target = 0.3
    '''
    
    print(integration_code)
    
    print("✅ Benefits of Phase 1 framework:")
    print("  • 🔧 Modular strategy development")
    print("  • 📊 Type-safe interfaces")
    print("  • 🔄 Dynamic strategy loading")
    print("  • ⚙️ Flexible configuration")
    print("  • 🧪 Easy testing and validation")
    print("  • 🚀 Extensible for Phase 2+")

async def main():
    """Run all integration demos"""
    print("🎭 PHASE 1 INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Demo 1: Strategy registration
    registry = demo_strategy_registration()
    
    # Demo 2: Strategy configuration 
    configs = demo_strategy_configuration()
    
    # Demo 3: Strategy execution
    results, order_manager = await demo_strategy_execution(registry, configs)
    
    # Demo 4: Future integration
    demo_future_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION DEMO SUMMARY")
    print("=" * 60)
    
    total_signals = sum(len(strategy_results) for strategy_results in results.values())
    print(f"🎯 Total signals generated: {total_signals}")
    
    for strategy_name, strategy_results in results.items():
        signals = [r.signal.value for r in strategy_results]
        print(f"  {strategy_name}: {len(signals)} signals - {signals}")
    
    print(f"📝 Total orders created: {len(order_manager.orders)}")
    
    print("\n🎉 Phase 1 framework successfully demonstrated!")
    print("✅ Ready for Phase 2: Technical Analysis Engine")

if __name__ == "__main__":
    asyncio.run(main()) 
