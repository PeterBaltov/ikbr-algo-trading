import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import random
from uuid import uuid4

from ..models.dashboard import (
    PortfolioSnapshot, StrategySnapshot, StrategyUpdate,
    StrategyType, StrategyStatus, StrategyPnL, StrategyMetrics,
    Position, Trade, PositionType, PositionSide, TradeAction, OrderType, OrderStatus
)

class DashboardIntegration:
    """
    Integration layer between ThetaGang system and Dashboard API.
    
    For Phase 1, this provides mock data for development.
    In Phase 3, this will integrate with the actual ThetaGang system.
    """
    
    def __init__(self):
        self.is_connected = False
        self.last_update = datetime.now(timezone.utc)
        self.mock_data_initialized = False
        self._init_mock_data()
    
    def _init_mock_data(self):
        """Initialize mock data for development"""
        self.mock_portfolio = PortfolioSnapshot(
            total_value=125450.23,
            day_pnl=1250.75,
            total_pnl=12340.50,
            cash_balance=25000.00,
            margin_used=15000.00,
            buying_power=85450.23,
            day_pnl_percent=1.01,
            total_pnl_percent=10.87,
            win_rate=73.2,
            active_strategies=5,
            last_updated=datetime.now(timezone.utc)
        )
        
        self.mock_strategies = [
            StrategySnapshot(
                name="enhanced_wheel",
                type=StrategyType.OPTIONS,
                status=StrategyStatus.ACTIVE,
                allocation=30.0,
                pnl=StrategyPnL(daily=450.25, total=2340.50, percentage=15.2),
                metrics=StrategyMetrics(
                    win_rate=78.5,
                    sharpe_ratio=1.8,
                    max_drawdown=-5.2,
                    total_trades=45,
                    avg_win=125.50,
                    avg_loss=-85.25
                ),
                positions=[],
                recent_trades=[],
                last_updated=datetime.now(timezone.utc)
            ),
            StrategySnapshot(
                name="momentum_scalper",
                type=StrategyType.STOCKS,
                status=StrategyStatus.ACTIVE,
                allocation=25.0,
                pnl=StrategyPnL(daily=300.15, total=890.25, percentage=8.7),
                metrics=StrategyMetrics(
                    win_rate=65.2,
                    sharpe_ratio=1.4,
                    max_drawdown=-8.1,
                    total_trades=128,
                    avg_win=85.75,
                    avg_loss=-65.50
                ),
                positions=[],
                recent_trades=[],
                last_updated=datetime.now(timezone.utc)
            ),
            StrategySnapshot(
                name="mean_reversion",
                type=StrategyType.STOCKS,
                status=StrategyStatus.PAUSED,
                allocation=20.0,
                pnl=StrategyPnL(daily=0.0, total=450.75, percentage=4.2),
                metrics=StrategyMetrics(
                    win_rate=58.8,
                    sharpe_ratio=0.9,
                    max_drawdown=-12.5,
                    total_trades=32,
                    avg_win=95.25,
                    avg_loss=-75.80
                ),
                positions=[],
                recent_trades=[],
                last_updated=datetime.now(timezone.utc)
            )
        ]
        
        self.mock_data_initialized = True
    
    async def connect(self) -> bool:
        """
        Connect to ThetaGang system.
        For Phase 1, this is a mock connection.
        """
        # Simulate connection delay
        await asyncio.sleep(0.1)
        self.is_connected = True
        print("📡 Connected to ThetaGang system (mock)")
        return True
    
    async def disconnect(self):
        """Disconnect from ThetaGang system"""
        self.is_connected = False
        print("📡 Disconnected from ThetaGang system")
    
    async def get_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Get current portfolio snapshot"""
        if not self.is_connected and not await self.connect():
            return None
        
        # Simulate some volatility in the portfolio values
        base_value = 125450.23
        volatility = random.uniform(-0.02, 0.02)  # ±2% volatility
        
        # Update mock portfolio with slight variations
        self.mock_portfolio.total_value = base_value * (1 + volatility)
        self.mock_portfolio.day_pnl = base_value * volatility
        self.mock_portfolio.day_pnl_percent = volatility * 100
        self.mock_portfolio.last_updated = datetime.now(timezone.utc)
        
        return self.mock_portfolio
    
    async def get_strategy_updates(self) -> List[StrategyUpdate]:
        """Get recent strategy updates"""
        if not self.is_connected:
            return []
        
        updates = []
        for strategy in self.mock_strategies:
            if strategy.status == StrategyStatus.ACTIVE:
                # Simulate small PnL changes
                pnl_change = random.uniform(-50, 100)
                updates.append(StrategyUpdate(
                    name=strategy.name,
                    status=strategy.status,
                    pnl_change=pnl_change,
                    last_trade=None  # We'll add trade simulation later
                ))
        
        return updates
    
    async def get_strategies(self) -> List[StrategySnapshot]:
        """Get all strategy snapshots"""
        if not self.is_connected and not await self.connect():
            return []
        
        # Update strategies with slight variations
        for strategy in self.mock_strategies:
            if strategy.status == StrategyStatus.ACTIVE:
                # Simulate PnL changes
                volatility = random.uniform(-0.05, 0.05)
                strategy.pnl.daily *= (1 + volatility)
                strategy.last_updated = datetime.now(timezone.utc)
        
        return self.mock_strategies
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        if not self.is_connected:
            return []
        
        # Mock positions
        return [
            Position(
                id=str(uuid4()),
                symbol="AAPL",
                quantity=100,
                avg_price=150.25,
                current_price=152.75,
                market_value=15275.0,
                unrealized_pnl=250.0,
                unrealized_pnl_percent=1.66,
                strategy="enhanced_wheel",
                type=PositionType.STOCK,
                side=PositionSide.LONG,
                open_date=datetime.now(timezone.utc) - timedelta(days=5)
            ),
            Position(
                id=str(uuid4()),
                symbol="TSLA",
                quantity=50,
                avg_price=220.50,
                current_price=225.25,
                market_value=11262.50,
                unrealized_pnl=237.50,
                unrealized_pnl_percent=2.15,
                strategy="momentum_scalper",
                type=PositionType.STOCK,
                side=PositionSide.LONG,
                open_date=datetime.now(timezone.utc) - timedelta(days=2)
            )
        ]
    
    async def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get recent trades"""
        if not self.is_connected:
            return []
        
        # Mock recent trades
        trades = []
        for i in range(limit):
            trades.append(Trade(
                id=str(uuid4()),
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=random.randint(1, 1440)),
                symbol=random.choice(["AAPL", "TSLA", "MSFT", "GOOGL"]),
                action=random.choice([TradeAction.BUY, TradeAction.SELL]),
                quantity=random.randint(10, 100),
                price=random.uniform(100, 300),
                commission=random.uniform(1, 5),
                pnl=random.uniform(-50, 150),
                strategy=random.choice(["enhanced_wheel", "momentum_scalper", "mean_reversion"]),
                order_type=random.choice([OrderType.MARKET, OrderType.LIMIT]),
                status=OrderStatus.FILLED
            ))
        
        return sorted(trades, key=lambda x: x.timestamp, reverse=True)
    
    async def pause_strategy(self, strategy_name: str) -> bool:
        """Pause a strategy"""
        for strategy in self.mock_strategies:
            if strategy.name == strategy_name:
                strategy.status = StrategyStatus.PAUSED
                strategy.last_updated = datetime.now(timezone.utc)
                return True
        return False
    
    async def resume_strategy(self, strategy_name: str) -> bool:
        """Resume a strategy"""
        for strategy in self.mock_strategies:
            if strategy.name == strategy_name:
                strategy.status = StrategyStatus.ACTIVE
                strategy.last_updated = datetime.now(timezone.utc)
                return True
        return False
    
    async def stop_strategy(self, strategy_name: str) -> bool:
        """Stop a strategy"""
        for strategy in self.mock_strategies:
            if strategy.name == strategy_name:
                strategy.status = StrategyStatus.STOPPED
                strategy.last_updated = datetime.now(timezone.utc)
                return True
        return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            "connected": self.is_connected,
            "last_update": self.last_update.isoformat(),
            "system": "mock" if not self.is_connected else "thetagang",
            "data_source": "mock_data" if self.mock_data_initialized else "live_data"
        } 
