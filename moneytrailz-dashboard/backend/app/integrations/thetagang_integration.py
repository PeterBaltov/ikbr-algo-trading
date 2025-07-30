import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from decimal import Decimal

from ..models.dashboard import PortfolioSnapshot, StrategyUpdate, Position, Trade, PositionType, PositionSide
from moneytrailz.portfolio_manager import PortfolioManager
from moneytrailz.config import Config
from ib_async import IB
from asyncio import Future
from uuid import uuid4

logger = logging.getLogger(__name__)

class DashboardIntegration:
    """Integration layer between ThetaGang and the dashboard"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the integration with ThetaGang system"""
        self.config = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.is_connected = False
        self.config_path = config_path or "moneytrailz.toml"
        self.last_snapshot_time = None
        
        # Initialize connection
        asyncio.create_task(self._initialize_connection())
    
    async def _initialize_connection(self):
        """Initialize connection to ThetaGang system"""
        try:
            logger.info("🔗 Attempting ThetaGang connection...")
            
            # For now, we'll start in mock mode since we need IB connection setup
            # In a real deployment, this would check for IB connection and credentials
            logger.info("🎭 Real ThetaGang connection requires IB setup - using mock mode")
            await self._initialize_mock_mode()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to ThetaGang: {e}")
            self.is_connected = False
            # Fall back to mock mode
            await self._initialize_mock_mode()
    
    async def _test_connection(self):
        """Test the connection to ThetaGang system"""
        if self.portfolio_manager:
            # Test basic portfolio access
            try:
                account_summary, portfolio_positions = await self.portfolio_manager.summarize_account()
                net_liquidation = account_summary.get("NetLiquidation")
                if net_liquidation:
                    logger.info(f"📈 Portfolio value: ${float(net_liquidation.value):,.2f}")
            except Exception as e:
                logger.warning(f"⚠️  Portfolio access test failed: {e}")
                raise
    
    async def _initialize_mock_mode(self):
        """Initialize with mock data when real connection fails"""
        logger.info("🎭 Initializing mock mode for development")
        self.is_connected = False  # Mark as mock mode
    
    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio state snapshot"""
        try:
            if self.is_connected and self.portfolio_manager:
                return await self._get_real_portfolio_snapshot()
            else:
                return await self._get_mock_portfolio_snapshot()
        except Exception as e:
            logger.error(f"Error getting portfolio snapshot: {e}")
            return await self._get_mock_portfolio_snapshot()
    
    async def _get_real_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get real portfolio data from MoneyTrailz"""
        if not self.portfolio_manager:
            return await self._get_mock_portfolio_snapshot()
            
        try:
            # Use the actual PortfolioManager API
            account_summary, portfolio_positions = await self.portfolio_manager.summarize_account()
            
            # Extract key metrics from account summary
            net_liquidation = float(account_summary.get("NetLiquidation", 0).value) if account_summary.get("NetLiquidation") else 0.0
            total_cash = float(account_summary.get("TotalCashValue", 0).value) if account_summary.get("TotalCashValue") else 0.0
            
            # Convert portfolio positions to dashboard format
            dashboard_positions = []
            for symbol, positions in portfolio_positions.items():
                for pos in positions:
                    dashboard_positions.append(Position(
                        symbol=pos.contract.symbol,
                        quantity=int(pos.position),
                        avg_price=float(pos.averageCost) if pos.averageCost else 0.0,
                        market_value=float(pos.marketValue) if pos.marketValue else 0.0,
                        unrealized_pnl=float(pos.unrealizedPNL) if pos.unrealizedPNL else 0.0
                    ))
            
            # Get active strategies from the strategy framework
            strategies = []
            if hasattr(self.portfolio_manager, 'active_strategies'):
                for strategy_name, strategy in self.portfolio_manager.active_strategies.items():
                    strategies.append({
                        "name": strategy_name,
                        "status": "active",  # Simplified for now
                        "pnl": 0.0,  # Would need to calculate from strategy
                        "positions": 0,  # Would need to count strategy positions
                        "last_update": datetime.now(timezone.utc).isoformat()
                    })
            
            return PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_value=net_liquidation,
                cash_balance=total_cash,
                day_pnl=0.0,  # Would need to calculate daily P&L
                total_pnl=0.0,  # Would need to calculate total P&L
                positions=dashboard_positions,
                strategies=strategies,
                connection_status="connected"
            )
            
        except Exception as e:
            logger.warning(f"Failed to get real portfolio data: {e}")
            return await self._get_mock_portfolio_snapshot()
    
    async def _get_mock_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get mock portfolio data for development/demo"""
        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_value=125450.23,
            cash_balance=15680.45,
            day_pnl=1250.75,
            total_pnl=8945.30,
            positions=[
                Position(
                    symbol="SPY",
                    quantity=100,
                    avg_price=445.20,
                    market_value=44650.00,
                    unrealized_pnl=430.00,
                    position_type="long"
                ),
                Position(
                    symbol="QQQ",
                    quantity=-50,
                    avg_price=385.40,
                    market_value=-19120.00,
                    unrealized_pnl=-180.50,
                    position_type="short"
                )
            ],
            strategies=[
                {
                    "name": "Enhanced Wheel",
                    "status": "active",
                    "pnl": 2340.50,
                    "positions": 8,
                    "last_update": datetime.now(timezone.utc).isoformat()
                },
                {
                    "name": "Momentum Scalper", 
                    "status": "paused",
                    "pnl": -125.30,
                    "positions": 3,
                    "last_update": datetime.now(timezone.utc).isoformat()
                }
            ],
            connection_status="mock" if not self.is_connected else "connected"
        )
    
    async def get_strategy_updates(self) -> List[StrategyUpdate]:
        """Get recent strategy updates"""
        try:
            if self.is_connected and self.portfolio_manager:
                return await self._get_real_strategy_updates()
            else:
                return await self._get_mock_strategy_updates()
        except Exception as e:
            logger.error(f"Error getting strategy updates: {e}")
            return await self._get_mock_strategy_updates()
    
    async def _get_real_strategy_updates(self) -> List[StrategyUpdate]:
        """Get real strategy updates from MoneyTrailz"""
        updates = []
        
        if hasattr(self.portfolio_manager, 'active_strategies'):
            for strategy in self.portfolio_manager.active_strategies:
                if hasattr(strategy, 'get_recent_updates'):
                    recent_updates = strategy.get_recent_updates()
                    for update in recent_updates:
                        updates.append(StrategyUpdate(
                            strategy_name=strategy.name,
                            action=update.get('action', 'update'),
                            details=update.get('details', {}),
                            timestamp=update.get('timestamp', datetime.now(timezone.utc).isoformat()),
                            pnl_impact=float(update.get('pnl_impact', 0))
                        ))
        
        return updates
    
    async def _get_mock_strategy_updates(self) -> List[StrategyUpdate]:
        """Get mock strategy updates for development"""
        return [
            StrategyUpdate(
                strategy_name="Enhanced Wheel",
                action="position_opened",
                details={"symbol": "AAPL", "strike": 175, "expiry": "2024-08-16"},
                timestamp=datetime.now(timezone.utc).isoformat(),
                pnl_impact=150.50
            ),
            StrategyUpdate(
                strategy_name="Momentum Scalper",
                action="position_closed",
                details={"symbol": "MSFT", "pnl": 85.25},
                timestamp=datetime.now(timezone.utc).isoformat(),
                pnl_impact=85.25
            )
        ]
    
    async def stream_updates(self, websocket_manager):
        """Stream real-time updates to connected clients"""
        logger.info("🔄 Starting real-time update stream...")
        
        while True:
            try:
                # Get current portfolio snapshot
                snapshot = await self.get_portfolio_snapshot()
                
                # Broadcast to all connected clients
                await websocket_manager.broadcast({
                    "type": "portfolio.update",
                    "data": snapshot.dict(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Get strategy updates if time for detailed update
                now = datetime.now(timezone.utc)
                if (self.last_snapshot_time is None or 
                    (now - self.last_snapshot_time).seconds >= 5):
                    
                    strategy_updates = await self.get_strategy_updates()
                    if strategy_updates:
                        await websocket_manager.broadcast({
                            "type": "strategy.update",
                            "data": [update.dict() for update in strategy_updates],
                            "timestamp": now.isoformat()
                        })
                    
                    self.last_snapshot_time = now
                
                # Wait before next update (1 second for portfolio, 5 seconds for strategies)
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in update stream: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def execute_strategy_action(self, strategy_name: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a strategy action (start, stop, modify)"""
        try:
            if not self.is_connected:
                return {
                    "success": False,
                    "message": "Not connected to ThetaGang system",
                    "mock_mode": True
                }
            
            # Find the strategy
            strategy = None
            if hasattr(self.portfolio_manager, 'active_strategies'):
                for s in self.portfolio_manager.active_strategies:
                    if s.name == strategy_name:
                        strategy = s
                        break
            
            if not strategy:
                return {
                    "success": False,
                    "message": f"Strategy '{strategy_name}' not found"
                }
            
            # Execute the action
            if action == "start":
                result = await strategy.start()
            elif action == "stop": 
                result = await strategy.stop()
            elif action == "pause":
                result = await strategy.pause()
            elif action == "modify":
                result = await strategy.modify_parameters(params)
            else:
                return {
                    "success": False,
                    "message": f"Unknown action: {action}"
                }
            
            return {
                "success": True,
                "message": f"Successfully executed {action} on {strategy_name}",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error executing strategy action: {e}")
            return {
                "success": False,
                "message": f"Error executing action: {str(e)}"
            }
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "connected": self.is_connected,
            "mode": "live" if self.is_connected else "mock",
            "config_path": self.config_path,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "portfolio_manager": self.portfolio_manager is not None
        } 
