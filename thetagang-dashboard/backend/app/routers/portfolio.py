from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime, timezone

from ..models.dashboard import (
    PortfolioResponse, PortfolioSnapshot, 
    Position, ApiResponse
)
from ..integrations.simple_integration import SimpleDashboardIntegration

router = APIRouter()

# Dependency to get the dashboard integration
async def get_dashboard_integration() -> SimpleDashboardIntegration:
    """Dependency to provide dashboard integration instance"""
    return SimpleDashboardIntegration()

@router.get("/", response_model=PortfolioResponse)
async def get_portfolio(
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get current portfolio overview"""
    try:
        portfolio_data = await integration.get_portfolio_snapshot()
        
        if not portfolio_data:
            raise HTTPException(status_code=503, detail="Portfolio data unavailable")
        
        return PortfolioResponse(
            success=True,
            data=portfolio_data,
            timestamp=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        return PortfolioResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now(timezone.utc)
        )

@router.get("/positions", response_model=ApiResponse)
async def get_positions(
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get current portfolio positions"""
    try:
        positions = await integration.get_positions()
        
        return ApiResponse(
            success=True,
            data=positions,
            timestamp=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now(timezone.utc)
        )

@router.get("/summary")
async def get_portfolio_summary(
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get portfolio summary with key metrics"""
    try:
        portfolio_data = await integration.get_portfolio_snapshot()
        
        if not portfolio_data:
            raise HTTPException(status_code=503, detail="Portfolio data unavailable")
        
        summary = {
            "total_value": portfolio_data.total_value,
            "day_pnl": portfolio_data.day_pnl,
            "day_pnl_percent": portfolio_data.day_pnl_percent,
            "win_rate": portfolio_data.win_rate,
            "active_strategies": portfolio_data.active_strategies,
            "cash_balance": portfolio_data.cash_balance,
            "buying_power": portfolio_data.buying_power,
            "last_updated": portfolio_data.last_updated.isoformat()
        }
        
        return ApiResponse(
            success=True,
            data=summary,
            timestamp=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now(timezone.utc)
        )

@router.get("/health")
async def portfolio_health_check(
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Health check for portfolio data connection"""
    try:
        connection_status = integration.get_connection_status()
        
        return {
            "status": "healthy" if connection_status["connected"] else "disconnected",
            "connection": connection_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        } 
