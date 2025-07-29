from fastapi import APIRouter, Depends
from datetime import datetime, timezone
from typing import List

from ..models.dashboard import StrategiesResponse, ApiResponse
from ..integrations.simple_integration import SimpleDashboardIntegration

router = APIRouter()

async def get_dashboard_integration() -> SimpleDashboardIntegration:
    return SimpleDashboardIntegration()

@router.get("/", response_model=StrategiesResponse)
async def get_strategies(
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get all strategy snapshots"""
    try:
        strategies = await integration.get_strategies()
        
        return StrategiesResponse(
            success=True,
            data=strategies,
            timestamp=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        return StrategiesResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now(timezone.utc)
        )

@router.post("/{strategy_name}/pause")
async def pause_strategy(
    strategy_name: str,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Pause a strategy"""
    try:
        success = await integration.pause_strategy(strategy_name)
        
        return ApiResponse(
            success=success,
            data={"strategy": strategy_name, "action": "paused"} if success else None,
            error="Strategy not found" if not success else None,
            timestamp=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now(timezone.utc)
        ) 
