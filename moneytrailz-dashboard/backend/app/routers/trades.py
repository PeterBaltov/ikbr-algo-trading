from fastapi import APIRouter, Depends
from datetime import datetime, timezone

from ..models.dashboard import TradesResponse
from ..integrations.simple_integration import SimpleDashboardIntegration

router = APIRouter()

async def get_dashboard_integration() -> SimpleDashboardIntegration:
    return SimpleDashboardIntegration()

@router.get("/", response_model=TradesResponse)
async def get_trades(
    limit: int = 10,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get recent trades"""
    try:
        trades = await integration.get_recent_trades(limit=limit)
        
        return TradesResponse(
            success=True,
            data=trades,
            timestamp=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        return TradesResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now(timezone.utc)
        ) 
