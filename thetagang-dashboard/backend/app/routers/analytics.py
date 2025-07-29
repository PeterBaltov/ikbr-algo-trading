from fastapi import APIRouter, Depends
from datetime import datetime, timezone

from ..models.dashboard import ApiResponse
from ..integrations.simple_integration import SimpleDashboardIntegration

router = APIRouter()

async def get_dashboard_integration() -> SimpleDashboardIntegration:
    return SimpleDashboardIntegration()

@router.get("/performance")
async def get_performance_metrics(
    timeframe: str = "1D",
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get performance metrics"""
    # Placeholder for Phase 4 implementation
    return ApiResponse(
        success=True,
        data={"message": "Performance analytics will be implemented in Phase 4"},
        timestamp=datetime.now(timezone.utc)
    ) 
