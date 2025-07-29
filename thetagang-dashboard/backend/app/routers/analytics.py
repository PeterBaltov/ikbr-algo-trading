from fastapi import APIRouter, Depends
from datetime import datetime, timezone

from ..models.dashboard import ApiResponse
from ..integrations.thetagang_integration import DashboardIntegration

router = APIRouter()

async def get_dashboard_integration() -> DashboardIntegration:
    return DashboardIntegration()

@router.get("/performance")
async def get_performance_metrics(
    timeframe: str = "1D",
    integration: DashboardIntegration = Depends(get_dashboard_integration)
):
    """Get performance metrics"""
    # Placeholder for Phase 4 implementation
    return ApiResponse(
        success=True,
        data={"message": "Performance analytics will be implemented in Phase 4"},
        timestamp=datetime.now(timezone.utc)
    ) 
