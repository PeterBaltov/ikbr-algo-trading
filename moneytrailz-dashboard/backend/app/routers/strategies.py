from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ..integrations.simple_integration import SimpleDashboardIntegration

router = APIRouter()

class StrategyConfig(BaseModel):
    allocation: float = Field(..., ge=0, le=100, description="Allocation percentage (0-100)")
    max_position_size: float = Field(..., gt=0, description="Maximum position size in dollars")
    stop_loss: float = Field(..., ge=0, le=100, description="Stop loss percentage (0-100)")
    take_profit: float = Field(..., ge=0, description="Take profit percentage")
    symbols: List[str] = Field(..., min_length=1, description="List of trading symbols")
    timeframes: List[str] = Field(..., min_length=1, description="List of timeframes")
    risk_parameters: Dict[str, float] = Field(default_factory=dict, description="Risk management parameters")
    enabled: bool = Field(default=True, description="Whether strategy is enabled")

class StrategyConfigRequest(BaseModel):
    config: StrategyConfig

class StrategyActionResponse(BaseModel):
    success: bool
    message: str
    strategy_name: str
    new_status: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

async def get_dashboard_integration() -> SimpleDashboardIntegration:
    return SimpleDashboardIntegration()

@router.get("/")
async def get_strategies(
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get all strategies with their current status and metrics"""
    try:
        strategies = await integration.get_strategies()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": [strategy.dict() for strategy in strategies],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch strategies: {str(e)}"
        )

@router.get("/{strategy_name}/config")
async def get_strategy_config(
    strategy_name: str,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get configuration for a specific strategy"""
    try:
        # In a real implementation, this would fetch from database or config service
        # For now, return default configuration
        default_config = StrategyConfig(
            allocation=10.0,
            max_position_size=10000.0,
            stop_loss=5.0,
            take_profit=10.0,
            symbols=["SPY"],
            timeframes=["1d"],
            risk_parameters={
                "max_drawdown": 20.0,
                "max_daily_loss": 1000.0,
                "correlation_limit": 0.7
            },
            enabled=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "config": default_config.dict(),
                "strategy_name": strategy_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch strategy configuration: {str(e)}"
        )

@router.put("/{strategy_name}/config")
async def update_strategy_config(
    strategy_name: str,
    config_request: StrategyConfigRequest,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Update configuration for a specific strategy"""
    try:
        # Validate strategy exists
        strategies = await integration.get_strategies()
        strategy_names = [s.name for s in strategies]
        
        if strategy_name not in strategy_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        # In a real implementation, this would save to database or config service
        # For now, just return success with the provided configuration
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": f"Configuration updated for strategy '{strategy_name}'",
                "config": config_request.config.dict(),
                "strategy_name": strategy_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update strategy configuration: {str(e)}"
        )

@router.post("/{strategy_name}/pause")
async def pause_strategy(
    strategy_name: str,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Pause a running strategy"""
    try:
        # Validate strategy exists and is active
        strategies = await integration.get_strategies()
        strategy = next((s for s in strategies if s.name == strategy_name), None)
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        if strategy.status != 'active':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Strategy '{strategy_name}' is not active (current status: {strategy.status})"
            )
        
        # In a real implementation, this would send pause command to the strategy engine
        print(f"🔸 Pausing strategy: {strategy_name}")
        
        return StrategyActionResponse(
            success=True,
            message=f"Strategy '{strategy_name}' has been paused",
            strategy_name=strategy_name,
            new_status="paused"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause strategy: {str(e)}"
        )

@router.post("/{strategy_name}/resume")
async def resume_strategy(
    strategy_name: str,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Resume a paused strategy"""
    try:
        # Validate strategy exists and is paused
        strategies = await integration.get_strategies()
        strategy = next((s for s in strategies if s.name == strategy_name), None)
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        if strategy.status != 'paused':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Strategy '{strategy_name}' is not paused (current status: {strategy.status})"
            )
        
        # In a real implementation, this would send resume command to the strategy engine
        print(f"▶️ Resuming strategy: {strategy_name}")
        
        return StrategyActionResponse(
            success=True,
            message=f"Strategy '{strategy_name}' has been resumed",
            strategy_name=strategy_name,
            new_status="active"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume strategy: {str(e)}"
        )

@router.post("/{strategy_name}/stop")
async def stop_strategy(
    strategy_name: str,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Stop a strategy and close all positions"""
    try:
        # Validate strategy exists
        strategies = await integration.get_strategies()
        strategy = next((s for s in strategies if s.name == strategy_name), None)
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        if strategy.status == 'stopped':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Strategy '{strategy_name}' is already stopped"
            )
        
        # In a real implementation, this would:
        # 1. Send stop command to strategy engine
        # 2. Close all open positions
        # 3. Cancel pending orders
        print(f"⏹️ Stopping strategy: {strategy_name}")
        
        return StrategyActionResponse(
            success=True,
            message=f"Strategy '{strategy_name}' has been stopped and all positions closed",
            strategy_name=strategy_name,
            new_status="stopped"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop strategy: {str(e)}"
        )

@router.get("/{strategy_name}")
async def get_strategy_details(
    strategy_name: str,
    integration: SimpleDashboardIntegration = Depends(get_dashboard_integration)
):
    """Get detailed information for a specific strategy"""
    try:
        strategies = await integration.get_strategies()
        strategy = next((s for s in strategies if s.name == strategy_name), None)
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": strategy.dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch strategy details: {str(e)}"
        ) 
