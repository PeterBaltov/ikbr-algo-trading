from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from datetime import datetime, timezone
from typing import List

from app.routers import portfolio, strategies, trades, analytics
from app.websockets.manager import ConnectionManager
from app.models.dashboard import PortfolioSnapshot, StrategyUpdate
from app.integrations.thetagang_integration import DashboardIntegration

# WebSocket connection manager
manager = ConnectionManager()

# ThetaGang integration instance
dashboard_integration: DashboardIntegration = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting ThetaGang Dashboard API...")
    
    # Initialize ThetaGang integration
    global dashboard_integration
    dashboard_integration = DashboardIntegration()
    
    # Start background tasks
    asyncio.create_task(broadcast_updates())
    
    yield
    
    # Shutdown
    print("🛑 Shutting down ThetaGang Dashboard API...")
    await manager.disconnect_all()

# FastAPI application
app = FastAPI(
    title="ThetaGang Dashboard API",
    description="Real-time algorithmic trading dashboard API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://your-domain.vercel.app",  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "services": {
            "api": "running",
            "websocket": "running",
            "thetagang_integration": "connected" if dashboard_integration else "disconnected"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ThetaGang Dashboard API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws"
    }

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
            elif message.get("type") == "subscribe":
                # Handle subscription requests
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "data": message.get("data", {}),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task for broadcasting updates
async def broadcast_updates():
    """Background task to broadcast real-time updates"""
    while True:
        try:
            if dashboard_integration and manager.active_connections:
                # Get portfolio snapshot
                portfolio_data = await dashboard_integration.get_portfolio_snapshot()
                
                # Broadcast to all connected clients
                await manager.broadcast({
                    "type": "portfolio.update",
                    "data": portfolio_data.dict() if portfolio_data else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Get strategy updates
                strategy_updates = await dashboard_integration.get_strategy_updates()
                if strategy_updates:
                    await manager.broadcast({
                        "type": "strategy.update",
                        "data": [update.dict() for update in strategy_updates],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
        except Exception as e:
            print(f"Error in broadcast_updates: {e}")
        
        # Wait 1 second before next update
        await asyncio.sleep(1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
