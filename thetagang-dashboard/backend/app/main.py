from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import socketio

from app.routers import portfolio, strategies, trades, analytics
from app.integrations.simple_integration import SimpleDashboardIntegration

# Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:3001",
        "https://your-domain.vercel.app"
    ],
    logger=True,
    engineio_logger=True
)

# ThetaGang integration instance  
dashboard_integration: SimpleDashboardIntegration | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting ThetaGang Dashboard API...")
    
    # Initialize ThetaGang integration
    global dashboard_integration
    dashboard_integration = SimpleDashboardIntegration()
    
    # Start background tasks
    asyncio.create_task(broadcast_updates())
    
    yield
    
    # Shutdown
    print("🛑 Shutting down ThetaGang Dashboard API...")

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
        "timestamp": "2024-01-01T00:00:00Z",
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
        "socket_io": "/socket.io/"
    }

# Socket.IO events
@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    print(f"🔗 Client connected: {sid}")
    await sio.emit('welcome', {
        'message': 'Connected to ThetaGang Dashboard',
        'sid': sid
    }, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"🔌 Client disconnected: {sid}")

@sio.event
async def subscribe(sid, data):
    """Handle subscription requests"""
    channels = data.get('channels', [])
    print(f"📡 Client {sid} subscribing to: {channels}")
    
    # Join rooms for subscriptions
    for channel in channels:
        await sio.enter_room(sid, channel)
    
    await sio.emit('subscribed', {
        'channels': channels,
        'message': f'Subscribed to {len(channels)} channels'
    }, room=sid)

@sio.event
async def ping(sid, data):
    """Handle ping requests"""
    await sio.emit('pong', {'timestamp': data.get('timestamp')}, room=sid)

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)

# Background task for broadcasting updates
async def broadcast_updates():
    """Background task to broadcast real-time updates"""
    print("🔄 Starting real-time broadcast service...")
    
    while True:
        try:
            if dashboard_integration:
                # Get portfolio snapshot
                portfolio_data = await dashboard_integration.get_portfolio_snapshot()
                
                # Broadcast to portfolio subscribers
                await sio.emit('portfolio.update', {
                    'data': portfolio_data.dict(),
                    'timestamp': '2024-01-01T00:00:00Z'
                }, room='portfolio')
                
                # Get strategy updates
                strategy_updates = await dashboard_integration.get_strategy_updates()
                if strategy_updates:
                    await sio.emit('strategy.update', {
                        'data': [update.dict() for update in strategy_updates],
                        'timestamp': '2024-01-01T00:00:00Z'
                    }, room='strategies')
                
        except Exception as e:
            print(f"Error in broadcast_updates: {e}")
        
        # Wait 2 seconds before next update
        await asyncio.sleep(2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:socket_app",  # Use socket_app instead of app
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
