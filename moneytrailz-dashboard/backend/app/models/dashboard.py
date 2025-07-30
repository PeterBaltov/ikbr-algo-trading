from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum

class StrategyType(str, Enum):
    OPTIONS = "options"
    STOCKS = "stocks"
    MIXED = "mixed"

class StrategyStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    ROLL = "roll"
    ASSIGN = "assign"
    EXERCISE = "exercise"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    COMBO = "combo"

class OrderStatus(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    PENDING = "pending"

class PositionType(str, Enum):
    STOCK = "stock"
    OPTION = "option"
    CASH = "cash"

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

# Core Models
class Position(BaseModel):
    id: str
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    strategy: str
    type: PositionType
    side: PositionSide
    open_date: datetime

class Trade(BaseModel):
    id: str
    timestamp: datetime
    symbol: str
    action: TradeAction
    quantity: int
    price: float
    commission: float
    pnl: Optional[float] = None
    strategy: str
    order_type: OrderType
    status: OrderStatus

class StrategyMetrics(BaseModel):
    win_rate: float = Field(..., ge=0, le=100, description="Win rate percentage")
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int = Field(..., ge=0)
    avg_win: float
    avg_loss: float

class StrategyPnL(BaseModel):
    daily: float
    total: float
    percentage: float

class StrategySnapshot(BaseModel):
    name: str
    type: StrategyType
    status: StrategyStatus
    allocation: float = Field(..., ge=0, le=100, description="Allocation percentage")
    pnl: StrategyPnL
    metrics: StrategyMetrics
    positions: List[Position] = []
    recent_trades: List[Trade] = []
    last_updated: datetime

class PortfolioSnapshot(BaseModel):
    total_value: float
    day_pnl: float
    total_pnl: float
    cash_balance: float
    margin_used: float
    buying_power: float
    day_pnl_percent: float
    total_pnl_percent: float
    win_rate: float = Field(..., ge=0, le=100)
    active_strategies: int = Field(..., ge=0)
    last_updated: datetime

class MarketData(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    bid: float
    ask: float
    high: float
    low: float
    open: float
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    timestamp: datetime

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(str, Enum):
    PRICE = "price"
    STRATEGY = "strategy"
    SYSTEM = "system"
    RISK = "risk"

class Alert(BaseModel):
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    strategy: Optional[str] = None
    symbol: Optional[str] = None

class PerformanceMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float = Field(..., ge=0, le=100)
    profit_factor: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    volatility: float

class ChartDataPoint(BaseModel):
    timestamp: datetime
    value: float
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None

# API Response Models
class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PortfolioResponse(ApiResponse):
    data: Optional[PortfolioSnapshot] = None

class StrategiesResponse(ApiResponse):
    data: Optional[List[StrategySnapshot]] = None

class TradesResponse(ApiResponse):
    data: Optional[List[Trade]] = None

class PerformanceResponse(ApiResponse):
    data: Optional[PerformanceMetrics] = None

# WebSocket Event Models
class WebSocketEvent(BaseModel):
    type: str
    data: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PortfolioUpdateEvent(WebSocketEvent):
    type: str = "portfolio.update"
    data: PortfolioSnapshot

class StrategyUpdateEvent(WebSocketEvent):
    type: str = "strategy.update"
    data: List[StrategySnapshot]

class TradeExecutedEvent(WebSocketEvent):
    type: str = "trade.executed"
    data: Trade

class MarketDataEvent(WebSocketEvent):
    type: str = "market.data"
    data: Dict[str, MarketData]

class AlertEvent(WebSocketEvent):
    type: str = "alert.new"
    data: Alert

# Update models for real-time events
class StrategyUpdate(BaseModel):
    name: str
    status: StrategyStatus
    pnl_change: float
    last_trade: Optional[Trade] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) 
