// Core Trading Types
export interface Portfolio {
  totalValue: number
  dayPnL: number
  totalPnL: number
  cashBalance: number
  marginUsed: number
  buyingPower: number
  dayPnLPercent: number
  totalPnLPercent: number
  winRate: number
  activeStrategies: number
  lastUpdated: string
}

export interface Strategy {
  name: string
  type: 'options' | 'stocks' | 'mixed'
  status: 'active' | 'paused' | 'stopped' | 'error'
  allocation: number
  pnl: {
    daily: number
    total: number
    percentage: number
  }
  metrics: {
    winRate: number
    sharpeRatio: number
    maxDrawdown: number
    totalTrades: number
    avgWin: number
    avgLoss: number
  }
  positions: Position[]
  recentTrades: Trade[]
  lastUpdated: string
}

export interface Position {
  id: string
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  marketValue: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  strategy: string
  type: 'stock' | 'option' | 'cash'
  side: 'long' | 'short'
  openDate: string
}

export interface Trade {
  id: string
  timestamp: string
  symbol: string
  action: 'buy' | 'sell' | 'roll' | 'assign' | 'exercise'
  quantity: number
  price: number
  commission: number
  pnl?: number
  strategy: string
  orderType: 'market' | 'limit' | 'stop' | 'combo'
  status: 'filled' | 'partial' | 'cancelled' | 'pending'
}

export interface PerformanceMetrics {
  totalReturn: number
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  var95: number
  cvar95: number
  beta: number
  alpha: number
  volatility: number
}

export interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  bid: number
  ask: number
  high: number
  low: number
  open: number
  impliedVolatility?: number
  delta?: number
  gamma?: number
  theta?: number
  vega?: number
  timestamp: string
}

export interface ChartDataPoint {
  timestamp: string
  value: number
  volume?: number
  high?: number
  low?: number
  open?: number
  close?: number
}

export interface Alert {
  id: string
  type: 'price' | 'strategy' | 'system' | 'risk'
  severity: 'info' | 'warning' | 'error' | 'critical'
  title: string
  message: string
  timestamp: string
  acknowledged: boolean
  strategy?: string
  symbol?: string
}

// Dashboard State Types
export interface DashboardState {
  portfolio: Portfolio | null
  strategies: Strategy[]
  alerts: Alert[]
  isConnected: boolean
  lastUpdate: string
  selectedTimeframe: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | 'ALL'
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: 'portfolio.update' | 'strategy.update' | 'trade.executed' | 'market.data' | 'alert.new'
  data: any
  timestamp: string
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean
  data: T
  error?: string
  timestamp: string
}

export interface StrategyConfig {
  name: string
  enabled: boolean
  allocation: number
  symbols: string[]
  timeframes: string[]
  parameters: Record<string, any>
  riskLimits: {
    maxPositionSize: number
    stopLoss?: number
    takeProfit?: number
    maxDrawdown: number
  }
}

// Chart Configuration Types
export interface ChartConfig {
  type: 'line' | 'candlestick' | 'bar' | 'area'
  timeframe: string
  indicators: string[]
  theme: 'light' | 'dark'
  height: number
}

// Table Configuration Types
export interface TableColumn<T> {
  key: keyof T
  label: string
  sortable?: boolean
  format?: (value: any) => string
  className?: string
}

export interface TableConfig<T> {
  columns: TableColumn<T>[]
  sortBy?: keyof T
  sortOrder?: 'asc' | 'desc'
  pageSize?: number
}

// Filter Types
export interface FilterConfig {
  strategies?: string[]
  symbols?: string[]
  dateRange?: {
    start: string
    end: string
  }
  status?: string[]
  minPnL?: number
  maxPnL?: number
} 
