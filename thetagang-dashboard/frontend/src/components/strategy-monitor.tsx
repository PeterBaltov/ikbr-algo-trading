'use client'

import React from 'react'
import { Play, Pause, Square, Settings, AlertTriangle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useRealtimeStrategies } from '@/hooks/useRealtimeData'
import { Strategy } from '@/types'
import { cn, formatCurrency, formatPercentage, getPnLColor } from '@/lib/utils'

export function StrategyMonitor() {
  const { strategies, isLoading, toggleStrategy } = useRealtimeStrategies()

  // Demo strategies for when backend is not connected
  const demoStrategies = [
    {
      name: 'Enhanced Wheel Strategy',
      type: 'options' as const,
      status: 'active' as const,
      allocation: 0.25,
      pnl: { daily: 1250.75, total: 8340.50, percentage: 15.2 },
      metrics: {
        winRate: 0.742,
        sharpeRatio: 1.85,
        maxDrawdown: -0.058,
        totalTrades: 127,
        avgWin: 285.50,
        avgLoss: -125.25
      },
      positions: [],
      recentTrades: [
        { id: '1', timestamp: new Date().toISOString(), symbol: 'SPY', action: 'sell' as const, quantity: 1, price: 450.25, commission: 1.0, strategy: 'wheel', orderType: 'limit' as const, status: 'filled' as const, pnl: 225.50 }
      ],
      lastUpdated: new Date().toISOString()
    },
    {
      name: 'Momentum Scalper',
      type: 'stocks' as const,
      status: 'paused' as const,
      allocation: 0.15,
      pnl: { daily: -85.25, total: 2150.75, percentage: 8.7 },
      metrics: {
        winRate: 0.681,
        sharpeRatio: 1.23,
        maxDrawdown: -0.125,
        totalTrades: 203,
        avgWin: 125.75,
        avgLoss: -85.50
      },
      positions: [],
      recentTrades: [],
      lastUpdated: new Date().toISOString()
    }
  ]

  const displayStrategies = strategies.length > 0 ? strategies : demoStrategies

  const handleToggleStrategy = async (strategyName: string) => {
    await toggleStrategy(strategyName)
  }

  const handleConfigureStrategy = (strategyName: string) => {
    // TODO: Implement strategy configuration modal
    console.log('Configure strategy:', strategyName)
  }

  if (isLoading) {
    const skeletonItems = ['strategy-1', 'strategy-2', 'strategy-3']
    return (
      <div className="space-y-4">
        {skeletonItems.map((id) => (
          <Card key={id} className="animate-pulse">
            <CardHeader>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Strategy Monitor</h2>
        <div className="text-sm text-muted-foreground">
          {displayStrategies.filter(s => s.status === 'active').length} of {displayStrategies.length} active
        </div>
      </div>
      
      <div className="space-y-4">
        {displayStrategies.map(strategy => (
          <StrategyCard
            key={strategy.name}
            strategy={strategy}
            onToggle={handleToggleStrategy}
            onConfigure={handleConfigureStrategy}
          />
        ))}
      </div>
    </div>
  )
}

interface StrategyCardProps {
  strategy: Strategy
  onToggle: (name: string) => void
  onConfigure: (name: string) => void
}

export function StrategyCard({ strategy, onToggle, onConfigure }: StrategyCardProps) {
  const getStatusColor = (status: Strategy['status']) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-950 dark:border-green-800'
      case 'paused':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-950 dark:border-yellow-800'
      case 'stopped':
        return 'text-gray-600 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-950 dark:border-gray-800'
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-950 dark:border-red-800'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-950 dark:border-gray-800'
    }
  }

  const getStatusIcon = (status: Strategy['status']) => {
    switch (status) {
      case 'active':
        return <Play className="h-4 w-4" />
      case 'paused':
        return <Pause className="h-4 w-4" />
      case 'stopped':
        return <Square className="h-4 w-4" />
      case 'error':
        return <AlertTriangle className="h-4 w-4" />
      default:
        return <Square className="h-4 w-4" />
    }
  }

  const getActionButton = () => {
    switch (strategy.status) {
      case 'active':
        return (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToggle(strategy.name)}
            className="gap-2"
          >
            <Pause className="h-4 w-4" />
            Pause
          </Button>
        )
      case 'paused':
        return (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToggle(strategy.name)}
            className="gap-2"
          >
            <Play className="h-4 w-4" />
            Resume
          </Button>
        )
      default:
        return (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToggle(strategy.name)}
            className="gap-2"
          >
            <Play className="h-4 w-4" />
            Start
          </Button>
        )
    }
  }

  return (
    <Card className={cn(
      "border-l-4 transition-all hover:shadow-md",
      strategy.status === 'active' && "border-l-green-500",
      strategy.status === 'paused' && "border-l-yellow-500",
      strategy.status === 'stopped' && "border-l-gray-500",
      strategy.status === 'error' && "border-l-red-500"
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <span>{strategy.name}</span>
            <span className={cn(
              "px-2 py-1 text-xs font-medium rounded-full border",
              getStatusColor(strategy.status)
            )}>
              {getStatusIcon(strategy.status)}
              <span className="ml-1 capitalize">{strategy.status}</span>
            </span>
          </CardTitle>
          
          <div className="flex items-center gap-2">
            {getActionButton()}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onConfigure(strategy.name)}
              className="gap-2"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {/* Type & Allocation */}
          <div>
            <div className="text-xs text-muted-foreground mb-1">Type</div>
            <div className="text-sm font-medium capitalize">{strategy.type}</div>
          </div>
          
          <div>
            <div className="text-xs text-muted-foreground mb-1">Allocation</div>
            <div className="text-sm font-medium">{formatPercentage(strategy.allocation)}</div>
          </div>
          
          {/* P&L Metrics */}
          <div>
            <div className="text-xs text-muted-foreground mb-1">Daily P&L</div>
            <div className={cn("text-sm font-medium", getPnLColor(strategy.pnl.daily))}>
              {formatCurrency(strategy.pnl.daily)}
            </div>
          </div>
          
          <div>
            <div className="text-xs text-muted-foreground mb-1">Total P&L</div>
            <div className={cn("text-sm font-medium", getPnLColor(strategy.pnl.total))}>
              {formatCurrency(strategy.pnl.total)}
            </div>
          </div>
          
          {/* Performance Metrics */}
          <div>
            <div className="text-xs text-muted-foreground mb-1">Win Rate</div>
            <div className="text-sm font-medium">{formatPercentage(strategy.metrics.winRate)}</div>
          </div>
          
          <div>
            <div className="text-xs text-muted-foreground mb-1">Sharpe Ratio</div>
            <div className="text-sm font-medium">{strategy.metrics.sharpeRatio.toFixed(2)}</div>
          </div>
        </div>
        
        {/* Additional Metrics Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t">
          <div>
            <div className="text-xs text-muted-foreground mb-1">Total Trades</div>
            <div className="text-sm font-medium">{strategy.metrics.totalTrades}</div>
          </div>
          
          <div>
            <div className="text-xs text-muted-foreground mb-1">Max Drawdown</div>
            <div className="text-sm font-medium text-red-600">
              {formatPercentage(strategy.metrics.maxDrawdown)}
            </div>
          </div>
          
          <div>
            <div className="text-xs text-muted-foreground mb-1">Avg Win</div>
            <div className="text-sm font-medium text-green-600">
              {formatCurrency(strategy.metrics.avgWin)}
            </div>
          </div>
          
          <div>
            <div className="text-xs text-muted-foreground mb-1">Avg Loss</div>
            <div className="text-sm font-medium text-red-600">
              {formatCurrency(strategy.metrics.avgLoss)}
            </div>
          </div>
        </div>
        
        {/* Recent Activity */}
        {strategy.recentTrades.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <div className="text-xs text-muted-foreground mb-2">Recent Activity</div>
            <div className="space-y-1">
              {strategy.recentTrades.slice(0, 3).map((trade) => (
                <div key={trade.id} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    {trade.symbol} {trade.action}
                  </span>
                  <span className={cn("font-medium", getPnLColor(trade.pnl || 0))}>
                    {trade.pnl ? formatCurrency(trade.pnl) : formatCurrency(trade.price * trade.quantity)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 
