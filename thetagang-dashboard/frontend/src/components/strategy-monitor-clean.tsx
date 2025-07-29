'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useRealtimeStrategies } from '@/hooks/useRealtimeData'
import { Strategy } from '@/types'
import { cn, formatCurrency, formatPercentage, getPnLColor } from '@/lib/utils'
import StrategyControls from './strategy-controls'

export function StrategyMonitor() {
  const { strategies, isLoading } = useRealtimeStrategies()

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
          />
        ))}
      </div>
    </div>
  )
}

interface StrategyCardProps {
  strategy: Strategy
}

function StrategyCard({ strategy }: StrategyCardProps) {
  return (
    <Card className="transition-all duration-200 hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-medium">
            {strategy.name}
          </CardTitle>
          <StrategyControls strategy={strategy} size="sm" />
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
            <div className="text-sm font-medium">{formatPercentage(strategy.allocation * 100)}</div>
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
            <div className="text-sm font-medium">{formatPercentage(strategy.metrics.winRate * 100)}</div>
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
              {formatPercentage(strategy.metrics.maxDrawdown * 100)}
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

export default StrategyMonitor 
