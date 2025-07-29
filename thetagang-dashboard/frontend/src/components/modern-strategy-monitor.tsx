'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  BarChart3
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useRealtimeStrategies } from '@/hooks/useRealtimeData'
import StrategyControls from './strategy-controls'

interface StrategyCardProps {
  strategy: {
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
      avgWin?: number
      avgLoss?: number
    }
    positions?: any[]
    recentTrades?: any[]
    lastUpdated: string
  }
}

function StrategyCard({ strategy }: StrategyCardProps) {
  const getStatusBadge = () => {
    const variants = {
      active: { variant: "default" as const, icon: CheckCircle, text: "Active" },
      paused: { variant: "secondary" as const, icon: Clock, text: "Paused" },
      stopped: { variant: "outline" as const, icon: Square, text: "Stopped" },
      error: { variant: "destructive" as const, icon: AlertTriangle, text: "Error" }
    }
    
    const config = variants[strategy.status]
    const Icon = config.icon
    
    return (
      <Badge variant={config.variant} className="flex items-center space-x-1">
        <Icon className="w-3 h-3" />
        <span>{config.text}</span>
      </Badge>
    )
  }

  const getTypeBadge = () => {
    const colors = {
      options: "bg-blue-100 text-blue-800 border-blue-200",
      stocks: "bg-green-100 text-green-800 border-green-200", 
      mixed: "bg-purple-100 text-purple-800 border-purple-200"
    }
    
    return (
      <Badge variant="outline" className={colors[strategy.type]}>
        {strategy.type.charAt(0).toUpperCase() + strategy.type.slice(1)}
      </Badge>
    )
  }

  return (
    <Card className="transition-all duration-200 hover:shadow-lg">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <CardTitle className="text-lg font-semibold">{strategy.name}</CardTitle>
              {getTypeBadge()}
            </div>
            <div className="flex items-center space-x-3">
              {getStatusBadge()}
              <span className="text-sm text-muted-foreground">
                {strategy.allocation}% allocation
              </span>
            </div>
          </div>
          <StrategyControls strategy={strategy} size="sm" />
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* P&L Section */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">Daily P&L</p>
            <div className="flex items-center space-x-2">
              <span className={cn(
                "text-xl font-bold",
                strategy.pnl.daily >= 0 ? "text-green-600" : "text-red-600"
              )}>
                ${Math.abs(strategy.pnl.daily).toLocaleString()}
              </span>
              {strategy.pnl.daily >= 0 ? (
                <TrendingUp className="w-4 h-4 text-green-600" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-600" />
              )}
            </div>
          </div>
          
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">Total P&L</p>
            <div className="flex items-center space-x-2">
              <span className={cn(
                "text-xl font-bold",
                strategy.pnl.total >= 0 ? "text-green-600" : "text-red-600"
              )}>
                ${Math.abs(strategy.pnl.total).toLocaleString()}
              </span>
              <Badge variant="outline" className={cn(
                strategy.pnl.percentage >= 0 ? "text-green-600" : "text-red-600"
              )}>
                {strategy.pnl.percentage >= 0 ? '+' : ''}{strategy.pnl.percentage.toFixed(1)}%
              </Badge>
            </div>
          </div>
        </div>

        <Separator />

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center space-y-1">
            <Target className="w-5 h-5 text-blue-600 mx-auto" />
            <p className="text-sm font-medium">{(strategy.metrics.winRate * 100).toFixed(1)}%</p>
            <p className="text-xs text-muted-foreground">Win Rate</p>
          </div>
          
          <div className="text-center space-y-1">
            <BarChart3 className="w-5 h-5 text-purple-600 mx-auto" />
            <p className="text-sm font-medium">{strategy.metrics.sharpeRatio.toFixed(2)}</p>
            <p className="text-xs text-muted-foreground">Sharpe</p>
          </div>
          
          <div className="text-center space-y-1">
            <TrendingDown className="w-5 h-5 text-red-600 mx-auto" />
            <p className="text-sm font-medium">{(Math.abs(strategy.metrics.maxDrawdown) * 100).toFixed(1)}%</p>
            <p className="text-xs text-muted-foreground">Max DD</p>
          </div>
          
          <div className="text-center space-y-1">
            <Activity className="w-5 h-5 text-green-600 mx-auto" />
            <p className="text-sm font-medium">{strategy.metrics.totalTrades}</p>
            <p className="text-xs text-muted-foreground">Trades</p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Avg Win:</span>
              <span className="font-medium text-green-600">
                ${strategy.metrics.avgWin?.toLocaleString() || '0'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Avg Loss:</span>
              <span className="font-medium text-red-600">
                ${Math.abs(strategy.metrics.avgLoss || 0).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function ModernStrategyMonitor() {
  const { strategies, isLoading } = useRealtimeStrategies()

  // Demo strategies for when backend is not connected
  const demoStrategies = [
    {
      name: 'Enhanced Wheel Strategy',
      type: 'options' as const,
      status: 'active' as const,
      allocation: 25,
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
      recentTrades: [],
      lastUpdated: new Date().toISOString()
    },
    {
      name: 'Momentum Scalper',
      type: 'stocks' as const,
      status: 'paused' as const,
      allocation: 15,
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
    },
    {
      name: 'Mean Reversion',
      type: 'mixed' as const,
      status: 'stopped' as const,
      allocation: 10,
      pnl: { daily: 0, total: -450.30, percentage: -2.1 },
      metrics: {
        winRate: 0.625,
        sharpeRatio: 0.87,
        maxDrawdown: -0.180,
        totalTrades: 89,
        avgWin: 95.30,
        avgLoss: -110.75
      },
      positions: [],
      recentTrades: [],
      lastUpdated: new Date().toISOString()
    }
  ]

  const displayStrategies = strategies.length > 0 ? strategies : demoStrategies

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Strategy Monitor</h2>
            <p className="text-muted-foreground">
              Manage and monitor your trading strategies
            </p>
          </div>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-4 bg-muted rounded w-3/4"></div>
                <div className="h-3 bg-muted rounded w-1/2"></div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="h-16 bg-muted rounded"></div>
                  <div className="h-12 bg-muted rounded"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const activeStrategies = displayStrategies.filter(s => s.status === 'active').length
  const totalPnL = displayStrategies.reduce((sum, s) => sum + s.pnl.daily, 0)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Strategy Monitor</h2>
          <p className="text-muted-foreground">
            Manage and monitor your trading strategies
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Badge variant="outline" className="flex items-center space-x-1">
            <Activity className="w-3 h-3" />
            <span>{activeStrategies} of {displayStrategies.length} active</span>
          </Badge>
          
          <Badge variant={totalPnL >= 0 ? "default" : "destructive"} className="flex items-center space-x-1">
            {totalPnL >= 0 ? (
              <TrendingUp className="w-3 h-3" />
            ) : (
              <TrendingDown className="w-3 h-3" />
            )}
            <span>
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toLocaleString()} today
            </span>
          </Badge>
        </div>
      </div>

      {/* Strategy Cards Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {displayStrategies.map((strategy) => (
          <StrategyCard key={strategy.name} strategy={strategy} />
        ))}
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5" />
            <span>Quick Actions</span>
          </CardTitle>
          <CardDescription>
            Manage all strategies at once
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2">
            <Button variant="outline" className="flex items-center space-x-2">
              <Play className="w-4 h-4" />
              <span>Start All</span>
            </Button>
            <Button variant="outline" className="flex items-center space-x-2">
              <Pause className="w-4 h-4" />
              <span>Pause All</span>
            </Button>
            <Button variant="outline" className="flex items-center space-x-2">
              <Square className="w-4 h-4" />
              <span>Stop All</span>
            </Button>
            <Button variant="outline" className="flex items-center space-x-2">
              <Settings className="w-4 h-4" />
              <span>Global Settings</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ModernStrategyMonitor 
