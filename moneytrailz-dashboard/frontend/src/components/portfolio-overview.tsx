'use client'

import React from "react"
import { DollarSign, Target, Wifi, WifiOff } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRealtimePortfolio, useConnectionStatus } from "@/hooks/useRealtimeData"

import { MetricCard, PnLCard, StrategyStatusCard } from "./ui/metric-card"

export function PortfolioOverview() {
  const { portfolio, isLoading } = useRealtimePortfolio()
  const { isConnected, lastUpdate } = useConnectionStatus()

  // Demo data for when backend is not connected
  const demoPortfolio = {
    totalValue: 125450.23,
    dayPnL: 1250.75,
    totalPnL: 12340.50,
    cashBalance: 25000.00,
    marginUsed: 15000.00,
    buyingPower: 85450.23,
    dayPnLPercent: 1.01,
    totalPnLPercent: 10.87,
    winRate: 73.2,
    activeStrategies: 5,
    lastUpdated: new Date().toISOString(),
  }

  const displayPortfolio = portfolio || demoPortfolio

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {['portfolio-value', 'pnl', 'win-rate', 'strategies'].map((metric) => (
            <div key={metric} className="h-32 bg-gray-200 dark:bg-gray-700 animate-pulse rounded-lg" />
          ))}
        </div>
      </div>
    )
  }



  return (
    <div className="space-y-6">
      {/* Enhanced Connection Status */}
      <EnhancedConnectionStatus isConnected={isConnected} lastUpdate={lastUpdate} />
      
      {/* Key Metrics Row */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Portfolio Value"
          value={displayPortfolio.totalValue}
          change={displayPortfolio.dayPnL}
          format="currency"
          icon={<DollarSign />}
          size="lg"
        />
        
        <PnLCard
          dailyPnL={displayPortfolio.dayPnL}
          totalPnL={displayPortfolio.totalPnL}
        />
        
        <MetricCard
          title="Win Rate"
          value={displayPortfolio.winRate}
          format="percentage"
          icon={<Target />}
          size="lg"
        />
        
        <StrategyStatusCard
          activeStrategies={displayPortfolio.activeStrategies}
          totalStrategies={10} // This should come from actual data
        />
      </div>

      {/* Additional Metrics Row */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
        <MetricCard
          title="Cash Balance"
          value={displayPortfolio.cashBalance}
          format="currency"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Margin Used"
          value={displayPortfolio.marginUsed}
          format="currency"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Buying Power"
          value={displayPortfolio.buyingPower}
          format="currency"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Day P&L %"
          value={displayPortfolio.dayPnLPercent}
          format="percentage"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Total P&L %"
          value={displayPortfolio.totalPnLPercent}
          format="percentage"
          size="sm"
          showTrend={false}
        />
        
        <div className="flex flex-col items-center justify-center p-4 bg-card rounded-lg border">
          <div className="text-sm font-medium text-muted-foreground mb-1">Last Updated</div>
          <div className="text-sm font-mono">
            {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : '--:--:--'}
          </div>
        </div>
      </div>
    </div>
  )
}

// Enhanced real-time connection status indicator
interface EnhancedConnectionStatusProps {
  isConnected: boolean
  lastUpdate: Date | null
}

export function EnhancedConnectionStatus({ isConnected, lastUpdate }: EnhancedConnectionStatusProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  return (
    <Card className={`border-l-4 ${isConnected ? 'border-l-green-500' : 'border-l-red-500'}`}>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          {isConnected ? (
            <>
              <Wifi className="h-5 w-5 text-green-500" />
              <span className="text-green-700 dark:text-green-400">MoneyTrailz Dashboard</span>
            </>
          ) : (
            <>
              <WifiOff className="h-5 w-5 text-red-500" />
              <span className="text-red-700 dark:text-red-400">MoneyTrailz Dashboard</span>
            </>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Algorithmic Trading System</span>
          <div className="flex items-center gap-2">
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            <span>•</span>
            <span>Live data</span>
            <span>•</span>
            <span>{lastUpdate ? formatTime(lastUpdate) : '--:--:--'}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Legacy connection status for backward compatibility
export function ConnectionStatus({ isConnected }: { isConnected: boolean }) {
  return (
    <div className="flex items-center space-x-2 text-sm">
      <div className={`w-2 h-2 rounded-full ${
        isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
      }`} />
      <span className="text-muted-foreground">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
      {isConnected && (
        <span className="text-xs text-muted-foreground">
          Live data
        </span>
      )}
    </div>
  )
} 
