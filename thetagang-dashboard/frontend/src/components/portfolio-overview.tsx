import React from "react"
import { DollarSign, TrendingUp, Target, Activity } from "lucide-react"

import { Portfolio } from "@/types"

import { MetricCard, PnLCard, StrategyStatusCard } from "./ui/metric-card"

interface PortfolioOverviewProps {
  portfolio: Portfolio | null
  isLoading?: boolean
}

export function PortfolioOverview({ portfolio, isLoading }: PortfolioOverviewProps) {
  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {['portfolio-value', 'pnl', 'win-rate', 'strategies'].map((metric) => (
          <div key={metric} className="h-32 bg-gray-200 animate-pulse rounded-lg" />
        ))}
      </div>
    )
  }

  if (!portfolio) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Portfolio Value"
          value={0}
          format="currency"
          icon={<DollarSign />}
        />
        <MetricCard
          title="Today's P&L"
          value={0}
          format="currency"
          icon={<TrendingUp />}
        />
        <MetricCard
          title="Win Rate"
          value={0}
          format="percentage"
          icon={<Target />}
        />
        <MetricCard
          title="Active Strategies"
          value={0}
          format="number"
          icon={<Activity />}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics Row */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Portfolio Value"
          value={portfolio.totalValue}
          change={portfolio.dayPnL}
          format="currency"
          icon={<DollarSign />}
          size="lg"
        />
        
        <PnLCard
          dailyPnL={portfolio.dayPnL}
          totalPnL={portfolio.totalPnL}
        />
        
        <MetricCard
          title="Win Rate"
          value={portfolio.winRate}
          format="percentage"
          icon={<Target />}
          size="lg"
        />
        
        <StrategyStatusCard
          activeStrategies={portfolio.activeStrategies}
          totalStrategies={10} // This should come from actual data
        />
      </div>

      {/* Additional Metrics Row */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
        <MetricCard
          title="Cash Balance"
          value={portfolio.cashBalance}
          format="currency"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Margin Used"
          value={portfolio.marginUsed}
          format="currency"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Buying Power"
          value={portfolio.buyingPower}
          format="currency"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Day P&L %"
          value={portfolio.dayPnLPercent}
          format="percentage"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Total P&L %"
          value={portfolio.totalPnLPercent}
          format="percentage"
          size="sm"
          showTrend={false}
        />
        
        <MetricCard
          title="Last Updated"
          value={0}
          format="number"
          size="sm"
          showTrend={false}
          className="flex items-center justify-center"
        />
      </div>
    </div>
  )
}

// Real-time connection status indicator
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
