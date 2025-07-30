'use client'

import React from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from './card'
import { Badge } from './badge'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

// Professional portfolio summary card inspired by Interactive Brokers

interface PortfolioSummaryCardProps {
  className?: string
}

interface AccountMetricProps {
  label: string
  value: number
  format?: 'currency' | 'percentage' | 'number'
  precision?: number
  showChange?: boolean
  change?: number
  className?: string
}

interface PositionSummaryProps {
  totalValue: number
  dayPnL: number
  dayPnLPercent: number
  unrealizedPnL: number
  realizedPnL: number
  availableCash: number
  buyingPower: number
  maintenanceMargin: number
  className?: string
}

function AccountMetric({ 
  label, 
  value, 
  format = 'currency', 
  precision = 2,
  showChange = false,
  change,
  className 
}: AccountMetricProps) {
  const formatValue = (val: number) => {
    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(val)
      
      case 'percentage':
        return new Intl.NumberFormat('en-US', {
          style: 'percent',
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(val / 100)
      
      case 'number':
        return new Intl.NumberFormat('en-US', {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(val)
      
      default:
        return String(val)
    }
  }

  const getValueColor = () => {
    if (format === 'currency' || format === 'number') {
      if (value > 0) return 'text-profit'
      if (value < 0) return 'text-loss'
    }
    return 'text-gray-900 dark:text-gray-100'
  }

  const getChangeColor = () => {
    if (change === undefined) return ''
    if (change > 0) return 'text-profit'
    if (change < 0) return 'text-loss'
    return 'text-gray-500 dark:text-gray-400'
  }

  const getChangeIcon = () => {
    if (change === undefined) return null
    if (change > 0) return <TrendingUp className="w-3 h-3" />
    if (change < 0) return <TrendingDown className="w-3 h-3" />
    return <Minus className="w-3 h-3" />
  }

  return (
    <div className={cn("space-y-1", className)}>
      <div className="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
        {label}
      </div>
      <div className="flex items-center justify-between">
        <div className={cn("text-sm font-semibold", getValueColor())}>
          {formatValue(value)}
        </div>
        {showChange && change !== undefined && (
          <div className={cn("flex items-center gap-1 text-xs", getChangeColor())}>
            {getChangeIcon()}
            <span>{formatValue(Math.abs(change))}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export function PositionSummary({
  totalValue,
  dayPnL,
  dayPnLPercent,
  unrealizedPnL,
  realizedPnL,
  availableCash,
  buyingPower,
  maintenanceMargin,
  className
}: PositionSummaryProps) {
  return (
    <Card className={cn("", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
          Account Summary
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Primary Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <AccountMetric
            label="Net Liquidation Value"
            value={totalValue}
            format="currency"
          />
          <AccountMetric
            label="Available Funds"
            value={availableCash}
            format="currency"
          />
        </div>

        {/* P&L Section */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="grid grid-cols-1 gap-3">
            <AccountMetric
              label="Day's P&L"
              value={dayPnL}
              format="currency"
              showChange={true}
              change={dayPnLPercent}
            />
            <div className="grid grid-cols-2 gap-4">
              <AccountMetric
                label="Unrealized P&L"
                value={unrealizedPnL}
                format="currency"
              />
              <AccountMetric
                label="Realized P&L"
                value={realizedPnL}
                format="currency"
              />
            </div>
          </div>
        </div>

        {/* Margin Information */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="grid grid-cols-2 gap-4">
            <AccountMetric
              label="Buying Power"
              value={buyingPower}
              format="currency"
            />
            <AccountMetric
              label="Maintenance Margin"
              value={maintenanceMargin}
              format="currency"
            />
          </div>
        </div>

        {/* Status Indicators */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex justify-between items-center">
            <Badge 
              variant={dayPnL >= 0 ? "default" : "destructive"}
              className="text-xs"
            >
              {dayPnL >= 0 ? "Profitable Day" : "Loss Day"}
            </Badge>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function PortfolioSummaryCard({ className }: PortfolioSummaryCardProps) {
  // Demo data - would be replaced with real data from props/hooks
  const portfolioData = {
    totalValue: 125450.23,
    dayPnL: 1250.75,
    dayPnLPercent: 1.01,
    unrealizedPnL: 3420.18,
    realizedPnL: 850.25,
    availableCash: 15680.45,
    buyingPower: 95000.00,
    maintenanceMargin: 5240.12
  }

  return (
    <div className={cn("", className)}>
      <PositionSummary {...portfolioData} />
    </div>
  )
}
