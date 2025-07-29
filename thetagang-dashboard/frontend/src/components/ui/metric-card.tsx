import React from "react"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"

import { cn, formatCurrency, formatPercentage, formatNumber, getPnLColor } from "@/lib/utils"

import { Card, CardContent, CardHeader, CardTitle } from "./card"

interface MetricCardProps {
  title: string
  value: number
  change?: number
  format?: "currency" | "percentage" | "number"
  icon?: React.ReactNode
  className?: string
  size?: "sm" | "md" | "lg"
  showTrend?: boolean
}

export function MetricCard({
  title,
  value,
  change,
  format = "number",
  icon,
  className,
  size = "md",
  showTrend = true,
}: MetricCardProps) {
  const formatValue = (val: number) => {
    switch (format) {
      case "currency":
        return formatCurrency(val)
      case "percentage":
        return formatPercentage(val)
      case "number":
        return formatNumber(val)
      default:
        return val.toString()
    }
  }

  const getTrendIcon = (changeValue: number) => {
    if (changeValue > 0) return <TrendingUp className="w-4 h-4" />
    if (changeValue < 0) return <TrendingDown className="w-4 h-4" />
    return <Minus className="w-4 h-4" />
  }

  const sizeClasses = {
    sm: "p-4",
    md: "p-6",
    lg: "p-8",
  }

  const valueSizeClasses = {
    sm: "text-xl",
    md: "text-2xl",
    lg: "text-3xl",
  }

  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <CardHeader className={cn("flex flex-row items-center justify-between space-y-0", sizeClasses[size])}>
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {icon && (
          <div className="w-4 h-4 text-muted-foreground">
            {icon}
          </div>
        )}
      </CardHeader>
      <CardContent className={cn("pt-0", sizeClasses[size])}>
        <div className="flex items-center space-x-2">
          <div className={cn("font-bold", valueSizeClasses[size])}>
            {formatValue(value)}
          </div>
          {change !== undefined && showTrend && (
            <div className={cn(
              "flex items-center space-x-1 text-sm font-medium",
              getPnLColor(change)
            )}>
              {getTrendIcon(change)}
              <span>
                {format === "percentage" 
                  ? formatPercentage(Math.abs(change))
                  : format === "currency"
                  ? formatCurrency(Math.abs(change))
                  : formatNumber(Math.abs(change))
                }
              </span>
            </div>
          )}
        </div>
        {change !== undefined && (
          <p className="text-xs text-muted-foreground mt-1">
            {change > 0 ? "↗" : change < 0 ? "↘" : "→"} from previous period
          </p>
        )}
      </CardContent>
    </Card>
  )
}

// Specialized variants for common trading metrics
export function PnLCard({ dailyPnL, totalPnL, className }: { 
  dailyPnL: number
  totalPnL: number 
  className?: string 
}) {
  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">P&L</CardTitle>
        <div className="w-4 h-4 text-muted-foreground">
          {dailyPnL > 0 ? <TrendingUp /> : dailyPnL < 0 ? <TrendingDown /> : <Minus />}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Today</span>
            <span className={cn("font-bold", getPnLColor(dailyPnL))}>
              {formatCurrency(dailyPnL)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Total</span>
            <span className={cn("font-bold", getPnLColor(totalPnL))}>
              {formatCurrency(totalPnL)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function StrategyStatusCard({ 
  activeStrategies, 
  totalStrategies, 
  className 
}: { 
  activeStrategies: number
  totalStrategies: number
  className?: string 
}) {
  const percentage = totalStrategies > 0 ? (activeStrategies / totalStrategies) * 100 : 0
  
  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Strategies</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="text-2xl font-bold">
            {activeStrategies}/{totalStrategies}
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex-1 bg-secondary rounded-full h-2">
              <div 
                className="bg-primary h-2 rounded-full transition-all duration-300" 
                style={{ width: `${percentage}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground">
              {formatPercentage(percentage)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground">
            {activeStrategies} active strategies
          </p>
        </div>
      </CardContent>
    </Card>
  )
} 
