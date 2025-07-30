'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { usePerformanceData } from '@/hooks/useRealtimeData'
import { PerformanceDataPoint } from '@/types'

interface PerformanceChartProps {
  height?: number
  showControls?: boolean
}

export function PerformanceChart({ height = 400, showControls = true }: PerformanceChartProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D')
  const { performanceData, isLoading } = usePerformanceData(selectedTimeframe)

  const timeframes = [
    { label: '1D', value: '1D' },
    { label: '1W', value: '1W' },
    { label: '1M', value: '1M' },
    { label: '3M', value: '3M' },
    { label: '6M', value: '6M' },
    { label: '1Y', value: '1Y' },
    { label: 'ALL', value: 'ALL' }
  ]

  const handleTimeframeChange = (timeframe: string) => {
    setSelectedTimeframe(timeframe)
  }

  const getPerformanceStats = () => {
    if (!performanceData.length) return null

    const firstValue = performanceData[0]?.value || 0
    const lastValue = performanceData[performanceData.length - 1]?.value || 0
    const change = lastValue - firstValue
    const changePercent = firstValue !== 0 ? (change / firstValue) * 100 : 0

    const maxValue = Math.max(...performanceData.map(p => p.value))
    const minValue = Math.min(...performanceData.map(p => p.value))
    const maxDrawdown = firstValue !== 0 ? ((maxValue - minValue) / maxValue) * 100 : 0

    return {
      change,
      changePercent,
      maxValue,
      minValue,
      maxDrawdown,
    }
  }

  const stats = getPerformanceStats()

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div 
            className="animate-pulse bg-gray-200 dark:bg-gray-700 rounded"
            style={{ height }}
          />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Portfolio Performance</CardTitle>
          {showControls && (
            <div className="flex items-center gap-1">
              {timeframes.map(tf => (
                <Button
                  key={tf.value}
                  variant={selectedTimeframe === tf.value ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => handleTimeframeChange(tf.value)}
                  className="px-3 py-1 text-xs"
                >
                  {tf.label}
                </Button>
              ))}
            </div>
          )}
        </div>
        
        {stats && (
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <div>
              <span className="font-medium">Change: </span>
              <span className={stats.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                ${stats.change.toFixed(2)} ({stats.changePercent.toFixed(2)}%)
              </span>
            </div>
            <div>
              <span className="font-medium">Max: </span>
              <span>${stats.maxValue.toFixed(2)}</span>
            </div>
            <div>
              <span className="font-medium">Min: </span>
              <span>${stats.minValue.toFixed(2)}</span>
            </div>
            <div>
              <span className="font-medium">Max Drawdown: </span>
              <span className="text-red-600">{stats.maxDrawdown.toFixed(2)}%</span>
            </div>
          </div>
        )}
      </CardHeader>
      
      <CardContent>
        <div 
          className="w-full bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950 dark:to-blue-950 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 flex items-center justify-center"
          style={{ height }}
        >
          {!performanceData.length ? (
            <div className="text-center text-muted-foreground">
              <div className="text-4xl mb-4">📈</div>
              <p className="text-lg font-semibold">No performance data available</p>
              <p className="text-sm mt-1">Data will appear when strategies are active</p>
            </div>
          ) : (
            <div className="text-center text-muted-foreground">
              <div className="text-4xl mb-4">📊</div>
              <p className="text-lg font-semibold">Interactive Chart</p>
              <p className="text-sm mt-1">Advanced charting will be implemented in Phase 3</p>
              <p className="text-xs mt-2">({performanceData.length} data points loaded)</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// Lightweight TradingView-style widget component
export function TradingViewWidget({ 
  symbol, 
  indicators = [], 
}: {
  symbol: string
  data: PerformanceDataPoint[]
  indicators?: string[]
  theme?: 'light' | 'dark'
}) {
  return (
    <div className="relative">
      <PerformanceChart height={384} />
      
      {/* Indicators overlay */}
      {indicators.length > 0 && (
        <div className="absolute top-2 left-2 bg-black/80 text-white text-xs px-2 py-1 rounded">
          {indicators.join(', ')}
        </div>
      )}
      
      {/* Symbol display */}
      <div className="absolute top-2 right-2 bg-black/80 text-white text-xs px-2 py-1 rounded">
        {symbol}
      </div>
    </div>
  )
} 
