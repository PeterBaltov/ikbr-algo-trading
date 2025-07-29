'use client'

import React, { useEffect, useRef, useState } from 'react'
import { createChart, ColorType, LineStyle, CrosshairMode } from 'lightweight-charts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  TrendingUp, 
  Calendar, 
  BarChart3, 
  Activity,
  Maximize2
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface TradingChartProps {
  data: Array<{ time: string; value: number }>
  title?: string
  height?: number
  className?: string
}

function TradingChartInner({ data, title = "Portfolio Performance", height = 400, className }: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const seriesRef = useRef<any>(null)
  const [timeframe, setTimeframe] = useState('1D')

  useEffect(() => {
    if (!chartContainerRef.current) return

    // Create the chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: 'hsl(var(--foreground))',
        fontSize: 12,
        fontFamily: 'Inter, system-ui, sans-serif'
      },
      grid: {
        vertLines: {
          color: 'hsl(var(--border))',
          style: LineStyle.Dotted,
        },
        horzLines: {
          color: 'hsl(var(--border))',
          style: LineStyle.Dotted,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: 'hsl(var(--primary))',
          width: 1,
          style: LineStyle.Solid,
        },
        horzLine: {
          color: 'hsl(var(--primary))',
          width: 1,
          style: LineStyle.Solid,
        },
      },
      rightPriceScale: {
        borderColor: 'hsl(var(--border))',
        textColor: 'hsl(var(--muted-foreground))',
      },
      timeScale: {
        borderColor: 'hsl(var(--border))',
        textColor: 'hsl(var(--muted-foreground))',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Add the area series
    const series = chart.addAreaSeries({
      lineColor: 'hsl(var(--primary))',
      topColor: 'hsl(var(--primary) / 0.3)',
      bottomColor: 'hsl(var(--primary) / 0.05)',
      lineWidth: 2,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: 'hsl(var(--primary))',
      crosshairMarkerBackgroundColor: 'hsl(var(--primary))',
    })

    // Set the data
    series.setData(data)

    // Store references
    chartRef.current = chart
    seriesRef.current = series

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
          width: chartContainerRef.current.clientWidth 
        })
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(chartContainerRef.current)

    // Cleanup
    return () => {
      resizeObserver.disconnect()
      if (chartRef.current) {
        chartRef.current.remove()
      }
    }
  }, [data, height])

  // Update data when it changes
  useEffect(() => {
    if (seriesRef.current && data) {
      seriesRef.current.setData(data)
    }
  }, [data])

  const timeframeOptions = [
    { label: '1D', value: '1D' },
    { label: '1W', value: '1W' },
    { label: '1M', value: '1M' },
    { label: '3M', value: '3M' },
    { label: '1Y', value: '1Y' },
  ]

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
          <div>
            <CardTitle className="text-lg">{title}</CardTitle>
            <p className="text-sm text-muted-foreground flex items-center">
              <Activity className="w-3 h-3 mr-1" />
              Real-time data
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Timeframe Selector */}
          <div className="flex bg-muted rounded-lg p-1">
            {timeframeOptions.map((option) => (
              <Button
                key={option.value}
                variant={timeframe === option.value ? "default" : "ghost"}
                size="sm"
                className="h-7 px-3 text-xs"
                onClick={() => setTimeframe(option.value)}
              >
                {option.label}
              </Button>
            ))}
          </div>
          
          <Button variant="outline" size="sm">
            <Maximize2 className="w-4 h-4" />
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="px-6 pb-2">
          <div className="flex items-baseline space-x-4">
            <div className="text-3xl font-bold">
              {data.length > 0 ? `$${data[data.length - 1]?.value.toLocaleString()}` : '$0'}
            </div>
            <Badge variant="outline" className="text-green-600 border-green-200">
              <TrendingUp className="w-3 h-3 mr-1" />
              +2.4% today
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Portfolio value as of {new Date().toLocaleDateString()}
          </p>
        </div>
        
        <div 
          ref={chartContainerRef} 
          className="w-full"
          style={{ height: `${height}px` }}
        />
      </CardContent>
    </Card>
  )
}

// Export with dynamic import to handle SSR
import dynamic from 'next/dynamic'

export const TradingChart = dynamic(
  () => Promise.resolve(TradingChartInner),
  { 
    ssr: false,
    loading: () => (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Portfolio Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-96 bg-muted/30 rounded-lg">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    )
  }
)

export default TradingChart 
