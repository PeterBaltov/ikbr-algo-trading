'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { PositionSummary } from '@/components/ui/portfolio-summary-card'
import { TradingTable, TradingTableHeader, TradingTableRow, TradingTableCell, DataCell } from '@/components/ui/trading-table'
import { StatusIndicator, StatusBadge } from '@/components/ui/status-indicator'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Activity, Play, Pause, Settings } from 'lucide-react'
import { cn } from '@/lib/utils'

// Professional dashboard layout inspired by Interactive Brokers

interface Position {
  symbol: string
  quantity: number
  avgCost: number
  marketPrice: number
  marketValue: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  dayChange: number
  dayChangePercent: number
}

interface Strategy {
  name: string
  status: 'working' | 'paused' | 'error'
  symbol: string
  dayPnL: number
  totalPnL: number
  positions: number
  lastAction: string
  lastUpdate: string
}

interface RecentTrade {
  id: string
  time: string
  symbol: string
  action: 'BUY' | 'SELL'
  quantity: number
  price: number
  status: 'filled' | 'pending' | 'partial'
}

export function ProfessionalDashboard() {
  // Demo data - in real app, this would come from props/hooks
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

  const positions: Position[] = [
    {
      symbol: 'AAPL',
      quantity: 100,
      avgCost: 150.25,
      marketPrice: 152.30,
      marketValue: 15230.00,
      unrealizedPnL: 205.00,
      unrealizedPnLPercent: 1.36,
      dayChange: 2.10,
      dayChangePercent: 1.40
    },
    {
      symbol: 'SPY',
      quantity: 50,
      avgCost: 420.80,
      marketPrice: 418.65,
      marketValue: 20932.50,
      unrealizedPnL: -107.50,
      unrealizedPnLPercent: -0.51,
      dayChange: -1.25,
      dayChangePercent: -0.30
    },
    {
      symbol: 'QQQ',
      quantity: 75,
      avgCost: 365.40,
      marketPrice: 368.90,
      marketValue: 27667.50,
      unrealizedPnL: 262.50,
      unrealizedPnLPercent: 0.96,
      dayChange: 1.85,
      dayChangePercent: 0.50
    }
  ]

  const strategies: Strategy[] = [
    {
      name: 'Wheel Strategy',
      status: 'working',
      symbol: 'AAPL',
      dayPnL: 125.50,
      totalPnL: 2840.25,
      positions: 2,
      lastAction: 'Sold PUT',
      lastUpdate: '10:30 AM'
    },
    {
      name: 'Iron Condor',
      status: 'paused',
      symbol: 'SPY',
      dayPnL: -45.20,
      totalPnL: 1250.80,
      positions: 4,
      lastAction: 'Adjusted spread',
      lastUpdate: '9:45 AM'
    },
    {
      name: 'Covered Call',
      status: 'working',
      symbol: 'QQQ',
      dayPnL: 85.30,
      totalPnL: 520.15,
      positions: 1,
      lastAction: 'Sold CALL',
      lastUpdate: '11:15 AM'
    }
  ]

  const recentTrades: RecentTrade[] = [
    {
      id: '1',
      time: '11:30:25',
      symbol: 'AAPL',
      action: 'SELL',
      quantity: 1,
      price: 3.45,
      status: 'filled'
    },
    {
      id: '2',
      time: '10:45:12',
      symbol: 'SPY',
      action: 'BUY',
      quantity: 2,
      price: 2.80,
      status: 'filled'
    },
    {
      id: '3',
      time: '09:30:08',
      symbol: 'QQQ',
      action: 'SELL',
      quantity: 1,
      price: 4.20,
      status: 'partial'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <PositionSummary {...portfolioData} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Positions Table */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Current Positions</span>
                <Badge variant="outline" className="text-xs">
                  {positions.length} positions
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Mobile-friendly table */}
              <div className="block lg:hidden space-y-4">
                {positions.map((position) => (
                  <div 
                    key={position.symbol}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 space-y-3"
                  >
                    <div className="flex justify-between items-center">
                      <h4 className="font-semibold text-lg">{position.symbol}</h4>
                      <Badge variant="outline" className="text-xs">
                        {position.quantity.toLocaleString()} shares
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Avg Cost:</span>
                        <div className="font-medium">${position.avgCost.toFixed(2)}</div>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Market Price:</span>
                        <div className="font-medium">${position.marketPrice.toFixed(2)}</div>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Market Value:</span>
                        <div className="font-medium">${position.marketValue.toLocaleString()}</div>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Unrealized P&L:</span>
                        <div className={cn(
                          "font-medium",
                          position.unrealizedPnL >= 0 ? "text-profit" : "text-loss"
                        )}>
                          ${position.unrealizedPnL.toFixed(2)} ({position.unrealizedPnLPercent.toFixed(2)}%)
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Desktop table */}
              <div className="hidden lg:block">
                <TradingTable>
                  <TradingTableHeader>
                    <tr>
                      <th className="px-3 py-2 text-left">Symbol</th>
                      <th className="px-3 py-2 text-right">Qty</th>
                      <th className="px-3 py-2 text-right">Avg Cost</th>
                      <th className="px-3 py-2 text-right">Market Price</th>
                      <th className="px-3 py-2 text-right">Market Value</th>
                      <th className="px-3 py-2 text-right">Unrealized P&L</th>
                      <th className="px-3 py-2 text-right">Day Change</th>
                    </tr>
                  </TradingTableHeader>
                  <tbody>
                    {positions.map((position) => (
                      <TradingTableRow key={position.symbol}>
                        <TradingTableCell className="font-medium">
                          {position.symbol}
                        </TradingTableCell>
                        <TradingTableCell align="right">
                          {position.quantity.toLocaleString()}
                        </TradingTableCell>
                        <DataCell 
                          value={position.avgCost}
                          type="currency"
                        />
                        <DataCell 
                          value={position.marketPrice}
                          type="currency"
                        />
                        <DataCell 
                          value={position.marketValue}
                          type="currency"
                        />
                        <TradingTableCell align="right">
                          <div className="space-y-1">
                            <div className={cn(
                              "font-semibold",
                              position.unrealizedPnL >= 0 ? "text-profit" : "text-loss"
                            )}>
                              ${position.unrealizedPnL.toFixed(2)}
                            </div>
                            <div className={cn(
                              "text-xs font-semibold",
                              position.unrealizedPnL >= 0 ? "text-profit" : "text-loss"
                            )}>
                              {position.unrealizedPnLPercent.toFixed(2)}%
                            </div>
                          </div>
                        </TradingTableCell>
                        <TradingTableCell align="right">
                          <div className="space-y-1">
                            <div className={cn(
                              "font-semibold",
                              position.dayChange >= 0 ? "text-profit" : "text-loss"
                            )}>
                              ${position.dayChange.toFixed(2)}
                            </div>
                            <div className={cn(
                              "text-xs font-semibold",
                              position.dayChange >= 0 ? "text-profit" : "text-loss"
                            )}>
                              {position.dayChangePercent.toFixed(2)}%
                            </div>
                          </div>
                        </TradingTableCell>
                      </TradingTableRow>
                    ))}
                  </tbody>
                </TradingTable>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Strategy Monitor */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Active Strategies</span>
                <Button variant="outline" size="sm">
                  <Settings className="w-4 h-4" />
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {strategies.map((strategy) => (
                <div 
                  key={strategy.name}
                  className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-sm">{strategy.name}</h4>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {strategy.symbol}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <StatusIndicator 
                        status={strategy.status} 
                        size="sm"
                      />
                      <Button 
                        variant="ghost" 
                        size="sm"
                        className="h-6 w-6 p-0"
                      >
                        {strategy.status === 'working' ? (
                          <Pause className="w-3 h-3" />
                        ) : (
                          <Play className="w-3 h-3" />
                        )}
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Day P&L:</span>
                      <div className={cn(
                        "font-medium",
                        strategy.dayPnL >= 0 ? "text-profit" : "text-loss"
                      )}>
                        ${strategy.dayPnL.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Total P&L:</span>
                      <div className={cn(
                        "font-medium",
                        strategy.totalPnL >= 0 ? "text-profit" : "text-loss"
                      )}>
                        ${strategy.totalPnL.toFixed(2)}
                      </div>
                    </div>
                  </div>

                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    <div>Last: {strategy.lastAction}</div>
                    <div>Updated: {strategy.lastUpdate}</div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Recent Trades */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Recent Trades
          </CardTitle>
        </CardHeader>
        <CardContent>
          <TradingTable>
            <TradingTableHeader>
              <tr>
                <th className="px-3 py-2 text-left">Time</th>
                <th className="px-3 py-2 text-left">Symbol</th>
                <th className="px-3 py-2 text-center">Action</th>
                <th className="px-3 py-2 text-right">Quantity</th>
                <th className="px-3 py-2 text-right">Price</th>
                <th className="px-3 py-2 text-center">Status</th>
              </tr>
            </TradingTableHeader>
            <tbody>
              {recentTrades.map((trade) => (
                <TradingTableRow key={trade.id}>
                  <TradingTableCell className="font-mono text-xs">
                    {trade.time}
                  </TradingTableCell>
                  <TradingTableCell className="font-medium">
                    {trade.symbol}
                  </TradingTableCell>
                  <TradingTableCell 
                    type={trade.action === 'BUY' ? 'buy' : 'sell'}
                    align="center"
                  >
                    {trade.action}
                  </TradingTableCell>
                  <TradingTableCell align="right">
                    {trade.quantity.toLocaleString()}
                  </TradingTableCell>
                  <DataCell 
                    value={trade.price}
                    type="currency"
                  />
                  <TradingTableCell align="center">
                    <StatusBadge status={trade.status} size="sm">
                      {trade.status.toUpperCase()}
                    </StatusBadge>
                  </TradingTableCell>
                </TradingTableRow>
              ))}
            </tbody>
          </TradingTable>
        </CardContent>
      </Card>
    </div>
  )
}
