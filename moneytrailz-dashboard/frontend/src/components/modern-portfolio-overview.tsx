'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  ChartContainer, 
  ChartTooltip, 
  ChartTooltipContent,
  ChartConfig 
} from '@/components/ui/chart'
import { 
  PieChart, 
  Pie, 
  Cell, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Users, 
  Target,
  Activity,
  Wifi,
  Calendar
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useRealtimePortfolio } from '@/hooks/useRealtimeData'

// Chart configurations
const portfolioChartConfig = {
  stocks: {
    label: "Stocks",
    color: "hsl(var(--chart-1))",
  },
  options: {
    label: "Options", 
    color: "hsl(var(--chart-2))",
  },
  crypto: {
    label: "Crypto",
    color: "hsl(var(--chart-3))",
  },
  cash: {
    label: "Cash",
    color: "hsl(var(--chart-4))",
  }
} satisfies ChartConfig

const performanceChartConfig = {
  value: {
    label: "Portfolio Value",
    color: "hsl(var(--chart-1))",
  }
} satisfies ChartConfig

// Sample data - replace with real data
const portfolioData = [
  { name: "Stocks", value: 45, fill: "hsl(var(--chart-1))" },
  { name: "Options", value: 30, fill: "hsl(var(--chart-2))" },
  { name: "Crypto", value: 15, fill: "hsl(var(--chart-3))" },
  { name: "Cash", value: 10, fill: "hsl(var(--chart-4))" }
]

const performanceData = [
  { date: "Jan", value: 120000 },
  { date: "Feb", value: 122000 },
  { date: "Mar", value: 118000 },
  { date: "Apr", value: 125000 },
  { date: "May", value: 128000 },
  { date: "Jun", value: 125450 }
]

const activeUsersData = [
  { day: "MON", users: 800 },
  { day: "TUE", users: 1200 },
  { day: "WED", users: 1150 },
  { day: "THU", users: 1300 },
  { day: "FRI", users: 1250 },
  { day: "SAT", users: 900 },
  { day: "SUN", users: 1100 }
]

interface MetricCardProps {
  title: string
  value: string
  change?: string
  icon: React.ReactNode
  trend?: 'up' | 'down' | 'neutral'
  className?: string
}

function MetricCard({ title, value, change, icon, trend, className }: MetricCardProps) {
  return (
    <Card className={cn("transition-all hover:shadow-md", className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              {icon}
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">{title}</p>
              <p className="text-2xl font-bold">{value}</p>
            </div>
          </div>
          {change && (
            <div className={cn(
              "flex items-center space-x-1 text-sm font-medium",
              trend === 'up' && "text-green-600",
              trend === 'down' && "text-red-600",
              trend === 'neutral' && "text-muted-foreground"
            )}>
              {trend === 'up' && <TrendingUp className="w-4 h-4" />}
              {trend === 'down' && <TrendingDown className="w-4 h-4" />}
              <span>{change}</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export function ModernPortfolioOverview() {
  const { portfolio, isConnected } = useRealtimePortfolio()

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Portfolio Overview</h1>
          <p className="text-muted-foreground">
            Monitor your algorithmic trading performance
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <Badge variant={isConnected ? "default" : "destructive"} className="flex items-center space-x-1">
            <Wifi className="w-3 h-3" />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </Badge>
          <Badge variant="outline" className="flex items-center space-x-1">
            <Activity className="w-3 h-3" />
            <span>Live</span>
          </Badge>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Portfolio Value"
          value={portfolio ? `$${portfolio.totalValue.toLocaleString()}` : "$125,450"}
          change="+1.2%"
          trend="up"
          icon={<DollarSign className="w-5 h-5 text-primary" />}
        />
        <MetricCard
          title="Today's P&L"
          value={portfolio ? `$${portfolio.dayPnL.toLocaleString()}` : "$1,250"}
          change="+0.8%"
          trend="up"
          icon={<TrendingUp className="w-5 h-5 text-green-600" />}
        />
        <MetricCard
          title="Win Rate"
          value="73.2%"
          change="+2.1%"
          trend="up"
          icon={<Target className="w-5 h-5 text-blue-600" />}
        />
        <MetricCard
          title="Active Strategies"
          value="2/10"
          change="20.0%"
          trend="neutral"
          icon={<Users className="w-5 h-5 text-purple-600" />}
        />
      </div>

      {/* Charts Section */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-7">
        {/* Portfolio Allocation Chart */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Portfolio Allocation</span>
              <Badge variant="secondary">Updated</Badge>
            </CardTitle>
            <CardDescription>
              Current asset allocation breakdown
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={portfolioChartConfig}
              className="mx-auto aspect-square max-h-[250px]"
            >
              <PieChart>
                <ChartTooltip
                  cursor={false}
                  content={<ChartTooltipContent hideLabel />}
                />
                <Pie
                  data={portfolioData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={60}
                  strokeWidth={5}
                >
                  {portfolioData.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Pie>
              </PieChart>
            </ChartContainer>
            
            {/* Legend */}
            <div className="grid grid-cols-2 gap-2 mt-4">
              {portfolioData.map((item) => (
                <div key={item.name} className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.fill }}
                  />
                  <span className="text-sm text-muted-foreground">
                    {item.name}: {item.value}%
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Performance Chart */}
        <Card className="lg:col-span-4">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Performance Trend</span>
              <div className="flex items-center space-x-2">
                <Calendar className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Last 6 months</span>
              </div>
            </CardTitle>
            <CardDescription>
              Portfolio value over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={performanceChartConfig} className="h-[200px]">
              <AreaChart data={performanceData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="date" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-background border rounded-lg shadow-lg p-3">
                          <p className="font-medium">{label}</p>
                          <p className="text-sm text-primary">
                            Value: ${payload[0].value?.toLocaleString()}
                          </p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="hsl(var(--chart-1))"
                  fill="url(#colorValue)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Additional Metrics Grid */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Cash Balance */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Cash Balance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$15,680.45</div>
            <p className="text-xs text-muted-foreground">Available for trading</p>
          </CardContent>
        </Card>

        {/* Margin Used */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Margin Used</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$12,000.00</div>
            <p className="text-xs text-muted-foreground">12.6% of buying power</p>
          </CardContent>
        </Card>

        {/* Buying Power */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Buying Power</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$95,000.00</div>
            <p className="text-xs text-muted-foreground">Available to trade</p>
          </CardContent>
        </Card>
      </div>

      {/* Activity Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Trading Activity</CardTitle>
          <CardDescription>
            Active users and trading volume this week
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={{ users: { label: "Active Users", color: "hsl(var(--chart-2))" } }} className="h-[200px]">
            <LineChart data={activeUsersData}>
              <XAxis 
                dataKey="day" 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12 }}
              />
              <ChartTooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-background border rounded-lg shadow-lg p-3">
                        <p className="font-medium">{label}</p>
                        <p className="text-sm text-primary">
                          Users: {payload[0].value?.toLocaleString()}
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Line
                type="monotone"
                dataKey="users"
                stroke="hsl(var(--chart-2))"
                strokeWidth={3}
                dot={{ fill: "hsl(var(--chart-2))", strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: "hsl(var(--chart-2))", strokeWidth: 2 }}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
}

export default ModernPortfolioOverview 
