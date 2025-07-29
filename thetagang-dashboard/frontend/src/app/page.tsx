import { PortfolioOverview, ConnectionStatus } from "@/components/portfolio-overview"
import { StrategyMonitor } from "@/components/strategy-monitor"
import { PerformanceChart } from "@/components/performance-chart"

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-foreground">
                🎯 ThetaGang Dashboard
              </h1>
              <div className="text-sm text-muted-foreground">
                Algorithmic Trading System
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <ConnectionStatus isConnected={true} />
              <div className="text-sm text-muted-foreground">
                {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="space-y-8">
          {/* Portfolio Overview Section */}
          <section>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-semibold text-foreground">
                  Portfolio Overview
                </h2>
                <p className="text-sm text-muted-foreground">
                  Real-time portfolio performance and key metrics
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-muted-foreground">Live</span>
              </div>
            </div>
            
            <PortfolioOverview />
          </section>

          {/* Strategy Monitor Section */}
          <section>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-semibold text-foreground">
                  Strategy Monitor
                </h2>
                <p className="text-sm text-muted-foreground">
                  Active trading strategies and their performance
                </p>
              </div>
            </div>
            
            <StrategyMonitor />
          </section>

          {/* Performance Chart Section */}
          <section>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-semibold text-foreground">
                  Performance Chart
                </h2>
                <p className="text-sm text-muted-foreground">
                  Portfolio value over time
                </p>
              </div>
            </div>
            
            <PerformanceChart />
          </section>
        </div>
      </main>
    </div>
  )
}
