import { PortfolioOverview, ConnectionStatus } from "@/components/portfolio-overview"
import { Portfolio } from "@/types"

// Mock data for development
const mockPortfolio: Portfolio = {
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
            
            <PortfolioOverview 
              portfolio={mockPortfolio} 
              isLoading={false}
            />
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
            
            <div className="grid gap-4">
              <div className="bg-card border rounded-lg p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <div>
                      <h3 className="font-semibold">Enhanced Wheel Strategy</h3>
                      <p className="text-sm text-muted-foreground">Options • Active</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-profit">+$2,340.50</div>
                    <div className="text-sm text-muted-foreground">+15.2% allocation</div>
                  </div>
                </div>
              </div>
              
              <div className="bg-card border rounded-lg p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div>
                      <h3 className="font-semibold">Momentum Scalper</h3>
                      <p className="text-sm text-muted-foreground">Stocks • Monitoring</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-profit">+$890.25</div>
                    <div className="text-sm text-muted-foreground">+8.7% allocation</div>
                  </div>
                </div>
              </div>
            </div>
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
            
            <div className="bg-card border rounded-lg p-6 h-96 flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <div className="text-4xl mb-4">📈</div>
                <div className="text-lg font-semibold mb-2">Performance Chart</div>
                <div className="text-sm">Chart component will be implemented in Phase 2</div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}
