import DashboardLayout from "@/components/layout/dashboard-layout"
import { ProfessionalDashboard } from "@/components/professional-dashboard"
import TradingChart from "@/components/trading-chart"

// Sample performance data for the TradingView chart
const performanceData = [
  { time: '2024-01-01', value: 120000 },
  { time: '2024-01-02', value: 121500 },
  { time: '2024-01-03', value: 119800 },
  { time: '2024-01-04', value: 122300 },
  { time: '2024-01-05', value: 123100 },
  { time: '2024-01-08', value: 121900 },
  { time: '2024-01-09', value: 124200 },
  { time: '2024-01-10', value: 125450 },
  { time: '2024-01-11', value: 126800 },
  { time: '2024-01-12', value: 125200 },
  { time: '2024-01-15', value: 127300 },
  { time: '2024-01-16', value: 128900 },
  { time: '2024-01-17', value: 127600 },
  { time: '2024-01-18', value: 129400 },
  { time: '2024-01-19', value: 131200 }
]

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Professional Dashboard with IB-inspired design */}
        <ProfessionalDashboard />

        {/* Advanced Trading Chart */}
        <section>
          <TradingChart 
            data={performanceData}
            title="Portfolio Performance"
            height={400}
          />
        </section>
      </div>
    </DashboardLayout>
  )
}
