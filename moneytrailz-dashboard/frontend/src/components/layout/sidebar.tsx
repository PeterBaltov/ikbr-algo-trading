'use client'

import { type ComponentType, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  BarChart3,
  TrendingUp,
  Settings,
  User,
  PieChart,
  Activity,
  Wallet,
  Target,
  ChevronLeft,
  ChevronRight,
  Home,
  LineChart
} from 'lucide-react'

interface SidebarProps {
  className?: string
}

interface NavigationItem {
  name: string
  href: string
  icon: ComponentType<{ className?: string }>
  current?: boolean
}

export function Sidebar({ className }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)
  const pathname = usePathname()

  const navigation: NavigationItem[] = [
    {
      name: 'Overview',
      href: '/',
      icon: Home,
      current: pathname === '/'
    },
    {
      name: 'Portfolio',
      href: '/portfolio',
      icon: PieChart,
      current: pathname === '/portfolio'
    },
    {
      name: 'Strategies',
      href: '/strategies',
      icon: Target,
      current: pathname === '/strategies'
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: BarChart3,
      current: pathname === '/analytics'
    },
    {
      name: 'Performance',
      href: '/performance',
      icon: TrendingUp,
      current: pathname === '/performance'
    },
    {
      name: 'Trades',
      href: '/trades',
      icon: Activity,
      current: pathname === '/trades'
    },
    {
      name: 'Balance',
      href: '/balance',
      icon: Wallet,
      current: pathname === '/balance'
    }
  ]

  const secondaryNavigation = [
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
      current: pathname === '/settings'
    },
    {
      name: 'Account',
      href: '/account',
      icon: User,
      current: pathname === '/account'
    }
  ]

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transition-all duration-300",
        collapsed ? "w-16" : "w-64",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
        <div className={cn("flex items-center space-x-3", collapsed && "justify-center")}>
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <LineChart className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <div>
              <h1 className="text-lg font-bold text-gray-900 dark:text-white">MoneyTrailz</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Trading Dashboard</p>
            </div>
          )}
        </div>
        
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Account Summary - Similar to Interactive Brokers */}
      {!collapsed && (
        <div className="p-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Account</span>
              <span className="text-xs text-gray-500 dark:text-gray-500">••••5432</span>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-xs text-gray-600 dark:text-gray-400">Total Value</span>
                <span className="text-xs font-medium text-gray-900 dark:text-white">$125,450.23</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-600 dark:text-gray-400">Available to Trade</span>
                <span className="text-xs font-medium text-gray-900 dark:text-white">$95,000.00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-600 dark:text-gray-400">Day P&L</span>
                <span className="text-xs font-medium text-green-600 dark:text-green-400">+$1,250.75</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-600 dark:text-gray-400">Unrealized P&L</span>
                <span className="text-xs font-medium text-green-600 dark:text-green-400">+$3,420.18</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <div className="space-y-1">
          {navigation.map((item) => {
            const Icon = item.icon
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200",
                  item.current
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20"
                    : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white"
                )}
              >
                <Icon 
                  className={cn(
                    "flex-shrink-0 w-5 h-5 transition-colors",
                    item.current ? "text-white" : "text-gray-500 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white"
                  )} 
                />
                {!collapsed && (
                  <span className="ml-3 truncate">{item.name}</span>
                )}
                {collapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 dark:bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                    {item.name}
                  </div>
                )}
              </Link>
            )
          })}
        </div>

        {/* Separator */}
        <div className="border-t border-gray-200 dark:border-gray-800 my-4" />

        {/* Secondary Navigation */}
        <div className="space-y-1">
          {secondaryNavigation.map((item) => {
            const Icon = item.icon
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200",
                  item.current
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20"
                    : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white"
                )}
              >
                <Icon 
                  className={cn(
                    "flex-shrink-0 w-5 h-5 transition-colors",
                    item.current ? "text-white" : "text-gray-500 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white"
                  )} 
                />
                {!collapsed && (
                  <span className="ml-3 truncate">{item.name}</span>
                )}
                {collapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 dark:bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                    {item.name}
                  </div>
                )}
              </Link>
            )
          })}
        </div>
      </nav>

      {/* User Profile Section */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-800">
        <div className={cn(
          "flex items-center px-3 py-2.5 rounded-lg bg-gray-100 dark:bg-gray-800/50",
          collapsed ? "justify-center" : "space-x-3"
        )}>
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <span className="text-white text-sm font-medium">MT</span>
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">MoneyTrailz</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 truncate">Algorithmic Trader</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Sidebar 
