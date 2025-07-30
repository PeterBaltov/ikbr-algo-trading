'use client'

import { type ReactNode, useState } from 'react'
import { Sidebar } from './sidebar'
import { Button } from '@/components/ui/button'
import { Menu, User } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ThemeToggle } from '@/components/ui/theme-toggle'

interface DashboardLayoutProps {
  children: ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true) // Start with sidebar open on desktop

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <div className="flex h-screen">
        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <button
            type="button"
            className="fixed inset-0 z-40 lg:hidden bg-black/50"
            onClick={() => setSidebarOpen(false)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setSidebarOpen(false)
            }}
            aria-label="Close sidebar"
          />
        )}

        {/* Sidebar - Always visible on desktop, overlay on mobile */}
        <div className={cn(
          "inset-y-0 left-0 z-50 w-64 transform transition-transform duration-300 ease-in-out",
          "lg:relative lg:translate-x-0 lg:block", // Always visible on desktop
          "fixed", // Fixed positioning on mobile
          sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0" // Responsive behavior
        )}>
          <Sidebar />
        </div>

        {/* Main content - Pushed by sidebar on desktop */}
        <div className={cn(
          "flex flex-col flex-1 overflow-hidden transition-all duration-300 ease-in-out",
          "lg:ml-0" // Content starts after sidebar on desktop
        )}>
          {/* Top header bar */}
          <div className="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-4 lg:px-6">
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden"
            >
              <Menu className="w-5 h-5" />
            </Button>

            {/* Desktop header content */}
            <div className="hidden lg:flex items-center space-x-4">
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Trading Dashboard
              </h1>
            </div>

            {/* Mobile title */}
            <div className="text-lg font-semibold lg:hidden">MoneyTrailz</div>

            {/* Header actions */}
            <div className="flex items-center space-x-3">
              <ThemeToggle />
              <Button variant="ghost" size="sm" className="text-gray-600 dark:text-gray-400">
                <User className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Page content */}
          <main className="flex-1 relative overflow-auto bg-gray-50 dark:bg-gray-950">
            <div className="h-full p-4 sm:p-6 lg:p-8">
              <div className="max-w-7xl mx-auto">
                {children}
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}

export default DashboardLayout 
