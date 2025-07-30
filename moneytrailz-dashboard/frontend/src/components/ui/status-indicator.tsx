'use client'

import React from 'react'
import { cn } from '@/lib/utils'

// Professional status indicators inspired by Interactive Brokers

export type StatusType = 
  | 'working'     // Green - Order/Strategy is active and working
  | 'pending'     // Light Blue - Transmitted but not confirmed
  | 'cancelled'   // Red - Cancelled or rejected
  | 'filled'      // Dark Blue - Completed/Filled
  | 'paused'      // Yellow - Paused or waiting
  | 'error'       // Red - Error state
  | 'simulated'   // Purple - Simulated order
  | 'partial'     // Light Green - Partially filled

interface StatusIndicatorProps {
  status: StatusType
  label?: string
  size?: 'sm' | 'md' | 'lg'
  showDot?: boolean
  className?: string
}

interface StatusBadgeProps {
  status: StatusType
  children: React.ReactNode
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const statusConfig = {
  working: {
    color: 'bg-green-500',
    textColor: 'text-green-700 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    borderColor: 'border-green-200 dark:border-green-800',
    label: 'Working'
  },
  pending: {
    color: 'bg-blue-400',
    textColor: 'text-blue-700 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    borderColor: 'border-blue-200 dark:border-blue-800',
    label: 'Pending'
  },
  cancelled: {
    color: 'bg-red-500',
    textColor: 'text-red-700 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    borderColor: 'border-red-200 dark:border-red-800',
    label: 'Cancelled'
  },
  filled: {
    color: 'bg-blue-700',
    textColor: 'text-blue-800 dark:text-blue-300',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    borderColor: 'border-blue-200 dark:border-blue-800',
    label: 'Filled'
  },
  paused: {
    color: 'bg-yellow-500',
    textColor: 'text-yellow-800 dark:text-yellow-400',
    bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
    borderColor: 'border-yellow-200 dark:border-yellow-800',
    label: 'Paused'
  },
  error: {
    color: 'bg-red-600',
    textColor: 'text-red-700 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    borderColor: 'border-red-200 dark:border-red-800',
    label: 'Error'
  },
  simulated: {
    color: 'bg-purple-500',
    textColor: 'text-purple-700 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    borderColor: 'border-purple-200 dark:border-purple-800',
    label: 'Simulated'
  },
  partial: {
    color: 'bg-green-400',
    textColor: 'text-green-700 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    borderColor: 'border-green-200 dark:border-green-800',
    label: 'Partial'
  }
}

export function StatusIndicator({ 
  status, 
  label, 
  size = 'md', 
  showDot = true,
  className 
}: StatusIndicatorProps) {
  const config = statusConfig[status]
  const displayLabel = label || config.label

  const sizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  }

  const dotSizes = {
    sm: 'w-2 h-2',
    md: 'w-2.5 h-2.5',
    lg: 'w-3 h-3'
  }

  return (
    <div className={cn(
      "inline-flex items-center gap-2",
      sizeClasses[size],
      config.textColor,
      className
    )}>
      {showDot && (
        <div className={cn(
          "rounded-full flex-shrink-0",
          dotSizes[size],
          config.color
        )} />
      )}
      <span className="font-medium">{displayLabel}</span>
    </div>
  )
}

export function StatusBadge({ 
  status, 
  children, 
  size = 'md',
  className 
}: StatusBadgeProps) {
  const config = statusConfig[status]

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  }

  return (
    <div className={cn(
      "inline-flex items-center rounded-full border font-medium",
      sizeClasses[size],
      config.textColor,
      config.bgColor,
      config.borderColor,
      className
    )}>
      {children}
    </div>
  )
}

// Progress status for orders with fill progress
interface ProgressStatusProps {
  filled: number
  total: number
  status: StatusType
  className?: string
}

export function ProgressStatus({ filled, total, status, className }: ProgressStatusProps) {
  const percentage = total > 0 ? (filled / total) * 100 : 0
  const config = statusConfig[status]

  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex justify-between items-center text-xs">
        <StatusIndicator status={status} size="sm" />
        <span className="text-gray-600 dark:text-gray-400">
          {filled.toLocaleString()} / {total.toLocaleString()}
        </span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
        <div 
          className={cn("h-1.5 rounded-full transition-all duration-300", config.color)}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400 text-right">
        {percentage.toFixed(1)}%
      </div>
    </div>
  )
}
