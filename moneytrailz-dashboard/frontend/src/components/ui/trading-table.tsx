'use client'

import React from 'react'
import { cn } from '@/lib/utils'

// Professional trading table inspired by Interactive Brokers

interface TradingTableProps {
  className?: string
  children: React.ReactNode
}

interface TradingTableHeaderProps {
  className?: string
  children: React.ReactNode
}

interface TradingTableRowProps {
  className?: string
  children: React.ReactNode
  selected?: boolean
  onClick?: () => void
}

interface TradingTableCellProps {
  className?: string
  children: React.ReactNode
  type?: 'default' | 'bid' | 'ask' | 'profit' | 'loss' | 'neutral' | 'buy' | 'sell' | 'last'
  align?: 'left' | 'center' | 'right'
}

export function TradingTable({ className, children }: TradingTableProps) {
  return (
    <div className={cn(
      "relative overflow-auto border border-gray-200 dark:border-gray-700 rounded-lg",
      "bg-white dark:bg-gray-900",
      className
    )}>
      <table className="w-full text-sm">
        {children}
      </table>
    </div>
  )
}

export function TradingTableHeader({ className, children }: TradingTableHeaderProps) {
  return (
    <thead className={cn(
      "bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700",
      "text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider",
      className
    )}>
      {children}
    </thead>
  )
}

export function TradingTableRow({ className, children, selected, onClick }: TradingTableRowProps) {
  return (
    <tr 
      className={cn(
        "border-b border-gray-100 dark:border-gray-800 transition-colors",
        "hover:bg-gray-50 dark:hover:bg-gray-800/50",
        selected && "bg-gray-100 dark:bg-gray-800",
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {children}
    </tr>
  )
}

export function TradingTableCell({ 
  className, 
  children, 
  type = 'default',
  align = 'left'
}: TradingTableCellProps) {
  const getTypeStyles = () => {
    switch (type) {
      case 'bid':
        return 'bg-neutral text-neutral-foreground font-medium'
      case 'ask':
        return 'bg-ask text-ask-foreground font-medium'
      case 'profit':
        return 'text-profit font-semibold'
      case 'loss':
        return 'text-loss font-semibold'
      case 'neutral':
        return 'text-neutral-dark font-medium'
      case 'buy':
        return 'bg-buy text-buy-foreground font-medium'
      case 'sell':
        return 'bg-sell text-sell-foreground font-medium'
      case 'last':
        return 'bg-trading-last text-white font-medium'
      default:
        return 'text-gray-900 dark:text-gray-100'
    }
  }

  const getAlignment = () => {
    switch (align) {
      case 'center':
        return 'text-center'
      case 'right':
        return 'text-right'
      default:
        return 'text-left'
    }
  }

  return (
    <td className={cn(
      "px-3 py-2 whitespace-nowrap",
      getTypeStyles(),
      getAlignment(),
      className
    )}>
      {children}
    </td>
  )
}

// Professional data cell with automatic formatting
interface DataCellProps {
  value: number | string
  type?: 'currency' | 'percentage' | 'number' | 'text'
  precision?: number
  showSign?: boolean
  className?: string
}

export function DataCell({ 
  value, 
  type = 'text', 
  precision = 2, 
  showSign = false,
  className 
}: DataCellProps) {
  const formatValue = () => {
    if (typeof value === 'string') return value
    
    const numValue = Number(value)
    
    switch (type) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(numValue)
      
      case 'percentage':
        return new Intl.NumberFormat('en-US', {
          style: 'percent',
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(numValue / 100)
      
      case 'number':
        const formatted = new Intl.NumberFormat('en-US', {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(numValue)
        return showSign && numValue > 0 ? `+${formatted}` : formatted
      
      default:
        return String(value)
    }
  }

  const getCellType = () => {
    if (typeof value === 'string') return 'default'
    
    const numValue = Number(value)
    if (numValue > 0) return 'profit'
    if (numValue < 0) return 'loss'
    return 'neutral'
  }

  return (
    <TradingTableCell 
      type={type === 'currency' || type === 'number' || type === 'percentage' ? getCellType() : 'default'}
      align="right"
      className={className}
    >
      {formatValue()}
    </TradingTableCell>
  )
}
