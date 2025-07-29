'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { useStrategyControls } from '@/hooks/useStrategyControls'
import { Strategy } from '@/types'
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  Loader2
} from 'lucide-react'
import { cn } from '@/lib/utils'
import StrategyConfigPanel from './strategy-config'

interface StrategyControlsProps {
  strategy: Strategy
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

export function StrategyControls({ strategy, className, size = 'md' }: StrategyControlsProps) {
  const { pauseStrategy, resumeStrategy, stopStrategy, isLoading } = useStrategyControls()
  const [showConfig, setShowConfig] = useState(false)
  const [operationLoading, setOperationLoading] = useState<string | null>(null)

  const handlePause = async () => {
    setOperationLoading('pause')
    try {
      await pauseStrategy(strategy.name)
    } catch (error) {
      console.error('Failed to pause strategy:', error)
    } finally {
      setOperationLoading(null)
    }
  }

  const handleResume = async () => {
    setOperationLoading('resume')
    try {
      await resumeStrategy(strategy.name)
    } catch (error) {
      console.error('Failed to resume strategy:', error)
    } finally {
      setOperationLoading(null)
    }
  }

  const handleStop = async () => {
    if (!confirm(`Are you sure you want to stop "${strategy.name}"? This will close all positions.`)) {
      return
    }

    setOperationLoading('stop')
    try {
      await stopStrategy(strategy.name)
    } catch (error) {
      console.error('Failed to stop strategy:', error)
    } finally {
      setOperationLoading(null)
    }
  }

  const openConfigModal = () => {
    setShowConfig(true)
  }

  const getStatusIcon = () => {
    switch (strategy.status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'paused':
        return <Clock className="w-4 h-4 text-yellow-500" />
      case 'stopped':
        return <Square className="w-4 h-4 text-gray-500" />
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      default:
        return null
    }
  }

  const getStatusColor = () => {
    switch (strategy.status) {
      case 'active':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'paused':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'stopped':
        return 'text-gray-600 bg-gray-50 border-gray-200'
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const buttonSizes = {
    sm: 'h-8 px-2 text-xs',
    md: 'h-9 px-3 text-sm',
    lg: 'h-10 px-4 text-base'
  }

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  }

  return (
    <>
      <div className={cn("flex items-center space-x-2", className)}>
        {/* Status Indicator */}
        <div className={cn(
          "flex items-center space-x-1 px-2 py-1 rounded-full border text-xs font-medium",
          getStatusColor()
        )}>
          {getStatusIcon()}
          <span className="capitalize">{strategy.status}</span>
        </div>

        {/* Control Buttons */}
        <div className="flex space-x-1">
          {strategy.status === 'active' && (
            <Button
              variant="outline"
              size="sm"
              onClick={handlePause}
              disabled={isLoading || operationLoading === 'pause'}
              className={cn(
                "border-yellow-300 text-yellow-700 hover:bg-yellow-50",
                buttonSizes[size]
              )}
            >
              {operationLoading === 'pause' ? (
                <Loader2 className={cn("animate-spin", iconSizes[size])} />
              ) : (
                <Pause className={iconSizes[size]} />
              )}
              {size !== 'sm' && <span className="ml-1">Pause</span>}
            </Button>
          )}

          {strategy.status === 'paused' && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleResume}
              disabled={isLoading || operationLoading === 'resume'}
              className={cn(
                "border-green-300 text-green-700 hover:bg-green-50",
                buttonSizes[size]
              )}
            >
              {operationLoading === 'resume' ? (
                <Loader2 className={cn("animate-spin", iconSizes[size])} />
              ) : (
                <Play className={iconSizes[size]} />
              )}
              {size !== 'sm' && <span className="ml-1">Resume</span>}
            </Button>
          )}

          {(strategy.status === 'active' || strategy.status === 'paused') && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleStop}
              disabled={isLoading || operationLoading === 'stop'}
              className={cn(
                "border-red-300 text-red-700 hover:bg-red-50",
                buttonSizes[size]
              )}
            >
              {operationLoading === 'stop' ? (
                <Loader2 className={cn("animate-spin", iconSizes[size])} />
              ) : (
                <Square className={iconSizes[size]} />
              )}
              {size !== 'sm' && <span className="ml-1">Stop</span>}
            </Button>
          )}

          <Button
            variant="outline"
            size="sm"
            onClick={openConfigModal}
            className={cn(
              "border-blue-300 text-blue-700 hover:bg-blue-50",
              buttonSizes[size]
            )}
          >
            <Settings className={iconSizes[size]} />
            {size !== 'sm' && <span className="ml-1">Configure</span>}
          </Button>
        </div>
      </div>

      {/* Configuration Modal */}
      <StrategyConfigPanel
        strategyName={strategy.name}
        isOpen={showConfig}
        onClose={() => setShowConfig(false)}
      />
    </>
  )
}

export default StrategyControls 
