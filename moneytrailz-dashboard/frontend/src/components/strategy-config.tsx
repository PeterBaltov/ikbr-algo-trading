'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useStrategyConfig } from '@/hooks/useStrategyConfig'
import { cn } from '@/lib/utils'
import { Save, X, AlertTriangle, CheckCircle } from 'lucide-react'

interface StrategyConfigPanelProps {
  strategyName: string
  isOpen: boolean
  onClose: () => void
}

interface StrategyConfig {
  allocation: number
  maxPositionSize: number
  stopLoss: number
  takeProfit: number
  symbols: string[]
  timeframes: string[]
  riskParameters: {
    maxDrawdown: number
    maxDailyLoss: number
    correlationLimit: number
  }
  enabled: boolean
}

export function StrategyConfigPanel({ strategyName, isOpen, onClose }: StrategyConfigPanelProps) {
  const { config, updateConfig, isLoading, error, saveConfig } = useStrategyConfig(strategyName)
  const [localConfig, setLocalConfig] = useState<StrategyConfig | null>(null)
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    if (config) {
      setLocalConfig(config)
      setHasChanges(false)
    }
  }, [config])

  const validateForm = (): boolean => {
    const errors: Record<string, string> = {}

    if (!localConfig) return false

    if (localConfig.allocation < 0 || localConfig.allocation > 100) {
      errors.allocation = 'Allocation must be between 0 and 100%'
    }

    if (localConfig.maxPositionSize <= 0) {
      errors.maxPositionSize = 'Max position size must be greater than 0'
    }

    if (localConfig.stopLoss < 0 || localConfig.stopLoss > 100) {
      errors.stopLoss = 'Stop loss must be between 0 and 100%'
    }

    if (localConfig.symbols.length === 0) {
      errors.symbols = 'At least one symbol must be selected'
    }

    if (localConfig.riskParameters.maxDrawdown < 0 || localConfig.riskParameters.maxDrawdown > 100) {
      errors.maxDrawdown = 'Max drawdown must be between 0 and 100%'
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleInputChange = (field: string, value: any) => {
    if (!localConfig) return

    setLocalConfig(prev => ({
      ...prev!,
      [field]: value
    }))
    setHasChanges(true)
  }

  const handleRiskParameterChange = (field: string, value: number) => {
    if (!localConfig) return

    setLocalConfig(prev => ({
      ...prev!,
      riskParameters: {
        ...prev!.riskParameters,
        [field]: value
      }
    }))
    setHasChanges(true)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!validateForm() || !localConfig) return

    try {
      await saveConfig(localConfig)
      setHasChanges(false)
      onClose()
    } catch (err) {
      console.error('Failed to save strategy config:', err)
    }
  }

  const handleSymbolToggle = (symbol: string) => {
    if (!localConfig) return

    const newSymbols = localConfig.symbols.includes(symbol)
      ? localConfig.symbols.filter(s => s !== symbol)
      : [...localConfig.symbols, symbol]

    handleInputChange('symbols', newSymbols)
  }

  const availableSymbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD']
  const availableTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-auto">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-xl font-semibold">
            Configure Strategy: {strategyName}
          </CardTitle>
          <Button variant="outline" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Configuration */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Basic Configuration</h3>
                
                <div>
                  <label htmlFor="allocation" className="block text-sm font-medium mb-2">
                    Allocation (%)
                  </label>
                  <input
                    id="allocation"
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={localConfig?.allocation || 0}
                    onChange={(e) => handleInputChange('allocation', Number(e.target.value))}
                    className={cn(
                      "w-full px-3 py-2 border rounded-md",
                      validationErrors.allocation 
                        ? "border-red-500 focus:ring-red-500" 
                        : "border-gray-300 focus:ring-blue-500"
                    )}
                  />
                  {validationErrors.allocation && (
                    <p className="text-red-500 text-sm mt-1 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-1" />
                      {validationErrors.allocation}
                    </p>
                  )}
                </div>

                <div>
                  <label htmlFor="maxPositionSize" className="block text-sm font-medium mb-2">
                    Max Position Size ($)
                  </label>
                  <input
                    id="maxPositionSize"
                    type="number"
                    min="1"
                    step="1"
                    value={localConfig?.maxPositionSize || 0}
                    onChange={(e) => handleInputChange('maxPositionSize', Number(e.target.value))}
                    className={cn(
                      "w-full px-3 py-2 border rounded-md",
                      validationErrors.maxPositionSize 
                        ? "border-red-500 focus:ring-red-500" 
                        : "border-gray-300 focus:ring-blue-500"
                    )}
                  />
                  {validationErrors.maxPositionSize && (
                    <p className="text-red-500 text-sm mt-1 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-1" />
                      {validationErrors.maxPositionSize}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="stopLoss" className="block text-sm font-medium mb-2">
                      Stop Loss (%)
                    </label>
                    <input
                      id="stopLoss"
                      type="number"
                      min="0"
                      max="100"
                      step="0.1"
                      value={localConfig?.stopLoss || 0}
                      onChange={(e) => handleInputChange('stopLoss', Number(e.target.value))}
                      className={cn(
                        "w-full px-3 py-2 border rounded-md",
                        validationErrors.stopLoss 
                          ? "border-red-500 focus:ring-red-500" 
                          : "border-gray-300 focus:ring-blue-500"
                      )}
                    />
                  </div>

                  <div>
                    <label htmlFor="takeProfit" className="block text-sm font-medium mb-2">
                      Take Profit (%)
                    </label>
                    <input
                      id="takeProfit"
                      type="number"
                      min="0"
                      step="0.1"
                      value={localConfig?.takeProfit || 0}
                      onChange={(e) => handleInputChange('takeProfit', Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Risk Parameters */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Risk Management</h3>
                
                <div>
                  <label htmlFor="maxDrawdown" className="block text-sm font-medium mb-2">
                    Max Drawdown (%)
                  </label>
                  <input
                    id="maxDrawdown"
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={localConfig?.riskParameters?.maxDrawdown || 0}
                    onChange={(e) => handleRiskParameterChange('maxDrawdown', Number(e.target.value))}
                    className={cn(
                      "w-full px-3 py-2 border rounded-md",
                      validationErrors.maxDrawdown 
                        ? "border-red-500 focus:ring-red-500" 
                        : "border-gray-300 focus:ring-blue-500"
                    )}
                  />
                </div>

                <div>
                  <label htmlFor="maxDailyLoss" className="block text-sm font-medium mb-2">
                    Max Daily Loss ($)
                  </label>
                  <input
                    id="maxDailyLoss"
                    type="number"
                    min="0"
                    step="1"
                    value={localConfig?.riskParameters?.maxDailyLoss || 0}
                    onChange={(e) => handleRiskParameterChange('maxDailyLoss', Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label htmlFor="correlationLimit" className="block text-sm font-medium mb-2">
                    Correlation Limit
                  </label>
                  <input
                    id="correlationLimit"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={localConfig?.riskParameters?.correlationLimit || 0}
                    onChange={(e) => handleRiskParameterChange('correlationLimit', Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Symbol Selection */}
            <div>
              <h3 className="text-lg font-medium mb-4">Symbol Selection</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                {availableSymbols.map(symbol => (
                  <button
                    key={symbol}
                    type="button"
                    onClick={() => handleSymbolToggle(symbol)}
                    className={cn(
                      "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                      localConfig?.symbols?.includes(symbol)
                        ? "bg-blue-500 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    )}
                  >
                    {symbol}
                  </button>
                ))}
              </div>
              {validationErrors.symbols && (
                <p className="text-red-500 text-sm mt-2 flex items-center">
                  <AlertTriangle className="w-4 h-4 mr-1" />
                  {validationErrors.symbols}
                </p>
              )}
            </div>

            {/* Timeframe Selection */}
            <div>
              <h3 className="text-lg font-medium mb-4">Timeframes</h3>
              <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
                {availableTimeframes.map(timeframe => (
                  <button
                    key={timeframe}
                    type="button"
                    onClick={() => {
                      const newTimeframes = localConfig?.timeframes?.includes(timeframe)
                        ? localConfig.timeframes.filter(t => t !== timeframe)
                        : [...(localConfig?.timeframes || []), timeframe]
                      handleInputChange('timeframes', newTimeframes)
                    }}
                    className={cn(
                      "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                      localConfig?.timeframes?.includes(timeframe)
                        ? "bg-green-500 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    )}
                  >
                    {timeframe}
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end space-x-4 pt-6 border-t">
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button 
                type="submit" 
                disabled={!hasChanges || isLoading}
                className="flex items-center"
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                ) : (
                  <Save className="w-4 h-4 mr-2" />
                )}
                Save Configuration
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

export default StrategyConfigPanel 
