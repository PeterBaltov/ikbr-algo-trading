import { useState, useEffect, useCallback } from 'react';

interface StrategyConfig {
  allocation: number;
  maxPositionSize: number;
  stopLoss: number;
  takeProfit: number;
  symbols: string[];
  timeframes: string[];
  riskParameters: {
    maxDrawdown: number;
    maxDailyLoss: number;
    correlationLimit: number;
  };
  enabled: boolean;
}

interface UseStrategyConfigReturn {
  config: StrategyConfig | null;
  isLoading: boolean;
  error: string | null;
  updateConfig: (newConfig: Partial<StrategyConfig>) => void;
  saveConfig: (config: StrategyConfig) => Promise<void>;
  resetConfig: () => void;
}

// Get default configuration
const getDefaultConfig = (): StrategyConfig => ({
  allocation: 10,
  maxPositionSize: 10000,
  stopLoss: 5,
  takeProfit: 10,
  symbols: ['SPY'],
  timeframes: ['1d'],
  riskParameters: {
    maxDrawdown: 20,
    maxDailyLoss: 1000,
    correlationLimit: 0.7,
  },
  enabled: true,
});

export function useStrategyConfig(strategyName: string): UseStrategyConfigReturn {
  const [config, setConfig] = useState<StrategyConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load initial configuration
  const loadConfig = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/strategies/${encodeURIComponent(strategyName)}/config`);
      
      if (!response.ok) {
        throw new Error(`Failed to load configuration: ${response.statusText}`);
      }

      const data = await response.json();
      setConfig(data.config || getDefaultConfig());
    } catch (err) {
      console.error('Error loading strategy config:', err);
      setError(err instanceof Error ? err.message : 'Failed to load configuration');
      
      // Use default config if loading fails
      setConfig(getDefaultConfig());
    } finally {
      setIsLoading(false);
    }
  }, [strategyName]);

  // Update local configuration
  const updateConfig = useCallback((newConfig: Partial<StrategyConfig>) => {
    setConfig(prev => prev ? { ...prev, ...newConfig } : null);
  }, []);

  // Save configuration to backend
  const saveConfig = useCallback(async (configToSave: StrategyConfig) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/strategies/${encodeURIComponent(strategyName)}/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ config: configToSave }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save configuration: ${response.statusText}`);
      }

      const data = await response.json();
      setConfig(data.config);
      
      // Show success notification (you could integrate with a toast system)
      console.log('Strategy configuration saved successfully');
    } catch (err) {
      console.error('Error saving strategy config:', err);
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
      throw err; // Re-throw so the component can handle it
    } finally {
      setIsLoading(false);
    }
  }, [strategyName]);

  // Reset to default configuration
  const resetConfig = useCallback(() => {
    setConfig(getDefaultConfig());
    setError(null);
  }, []);

  // Load configuration on mount or when strategy name changes
  useEffect(() => {
    if (strategyName) {
      loadConfig();
    }
  }, [strategyName, loadConfig]);

  return {
    config,
    isLoading,
    error,
    updateConfig,
    saveConfig,
    resetConfig,
  };
}

export default useStrategyConfig; 
