import { useState, useCallback } from 'react';

interface UseStrategyControlsReturn {
  pauseStrategy: (strategyName: string) => Promise<void>;
  resumeStrategy: (strategyName: string) => Promise<void>;
  stopStrategy: (strategyName: string) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

export function useStrategyControls(): UseStrategyControlsReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStrategyAction = useCallback(async (
    strategyName: string, 
    action: 'pause' | 'resume' | 'stop'
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/strategies/${encodeURIComponent(strategyName)}/${action}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Failed to ${action} strategy: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Log success
      console.log(`Strategy ${strategyName} ${action}d successfully:`, result);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : `Failed to ${action} strategy`;
      setError(errorMessage);
      console.error(`Error ${action}ing strategy:`, err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const pauseStrategy = useCallback(async (strategyName: string) => {
    return handleStrategyAction(strategyName, 'pause');
  }, [handleStrategyAction]);

  const resumeStrategy = useCallback(async (strategyName: string) => {
    return handleStrategyAction(strategyName, 'resume');
  }, [handleStrategyAction]);

  const stopStrategy = useCallback(async (strategyName: string) => {
    return handleStrategyAction(strategyName, 'stop');
  }, [handleStrategyAction]);

  return {
    pauseStrategy,
    resumeStrategy,
    stopStrategy,
    isLoading,
    error,
  };
}

export default useStrategyControls; 
