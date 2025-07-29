import { useState, useEffect, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { Portfolio, Strategy, PerformanceDataPoint } from '@/types';

// WebSocket connection
let socket: Socket | null = null;

const getSocket = () => {
  if (!socket) {
    socket = io(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000', {
      transports: ['websocket'],
      autoConnect: false, // Disable auto-connect for now
      timeout: 5000,
    });
  }
  return socket;
};

// Portfolio real-time hook
export function useRealtimePortfolio() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const ws = getSocket();
    setIsLoading(true);

    const handleConnect = () => {
      console.log('🔗 Connected to WebSocket');
      setIsConnected(true);
      ws.emit('subscribe', ['portfolio']);
    };

    const handleDisconnect = () => {
      console.log('🔌 Disconnected from WebSocket');
      setIsConnected(false);
    };

    const handlePortfolioUpdate = (data: Portfolio) => {
      setPortfolio(data);
      setIsLoading(false);
    };

    const handleError = (error: any) => {
      console.error('WebSocket error:', error);
      setIsLoading(false);
    };

    // Event listeners
    ws.on('connect', handleConnect);
    ws.on('disconnect', handleDisconnect);
    ws.on('portfolio.update', handlePortfolioUpdate);
    ws.on('error', handleError);

    // Initial data fetch
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/portfolio/summary`)
      .then(res => res.json())
      .then(data => {
        setPortfolio(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch initial portfolio data:', err);
        setIsLoading(false);
      });

    return () => {
      ws.off('connect', handleConnect);
      ws.off('disconnect', handleDisconnect);
      ws.off('portfolio.update', handlePortfolioUpdate);
      ws.off('error', handleError);
    };
  }, []);

  return { portfolio, isLoading, isConnected };
}

// Strategies real-time hook
export function useRealtimeStrategies() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const toggleStrategy = useCallback(async (strategyName: string) => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/strategies/${strategyName}/pause`,
        { method: 'POST' }
      );
      
      if (response.ok) {
        const updatedStrategy = await response.json();
        setStrategies(prev => 
          prev.map(s => s.name === strategyName ? { ...s, ...updatedStrategy } : s)
        );
      }
    } catch (error) {
      console.error('Failed to toggle strategy:', error);
    }
  }, []);

  useEffect(() => {
    const ws = getSocket();
    setIsLoading(true);

    const handleStrategiesUpdate = (data: Strategy[]) => {
      setStrategies(data);
      setIsLoading(false);
    };

    const handleStrategyUpdate = (data: Strategy) => {
      setStrategies(prev => 
        prev.map(s => s.name === data.name ? data : s)
      );
    };

    // Event listeners
    ws.on('strategies.update', handleStrategiesUpdate);
    ws.on('strategy.update', handleStrategyUpdate);
    ws.emit('subscribe', ['strategies']);

    // Initial data fetch
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/strategies`)
      .then(res => res.json())
      .then(data => {
        setStrategies(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch initial strategies data:', err);
        setIsLoading(false);
      });

    return () => {
      ws.off('strategies.update', handleStrategiesUpdate);
      ws.off('strategy.update', handleStrategyUpdate);
    };
  }, []);

  return { strategies, isLoading, toggleStrategy };
}

// Performance data hook
export function usePerformanceData(timeframe: string = '1D') {
  const [performanceData, setPerformanceData] = useState<PerformanceDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const ws = getSocket();
    setIsLoading(true);

    const handlePerformanceUpdate = (data: PerformanceDataPoint[]) => {
      setPerformanceData(data);
      setIsLoading(false);
    };

    // Event listeners
    ws.on('performance.update', handlePerformanceUpdate);
    ws.emit('subscribe', [`performance.${timeframe}`]);

    // Initial data fetch
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/analytics/performance?timeframe=${timeframe}`)
      .then(res => res.json())
      .then(data => {
        setPerformanceData(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch performance data:', err);
        setIsLoading(false);
      });

    return () => {
      ws.off('performance.update', handlePerformanceUpdate);
    };
  }, [timeframe]);

  return { performanceData, isLoading };
}

// Market data hook for real-time price updates
export function useMarketData(symbols: string[]) {
  const [marketData, setMarketData] = useState<Record<string, any>>({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!symbols.length) return;

    const ws = getSocket();
    setIsLoading(true);

    const handleMarketUpdate = (data: Record<string, any>) => {
      setMarketData(prev => ({ ...prev, ...data }));
      setIsLoading(false);
    };

    // Event listeners
    ws.on('market.update', handleMarketUpdate);
    ws.emit('subscribe', symbols.map(symbol => `market.${symbol}`));

    return () => {
      ws.off('market.update', handleMarketUpdate);
    };
  }, [symbols]);

  return { marketData, isLoading };
}

// Connection status hook
export function useConnectionStatus() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    const ws = getSocket();

    const handleConnect = () => {
      setIsConnected(true);
      setLastUpdate(new Date());
    };

    const handleDisconnect = () => {
      setIsConnected(false);
    };

    const handleAnyUpdate = () => {
      setLastUpdate(new Date());
    };

    // Event listeners
    ws.on('connect', handleConnect);
    ws.on('disconnect', handleDisconnect);
    ws.onAny(handleAnyUpdate);

    // Check initial connection
    setIsConnected(ws.connected);

    return () => {
      ws.off('connect', handleConnect);
      ws.off('disconnect', handleDisconnect);
      ws.offAny(handleAnyUpdate);
    };
  }, []);

  return { isConnected, lastUpdate };
} 
