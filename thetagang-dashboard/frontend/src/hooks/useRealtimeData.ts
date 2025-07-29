import { useState, useEffect, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { Portfolio, Strategy, PerformanceDataPoint } from '@/types';

// WebSocket connection
let socket: Socket | null = null;

const getSocket = () => {
  if (!socket) {
    socket = io(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000', {
      transports: ['websocket'],
      autoConnect: true,
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
      ws.emit('subscribe', { channels: ['portfolio'] });
    };

    const handleDisconnect = () => {
      console.log('🔌 Disconnected from WebSocket');
      setIsConnected(false);
    };

    const handlePortfolioUpdate = (data: any) => {
      console.log('📊 Portfolio update received:', data);
      setPortfolio(data.data);
      setIsLoading(false);
    };

    const handleError = (error: unknown) => {
      console.error('WebSocket error:', error);
      setIsLoading(false);
    };

    // Event listeners
    ws.on('connect', handleConnect);
    ws.on('disconnect', handleDisconnect);
    ws.on('portfolio.update', handlePortfolioUpdate);
    ws.on('error', handleError);

    // Connect to socket
    ws.connect();

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

    const handleStrategiesUpdate = (data: any) => {
      console.log('🎛️ Strategies update received:', data);
      setStrategies(data.data);
      setIsLoading(false);
    };

    const handleStrategyUpdate = (data: any) => {
      console.log('🔄 Strategy update received:', data);
      // Update individual strategy data
      setStrategies(prev => 
        prev.map(s => {
          const update = data.data.find((u: any) => u.name === s.name);
          return update ? { ...s, ...update } : s;
        })
      );
    };

    // Event listeners
    ws.on('strategies.update', handleStrategiesUpdate);
    ws.on('strategy.update', handleStrategyUpdate);
    ws.emit('subscribe', { channels: ['strategies'] });

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
  const [marketData, setMarketData] = useState<Record<string, unknown>>({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!symbols.length) return;

    const ws = getSocket();
    setIsLoading(true);

    const handleMarketUpdate = (data: Record<string, unknown>) => {
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
