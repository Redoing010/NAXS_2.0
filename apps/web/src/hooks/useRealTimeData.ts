// 实时数据获取Hook
// 支持WebSocket和HTTP轮询的实时数据获取

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import apiService from '../services/api';
import { WEBSOCKET_URL } from '../config/environment';

// 数据类型定义
interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: Date;
}

interface FactorData {
  factorId: string;
  name: string;
  value: number;
  ic: number;
  icIr: number;
  weight: number;
  status: 'active' | 'inactive' | 'deprecated';
  lastUpdate: Date;
}

// Hook配置
interface UseRealTimeDataConfig {
  symbols?: string[];
  type?: 'market' | 'factors' | 'news';
  interval?: number;
  enabled?: boolean;
  useWebSocket?: boolean;
  onData?: (data: any) => void;
  onError?: (error: Error) => void;
}

// Hook返回值
interface UseRealTimeDataReturn {
  marketData?: MarketData[];
  factors?: FactorData[];
  data?: any;
  isLoading: boolean;
  error?: Error;
  lastUpdate?: Date;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  refetch: () => Promise<void>;
}

export const useRealTimeData = (config: UseRealTimeDataConfig): UseRealTimeDataReturn => {
  const {
    symbols = [],
    type = 'market',
    interval = 5000,
    enabled = true,
    useWebSocket: useWS = true,
    onData,
    onError
  } = config;

  const [marketData, setMarketData] = useState<MarketData[]>();
  const [factors, setFactors] = useState<FactorData[]>();
  const [data, setData] = useState<any>();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error>();
  const [lastUpdate, setLastUpdate] = useState<Date>();
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');

  const intervalRef = useRef<NodeJS.Timeout>();
  const retryCountRef = useRef(0);
  const maxRetries = 3;

  // WebSocket连接
  const { isConnected, sendMessage } = useWebSocket({
    url: WEBSOCKET_URL,
    enabled: enabled && useWS,
    onOpen: () => {
      setConnectionStatus('connected');
      retryCountRef.current = 0;
      
      // 订阅数据
      if (type === 'market' && symbols.length > 0) {
        sendMessage({
          type: 'subscribe',
          subscription_type: 'market_data',
          symbols: symbols
        });
      } else if (type === 'factors') {
        sendMessage({
          type: 'subscribe',
          subscription_type: 'factor_data'
        });
      }
    },
    onMessage: (message) => {
      try {
        const parsedData = typeof message === 'string' ? JSON.parse(message) : message;
        handleWebSocketData(parsedData);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    },
    onClose: () => {
      setConnectionStatus('disconnected');
    },
    onError: (err) => {
      setConnectionStatus('error');
      setError(new Error(`WebSocket error: ${err.message}`));
      if (onError) onError(err);
    }
  });

  // 处理WebSocket数据
  const handleWebSocketData = useCallback((wsData: any) => {
    try {
      if (wsData.type === 'market_data') {
        const marketItems: MarketData[] = wsData.data.map((item: any) => ({
          symbol: item.symbol,
          price: item.price,
          change: item.change,
          changePercent: item.change_percent,
          volume: item.volume,
          timestamp: new Date(item.timestamp)
        }));
        setMarketData(marketItems);
        setLastUpdate(new Date());
        if (onData) onData(marketItems);
      } else if (wsData.type === 'factor_data') {
        const factorItems: FactorData[] = wsData.data.map((item: any) => ({
          factorId: item.factor_id,
          name: item.name,
          value: item.value,
          ic: item.ic,
          icIr: item.ic_ir,
          weight: item.weight,
          status: item.status,
          lastUpdate: new Date(item.last_update)
        }));
        setFactors(factorItems);
        setLastUpdate(new Date());
        if (onData) onData(factorItems);
      }
    } catch (err) {
      console.error('Failed to process WebSocket data:', err);
      setError(err as Error);
    }
  }, [onData]);

  // HTTP轮询获取数据
  const fetchData = useCallback(async () => {
    if (!enabled) return;

    setIsLoading(true);
    setError(undefined);

    try {
      let response;
      
      if (type === 'market') {
        response = await apiService.getMarketData({ symbols });
        if (response.success) {
          const marketItems: MarketData[] = response.data.map((item: any) => ({
            symbol: item.symbol,
            price: item.price,
            change: item.change,
            changePercent: item.change_percent,
            volume: item.volume,
            timestamp: new Date(item.timestamp)
          }));
          setMarketData(marketItems);
          if (onData) onData(marketItems);
        }
      } else if (type === 'factors') {
        response = await apiService.getFactorData();
        if (response.success) {
          const factorItems: FactorData[] = response.data.map((item: any) => ({
            factorId: item.factor_id,
            name: item.name,
            value: item.value,
            ic: item.ic,
            icIr: item.ic_ir,
            weight: item.weight,
            status: item.status,
            lastUpdate: new Date(item.last_update)
          }));
          setFactors(factorItems);
          if (onData) onData(factorItems);
        }
      }
      
      setLastUpdate(new Date());
      retryCountRef.current = 0;
      
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setError(err as Error);
      if (onError) onError(err as Error);
      
      // 重试逻辑
      if (retryCountRef.current < maxRetries) {
        retryCountRef.current++;
        setTimeout(() => fetchData(), 1000 * retryCountRef.current);
      }
    } finally {
      setIsLoading(false);
    }
  }, [enabled, type, symbols, onData, onError]);

  // 手动刷新
  const refetch = useCallback(async () => {
    await fetchData();
  }, [fetchData]);

  // 设置轮询
  useEffect(() => {
    if (!enabled) return;

    // 如果WebSocket连接成功，则不使用HTTP轮询
    if (useWS && isConnected) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = undefined;
      }
      return;
    }

    // 立即获取一次数据
    fetchData();

    // 设置定时轮询
    if (interval > 0) {
      intervalRef.current = setInterval(fetchData, interval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = undefined;
      }
    };
  }, [enabled, useWS, isConnected, interval, fetchData]);

  // 更新连接状态
  useEffect(() => {
    if (useWS) {
      if (isConnected) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('disconnected');
      }
    } else {
      setConnectionStatus('connected'); // HTTP模式始终显示为连接
    }
  }, [useWS, isConnected]);

  return {
    marketData,
    factors,
    data,
    isLoading,
    error,
    lastUpdate,
    connectionStatus,
    refetch
  };
};

export default useRealTimeData;