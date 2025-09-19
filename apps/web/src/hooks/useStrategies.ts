import { useState, useEffect } from 'react';
import type { Strategy } from '../store';

// 策略状态常量
export const STRATEGY_STATUS = {
  ACTIVE: 'active',
  INACTIVE: 'inactive',
  PAUSED: 'paused',
  STOPPED: 'stopped',
  ERROR: 'error',
  BACKTESTING: 'backtesting',
} as const;

// 策略类型常量
export const STRATEGY_TYPES = {
  MOMENTUM: 'momentum',
  MEAN_REVERSION: 'mean_reversion',
  ARBITRAGE: 'arbitrage',
  MULTI_FACTOR: 'multi_factor',
  ML_BASED: 'ml_based',
} as const;

// 模拟策略数据
const mockStrategies: Strategy[] = [
  {
    id: '1',
    name: '多因子量化策略',
    description: '基于多个技术和基本面因子的量化投资策略',
    type: 'multi_factor',
    status: 'active',
    performance: {
      totalReturn: 0.156,
      annualReturn: 0.142,
      sharpeRatio: 1.85,
      maxDrawdown: 0.08,
      winRate: 0.68,
    },
    config: {
      capital: 1000000,
      max_position: 0.1,
      stop_loss: 0.05,
      take_profit: 0.15,
      rebalance_frequency: 'daily',
    },
    factors: [
      { name: '动量因子', weight: 0.25 },
      { name: '价值因子', weight: 0.20 },
      { name: '质量因子', weight: 0.18 },
      { name: '成长因子', weight: 0.15 },
      { name: '低波因子', weight: 0.12 },
      { name: '规模因子', weight: 0.10 },
    ],
    created_at: '2024-01-15T08:00:00Z',
    updated_at: '2024-01-20T10:30:00Z',
    lastUpdated: '2024-01-20T10:30:00Z',
  },
  {
    id: '2',
    name: '动量策略',
    description: '基于价格动量的趋势跟踪策略',
    type: 'momentum',
    status: 'inactive',
    performance: {
      totalReturn: 0.089,
      annualReturn: 0.082,
      sharpeRatio: 1.32,
      maxDrawdown: 0.12,
      winRate: 0.61,
    },
    config: {
      capital: 500000,
      max_position: 0.15,
      stop_loss: 0.08,
      take_profit: 0.20,
      rebalance_frequency: 'weekly',
    },
    factors: [
      { name: '价格动量', weight: 0.40 },
      { name: '成交量动量', weight: 0.30 },
      { name: '相对强弱', weight: 0.30 },
    ],
    created_at: '2024-01-10T09:00:00Z',
    updated_at: '2024-01-18T14:20:00Z',
    lastUpdated: '2024-01-18T14:20:00Z',
  },
  {
    id: '3',
    name: '均值回归策略',
    description: '基于价格均值回归的反转策略',
    type: 'mean_reversion',
    status: 'backtesting',
    performance: {
      totalReturn: 0.034,
      annualReturn: 0.031,
      sharpeRatio: 0.95,
      maxDrawdown: 0.06,
      winRate: 0.55,
    },
    config: {
      capital: 800000,
      max_position: 0.08,
      stop_loss: 0.04,
      take_profit: 0.12,
      rebalance_frequency: 'daily',
    },
    factors: [
      { name: '价格偏离度', weight: 0.35 },
      { name: '波动率', weight: 0.25 },
      { name: '成交量异常', weight: 0.20 },
      { name: '技术指标', weight: 0.20 },
    ],
    created_at: '2024-01-12T11:00:00Z',
    updated_at: '2024-01-19T16:45:00Z',
    lastUpdated: '2024-01-19T16:45:00Z',
  },
];

// 策略数据 Hook
export const useStrategies = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStrategies = async () => {
    setLoading(true);
    setError(null);
    try {
      // 暂时使用模拟数据
      await new Promise(resolve => setTimeout(resolve, 500)); // 模拟网络延迟
      setStrategies(mockStrategies);
      
      // TODO: 替换为真实API调用
      // const response = await apiService.getStrategies();
      // if (response.status === 'ok') {
      //   setStrategies(response.data);
      // } else {
      //   setError(response.error || 'Failed to load strategies');
      // }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStrategies();
  }, []);

  return {
    list: strategies,
    loading,
    error,
    reload: loadStrategies,
  };
};

// 策略操作 Hook
export const useStrategyActions = () => {
  const loadStrategies = async () => {
    // TODO: 实现加载策略逻辑
    console.log('Loading strategies...');
  };

  const createStrategy = async (strategyData: Partial<Strategy>) => {
    try {
      // TODO: 实现创建策略API调用
      console.log('Creating strategy:', strategyData);
      
      // const response = await apiService.createStrategy(strategyData);
      // if (response.status === 'ok') {
      //   return response.data;
      // } else {
      //   throw new Error(response.error || 'Failed to create strategy');
      // }
    } catch (error) {
      console.error('Failed to create strategy:', error);
      throw error;
    }
  };

  const updateStrategy = async (id: string, updates: Partial<Strategy>) => {
    try {
      // TODO: 实现更新策略API调用
      console.log('Updating strategy:', id, updates);
      
      // const response = await apiService.updateStrategy(id, updates);
      // if (response.status === 'ok') {
      //   return response.data;
      // } else {
      //   throw new Error(response.error || 'Failed to update strategy');
      // }
    } catch (error) {
      console.error('Failed to update strategy:', error);
      throw error;
    }
  };

  const deleteStrategy = async (id: string) => {
    try {
      // TODO: 实现删除策略API调用
      console.log('Deleting strategy:', id);
      
      // const response = await apiService.deleteStrategy(id);
      // if (response.status !== 'ok') {
      //   throw new Error(response.error || 'Failed to delete strategy');
      // }
    } catch (error) {
      console.error('Failed to delete strategy:', error);
      throw error;
    }
  };

  return {
    loadStrategies,
    createStrategy,
    updateStrategy,
    deleteStrategy,
  };
};