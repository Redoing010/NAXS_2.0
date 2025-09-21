// @ts-nocheck
// 实时数据流组件
// 展示实时市场指数、价格变动和数据流

import React, { useState, useEffect, useRef } from 'react';
import { clsx } from 'clsx';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Wifi,
  WifiOff,
  RefreshCw,
  Pause,
  Play,
  Settings,
  BarChart3,
  LineChart,
  PieChart,
  AlertTriangle,
  CheckCircle,
  Clock,
  Globe,
  Database,
  Signal,
} from 'lucide-react';
import { Line, Bar } from 'recharts';
import { LineChart as RechartsLineChart, BarChart as RechartsBarChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// 市场指数数据接口
interface MarketIndex {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: Date;
  status: 'trading' | 'closed' | 'pre-market' | 'after-hours';
}

// 实时数据点
interface DataPoint {
  timestamp: Date;
  value: number;
  volume?: number;
}

// 数据流状态
interface StreamStatus {
  connected: boolean;
  latency: number;
  messagesPerSecond: number;
  lastUpdate: Date;
  errors: number;
}

// 主要市场指数
const MAJOR_INDICES = [
  { symbol: 'SH000001', name: '上证指数', market: 'SH' },
  { symbol: 'SZ399001', name: '深证成指', market: 'SZ' },
  { symbol: 'SZ399006', name: '创业板指', market: 'SZ' },
  { symbol: 'SH000300', name: '沪深300', market: 'SH' },
  { symbol: 'SZ399905', name: '中证500', market: 'SZ' },
  { symbol: 'SH000016', name: '上证50', market: 'SH' },
];

interface RealTimeDataProps {
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const RealTimeData: React.FC<RealTimeDataProps> = ({
  className,
  autoRefresh = true,
  refreshInterval = 1000,
}) => {
  const [indices, setIndices] = useState<MarketIndex[]>([]);
  const [historicalData, setHistoricalData] = useState<Record<string, DataPoint[]>>({});
  const [streamStatus, setStreamStatus] = useState<StreamStatus>({
    connected: false,
    latency: 0,
    messagesPerSecond: 0,
    lastUpdate: new Date(),
    errors: 0,
  });
  const [selectedIndex, setSelectedIndex] = useState<string>('SH000001');
  const [isPaused, setIsPaused] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'chart'>('grid');
  
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const messageCountRef = useRef(0);
  const lastSecondRef = useRef(Date.now());

  // 初始化WebSocket连接
  useEffect(() => {
    if (autoRefresh && !isPaused) {
      connectWebSocket();
    }
    
    return () => {
      disconnectWebSocket();
    };
  }, [autoRefresh, isPaused]);

  // 连接WebSocket
  const connectWebSocket = () => {
    try {
      // 这里应该连接到实际的WebSocket服务
      // wsRef.current = new WebSocket('ws://localhost:8080/realtime');
      
      // 模拟WebSocket连接
      setStreamStatus(prev => ({ ...prev, connected: true }));
      
      // 模拟数据流
      intervalRef.current = setInterval(() => {
        if (!isPaused) {
          generateMockData();
          updateStreamMetrics();
        }
      }, refreshInterval);
      
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setStreamStatus(prev => ({ 
        ...prev, 
        connected: false, 
        errors: prev.errors + 1 
      }));
    }
  };

  // 断开WebSocket
  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setStreamStatus(prev => ({ ...prev, connected: false }));
  };

  // 生成模拟数据
  const generateMockData = () => {
    const now = new Date();
    const newIndices: MarketIndex[] = MAJOR_INDICES.map(index => {
      const basePrice = 3000 + Math.random() * 2000;
      const change = (Math.random() - 0.5) * 100;
      const changePercent = (change / basePrice) * 100;
      
      return {
        symbol: index.symbol,
        name: index.name,
        price: basePrice + change,
        change,
        changePercent,
        volume: Math.floor(Math.random() * 1000000000),
        timestamp: now,
        status: getMarketStatus(),
      };
    });
    
    setIndices(newIndices);
    
    // 更新历史数据
    setHistoricalData(prev => {
      const updated = { ...prev };
      newIndices.forEach(index => {
        if (!updated[index.symbol]) {
          updated[index.symbol] = [];
        }
        
        updated[index.symbol].push({
          timestamp: now,
          value: index.price,
          volume: index.volume,
        });
        
        // 保持最近100个数据点
        if (updated[index.symbol].length > 100) {
          updated[index.symbol] = updated[index.symbol].slice(-100);
        }
      });
      return updated;
    });
  };

  // 获取市场状态
  const getMarketStatus = (): MarketIndex['status'] => {
    const now = new Date();
    const hour = now.getHours();
    
    if (hour >= 9 && hour < 15) {
      return 'trading';
    } else if (hour >= 15 && hour < 17) {
      return 'after-hours';
    } else if (hour >= 8 && hour < 9) {
      return 'pre-market';
    } else {
      return 'closed';
    }
  };

  // 更新流指标
  const updateStreamMetrics = () => {
    messageCountRef.current++;
    const now = Date.now();
    
    if (now - lastSecondRef.current >= 1000) {
      const messagesPerSecond = messageCountRef.current;
      messageCountRef.current = 0;
      lastSecondRef.current = now;
      
      setStreamStatus(prev => ({
        ...prev,
        messagesPerSecond,
        latency: Math.random() * 50 + 10, // 模拟延迟
        lastUpdate: new Date(),
      }));
    }
  };

  // 暂停/恢复数据流
  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  // 手动刷新
  const handleManualRefresh = () => {
    generateMockData();
  };

  // 格式化数字
  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toLocaleString('zh-CN', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  // 格式化变化
  const formatChange = (change: number, isPercent: boolean = false) => {
    const formatted = formatNumber(Math.abs(change), isPercent ? 2 : 2);
    const sign = change >= 0 ? '+' : '-';
    return `${sign}${formatted}${isPercent ? '%' : ''}`;
  };

  // 渲染连接状态
  const renderConnectionStatus = () => (
    <div className="flex items-center space-x-4 text-sm">
      <div className="flex items-center space-x-2">
        {streamStatus.connected ? (
          <>
            <Wifi className="w-4 h-4 text-green-500" />
            <span className="text-green-600">已连接</span>
          </>
        ) : (
          <>
            <WifiOff className="w-4 h-4 text-red-500" />
            <span className="text-red-600">未连接</span>
          </>
        )}
      </div>
      
      {streamStatus.connected && (
        <>
          <div className="flex items-center space-x-1">
            <Signal className="w-4 h-4 text-blue-500" />
            <span className="text-gray-600">
              {streamStatus.latency.toFixed(0)}ms
            </span>
          </div>
          
          <div className="flex items-center space-x-1">
            <Activity className="w-4 h-4 text-purple-500" />
            <span className="text-gray-600">
              {streamStatus.messagesPerSecond}/s
            </span>
          </div>
          
          <div className="flex items-center space-x-1">
            <Clock className="w-4 h-4 text-orange-500" />
            <span className="text-gray-600">
              {streamStatus.lastUpdate.toLocaleTimeString()}
            </span>
          </div>
        </>
      )}
    </div>
  );

  // 渲染指数卡片
  const renderIndexCard = (index: MarketIndex) => {
    const isPositive = index.change >= 0;
    const statusColor = {
      trading: 'text-green-600',
      closed: 'text-gray-600',
      'pre-market': 'text-blue-600',
      'after-hours': 'text-orange-600',
    }[index.status];
    
    return (
      <div
        key={index.symbol}
        className={clsx(
          'bg-white rounded-lg border p-4 cursor-pointer transition-all hover:shadow-md',
          selectedIndex === index.symbol ? 'border-blue-500 shadow-md' : 'border-gray-200'
        )}
        onClick={() => setSelectedIndex(index.symbol)}
      >
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="font-semibold text-gray-900">{index.name}</h3>
            <p className="text-sm text-gray-500">{index.symbol}</p>
          </div>
          <div className={clsx('text-xs font-medium', statusColor)}>
            {{
              trading: '交易中',
              closed: '已收盘',
              'pre-market': '盘前',
              'after-hours': '盘后',
            }[index.status]}
          </div>
        </div>
        
        <div className="space-y-1">
          <div className="text-2xl font-bold text-gray-900">
            {formatNumber(index.price)}
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={clsx(
              'flex items-center space-x-1 text-sm font-medium',
              isPositive ? 'text-red-600' : 'text-green-600'
            )}>
              {isPositive ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              <span>{formatChange(index.change)}</span>
            </div>
            
            <div className={clsx(
              'text-sm font-medium',
              isPositive ? 'text-red-600' : 'text-green-600'
            )}>
              ({formatChange(index.changePercent, true)})
            </div>
          </div>
          
          <div className="text-xs text-gray-500">
            成交量: {formatNumber(index.volume / 100000000, 1)}亿
          </div>
        </div>
      </div>
    );
  };

  // 渲染图表
  const renderChart = () => {
    const data = historicalData[selectedIndex] || [];
    const chartData = data.slice(-30).map((point, index) => ({
      time: point.timestamp.toLocaleTimeString(),
      price: point.value,
      volume: point.volume || 0,
    }));
    
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            {MAJOR_INDICES.find(i => i.symbol === selectedIndex)?.name} 实时走势
          </h3>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode('grid')}
              className={clsx(
                'p-2 rounded-md',
                viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600'
              )}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('chart')}
              className={clsx(
                'p-2 rounded-md',
                viewMode === 'chart' ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600'
              )}
            >
              <LineChart className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <RechartsLineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 12 }}
                interval="preserveStartEnd"
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                domain={['dataMin - 10', 'dataMax + 10']}
              />
              <Tooltip 
                formatter={(value: any) => [formatNumber(value), '价格']}
                labelFormatter={(label) => `时间: ${label}`}
              />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </RechartsLineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <div className={clsx('space-y-6', className)}>
      {/* 控制面板 */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-gray-900">实时行情</h2>
            {renderConnectionStatus()}
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={togglePause}
              className={clsx(
                'p-2 rounded-md transition-colors',
                isPaused
                  ? 'bg-green-100 text-green-600 hover:bg-green-200'
                  : 'bg-yellow-100 text-yellow-600 hover:bg-yellow-200'
              )}
            >
              {isPaused ? (
                <Play className="w-4 h-4" />
              ) : (
                <Pause className="w-4 h-4" />
              )}
            </button>
            
            <button
              onClick={handleManualRefresh}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* 主要指数网格 */}
      {viewMode === 'grid' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {indices.map(renderIndexCard)}
        </div>
      )}

      {/* 图表视图 */}
      {viewMode === 'chart' && renderChart()}

      {/* 数据流统计 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Database className="w-5 h-5 text-blue-600" />
            <h3 className="font-medium text-gray-900">数据点</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {Object.values(historicalData).reduce((sum, data) => sum + data.length, 0)}
          </p>
          <p className="text-sm text-gray-500">总计</p>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="w-5 h-5 text-green-600" />
            <h3 className="font-medium text-gray-900">更新频率</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {streamStatus.messagesPerSecond}
          </p>
          <p className="text-sm text-gray-500">次/秒</p>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Signal className="w-5 h-5 text-purple-600" />
            <h3 className="font-medium text-gray-900">延迟</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {streamStatus.latency.toFixed(0)}
          </p>
          <p className="text-sm text-gray-500">毫秒</p>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <h3 className="font-medium text-gray-900">错误</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {streamStatus.errors}
          </p>
          <p className="text-sm text-gray-500">次</p>
        </div>
      </div>
    </div>
  );
};

export default RealTimeData;