// @ts-nocheck
// 性能优化的仪表板页面

import React, { useEffect, useState, useMemo, useCallback, lazy, Suspense } from 'react';
import { clsx } from 'clsx';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  BarChart3,
  PieChart,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

// 性能优化组件
import {
  PerformanceMonitor,
  VirtualScrollList,
  LazyImage,
  CodeSplitWrapper,
  DebouncedInput,
  withPerformanceOptimization,
  DataPreloader,
  OptimizedTable,
  PerformanceStats
} from '../Performance/PerformanceOptimizer';

// 性能优化Hooks
import {
  useDebounce,
  useThrottle,
  useOptimizedFetch,
  usePerformanceMonitor
} from '../../hooks/usePerformance';

// Store
import { useMarketData, useMarketActions, useStrategies } from '../../store';
import apiService from '../../services/api';

// 懒加载组件
const RealTimeData = lazy(() => import('../RealTime/RealTimeData'));
const AdvancedCharts = lazy(() => import('./AdvancedCharts'));

// 市场指数数据
const MARKET_INDICES = [
  { symbol: 'SH000001', name: '上证指数', color: '#ef4444' },
  { symbol: 'SZ399001', name: '深证成指', color: '#10b981' },
  { symbol: 'SZ399006', name: '创业板指', color: '#3b82f6' },
];

// 图表颜色
const CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

interface OptimizedDashboardProps {
  className?: string;
}

const OptimizedDashboard: React.FC<OptimizedDashboardProps> = ({ className }) => {
  const { indices, watchlist, lastUpdated } = useMarketData();
  const { list: strategies } = useStrategies();
  const { updateMarketData } = useMarketActions();
  
  const [loading, setLoading] = useState(false);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [portfolioData, setPortfolioData] = useState<any[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  
  // 性能监控
  const { startRender, endRender, getMetrics } = usePerformanceMonitor();
  
  // 优化的数据获取
  const { fetchWithCache, clearCache } = useOptimizedFetch();
  
  // 防抖的刷新函数
  const debouncedRefresh = useDebounce(
    useCallback(() => {
      setRefreshKey(prev => prev + 1);
      loadMarketData();
    }, []),
    1000
  );
  
  // 节流的滚动处理
  const throttledScroll = useThrottle(
    useCallback((scrollTop: number) => {
      // 处理滚动事件
      console.log('Scroll position:', scrollTop);
    }, []),
    100
  );
  
  // 加载市场数据 - 使用缓存优化
  const loadMarketData = useCallback(async () => {
    try {
      setLoading(true);
      startRender();
      
      // 使用缓存的API调用
      const indicesResponse = await fetchWithCache(
        '/api/market/indices',
        { method: 'GET' },
        5 * 60 * 1000 // 5分钟缓存
      );
      
      if (indicesResponse?.status === 'ok' && indicesResponse.data) {
        updateMarketData('indices', indicesResponse.data);
      }
      
      // 生成模拟数据（实际应用中从API获取）
      generateMockData();
      
    } catch (error) {
      console.error('Failed to load market data:', error);
    } finally {
      setLoading(false);
      endRender();
    }
  }, [fetchWithCache, updateMarketData, startRender, endRender]);
  
  // 生成模拟数据 - 使用useMemo优化
  const generateMockData = useCallback(() => {
    // 生成性能数据
    const performanceData = [];
    const now = new Date();
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      performanceData.push({
        date: format(date, 'MM-dd'),
        portfolio: 100000 + Math.random() * 20000 - 10000,
        benchmark: 100000 + Math.random() * 15000 - 7500,
      });
    }
    setPerformanceData(performanceData);
    
    // 生成组合数据
    const portfolioData = [
      { name: '股票', value: 65, color: '#3b82f6' },
      { name: '债券', value: 20, color: '#10b981' },
      { name: '现金', value: 10, color: '#f59e0b' },
      { name: '其他', value: 5, color: '#ef4444' },
    ];
    setPortfolioData(portfolioData);
  }, []);
  
  // 计算市场统计 - 使用useMemo优化
  const marketStats = useMemo(() => {
    const activeStrategies = (strategies || []).filter(s => s.status === 'active');
    
    return {
      totalValue: 1250000,
      dailyChange: 15600,
      dailyChangePercent: 1.26,
      totalReturn: 156000,
      totalReturnPercent: 14.2,
      activeStrategiesCount: activeStrategies.length,
    };
  }, [strategies]);
  
  // 预加载数据URLs
  const preloadUrls = useMemo(() => [
    '/api/market/indices',
    '/api/portfolio/performance',
    '/api/strategies/active'
  ], []);
  
  // 初始化数据
  useEffect(() => {
    loadMarketData();
    
    // 设置定时刷新 - 使用更长的间隔减少服务器压力
    const interval = setInterval(loadMarketData, 60000); // 60秒刷新一次
    return () => clearInterval(interval);
  }, [loadMarketData]);
  
  // 表格列配置
  const tableColumns = useMemo(() => [
    {
      key: 'symbol',
      title: '代码',
      render: (value: string) => (
        <span className="font-mono text-sm">{value}</span>
      )
    },
    {
      key: 'name',
      title: '名称'
    },
    {
      key: 'price',
      title: '价格',
      render: (value: number) => (
        <span className="font-medium">¥{value.toFixed(2)}</span>
      )
    },
    {
      key: 'change',
      title: '涨跌',
      render: (value: number, row: any) => (
        <span className={clsx(
          'font-medium',
          value >= 0 ? 'text-green-600' : 'text-red-600'
        )}>
          {value >= 0 ? '+' : ''}{value.toFixed(2)}%
        </span>
      )
    }
  ], []);
  
  return (
    <PerformanceMonitor componentName="OptimizedDashboard" enableLogging={true}>
      <DataPreloader urls={preloadUrls}>
        <div className={clsx('p-6 space-y-6', className)}>
          {/* 页面标题 */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">智能投资仪表板</h1>
              <p className="text-gray-600 mt-1">
                最后更新: {lastUpdated ? format(new Date(lastUpdated), 'yyyy-MM-dd HH:mm:ss', { locale: zhCN }) : '未知'}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <PerformanceStats className="mr-4" />
              <button
                onClick={debouncedRefresh}
                disabled={loading}
                className="btn btn-outline btn-sm"
              >
                <RefreshCw className={clsx('w-4 h-4 mr-2', loading && 'animate-spin')} />
                刷新数据
              </button>
            </div>
          </div>
          
          {/* 关键指标卡片 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <OptimizedMetricCard
              title="总资产"
              value={`¥${marketStats.totalValue.toLocaleString()}`}
              change={marketStats.dailyChange}
              changePercent={marketStats.dailyChangePercent}
              icon={<DollarSign className="w-6 h-6" />}
              color="blue"
            />
            <OptimizedMetricCard
              title="总收益"
              value={`¥${marketStats.totalReturn.toLocaleString()}`}
              change={marketStats.totalReturn}
              changePercent={marketStats.totalReturnPercent}
              icon={<TrendingUp className="w-6 h-6" />}
              color="green"
            />
            <OptimizedMetricCard
              title="活跃策略"
              value={marketStats.activeStrategiesCount.toString()}
              change={2}
              changePercent={25}
              icon={<Activity className="w-6 h-6" />}
              color="purple"
            />
            <OptimizedMetricCard
              title="风险评分"
              value="7.2"
              change={-0.3}
              changePercent={-4.0}
              icon={<AlertTriangle className="w-6 h-6" />}
              color="orange"
            />
          </div>
          
          {/* 实时数据流 - 懒加载 */}
          <Suspense fallback={
            <div className="h-32 bg-gray-100 rounded-lg animate-pulse flex items-center justify-center">
              <span className="text-gray-500">加载实时数据中...</span>
            </div>
          }>
            <RealTimeData className="mb-8" />
          </Suspense>
          
          {/* 图表区域 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 组合表现 */}
            <OptimizedPerformanceChart 
              data={performanceData}
              loading={loading}
            />
            
            {/* 资产配置 */}
            <OptimizedPortfolioChart 
              data={portfolioData}
              loading={loading}
            />
          </div>
          
          {/* 市场数据表格 - 虚拟滚动优化 */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">市场数据</h2>
            <OptimizedTable
              data={indices || []}
              columns={tableColumns}
              pageSize={20}
              className="w-full"
            />
          </div>
          
          {/* 高级图表 - 代码分割 */}
          <CodeSplitWrapper
            loader={() => import('./AdvancedCharts')}
            fallback={
              <div className="h-64 bg-gray-100 rounded-lg animate-pulse flex items-center justify-center">
                <span className="text-gray-500">加载高级图表中...</span>
              </div>
            }
            props={{ data: performanceData }}
          />
          
          {/* 快速操作 */}
          <OptimizedQuickActions />
        </div>
      </DataPreloader>
    </PerformanceMonitor>
  );
};

// 优化的指标卡片组件
interface OptimizedMetricCardProps {
  title: string;
  value: string;
  change: number;
  changePercent: number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'purple' | 'orange';
}

const OptimizedMetricCard = React.memo<OptimizedMetricCardProps>(({ 
  title, 
  value, 
  change, 
  changePercent, 
  icon, 
  color 
}) => {
  const colorClasses = useMemo(() => ({
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    purple: 'text-purple-600 bg-purple-100',
    orange: 'text-orange-600 bg-orange-100',
  }), []);
  
  const isPositive = change >= 0;
  
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className={clsx('p-2 rounded-lg', colorClasses[color])}>
          {icon}
        </div>
        <div className={clsx(
          'flex items-center text-sm',
          isPositive ? 'text-success-600' : 'text-danger-600'
        )}>
          {isPositive ? (
            <TrendingUp className="w-4 h-4 mr-1" />
          ) : (
            <TrendingDown className="w-4 h-4 mr-1" />
          )}
          <span>{changePercent >= 0 ? '+' : ''}{changePercent.toFixed(1)}%</span>
        </div>
      </div>
      
      <div>
        <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
        <div className="text-2xl font-bold text-gray-900">{value}</div>
        <div className={clsx(
          'text-sm mt-1',
          isPositive ? 'text-success-600' : 'text-danger-600'
        )}>
          {isPositive ? '+' : ''}{typeof change === 'number' ? change.toLocaleString() : change}
        </div>
      </div>
    </div>
  );
});

// 优化的性能图表组件
const OptimizedPerformanceChart = React.memo<{
  data: any[];
  loading: boolean;
}>(({ data, loading }) => {
  if (loading) {
    return (
      <div className="card p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">组合表现</h2>
        <BarChart3 className="w-5 h-5 text-gray-400" />
      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis 
              dataKey="date" 
              stroke="#6b7280"
              fontSize={12}
            />
            <YAxis 
              stroke="#6b7280"
              fontSize={12}
              tickFormatter={(value) => `¥${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
              formatter={(value: number, name: string) => [
                `¥${value.toLocaleString()}`,
                name === 'portfolio' ? '组合价值' : '基准价值'
              ]}
            />
            <Area
              type="monotone"
              dataKey="portfolio"
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.1}
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="benchmark"
              stroke="#10b981"
              fill="#10b981"
              fillOpacity={0.1}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
});

// 优化的组合图表组件
const OptimizedPortfolioChart = React.memo<{
  data: any[];
  loading: boolean;
}>(({ data, loading }) => {
  if (loading) {
    return (
      <div className="card p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">资产配置</h2>
        <PieChart className="w-5 h-5 text-gray-400" />
      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsPieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={5}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip 
              formatter={(value: number) => [`${value}%`, '占比']}
            />
          </RechartsPieChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-2">
        {data.map((item, index) => (
          <div key={index} className="flex items-center">
            <div
              className="w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-sm text-gray-600">{item.name}</span>
            <span className="ml-auto text-sm font-medium">{item.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
});

// 优化的快速操作组件
const OptimizedQuickActions = React.memo(() => {
  const actions = useMemo(() => [
    {
      icon: <TrendingUp className="w-5 h-5 text-primary-600 mr-2" />,
      title: '策略回测',
      description: '测试投资策略的历史表现',
      onClick: () => console.log('策略回测')
    },
    {
      icon: <Activity className="w-5 h-5 text-success-600 mr-2" />,
      title: '实时监控',
      description: '监控市场动态和持仓变化',
      onClick: () => console.log('实时监控')
    },
    {
      icon: <BarChart3 className="w-5 h-5 text-warning-600 mr-2" />,
      title: '风险分析',
      description: '评估组合风险和优化建议',
      onClick: () => console.log('风险分析')
    }
  ], []);
  
  return (
    <div className="card p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">快速操作</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {actions.map((action, index) => (
          <button 
            key={index}
            onClick={action.onClick}
            className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <div className="flex items-center mb-2">
              {action.icon}
              <span className="font-medium">{action.title}</span>
            </div>
            <p className="text-sm text-gray-600">{action.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
});

// 使用性能优化HOC包装组件
export default withPerformanceOptimization(OptimizedDashboard, {
  componentName: 'OptimizedDashboard',
  enableMemo: true,
  enableMonitoring: true
});