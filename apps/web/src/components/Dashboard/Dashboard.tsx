// 仪表板页面

import React, { useEffect, useState } from 'react';
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
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import RealTimeData from '../RealTime/RealTimeData';
import { useMarketData, useMarketActions, useStrategies } from '../../store';
import apiService from '../../services/api';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

// 市场指数数据
const MARKET_INDICES = [
  { symbol: 'SH000001', name: '上证指数', color: '#ef4444' },
  { symbol: 'SZ399001', name: '深证成指', color: '#10b981' },
  { symbol: 'SZ399006', name: '创业板指', color: '#3b82f6' },
];

// 图表颜色
const CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

interface DashboardProps {
  className?: string;
}

export const Dashboard: React.FC<DashboardProps> = ({ className }) => {
  const { indices, watchlist, lastUpdated } = useMarketData();
  const { list: strategies } = useStrategies();
  const { updateMarketData } = useMarketActions();
  
  const [loading, setLoading] = useState(false);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [portfolioData, setPortfolioData] = useState<any[]>([]);

  // 加载市场数据
  const loadMarketData = async () => {
    try {
      setLoading(true);
      
      // 获取市场指数数据
      const indicesResponse = await apiService.getMarketIndices();
      if (indicesResponse.status === 'ok' && indicesResponse.data) {
        updateMarketData('indices', indicesResponse.data);
      }
      
      // 生成模拟数据（实际应用中从API获取）
      generateMockData();
      
    } catch (error) {
      console.error('Failed to load market data:', error);
    } finally {
      setLoading(false);
    }
  };

  // 生成模拟数据
  const generateMockData = () => {
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
  };

  // 初始化数据
  useEffect(() => {
    loadMarketData();
    
    // 设置定时刷新
    const interval = setInterval(loadMarketData, 30000); // 30秒刷新一次
    return () => clearInterval(interval);
  }, []);

  // 计算市场统计
  const marketStats = {
    totalValue: 1250000,
    dailyChange: 15600,
    dailyChangePercent: 1.26,
    totalReturn: 156000,
    totalReturnPercent: 14.2,
  };

  return (
    <div className={clsx('p-6 space-y-6', className)}>
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">投资仪表板</h1>
          <p className="text-gray-600 mt-1">
            最后更新: {lastUpdated ? format(new Date(lastUpdated), 'yyyy-MM-dd HH:mm:ss', { locale: zhCN }) : '未知'}
          </p>
        </div>
        <button
          onClick={loadMarketData}
          disabled={loading}
          className="btn btn-outline btn-sm"
        >
          <RefreshCw className={clsx('w-4 h-4 mr-2', loading && 'animate-spin')} />
          刷新数据
        </button>
      </div>

      {/* 关键指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="总资产"
          value={`¥${marketStats.totalValue.toLocaleString()}`}
          change={marketStats.dailyChange}
          changePercent={marketStats.dailyChangePercent}
          icon={<DollarSign className="w-6 h-6" />}
          color="blue"
        />
        <MetricCard
          title="总收益"
          value={`¥${marketStats.totalReturn.toLocaleString()}`}
          change={marketStats.totalReturn}
          changePercent={marketStats.totalReturnPercent}
          icon={<TrendingUp className="w-6 h-6" />}
          color="green"
        />
        <MetricCard
          title="活跃策略"
          value={(strategies || []).filter(s => s.status === 'active').length.toString()}
          change={2}
          changePercent={25}
          icon={<Activity className="w-6 h-6" />}
          color="purple"
        />
        <MetricCard
          title="风险评分"
          value="7.2"
          change={-0.3}
          changePercent={-4.0}
          icon={<AlertTriangle className="w-6 h-6" />}
          color="orange"
        />
      </div>

      {/* 实时数据流 */}
      <RealTimeData className="mb-8" />

      {/* 图表区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 组合表现 */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">组合表现</h2>
            <BarChart3 className="w-5 h-5 text-gray-400" />
          </div>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={performanceData}>
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

        {/* 资产配置 */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">资产配置</h2>
            <PieChart className="w-5 h-5 text-gray-400" />
          </div>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={portfolioData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {portfolioData.map((entry, index) => (
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
            {portfolioData.map((item, index) => (
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
      </div>

      {/* 快速操作 */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">快速操作</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="flex items-center mb-2">
              <TrendingUp className="w-5 h-5 text-primary-600 mr-2" />
              <span className="font-medium">策略回测</span>
            </div>
            <p className="text-sm text-gray-600">测试投资策略的历史表现</p>
          </button>
          
          <button className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="flex items-center mb-2">
              <Activity className="w-5 h-5 text-success-600 mr-2" />
              <span className="font-medium">实时监控</span>
            </div>
            <p className="text-sm text-gray-600">监控市场动态和持仓变化</p>
          </button>
          
          <button className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="flex items-center mb-2">
              <BarChart3 className="w-5 h-5 text-warning-600 mr-2" />
              <span className="font-medium">风险分析</span>
            </div>
            <p className="text-sm text-gray-600">评估组合风险和优化建议</p>
          </button>
        </div>
      </div>
    </div>
  );
};

// 指标卡片组件
interface MetricCardProps {
  title: string;
  value: string;
  change: number;
  changePercent: number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'purple' | 'orange';
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changePercent,
  icon,
  color,
}) => {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    purple: 'text-purple-600 bg-purple-100',
    orange: 'text-orange-600 bg-orange-100',
  };

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className={clsx('p-2 rounded-lg', colorClasses[color])}>
          {icon}
        </div>
        <div className={clsx(
          'flex items-center text-sm',
          change >= 0 ? 'text-success-600' : 'text-danger-600'
        )}>
          {change >= 0 ? (
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
          change >= 0 ? 'text-success-600' : 'text-danger-600'
        )}>
          {change >= 0 ? '+' : ''}{typeof change === 'number' ? change.toLocaleString() : change}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;