// 策略管理页面
// 实现策略配置、回测、监控和优化功能

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Target,
  Brain,
  Zap,
  CheckCircle,
  Clock,
  Pause,
  Square,
  AlertTriangle,
  Plus,
  RefreshCw,
  Play,
  Edit,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { useStrategies, useStrategyActions } from '../../hooks/useStrategies';
import type { Strategy } from '../../store';
import apiService from '../../services/api';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

// 策略状态枚举
const STRATEGY_STATUS = {
  DRAFT: 'draft',
  ACTIVE: 'active',
  PAUSED: 'paused',
  STOPPED: 'stopped',
  ERROR: 'error',
} as const;

// 策略类型
const STRATEGY_TYPES = {
  MOMENTUM: 'momentum',
  MEAN_REVERSION: 'mean_reversion',
  ARBITRAGE: 'arbitrage',
  MULTI_FACTOR: 'multi_factor',
  ML_BASED: 'ml_based',
} as const;



interface StrategiesPageProps {
  className?: string;
}

export const StrategiesPage: React.FC<StrategiesPageProps> = ({ className }) => {
  const { list: strategies, loading } = useStrategies();
  const { loadStrategies, createStrategy } = useStrategyActions();
  
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showBacktestModal, setShowBacktestModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'performance' | 'config' | 'factors'>('overview');
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [factorWeights, setFactorWeights] = useState<any[]>([]);

  // 加载策略数据
  useEffect(() => {
    loadStrategies();
    generateMockData();
  }, []);

  // 生成模拟数据
  const generateMockData = () => {
    // 生成性能数据
    const performanceData = [];
    const now = new Date();
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      performanceData.push({
        date: format(date, 'MM-dd'),
        strategy: 100 + Math.random() * 20 - 10,
        benchmark: 100 + Math.random() * 15 - 7.5,
        drawdown: -(Math.random() * 5),
      });
    }
    setPerformanceData(performanceData);

    // 生成因子权重数据
    const factorWeights = [
      { name: '动量因子', weight: 0.25, performance: 0.12 },
      { name: '价值因子', weight: 0.20, performance: 0.08 },
      { name: '质量因子', weight: 0.18, performance: 0.15 },
      { name: '成长因子', weight: 0.15, performance: 0.10 },
      { name: '低波因子', weight: 0.12, performance: 0.06 },
      { name: '规模因子', weight: 0.10, performance: 0.04 },
    ];
    setFactorWeights(factorWeights);
  };

  // 策略状态图标
  const getStatusIcon = (status: string) => {
    switch (status) {
      case STRATEGY_STATUS.ACTIVE:
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case STRATEGY_STATUS.PAUSED:
        return <Pause className="w-4 h-4 text-yellow-500" />;
      case STRATEGY_STATUS.STOPPED:
        return <Square className="w-4 h-4 text-gray-500" />;
      case STRATEGY_STATUS.ERROR:
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-blue-500" />;
    }
  };

  // 策略类型图标
  const getTypeIcon = (type: string) => {
    switch (type) {
      case STRATEGY_TYPES.MOMENTUM:
        return <TrendingUp className="w-4 h-4" />;
      case STRATEGY_TYPES.MEAN_REVERSION:
        return <TrendingDown className="w-4 h-4" />;
      case STRATEGY_TYPES.ARBITRAGE:
        return <Target className="w-4 h-4" />;
      case STRATEGY_TYPES.MULTI_FACTOR:
        return <BarChart3 className="w-4 h-4" />;
      case STRATEGY_TYPES.ML_BASED:
        return <Brain className="w-4 h-4" />;
      default:
        return <Zap className="w-4 h-4" />;
    }
  };

  // 启动策略
  const handleStartStrategy = async (strategyId: string) => {
    try {
      // TODO: 实现启动策略API
      console.log('Starting strategy:', strategyId);
      await loadStrategies();
    } catch (error) {
      console.error('Failed to start strategy:', error);
    }
  };

  // 暂停策略
  const handlePauseStrategy = async (strategyId: string) => {
    try {
      // TODO: 实现暂停策略API
      console.log('Pausing strategy:', strategyId);
      await loadStrategies();
    } catch (error) {
      console.error('Failed to pause strategy:', error);
    }
  };

  // 运行回测
  const handleRunBacktest = async (strategyId: string) => {
    try {
      setShowBacktestModal(true);
      const response = await apiService.runBacktest(strategyId, {
        start_date: '2023-01-01',
        end_date: '2024-01-01',
        initial_capital: 1000000,
      });
      
      if (response.status === 'ok') {
        // 更新策略的回测结果
        console.log('Backtest completed:', response.data);
      }
    } catch (error) {
      console.error('Failed to run backtest:', error);
    } finally {
      setShowBacktestModal(false);
    }
  };

  return (
    <div className={clsx('p-6 space-y-6', className)}>
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">策略管理</h1>
          <p className="text-gray-600 mt-1">管理和监控您的量化投资策略</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => loadStrategies()}
            disabled={loading}
            className="btn btn-outline btn-sm"
          >
            <RefreshCw className={clsx('w-4 h-4 mr-2', loading && 'animate-spin')} />
            刷新
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary btn-sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            创建策略
          </button>
        </div>
      </div>

      {/* 策略概览卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">总策略数</p>
              <p className="text-2xl font-bold text-gray-900">{strategies.length}</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <BarChart3 className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">活跃策略</p>
              <p className="text-2xl font-bold text-gray-900">
                {(strategies || []).filter(s => s.status === STRATEGY_STATUS.ACTIVE).length}
              </p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <Play className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">平均收益率</p>
              <p className="text-2xl font-bold text-green-600">+12.5%</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">夏普比率</p>
              <p className="text-2xl font-bold text-gray-900">1.85</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-full">
              <Target className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 策略列表 */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">策略列表</h2>
            </div>
            
            <div className="divide-y divide-gray-200">
              {(strategies || []).map((strategy) => (
                <div
                  key={strategy.id}
                  className={clsx(
                    'p-6 hover:bg-gray-50 cursor-pointer transition-colors',
                    selectedStrategy?.id === strategy.id && 'bg-blue-50 border-l-4 border-blue-500'
                  )}
                  onClick={() => setSelectedStrategy(strategy)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-10 h-10 bg-gray-100 rounded-lg">
                        {getTypeIcon(strategy.type)}
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-gray-900">{strategy.name}</h3>
                        <p className="text-sm text-gray-500">{strategy.description}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">
                          {strategy.performance.totalReturn > 0 ? '+' : ''}
                          {(strategy.performance.totalReturn * 100).toFixed(2)}%
                        </p>
                        <p className="text-xs text-gray-500">总收益</p>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        {getStatusIcon(strategy.status)}
                        <span className="text-xs text-gray-500 capitalize">
                          {strategy.status}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        {strategy.status === STRATEGY_STATUS.ACTIVE ? (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handlePauseStrategy(strategy.id);
                            }}
                            className="p-1 text-gray-400 hover:text-yellow-600"
                          >
                            <Pause className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartStrategy(strategy.id);
                            }}
                            className="p-1 text-gray-400 hover:text-green-600"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                        )}
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRunBacktest(strategy.id);
                          }}
                          className="p-1 text-gray-400 hover:text-blue-600"
                        >
                          <BarChart3 className="w-4 h-4" />
                        </button>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // 编辑策略
                          }}
                          className="p-1 text-gray-400 hover:text-gray-600"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 策略详情 */}
        <div className="space-y-6">
          {selectedStrategy ? (
            <>
              {/* 策略信息卡片 */}
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {selectedStrategy.name}
                  </h3>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(selectedStrategy.status)}
                    <span className="text-sm text-gray-500 capitalize">
                      {selectedStrategy.status}
                    </span>
                  </div>
                </div>
                
                <p className="text-sm text-gray-600 mb-4">
                  {selectedStrategy.description}
                </p>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">策略类型</span>
                    <span className="text-sm font-medium text-gray-900 capitalize">
                      {selectedStrategy.type.replace('_', ' ')}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">创建时间</span>
                    <span className="text-sm font-medium text-gray-900">
                      {format(new Date(selectedStrategy.created_at), 'yyyy-MM-dd', { locale: zhCN })}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">资金规模</span>
                    <span className="text-sm font-medium text-gray-900">
                      ¥{selectedStrategy.config.capital.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* 性能指标 */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">性能指标</h3>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">总收益率</span>
                    <span className={clsx(
                      'text-sm font-medium',
                      selectedStrategy.performance.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
                    )}>
                      {selectedStrategy.performance.totalReturn > 0 ? '+' : ''}
                      {(selectedStrategy.performance.totalReturn * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">年化收益率</span>
                    <span className="text-sm font-medium text-gray-900">
                      {(selectedStrategy.performance.annualReturn * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">夏普比率</span>
                    <span className="text-sm font-medium text-gray-900">
                      {selectedStrategy.performance.sharpeRatio.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">最大回撤</span>
                    <span className="text-sm font-medium text-red-600">
                      {(selectedStrategy.performance.maxDrawdown * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">胜率</span>
                    <span className="text-sm font-medium text-gray-900">
                      {(selectedStrategy.performance.winRate * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* 因子权重 */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">因子权重</h3>
                
                <div className="space-y-2">
                  {(factorWeights || []).map((factor) => (
                    <div key={factor.name} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">{factor.name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${factor.weight * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-gray-900 w-12 text-right">
                          {(factor.weight * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="card p-6">
              <div className="text-center py-8">
                <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">选择策略</h3>
                <p className="text-gray-500">点击左侧策略查看详细信息</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 性能图表 */}
      {selectedStrategy && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">策略表现</h2>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setActiveTab('overview')}
                className={clsx(
                  'px-3 py-1 text-sm rounded-md transition-colors',
                  activeTab === 'overview'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                )}
              >
                概览
              </button>
              <button
                onClick={() => setActiveTab('performance')}
                className={clsx(
                  'px-3 py-1 text-sm rounded-md transition-colors',
                  activeTab === 'performance'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                )}
              >
                收益曲线
              </button>
              <button
                onClick={() => setActiveTab('factors')}
                className={clsx(
                  'px-3 py-1 text-sm rounded-md transition-colors',
                  activeTab === 'factors'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                )}
              >
                因子分析
              </button>
            </div>
          </div>

          {activeTab === 'performance' && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="strategy"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    name="策略收益"
                  />
                  <Line
                    type="monotone"
                    dataKey="benchmark"
                    stroke="#10b981"
                    strokeWidth={2}
                    name="基准收益"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === 'factors' && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={factorWeights}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="weight" fill="#3b82f6" name="权重" />
                  <Bar dataKey="performance" fill="#10b981" name="表现" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* 创建策略模态框 */}
      {showCreateModal && (
        <CreateStrategyModal
          onClose={() => setShowCreateModal(false)}
          onSubmit={(strategyData) => {
            createStrategy(strategyData);
            setShowCreateModal(false);
          }}
        />
      )}

      {/* 回测模态框 */}
      {showBacktestModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">运行回测中...</h3>
              <p className="text-gray-500">正在分析历史数据，请稍候</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// 创建策略模态框组件
interface CreateStrategyModalProps {
  onClose: () => void;
  onSubmit: (data: any) => void;
}

const CreateStrategyModal: React.FC<CreateStrategyModalProps> = ({ onClose, onSubmit }) => {
  const [formData, setFormData] = useState({
    name: '',
    type: STRATEGY_TYPES.MULTI_FACTOR,
    description: '',
    capital: 1000000,
    max_position: 0.1,
    stop_loss: 0.05,
    take_profit: 0.15,
    rebalance_frequency: 'daily',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">创建新策略</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              策略名称
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="输入策略名称"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              策略类型
            </label>
            <select
              value={formData.type}
              onChange={(e) => setFormData({ ...formData, type: e.target.value as 'multi_factor' })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={STRATEGY_TYPES.MOMENTUM}>动量策略</option>
              <option value={STRATEGY_TYPES.MEAN_REVERSION}>均值回归</option>
              <option value={STRATEGY_TYPES.ARBITRAGE}>套利策略</option>
              <option value={STRATEGY_TYPES.MULTI_FACTOR}>多因子策略</option>
              <option value={STRATEGY_TYPES.ML_BASED}>机器学习策略</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              策略描述
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
              placeholder="描述策略的核心逻辑和特点"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                初始资金
              </label>
              <input
                type="number"
                value={formData.capital}
                onChange={(e) => setFormData({ ...formData, capital: Number(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="10000"
                step="10000"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                最大仓位 (%)
              </label>
              <input
                type="number"
                value={formData.max_position * 100}
                onChange={(e) => setFormData({ ...formData, max_position: Number(e.target.value) / 100 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="1"
                max="100"
                step="1"
              />
            </div>
          </div>

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
            >
              取消
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              创建策略
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default StrategiesPage;