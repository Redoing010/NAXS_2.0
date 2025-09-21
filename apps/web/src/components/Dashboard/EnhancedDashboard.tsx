// 增强仪表板组件
// 集成实时数据展示、Agent状态监控和系统性能可视化

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { clsx } from 'clsx';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  LineChart,
  Zap,
  Database,
  Cpu,
  Memory,
  Wifi,
  WifiOff,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Bot,
  Settings,
  RefreshCw,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Play,
  Pause,
  Square,
  Filter,
  Download,
  Upload,
  Bell,
  BellOff,
  Target,
  Layers,
  Brain,
  Gauge,
} from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useRealTimeData } from '../../hooks/useRealTimeData';
import { useAgentStatus } from '../../hooks/useAgentStatus';
import { useSystemMetrics } from '../../hooks/useSystemMetrics';
import { WEBSOCKET_URL } from '../../config/environment';

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

interface AgentMetrics {
  agentId: string;
  name: string;
  status: 'idle' | 'running' | 'error' | 'offline';
  currentTask?: string;
  progress?: number;
  toolsUsed: number;
  successRate: number;
  avgResponseTime: number;
  lastActivity: Date;
}

interface SystemHealth {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  activeConnections: number;
  errorRate: number;
  uptime: number;
}

interface AlertItem {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}

// 组件属性
interface EnhancedDashboardProps {
  className?: string;
}

// 实时数据卡片组件
const RealTimeDataCard: React.FC<{
  title: string;
  data: MarketData[];
  isLoading: boolean;
}> = ({ title, data, isLoading }) => {
  const [expanded, setExpanded] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={clsx(
              'p-2 rounded-md transition-colors',
              autoRefresh
                ? 'bg-green-100 text-green-600 hover:bg-green-200'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            )}
          >
            {autoRefresh ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-2 rounded-md bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
          >
            {expanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
          <span className="ml-2 text-gray-500">加载中...</span>
        </div>
      ) : (
        <div className="space-y-3">
          {data.slice(0, expanded ? data.length : 5).map((item, index) => (
            <div key={item.symbol} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                <span className="font-medium text-gray-900">{item.symbol}</span>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-lg font-semibold text-gray-900">
                  ¥{item.price.toFixed(2)}
                </span>
                <div className={clsx(
                  'flex items-center space-x-1 px-2 py-1 rounded-full text-sm font-medium',
                  item.change >= 0
                    ? 'bg-green-100 text-green-800'
                    : 'bg-red-100 text-red-800'
                )}>
                  {item.change >= 0 ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  <span>{item.changePercent.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          ))}
          {!expanded && data.length > 5 && (
            <button
              onClick={() => setExpanded(true)}
              className="w-full py-2 text-sm text-gray-500 hover:text-gray-700 transition-colors"
            >
              显示更多 ({data.length - 5} 项)
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// 因子监控卡片组件
const FactorMonitorCard: React.FC<{
  factors: FactorData[];
  isLoading: boolean;
}> = ({ factors, isLoading }) => {
  const [sortBy, setSortBy] = useState<'ic' | 'weight' | 'name'>('ic');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'inactive'>('all');

  const filteredAndSortedFactors = useMemo(() => {
    let filtered = factors || [];
    
    if (filterStatus !== 'all') {
      filtered = (factors || []).filter(f => f.status === filterStatus);
    }
    
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'ic':
          return Math.abs(b.ic) - Math.abs(a.ic);
        case 'weight':
          return b.weight - a.weight;
        case 'name':
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });
  }, [factors, sortBy, filterStatus]);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">因子监控</h3>
        <div className="flex items-center space-x-2">
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="all">全部</option>
            <option value="active">活跃</option>
            <option value="inactive">非活跃</option>
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="ic">按IC排序</option>
            <option value="weight">按权重排序</option>
            <option value="name">按名称排序</option>
          </select>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
          <span className="ml-2 text-gray-500">加载中...</span>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredAndSortedFactors.map((factor) => (
            <div key={factor.factorId} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
              <div className="flex items-center space-x-3">
                <div className={clsx(
                  'w-3 h-3 rounded-full',
                  factor.status === 'active' ? 'bg-green-500' :
                  factor.status === 'inactive' ? 'bg-yellow-500' : 'bg-red-500'
                )} />
                <span className="font-medium text-gray-900">{factor.name}</span>
              </div>
              <div className="flex items-center space-x-4 text-sm">
                <div className="text-center">
                  <div className="text-gray-500">IC</div>
                  <div className={clsx(
                    'font-semibold',
                    Math.abs(factor.ic) > 0.05 ? 'text-green-600' :
                    Math.abs(factor.ic) > 0.02 ? 'text-yellow-600' : 'text-red-600'
                  )}>
                    {factor.ic.toFixed(3)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-500">权重</div>
                  <div className="font-semibold text-gray-900">
                    {(factor.weight * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-500">IC IR</div>
                  <div className={clsx(
                    'font-semibold',
                    factor.icIr > 0.5 ? 'text-green-600' :
                    factor.icIr > 0.3 ? 'text-yellow-600' : 'text-red-600'
                  )}>
                    {factor.icIr.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Agent状态监控组件
const AgentStatusCard: React.FC<{
  agents: AgentMetrics[];
  isLoading: boolean;
}> = ({ agents, isLoading }) => {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500';
      case 'idle': return 'bg-blue-500';
      case 'error': return 'bg-red-500';
      case 'offline': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return '运行中';
      case 'idle': return '空闲';
      case 'error': return '错误';
      case 'offline': return '离线';
      default: return '未知';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Agent状态</h3>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 text-sm text-gray-500">
            <Bot className="w-4 h-4" />
            <span>{agents.length} 个Agent</span>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
          <span className="ml-2 text-gray-500">加载中...</span>
        </div>
      ) : (
        <div className="space-y-3">
          {agents.map((agent) => (
            <div key={agent.agentId} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <div className={clsx('w-3 h-3 rounded-full', getStatusColor(agent.status))} />
                  <span className="font-medium text-gray-900">{agent.name}</span>
                  <span className="text-sm text-gray-500">{getStatusText(agent.status)}</span>
                </div>
                <button
                  onClick={() => setSelectedAgent(
                    selectedAgent === agent.agentId ? null : agent.agentId
                  )}
                  className="p-1 rounded-md hover:bg-gray-100 transition-colors"
                >
                  {selectedAgent === agent.agentId ? (
                    <EyeOff className="w-4 h-4 text-gray-500" />
                  ) : (
                    <Eye className="w-4 h-4 text-gray-500" />
                  )}
                </button>
              </div>

              {agent.currentTask && (
                <div className="mb-2">
                  <div className="text-sm text-gray-600 mb-1">当前任务: {agent.currentTask}</div>
                  {agent.progress !== undefined && (
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${agent.progress}%` }}
                      />
                    </div>
                  )}
                </div>
              )}

              {selectedAgent === agent.agentId && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">工具使用次数:</span>
                      <span className="ml-2 font-medium">{agent.toolsUsed}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">成功率:</span>
                      <span className="ml-2 font-medium">{(agent.successRate * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500">平均响应时间:</span>
                      <span className="ml-2 font-medium">{agent.avgResponseTime.toFixed(0)}ms</span>
                    </div>
                    <div>
                      <span className="text-gray-500">最后活动:</span>
                      <span className="ml-2 font-medium">
                        {agent.lastActivity.toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// 系统健康监控组件
const SystemHealthCard: React.FC<{
  health: SystemHealth;
  isLoading: boolean;
}> = ({ health, isLoading }) => {
  const getHealthColor = (value: number, thresholds: [number, number]) => {
    if (value < thresholds[0]) return 'text-green-600';
    if (value < thresholds[1]) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHealthBg = (value: number, thresholds: [number, number]) => {
    if (value < thresholds[0]) return 'bg-green-500';
    if (value < thresholds[1]) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}天 ${hours}小时 ${minutes}分钟`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">系统健康</h3>
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-gray-500" />
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
          <span className="ml-2 text-gray-500">加载中...</span>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          {/* CPU使用率 */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">CPU</span>
              </div>
              <span className={clsx(
                'text-sm font-semibold',
                getHealthColor(health.cpuUsage, [70, 90])
              )}>
                {health.cpuUsage.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={clsx('h-2 rounded-full transition-all duration-300', getHealthBg(health.cpuUsage, [70, 90]))}
                style={{ width: `${health.cpuUsage}%` }}
              />
            </div>
          </div>

          {/* 内存使用率 */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Memory className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">内存</span>
              </div>
              <span className={clsx(
                'text-sm font-semibold',
                getHealthColor(health.memoryUsage, [80, 95])
              )}>
                {health.memoryUsage.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={clsx('h-2 rounded-full transition-all duration-300', getHealthBg(health.memoryUsage, [80, 95]))}
                style={{ width: `${health.memoryUsage}%` }}
              />
            </div>
          </div>

          {/* 网络延迟 */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Wifi className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">网络延迟</span>
              </div>
              <span className={clsx(
                'text-sm font-semibold',
                getHealthColor(health.networkLatency, [100, 500])
              )}>
                {health.networkLatency.toFixed(0)}ms
              </span>
            </div>
          </div>

          {/* 活跃连接 */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">活跃连接</span>
              </div>
              <span className="text-sm font-semibold text-gray-900">
                {health.activeConnections}
              </span>
            </div>
          </div>

          {/* 错误率 */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">错误率</span>
              </div>
              <span className={clsx(
                'text-sm font-semibold',
                getHealthColor(health.errorRate, [1, 5])
              )}>
                {health.errorRate.toFixed(2)}%
              </span>
            </div>
          </div>

          {/* 运行时间 */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">运行时间</span>
              </div>
            </div>
            <div className="text-xs text-gray-600">
              {formatUptime(health.uptime)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// 告警中心组件
const AlertCenterCard: React.FC<{
  alerts: AlertItem[];
  onAcknowledge: (alertId: string) => void;
}> = ({ alerts, onAcknowledge }) => {
  const [filter, setFilter] = useState<'all' | 'unacknowledged'>('unacknowledged');
  const [showNotifications, setShowNotifications] = useState(true);

  const filteredAlerts = useMemo(() => {
    return (alerts || []).filter(alert => 
      filter === 'all' || !alert.acknowledged
    ).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }, [alerts, filter]);

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      default: return <AlertTriangle className="w-4 h-4 text-blue-500" />;
    }
  };

  const getAlertBg = (type: string) => {
    switch (type) {
      case 'error': return 'bg-red-50 border-red-200';
      case 'warning': return 'bg-yellow-50 border-yellow-200';
      case 'success': return 'bg-green-50 border-green-200';
      default: return 'bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">告警中心</h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className={clsx(
              'p-2 rounded-md transition-colors',
              showNotifications
                ? 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            )}
          >
            {showNotifications ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
          </button>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="unacknowledged">未确认</option>
            <option value="all">全部</option>
          </select>
        </div>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
            <p>暂无告警</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={clsx(
                'p-4 rounded-lg border transition-all duration-200',
                getAlertBg(alert.type),
                alert.acknowledged ? 'opacity-60' : ''
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{alert.title}</h4>
                    <p className="text-sm text-gray-600 mt-1">{alert.message}</p>
                    <p className="text-xs text-gray-500 mt-2">
                      {alert.timestamp.toLocaleString()}
                    </p>
                  </div>
                </div>
                {!alert.acknowledged && (
                  <button
                    onClick={() => onAcknowledge(alert.id)}
                    className="px-3 py-1 text-xs bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                  >
                    确认
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

// 主仪表板组件
export const EnhancedDashboard: React.FC<EnhancedDashboardProps> = ({ className }) => {
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5秒刷新
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  
  // 使用自定义hooks获取数据
  const { marketData, isLoading: marketLoading } = useRealTimeData({
    symbols: ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'],
    interval: refreshInterval,
    enabled: isAutoRefresh
  });
  
  const { factors, isLoading: factorLoading } = useRealTimeData({
    type: 'factors',
    interval: refreshInterval,
    enabled: isAutoRefresh
  });
  
  const { agents, isLoading: agentLoading } = useAgentStatus({
    interval: refreshInterval,
    enabled: isAutoRefresh
  });
  
  const { systemHealth, isLoading: systemLoading } = useSystemMetrics({
    interval: refreshInterval,
    enabled: isAutoRefresh
  });
  
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  
  // WebSocket连接状态
  const { isConnected, connectionStatus } = useWebSocket({
    url: WEBSOCKET_URL,
    onMessage: (data) => {
      // 处理实时数据更新
      console.log('WebSocket data:', data);
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    }
  });
  
  // 确认告警
  const handleAcknowledgeAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  }, []);
  
  // 手动刷新
  const handleManualRefresh = useCallback(() => {
    // 触发数据刷新
    window.location.reload();
  }, []);
  
  return (
    <div className={clsx('p-6 bg-gray-50 min-h-screen', className)}>
      {/* 头部控制栏 */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">NAXS 智能投研系统</h1>
            <p className="text-gray-600 mt-1">实时监控 · 智能分析 · 决策支持</p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* 连接状态 */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <Wifi className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-green-600">已连接</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5 text-red-500" />
                  <span className="text-sm text-red-600">连接断开</span>
                </>
              )}
            </div>
            
            {/* 自动刷新控制 */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                className={clsx(
                  'px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isAutoRefresh
                    ? 'bg-green-100 text-green-700 hover:bg-green-200'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                )}
              >
                {isAutoRefresh ? '自动刷新' : '手动刷新'}
              </button>
              
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                disabled={!isAutoRefresh}
              >
                <option value={1000}>1秒</option>
                <option value={5000}>5秒</option>
                <option value={10000}>10秒</option>
                <option value={30000}>30秒</option>
              </select>
              
              <button
                onClick={handleManualRefresh}
                className="p-2 bg-blue-100 text-blue-600 rounded-md hover:bg-blue-200 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* 主要内容区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* 实时市场数据 */}
        <div className="xl:col-span-1">
          <RealTimeDataCard
            title="实时行情"
            data={marketData || []}
            isLoading={marketLoading}
          />
        </div>
        
        {/* 因子监控 */}
        <div className="xl:col-span-1">
          <FactorMonitorCard
            factors={factors || []}
            isLoading={factorLoading}
          />
        </div>
        
        {/* Agent状态 */}
        <div className="xl:col-span-1">
          <AgentStatusCard
            agents={agents || []}
            isLoading={agentLoading}
          />
        </div>
        
        {/* 系统健康 */}
        <div className="lg:col-span-1">
          <SystemHealthCard
            health={systemHealth || {
              cpuUsage: 0,
              memoryUsage: 0,
              diskUsage: 0,
              networkLatency: 0,
              activeConnections: 0,
              errorRate: 0,
              uptime: 0
            }}
            isLoading={systemLoading}
          />
        </div>
        
        {/* 告警中心 */}
        <div className="lg:col-span-1 xl:col-span-2">
          <AlertCenterCard
            alerts={alerts}
            onAcknowledge={handleAcknowledgeAlert}
          />
        </div>
      </div>
    </div>
  );
};

export default EnhancedDashboard;