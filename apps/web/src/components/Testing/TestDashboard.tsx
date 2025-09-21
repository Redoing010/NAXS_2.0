// @ts-nocheck
import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Database,
  Globe,
  Monitor,
  Play,
  RefreshCw,
  Server,
  TrendingUp,
  Zap,
  BarChart3,
  Cpu,
  HardDrive,
  Wifi,
  AlertTriangle,
  Download
} from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// 系统指标接口
interface SystemMetrics {
  api: {
    responseTime: number;
    successRate: number;
    throughput: number;
    errorRate: number;
  };
  database: {
    connectionCount: number;
    queryTime: number;
    cacheHitRate: number;
    status: 'healthy' | 'warning' | 'error';
  };
  system: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    uptime: number;
  };
  websocket: {
    connected: boolean;
    activeConnections: number;
    messageRate: number;
  };
}

// 测试结果接口
interface TestResult {
  id: string;
  name: string;
  status: 'running' | 'passed' | 'failed' | 'pending';
  duration: number;
  timestamp: string;
  details?: string;
  metrics?: {
    requests: number;
    failures: number;
    avgResponseTime: number;
    maxResponseTime: number;
  };
}

// 状态指示器组件
const StatusIndicator: React.FC<{ status: 'healthy' | 'warning' | 'error' | 'unknown' }> = ({ status }) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'healthy':
        return { color: 'text-green-500', bg: 'bg-green-100', icon: CheckCircle };
      case 'warning':
        return { color: 'text-yellow-500', bg: 'bg-yellow-100', icon: AlertTriangle };
      case 'error':
        return { color: 'text-red-500', bg: 'bg-red-100', icon: AlertCircle };
      default:
        return { color: 'text-gray-500', bg: 'bg-gray-100', icon: Clock };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div className={`inline-flex items-center px-2 py-1 rounded-full ${config.bg}`}>
      <Icon className={`w-4 h-4 ${config.color} mr-1`} />
      <span className={`text-sm font-medium ${config.color} capitalize`}>{status}</span>
    </div>
  );
};

// 指标卡片组件
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ComponentType<{ className?: string }>;
  status?: 'healthy' | 'warning' | 'error';
  trend?: number;
}> = ({ title, value, unit, icon: Icon, status, trend }) => {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center">
          <Icon className="w-5 h-5 text-gray-500 mr-2" />
          <span className="text-sm font-medium text-gray-600">{title}</span>
        </div>
        {status && <StatusIndicator status={status} />}
      </div>
      <div className="flex items-end justify-between">
        <div>
          <span className="text-2xl font-bold text-gray-900">{value}</span>
          {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
        </div>
        {trend !== undefined && (
          <div className={`flex items-center text-sm ${
            trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-600' : 'text-gray-500'
          }`}>
            <TrendingUp className={`w-4 h-4 mr-1 ${
              trend < 0 ? 'transform rotate-180' : ''
            }`} />
            {Math.abs(trend)}%
          </div>
        )}
      </div>
    </div>
  );
};

// 测试结果组件
const TestResultItem: React.FC<{ result: TestResult }> = ({ result }) => {
  const getStatusIcon = () => {
    switch (result.status) {
      case 'passed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'running':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="flex items-center justify-between p-3 border-b border-gray-100 last:border-b-0">
      <div className="flex items-center">
        {getStatusIcon()}
        <div className="ml-3">
          <div className="text-sm font-medium text-gray-900">{result.name}</div>
          <div className="text-xs text-gray-500">
            {new Date(result.timestamp).toLocaleTimeString()} • {result.duration}ms
          </div>
        </div>
      </div>
      {result.metrics && (
        <div className="text-right">
          <div className="text-sm text-gray-900">
            {result.metrics.requests} requests
          </div>
          <div className="text-xs text-gray-500">
            {result.metrics.failures} failures
          </div>
        </div>
      )}
    </div>
  );
};

// 主测试仪表板组件
export const TestDashboard: React.FC = () => {
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [selectedTestType, setSelectedTestType] = useState<'all' | 'api' | 'stress' | 'integration'>('all');
  const queryClient = useQueryClient();

  // 获取系统指标
  const { data: metrics, isLoading: metricsLoading, error: metricsError } = useQuery<SystemMetrics>({
    queryKey: ['system-metrics'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/system/metrics');
      if (!response.ok) {
        throw new Error(`获取系统指标失败: ${response.status} ${response.statusText}`);
      }
      return response.json();
    },
    refetchInterval: 5000, // 每5秒刷新一次
    retry: 3,
    retryDelay: 1000,
  });

  // 获取测试结果
  const { data: testResults, isLoading: testsLoading, error: testsError } = useQuery<TestResult[]>({
    queryKey: ['test-results'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/system/tests/results');
      if (!response.ok) {
        throw new Error(`获取测试结果失败: ${response.status} ${response.statusText}`);
      }
      return response.json();
    },
    refetchInterval: 3000, // 每3秒刷新一次
    retry: 2,
    retryDelay: 1000,
  });

  // 运行测试
  const runTestsMutation = useMutation({
    mutationFn: async (testType: string) => {
      const response = await fetch(`http://localhost:8000/system/tests/run/${testType}`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`启动测试失败: ${response.status} ${response.statusText}`);
      }
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['test-results'] });
      setIsRunningTests(false);
      // 可以添加成功提示
      console.log('测试启动成功:', data.message);
    },
    onError: (error) => {
      setIsRunningTests(false);
      console.error('测试启动失败:', error.message);
    },
  });

  // 处理运行测试
  const handleRunTests = async (testType: string) => {
    setIsRunningTests(true);
    runTestsMutation.mutate(testType);
  };

  // 导出报告
  const handleExportReport = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      metrics,
      testResults,
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: 'application/json',
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `naxs-system-report-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // 错误状态显示
  if (metricsError) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <AlertCircle className="w-12 h-12 text-red-500" />
        <div className="text-center">
          <h3 className="text-lg font-medium text-gray-900 mb-2">系统指标加载失败</h3>
          <p className="text-sm text-gray-600 mb-4">{metricsError.message}</p>
          <button
            onClick={() => queryClient.invalidateQueries({ queryKey: ['system-metrics'] })}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            重新加载
          </button>
        </div>
      </div>
    );
  }

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
        <span className="ml-2 text-gray-600">加载系统指标中...</span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">系统测试监控</h1>
          <p className="text-gray-600 mt-1">实时监控系统性能和测试状态</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={handleExportReport}
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <Download className="w-4 h-4 mr-2" />
            导出报告
          </button>
          <button
            onClick={() => handleRunTests(selectedTestType)}
            disabled={isRunningTests}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
          >
            {isRunningTests ? (
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Play className="w-4 h-4 mr-2" />
            )}
            {isRunningTests ? '运行中...' : '运行测试'}
          </button>
        </div>
      </div>

      {/* 系统状态概览 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="API响应时间"
          value={metrics?.api.responseTime || 0}
          unit="ms"
          icon={Zap}
          status={metrics?.api.responseTime && metrics.api.responseTime < 200 ? 'healthy' : 'warning'}
          trend={-5.2}
        />
        <MetricCard
          title="成功率"
          value={metrics?.api.successRate || 0}
          unit="%"
          icon={CheckCircle}
          status={metrics?.api.successRate && metrics.api.successRate > 95 ? 'healthy' : 'error'}
          trend={2.1}
        />
        <MetricCard
          title="CPU使用率"
          value={metrics?.system.cpuUsage || 0}
          unit="%"
          icon={Cpu}
          status={metrics?.system.cpuUsage && metrics.system.cpuUsage < 70 ? 'healthy' : 'warning'}
        />
        <MetricCard
          title="内存使用率"
          value={metrics?.system.memoryUsage || 0}
          unit="%"
          icon={Monitor}
          status={metrics?.system.memoryUsage && metrics.system.memoryUsage < 80 ? 'healthy' : 'warning'}
        />
      </div>

      {/* 详细指标 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* API性能 */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center mb-4">
            <Globe className="w-5 h-5 text-blue-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">API性能</h3>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">吞吐量</span>
              <span className="text-sm font-medium">{metrics?.api.throughput || 0} req/s</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">错误率</span>
              <span className="text-sm font-medium text-red-600">{metrics?.api.errorRate || 0}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full" 
                style={{ width: `${Math.min((metrics?.api.successRate || 0), 100)}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* 数据库状态 */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Database className="w-5 h-5 text-green-500 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">数据库</h3>
            </div>
            <StatusIndicator status={metrics?.database.status || 'unknown'} />
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">连接数</span>
              <span className="text-sm font-medium">{metrics?.database.connectionCount || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">查询时间</span>
              <span className="text-sm font-medium">{metrics?.database.queryTime || 0}ms</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">缓存命中率</span>
              <span className="text-sm font-medium text-green-600">{metrics?.database.cacheHitRate || 0}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* 测试结果 */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <BarChart3 className="w-5 h-5 text-purple-500 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">测试结果</h3>
            </div>
            <div className="flex space-x-2">
              {['all', 'api', 'stress', 'integration'].map((type) => (
                <button
                  key={type}
                  onClick={() => setSelectedTestType(type as any)}
                  className={`px-3 py-1 text-sm rounded-md ${
                    selectedTestType === type
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {type === 'all' ? '全部' : type.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="max-h-96 overflow-y-auto">
          {testsError ? (
            <div className="flex flex-col items-center justify-center py-8 space-y-4">
              <AlertCircle className="w-8 h-8 text-red-500" />
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-2">{testsError.message}</p>
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['test-results'] })}
                  className="text-sm text-blue-600 hover:text-blue-700"
                >
                  重新加载测试结果
                </button>
              </div>
            </div>
          ) : testsLoading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 text-blue-500 animate-spin mr-2" />
              <span className="text-gray-600">加载测试结果中...</span>
            </div>
          ) : testResults && testResults.length > 0 ? (
            testResults.map((result) => (
              <TestResultItem key={result.id} result={result} />
            ))
          ) : (
            <div className="text-center py-8 text-gray-500">
              <BarChart3 className="w-8 h-8 mx-auto mb-2 text-gray-300" />
              <p className="text-sm">暂无测试结果</p>
              <p className="text-xs text-gray-400 mt-1">点击"运行测试"按钮开始测试</p>
            </div>
          )}
        </div>
      </div>

      {/* WebSocket状态 */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <Wifi className="w-5 h-5 text-indigo-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">实时连接</h3>
          </div>
          <StatusIndicator status={metrics?.websocket.connected ? 'healthy' : 'error'} />
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{metrics?.websocket.activeConnections || 0}</div>
            <div className="text-sm text-gray-600">活跃连接</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{metrics?.websocket.messageRate || 0}</div>
            <div className="text-sm text-gray-600">消息/秒</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{Math.floor((metrics?.system.uptime || 0) / 3600)}</div>
            <div className="text-sm text-gray-600">运行时间(小时)</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestDashboard;
// @ts-nocheck