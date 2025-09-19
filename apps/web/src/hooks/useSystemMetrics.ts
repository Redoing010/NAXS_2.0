// 系统指标监控Hook
// 实时监控系统性能、资源使用情况和健康状态

import { useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/api';

// 系统健康数据类型
interface SystemHealth {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  activeConnections: number;
  errorRate: number;
  uptime: number;
  loadAverage: number[];
  diskIO: {
    readRate: number;
    writeRate: number;
  };
  networkIO: {
    inboundRate: number;
    outboundRate: number;
  };
}

// 服务状态
interface ServiceStatus {
  serviceName: string;
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping';
  port?: number;
  pid?: number;
  uptime: number;
  memoryUsage: number;
  cpuUsage: number;
  restartCount: number;
  lastRestart?: Date;
  healthCheck?: {
    status: 'healthy' | 'unhealthy' | 'unknown';
    lastCheck: Date;
    responseTime: number;
  };
}

// 数据库状态
interface DatabaseStatus {
  type: 'postgresql' | 'redis' | 'mongodb' | 'mysql';
  status: 'connected' | 'disconnected' | 'error';
  connectionCount: number;
  maxConnections: number;
  queryRate: number;
  avgQueryTime: number;
  slowQueries: number;
  cacheHitRate?: number;
  replicationLag?: number;
}

// 性能指标历史
interface MetricHistory {
  timestamp: Date;
  cpuUsage: number;
  memoryUsage: number;
  networkLatency: number;
  errorRate: number;
  activeConnections: number;
}

// Hook配置
interface UseSystemMetricsConfig {
  interval?: number;
  enabled?: boolean;
  includeServices?: boolean;
  includeDatabases?: boolean;
  includeHistory?: boolean;
  historyLimit?: number;
  onAlert?: (alert: SystemAlert) => void;
  onError?: (error: Error) => void;
}

// 系统告警
interface SystemAlert {
  id: string;
  type: 'cpu' | 'memory' | 'disk' | 'network' | 'service' | 'database';
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  value: number;
  threshold: number;
  timestamp: Date;
}

// Hook返回值
interface UseSystemMetricsReturn {
  systemHealth?: SystemHealth;
  services: ServiceStatus[];
  databases: DatabaseStatus[];
  metricHistory: MetricHistory[];
  alerts: SystemAlert[];
  isLoading: boolean;
  error?: Error;
  lastUpdate?: Date;
  overallHealth: 'healthy' | 'warning' | 'critical';
  refetch: () => Promise<void>;
  getServiceByName: (serviceName: string) => ServiceStatus | undefined;
  getDatabaseByType: (type: string) => DatabaseStatus | undefined;
}

export const useSystemMetrics = (config: UseSystemMetricsConfig = {}): UseSystemMetricsReturn => {
  const {
    interval = 10000, // 10秒
    enabled = true,
    includeServices = true,
    includeDatabases = true,
    includeHistory = true,
    historyLimit = 100,
    onAlert,
    onError
  } = config;

  const [systemHealth, setSystemHealth] = useState<SystemHealth>();
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [databases, setDatabases] = useState<DatabaseStatus[]>([]);
  const [metricHistory, setMetricHistory] = useState<MetricHistory[]>([]);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error>();
  const [lastUpdate, setLastUpdate] = useState<Date>();

  const intervalRef = useRef<NodeJS.Timeout>();
  const alertThresholds = useRef({
    cpu: { warning: 70, critical: 90 },
    memory: { warning: 80, critical: 95 },
    disk: { warning: 85, critical: 95 },
    errorRate: { warning: 1, critical: 5 },
    networkLatency: { warning: 100, critical: 500 }
  });

  // 检查告警条件
  const checkAlerts = useCallback((health: SystemHealth) => {
    const newAlerts: SystemAlert[] = [];
    const now = new Date();

    // CPU使用率告警
    if (health.cpuUsage > alertThresholds.current.cpu.critical) {
      newAlerts.push({
        id: `cpu_${now.getTime()}`,
        type: 'cpu',
        level: 'critical',
        message: `CPU使用率过高: ${health.cpuUsage.toFixed(1)}%`,
        value: health.cpuUsage,
        threshold: alertThresholds.current.cpu.critical,
        timestamp: now
      });
    } else if (health.cpuUsage > alertThresholds.current.cpu.warning) {
      newAlerts.push({
        id: `cpu_${now.getTime()}`,
        type: 'cpu',
        level: 'warning',
        message: `CPU使用率较高: ${health.cpuUsage.toFixed(1)}%`,
        value: health.cpuUsage,
        threshold: alertThresholds.current.cpu.warning,
        timestamp: now
      });
    }

    // 内存使用率告警
    if (health.memoryUsage > alertThresholds.current.memory.critical) {
      newAlerts.push({
        id: `memory_${now.getTime()}`,
        type: 'memory',
        level: 'critical',
        message: `内存使用率过高: ${health.memoryUsage.toFixed(1)}%`,
        value: health.memoryUsage,
        threshold: alertThresholds.current.memory.critical,
        timestamp: now
      });
    } else if (health.memoryUsage > alertThresholds.current.memory.warning) {
      newAlerts.push({
        id: `memory_${now.getTime()}`,
        type: 'memory',
        level: 'warning',
        message: `内存使用率较高: ${health.memoryUsage.toFixed(1)}%`,
        value: health.memoryUsage,
        threshold: alertThresholds.current.memory.warning,
        timestamp: now
      });
    }

    // 磁盘使用率告警
    if (health.diskUsage > alertThresholds.current.disk.critical) {
      newAlerts.push({
        id: `disk_${now.getTime()}`,
        type: 'disk',
        level: 'critical',
        message: `磁盘使用率过高: ${health.diskUsage.toFixed(1)}%`,
        value: health.diskUsage,
        threshold: alertThresholds.current.disk.critical,
        timestamp: now
      });
    } else if (health.diskUsage > alertThresholds.current.disk.warning) {
      newAlerts.push({
        id: `disk_${now.getTime()}`,
        type: 'disk',
        level: 'warning',
        message: `磁盘使用率较高: ${health.diskUsage.toFixed(1)}%`,
        value: health.diskUsage,
        threshold: alertThresholds.current.disk.warning,
        timestamp: now
      });
    }

    // 网络延迟告警
    if (health.networkLatency > alertThresholds.current.networkLatency.critical) {
      newAlerts.push({
        id: `network_${now.getTime()}`,
        type: 'network',
        level: 'critical',
        message: `网络延迟过高: ${health.networkLatency.toFixed(0)}ms`,
        value: health.networkLatency,
        threshold: alertThresholds.current.networkLatency.critical,
        timestamp: now
      });
    } else if (health.networkLatency > alertThresholds.current.networkLatency.warning) {
      newAlerts.push({
        id: `network_${now.getTime()}`,
        type: 'network',
        level: 'warning',
        message: `网络延迟较高: ${health.networkLatency.toFixed(0)}ms`,
        value: health.networkLatency,
        threshold: alertThresholds.current.networkLatency.warning,
        timestamp: now
      });
    }

    // 错误率告警
    if (health.errorRate > alertThresholds.current.errorRate.critical) {
      newAlerts.push({
        id: `error_${now.getTime()}`,
        type: 'service',
        level: 'critical',
        message: `系统错误率过高: ${health.errorRate.toFixed(2)}%`,
        value: health.errorRate,
        threshold: alertThresholds.current.errorRate.critical,
        timestamp: now
      });
    } else if (health.errorRate > alertThresholds.current.errorRate.warning) {
      newAlerts.push({
        id: `error_${now.getTime()}`,
        type: 'service',
        level: 'warning',
        message: `系统错误率较高: ${health.errorRate.toFixed(2)}%`,
        value: health.errorRate,
        threshold: alertThresholds.current.errorRate.warning,
        timestamp: now
      });
    }

    // 触发告警回调
    newAlerts.forEach(alert => {
      if (onAlert) onAlert(alert);
    });

    // 更新告警列表（保留最近100条）
    setAlerts(prev => [...newAlerts, ...prev].slice(0, 100));
  }, [onAlert]);

  // 获取系统指标
  const fetchSystemMetrics = useCallback(async () => {
    if (!enabled) return;

    setIsLoading(true);
    setError(undefined);

    try {
      const promises = [];

      // 获取系统健康状态
      promises.push(apiService.getSystemHealth());

      // 获取服务状态
      if (includeServices) {
        promises.push(apiService.getServiceStatus());
      }

      // 获取数据库状态
      if (includeDatabases) {
        promises.push(apiService.getDatabaseStatus());
      }

      const results = await Promise.all(promises);
      let resultIndex = 0;

      // 处理系统健康数据
      const healthResponse = results[resultIndex++];
      if (healthResponse.success) {
        const healthData: SystemHealth = {
          cpuUsage: healthResponse.data.cpu_usage || 0,
          memoryUsage: healthResponse.data.memory_usage || 0,
          diskUsage: healthResponse.data.disk_usage || 0,
          networkLatency: healthResponse.data.network_latency || 0,
          activeConnections: healthResponse.data.active_connections || 0,
          errorRate: healthResponse.data.error_rate || 0,
          uptime: healthResponse.data.uptime || 0,
          loadAverage: healthResponse.data.load_average || [0, 0, 0],
          diskIO: {
            readRate: healthResponse.data.disk_io?.read_rate || 0,
            writeRate: healthResponse.data.disk_io?.write_rate || 0
          },
          networkIO: {
            inboundRate: healthResponse.data.network_io?.inbound_rate || 0,
            outboundRate: healthResponse.data.network_io?.outbound_rate || 0
          }
        };

        setSystemHealth(healthData);

        // 检查告警
        checkAlerts(healthData);

        // 更新历史记录
        if (includeHistory) {
          const historyItem: MetricHistory = {
            timestamp: new Date(),
            cpuUsage: healthData.cpuUsage,
            memoryUsage: healthData.memoryUsage,
            networkLatency: healthData.networkLatency,
            errorRate: healthData.errorRate,
            activeConnections: healthData.activeConnections
          };

          setMetricHistory(prev => [historyItem, ...prev].slice(0, historyLimit));
        }
      }

      // 处理服务状态
      if (includeServices && results[resultIndex]) {
        const serviceResponse = results[resultIndex];
        if (serviceResponse.success) {
          const serviceData: ServiceStatus[] = serviceResponse.data.map((item: any) => ({
            serviceName: item.service_name,
            status: item.status,
            port: item.port,
            pid: item.pid,
            uptime: item.uptime || 0,
            memoryUsage: item.memory_usage || 0,
            cpuUsage: item.cpu_usage || 0,
            restartCount: item.restart_count || 0,
            lastRestart: item.last_restart ? new Date(item.last_restart) : undefined,
            healthCheck: item.health_check ? {
              status: item.health_check.status,
              lastCheck: new Date(item.health_check.last_check),
              responseTime: item.health_check.response_time
            } : undefined
          }));

          setServices(serviceData);
        }
        resultIndex++;
      }

      // 处理数据库状态
      if (includeDatabases && results[resultIndex]) {
        const dbResponse = results[resultIndex];
        if (dbResponse.success) {
          const dbData: DatabaseStatus[] = dbResponse.data.map((item: any) => ({
            type: item.type,
            status: item.status,
            connectionCount: item.connection_count || 0,
            maxConnections: item.max_connections || 0,
            queryRate: item.query_rate || 0,
            avgQueryTime: item.avg_query_time || 0,
            slowQueries: item.slow_queries || 0,
            cacheHitRate: item.cache_hit_rate,
            replicationLag: item.replication_lag
          }));

          setDatabases(dbData);
        }
      }

      setLastUpdate(new Date());

    } catch (err) {
      console.error('Failed to fetch system metrics:', err);
      const fetchError = err as Error;
      setError(fetchError);
      if (onError) onError(fetchError);
    } finally {
      setIsLoading(false);
    }
  }, [enabled, includeServices, includeDatabases, includeHistory, historyLimit, checkAlerts, onError]);

  // 手动刷新
  const refetch = useCallback(async () => {
    await fetchSystemMetrics();
  }, [fetchSystemMetrics]);

  // 根据名称获取服务
  const getServiceByName = useCallback((serviceName: string): ServiceStatus | undefined => {
    return services.find(service => service.serviceName === serviceName);
  }, [services]);

  // 根据类型获取数据库
  const getDatabaseByType = useCallback((type: string): DatabaseStatus | undefined => {
    return databases.find(db => db.type === type);
  }, [databases]);

  // 计算整体健康状态
  const overallHealth: 'healthy' | 'warning' | 'critical' = (() => {
    if (!systemHealth) return 'healthy';

    const criticalConditions = [
      systemHealth.cpuUsage > 90,
      systemHealth.memoryUsage > 95,
      systemHealth.diskUsage > 95,
      systemHealth.errorRate > 5,
      services.some(s => s.status === 'error'),
      databases.some(db => db.status === 'error')
    ];

    const warningConditions = [
      systemHealth.cpuUsage > 70,
      systemHealth.memoryUsage > 80,
      systemHealth.diskUsage > 85,
      systemHealth.errorRate > 1,
      systemHealth.networkLatency > 100,
      services.some(s => s.status === 'stopped'),
      databases.some(db => db.status === 'disconnected')
    ];

    if (criticalConditions.some(condition => condition)) {
      return 'critical';
    } else if (warningConditions.some(condition => condition)) {
      return 'warning';
    } else {
      return 'healthy';
    }
  })();

  // 设置定时轮询
  useEffect(() => {
    if (!enabled) return;

    // 立即获取一次数据
    fetchSystemMetrics();

    // 设置定时轮询
    if (interval > 0) {
      intervalRef.current = setInterval(fetchSystemMetrics, interval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, interval, fetchSystemMetrics]);

  return {
    systemHealth,
    services,
    databases,
    metricHistory,
    alerts,
    isLoading,
    error,
    lastUpdate,
    overallHealth,
    refetch,
    getServiceByName,
    getDatabaseByType
  };
};

export default useSystemMetrics;