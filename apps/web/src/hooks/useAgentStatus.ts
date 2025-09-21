// @ts-nocheck
// Agent状态监控Hook
// 实时监控Agent运行状态、性能指标和任务执行情况

import { useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/api';

// Agent指标数据类型
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
  totalExecutions: number;
  errorCount: number;
  uptime: number;
}

// 工具执行记录
interface ToolExecution {
  executionId: string;
  toolName: string;
  agentId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  duration?: number;
  error?: string;
}

// Agent任务信息
interface AgentTask {
  taskId: string;
  agentId: string;
  type: string;
  description: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  progress: number;
  result?: any;
  error?: string;
}

// Hook配置
interface UseAgentStatusConfig {
  interval?: number;
  enabled?: boolean;
  agentIds?: string[];
  includeMetrics?: boolean;
  includeTasks?: boolean;
  includeExecutions?: boolean;
  onStatusChange?: (agentId: string, oldStatus: string, newStatus: string) => void;
  onTaskComplete?: (task: AgentTask) => void;
  onError?: (error: Error) => void;
}

// Hook返回值
interface UseAgentStatusReturn {
  agents: AgentMetrics[];
  tasks: AgentTask[];
  executions: ToolExecution[];
  isLoading: boolean;
  error?: Error;
  lastUpdate?: Date;
  totalAgents: number;
  activeAgents: number;
  errorAgents: number;
  avgResponseTime: number;
  totalExecutions: number;
  successRate: number;
  refetch: () => Promise<void>;
  getAgentById: (agentId: string) => AgentMetrics | undefined;
  getAgentTasks: (agentId: string) => AgentTask[];
  getAgentExecutions: (agentId: string) => ToolExecution[];
}

export const useAgentStatus = (config: UseAgentStatusConfig = {}): UseAgentStatusReturn => {
  const {
    interval = 5000,
    enabled = true,
    agentIds,
    includeMetrics = true,
    includeTasks = true,
    includeExecutions = true,
    onStatusChange,
    onTaskComplete,
    onError
  } = config;

  const [agents, setAgents] = useState<AgentMetrics[]>([]);
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [executions, setExecutions] = useState<ToolExecution[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error>();
  const [lastUpdate, setLastUpdate] = useState<Date>();

  const intervalRef = useRef<NodeJS.Timeout>();
  const previousAgentsRef = useRef<Map<string, string>>(new Map());
  const previousTasksRef = useRef<Map<string, string>>(new Map());

  // 获取Agent状态数据
  const fetchAgentStatus = useCallback(async () => {
    if (!enabled) return;

    setIsLoading(true);
    setError(undefined);

    try {
      const promises = [];

      // 获取Agent指标
      if (includeMetrics) {
        promises.push(apiService.getAgentMetrics({ agentIds }));
      }

      // 获取任务列表
      if (includeTasks) {
        promises.push(apiService.getAgentTasks({ agentIds, limit: 100 }));
      }

      // 获取执行记录
      if (includeExecutions) {
        promises.push(apiService.getToolExecutions({ agentIds, limit: 50 }));
      }

      const results = await Promise.all(promises);
      let resultIndex = 0;

      // 处理Agent指标
      if (includeMetrics && results[resultIndex]) {
        const agentResponse = results[resultIndex];
        if (agentResponse.success) {
          const agentData: AgentMetrics[] = agentResponse.data.map((item: any) => ({
            agentId: item.agent_id,
            name: item.name,
            status: item.status,
            currentTask: item.current_task,
            progress: item.progress,
            toolsUsed: item.tools_used || 0,
            successRate: item.success_rate || 0,
            avgResponseTime: item.avg_response_time || 0,
            lastActivity: new Date(item.last_activity),
            totalExecutions: item.total_executions || 0,
            errorCount: item.error_count || 0,
            uptime: item.uptime || 0
          }));

          // 检查状态变化
          agentData.forEach(agent => {
            const previousStatus = previousAgentsRef.current.get(agent.agentId);
            if (previousStatus && previousStatus !== agent.status && onStatusChange) {
              onStatusChange(agent.agentId, previousStatus, agent.status);
            }
            previousAgentsRef.current.set(agent.agentId, agent.status);
          });

          setAgents(agentData);
        }
        resultIndex++;
      }

      // 处理任务数据
      if (includeTasks && results[resultIndex]) {
        const taskResponse = results[resultIndex];
        if (taskResponse.success) {
          const taskData: AgentTask[] = taskResponse.data.map((item: any) => ({
            taskId: item.task_id,
            agentId: item.agent_id,
            type: item.type,
            description: item.description,
            status: item.status,
            priority: item.priority,
            createdAt: new Date(item.created_at),
            startedAt: item.started_at ? new Date(item.started_at) : undefined,
            completedAt: item.completed_at ? new Date(item.completed_at) : undefined,
            progress: item.progress || 0,
            result: item.result,
            error: item.error
          }));

          // 检查任务完成
          taskData.forEach(task => {
            const previousStatus = previousTasksRef.current.get(task.taskId);
            if (previousStatus !== 'completed' && task.status === 'completed' && onTaskComplete) {
              onTaskComplete(task);
            }
            previousTasksRef.current.set(task.taskId, task.status);
          });

          setTasks(taskData);
        }
        resultIndex++;
      }

      // 处理执行记录
      if (includeExecutions && results[resultIndex]) {
        const executionResponse = results[resultIndex];
        if (executionResponse.success) {
          const executionData: ToolExecution[] = executionResponse.data.map((item: any) => ({
            executionId: item.execution_id,
            toolName: item.tool_name,
            agentId: item.agent_id,
            status: item.status,
            startTime: new Date(item.start_time),
            endTime: item.end_time ? new Date(item.end_time) : undefined,
            duration: item.duration,
            error: item.error
          }));

          setExecutions(executionData);
        }
      }

      setLastUpdate(new Date());

    } catch (err) {
      console.error('Failed to fetch agent status:', err);
      const fetchError = err as Error;
      setError(fetchError);
      if (onError) onError(fetchError);
    } finally {
      setIsLoading(false);
    }
  }, [enabled, agentIds, includeMetrics, includeTasks, includeExecutions, onStatusChange, onTaskComplete, onError]);

  // 手动刷新
  const refetch = useCallback(async () => {
    await fetchAgentStatus();
  }, [fetchAgentStatus]);

  // 根据ID获取Agent
  const getAgentById = useCallback((agentId: string): AgentMetrics | undefined => {
    return agents.find(agent => agent.agentId === agentId);
  }, [agents]);

  // 获取Agent的任务
  const getAgentTasks = useCallback((agentId: string): AgentTask[] => {
    return tasks.filter(task => task.agentId === agentId);
  }, [tasks]);

  // 获取Agent的执行记录
  const getAgentExecutions = useCallback((agentId: string): ToolExecution[] => {
    return executions.filter(execution => execution.agentId === agentId);
  }, [executions]);

  // 计算统计指标
  const totalAgents = agents.length;
  const activeAgents = agents.filter(agent => agent.status === 'running').length;
  const errorAgents = agents.filter(agent => agent.status === 'error').length;
  const avgResponseTime = agents.length > 0 
    ? agents.reduce((sum, agent) => sum + agent.avgResponseTime, 0) / agents.length 
    : 0;
  const totalExecutions = agents.reduce((sum, agent) => sum + agent.totalExecutions, 0);
  const successRate = agents.length > 0
    ? agents.reduce((sum, agent) => sum + agent.successRate, 0) / agents.length
    : 0;

  // 设置定时轮询
  useEffect(() => {
    if (!enabled) return;

    // 立即获取一次数据
    fetchAgentStatus();

    // 设置定时轮询
    if (interval > 0) {
      intervalRef.current = setInterval(fetchAgentStatus, interval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, interval, fetchAgentStatus]);

  return {
    agents,
    tasks,
    executions,
    isLoading,
    error,
    lastUpdate,
    totalAgents,
    activeAgents,
    errorAgents,
    avgResponseTime,
    totalExecutions,
    successRate,
    refetch,
    getAgentById,
    getAgentTasks,
    getAgentExecutions
  };
};

export default useAgentStatus;
// @ts-nocheck