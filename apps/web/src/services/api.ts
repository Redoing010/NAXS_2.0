// API服务层 - 统一管理所有API调用

import axios from 'axios';
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import type { MarketData, Strategy, StockRecommendation, ChatMessage } from '../store';

// API基础地址，支持通过Vite环境变量覆盖
const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:3001/api';

// 仪表盘概览数据接口
export interface DashboardOverview {
  headline: string;
  subheadline: string;
  timestamp: string;
  account: {
    net_asset: number;
    week_pnl: number;
    available_cash: number;
    score: string;
    score_trend: number;
    goal: { completed: number; target: number };
    risk_level: string;
    risk_score: number;
    risk_comment: string;
  };
  market_heat: {
    score: number;
    change: number;
    grade: string;
    north_bound: number;
    ai_sentiment: string;
  };
  performance_trend: Array<{ date: string; portfolio: number; benchmark: number }>;
  fund_flow: Array<{ date: string; net_flow: number; forecast: number }>;
  ai_insights: Array<{ title: string; detail: string }>;
  quick_prompts: string[];
  profile: {
    name: string;
    avatar?: string;
    badges?: string[];
    greeting?: string;
  };
  news: Array<{ title: string; source: string; timestamp: string }>;
}

// 用户档案
export interface UserProfileSummary {
  id?: string;
  name: string;
  phone?: string;
  avatar?: string;
  city?: string;
  role?: string;
  greeting?: string;
  badges?: string[];
  updated_at?: string;
}

// 个性化偏好
export interface UserPersonalPreferences {
  experience_level: string;
  experience_label: string;
  asset_scale: string;
  risk_score: number;
  risk_attitude: string;
  investment_horizon: string;
  strategy_style: string;
  focus_industries: string[];
  excluded_industries: string[];
  ai_auto_adjust: boolean;
  notes?: string;
  updated_at?: string;
}

// 助手聊天响应
export interface AssistantChatResult {
  reply: string;
  insights: string[];
  timestamp: string;
}

// API响应接口
interface ApiResponse<T = any> {
  status: 'ok' | 'error';
  data?: T;
  message?: string;
  error?: string;
}

// 分页响应接口
interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
}

// 训练配置接口
interface TrainConfig {
  region?: string;
  provider_uri?: string | null;
  instruments?: string;
  start_time?: string;
  end_time?: string;
  fit_start_time?: string | null;
  fit_end_time?: string;
  work_dir?: string;
}

// 回测报告接口
interface BacktestReport {
  metrics: {
    total_return: number;
    annual_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    volatility: number;
  };
  prices_used: boolean;
  positions: Array<{
    date: string;
    symbol: string;
    position: number;
    price: number;
  }>;
  performance: Array<{
    date: string;
    portfolio_value: number;
    benchmark_value: number;
  }>;
}

// LLM聊天请求接口
interface ChatRequest {
  message: string;
  model?: string;
  context?: string[];
  tools?: string[];
  stream?: boolean;
}

// LLM聊天响应接口
interface ChatResponse {
  message: string;
  model: string;
  tokens_used: number;
  execution_time: number;
  tools_used?: string[];
  metadata?: Record<string, any>;
}

class ApiService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        // 添加认证token（如果有）
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        
        // 处理认证错误
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        
        return Promise.reject(error);
      }
    );
  }

  // 通用请求方法
  private async request<T = any>(
    config: AxiosRequestConfig,
    baseURL?: string
  ): Promise<ApiResponse<T>> {
    try {
      const requestConfig = baseURL ? { ...config, baseURL } : config;
      const response: AxiosResponse<ApiResponse<T> | T> = await this.client.request(requestConfig);
      const payload = response.data as any;

      if (payload && typeof payload === 'object' && 'status' in payload) {
        return payload as ApiResponse<T>;
      }

      return {
        status: 'ok',
        data: payload as T,
      };
    } catch (error: any) {
      if (error.response?.data) {
        return error.response.data;
      }
      return {
        status: 'error',
        error: error.message || 'Network error',
      };
    }
  }

  // GET请求
  private async get<T = any>(url: string, params?: any): Promise<ApiResponse<T>> {
    return this.request<T>({ method: 'GET', url, params });
  }

  // POST请求
  private async post<T = any>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>({ method: 'POST', url, data });
  }

  // PUT请求
  private async put<T = any>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>({ method: 'PUT', url, data });
  }

  // DELETE请求
  private async delete<T = any>(url: string): Promise<ApiResponse<T>> {
    return this.request<T>({ method: 'DELETE', url });
  }

  // ==================== 认证相关 ====================
  
  async login(username: string, password: string): Promise<ApiResponse<{ token: string; user: any }>> {
    return this.post('/auth/login', { username, password });
  }

  async logout(): Promise<ApiResponse> {
    return this.post('/auth/logout');
  }

  async refreshToken(): Promise<ApiResponse<{ token: string }>> {
    return this.post('/auth/refresh');
  }

  // ==================== 量化相关 ====================
  
  async trainModel(config: TrainConfig): Promise<ApiResponse> {
    return this.post('/naxs/alpha/train', config);
  }

  async predict(config: TrainConfig): Promise<ApiResponse> {
    return this.post('/naxs/alpha/predict', config);
  }

  async getLatestSignals(limit: number = 100): Promise<ApiResponse<PaginatedResponse<any>>> {
    return this.get('/naxs/alpha/latest', { limit });
  }

  async getBacktestReport(): Promise<ApiResponse<BacktestReport>> {
    return this.get('/naxs/backtest/report');
  }

  async refreshPrices(symbols: string[], start: string, end: string): Promise<ApiResponse> {
    return this.post('/naxs/data/refresh_prices', { symbols, start, end });
  }

  async getPrices(limit: number = 50): Promise<ApiResponse<PaginatedResponse<any>>> {
    return this.get('/naxs/data/prices', { limit });
  }

  // ==================== 市场数据相关 ====================
  
  async getMarketIndices(): Promise<ApiResponse<MarketData[]>> {
    return this.get('/market/indices');
  }

  async getWatchlistData(symbols: string[]): Promise<ApiResponse<MarketData[]>> {
    return this.post('/market/watchlist', { symbols });
  }

  async getStockQuote(symbol: string): Promise<ApiResponse<MarketData>> {
    return this.get(`/market/quote/${symbol}`);
  }

  async getHistoricalData(
    symbol: string,
    period: string = '1y',
    interval: string = '1d'
  ): Promise<ApiResponse<any[]>> {
    return this.get(`/market/history/${symbol}`, { period, interval });
  }

  async searchStocks(query: string): Promise<ApiResponse<Array<{ symbol: string; name: string }>>> {
    return this.get('/market/search', { q: query });
  }

  // ==================== 策略相关 ====================
  
  async getStrategies(): Promise<ApiResponse<Strategy[]>> {
    return this.get('/strategies');
  }

  async getStrategy(id: string): Promise<ApiResponse<Strategy>> {
    return this.get(`/strategies/${id}`);
  }

  async createStrategy(strategy: Omit<Strategy, 'id'>): Promise<ApiResponse<Strategy>> {
    return this.post('/strategies', strategy);
  }

  async updateStrategy(id: string, updates: Partial<Strategy>): Promise<ApiResponse<Strategy>> {
    return this.put(`/strategies/${id}`, updates);
  }

  async deleteStrategy(id: string): Promise<ApiResponse> {
    return this.delete(`/strategies/${id}`);
  }

  async backtestStrategy(
    id: string,
    config: {
      start_date: string;
      end_date: string;
      initial_capital: number;
    }
  ): Promise<ApiResponse<BacktestReport>> {
    return this.post(`/strategies/${id}/backtest`, config);
  }

  async runBacktest(
    strategyId: string,
    config: {
      start_date: string;
      end_date: string;
      initial_capital: number;
    }
  ): Promise<ApiResponse<BacktestReport>> {
    return this.post(`/strategies/${strategyId}/backtest`, config);
  }

  // ==================== 选股推荐相关 ====================
  
  async getRecommendations(filters?: {
    min_score?: number;
    max_risk?: string;
    sectors?: string[];
  }): Promise<ApiResponse<StockRecommendation[]>> {
    return this.get('/recommendations', filters);
  }

  async generateRecommendations(config: {
    user_preferences: any;
    market_conditions: any;
  }): Promise<ApiResponse<StockRecommendation[]>> {
    return this.post('/recommendations/generate', config);
  }

  async getRecommendationDetails(symbol: string): Promise<ApiResponse<{
    recommendation: StockRecommendation;
    analysis: {
      technical_analysis: any;
      fundamental_analysis: any;
      sentiment_analysis: any;
      risk_analysis: any;
    };
  }>> {
    return this.get(`/recommendations/${symbol}`);
  }

  // ==================== LLM聊天相关 ====================
  
  async sendChatMessage(request: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    return this.post('/chat/message', request);
  }

  // ==================== NAXS 2.0 UI增强 ====================

  async getDashboardOverview(): Promise<ApiResponse<DashboardOverview>> {
    return this.get('/dashboard/overview');
  }

  async getDashboardBriefing(): Promise<ApiResponse<{ title: string; highlights: string[] }>> {
    return this.get('/dashboard/briefing');
  }

  async getUserProfile(): Promise<ApiResponse<UserProfileSummary>> {
    return this.get('/user/profile');
  }

  async getUserPreferences(): Promise<ApiResponse<UserPersonalPreferences>> {
    return this.get('/user/preferences');
  }

  async updateUserPreferences(preferences: Partial<UserPersonalPreferences>): Promise<ApiResponse<UserPersonalPreferences>> {
    return this.post('/user/preferences', preferences);
  }

  async resetUserPreferences(): Promise<ApiResponse<UserPersonalPreferences>> {
    return this.post('/user/preferences/reset');
  }

  async getAssistantPrompts(): Promise<ApiResponse<{ items: string[] }>> {
    return this.get('/assistant/prompts');
  }

  async sendAssistantMessage(message: string): Promise<ApiResponse<AssistantChatResult>> {
    return this.post('/assistant/chat', { message });
  }

  // 发送Agent消息
  async sendAgentMessage(data: {
    message: string;
    model?: string;
    context?: Array<{role: string; content: string}>;
    tools_enabled?: boolean;
    stream?: boolean;
  }): Promise<ApiResponse<any> & Record<string, any>> {
    const assistant = await this.sendAssistantMessage(data.message);

    if (assistant.status !== 'ok' || !assistant.data) {
      return assistant as ApiResponse<any>;
    }

    const reply = assistant.data;
    const payload = {
      content: reply.reply,
      message: reply.reply,
      model: data.model ?? 'naxs-assistant',
      tokens_used: reply.reply.length,
      execution_time: 0,
      tools_used: reply.insights ?? [],
      confidence: 0.86,
      usage: { total_tokens: reply.reply.length },
      reasoning: (reply.insights ?? []).join('；'),
    };

    return {
      status: 'ok',
      data: payload,
      ...payload,
    };
  }

  // 执行工具调用
  async executeTool(data: {
    tool_name: string;
    parameters: any;
  }): Promise<ApiResponse<any>> {
    return this.post('/agent/tools/execute', data);
  }

  // 设置管理
  async getSettings(): Promise<ApiResponse<any>> {
    return this.get('/settings');
  }

  async updateSettings(settings: any): Promise<ApiResponse<any>> {
    return this.put('/settings', settings);
  }

  async resetSettings(): Promise<ApiResponse> {
    return this.post('/settings/reset');
  }

  // 模型连接测试
  async testModelConnection(providerId: string): Promise<ApiResponse<any>> {
    return this.post('/models/test-connection', { provider: providerId });
  }

  // 获取可用模型
  async getAvailableModels(): Promise<ApiResponse<string[]>> {
    return this.get('/models/available');
  }

  // 模型使用统计
  async getModelUsageStats(timeRange?: string): Promise<ApiResponse<any>> {
    return this.get('/models/usage', timeRange ? { range: timeRange } : undefined);
  }

  async getChatHistory(limit: number = 50): Promise<ApiResponse<ChatMessage[]>> {
    return this.get('/chat/history', { limit });
  }

  async clearChatHistory(): Promise<ApiResponse> {
    return this.delete('/chat/history');
  }



  async getAvailableTools(): Promise<ApiResponse<Array<{
    name: string;
    description: string;
    parameters: any;
  }>>> {
    return this.get('/chat/tools');
  }

  // ==================== Agent相关 ====================
  
  async executeAgentTask(task: {
    type: string;
    parameters: any;
    context?: any;
  }): Promise<ApiResponse<any>> {
    return this.post('/agent/execute', task);
  }

  async getAgentStatus(): Promise<ApiResponse<{
    status: string;
    active_tasks: number;
    completed_tasks: number;
    failed_tasks: number;
  }>> {
    return this.get('/agent/status');
  }

  async getAgentTools(): Promise<ApiResponse<Array<{
    name: string;
    description: string;
    category: string;
    enabled: boolean;
  }>>> {
    return this.get('/agent/tools');
  }

  // ==================== 知识图谱相关 ====================
  
  async searchKnowledgeGraph(query: {
    entity?: string;
    relation?: string;
    limit?: number;
  }): Promise<ApiResponse<any[]>> {
    return this.post('/knowledge/search', query);
  }

  async getEntityDetails(entityId: string): Promise<ApiResponse<any>> {
    return this.get(`/knowledge/entity/${entityId}`);
  }

  async getRelatedEntities(
    entityId: string,
    relationTypes?: string[]
  ): Promise<ApiResponse<any[]>> {
    return this.get(`/knowledge/entity/${entityId}/related`, { relationTypes });
  }

  // ==================== 实时数据相关 ====================
  
  async subscribeToRealTimeData(subscriptions: {
    type: string;
    symbols?: string[];
    parameters?: any;
  }[]): Promise<ApiResponse<{ subscription_id: string }>> {
    return this.post('/realtime/subscribe', { subscriptions });
  }

  async unsubscribeFromRealTimeData(subscriptionId: string): Promise<ApiResponse> {
    return this.delete(`/realtime/subscribe/${subscriptionId}`);
  }

  async getRealtimeServiceStatus(): Promise<ApiResponse<{
    status: string;
    connections: number;
    subscriptions: number;
    uptime: number;
  }>> {
    return this.get('/realtime/status');
  }

  // ==================== 系统相关 ====================
  
  async getSystemStatus(): Promise<ApiResponse<{
    status: string;
    version: string;
    uptime: number;
    services: Record<string, {
      status: string;
      last_check: string;
    }>;
  }>> {
    return this.get('/system/status');
  }

  async getSystemMetrics(): Promise<ApiResponse<{
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_io: {
      bytes_sent: number;
      bytes_received: number;
    };
  }>> {
    return this.get('/system/metrics');
  }

  async getSystemLogs(
    level?: string,
    limit?: number
  ): Promise<ApiResponse<Array<{
    timestamp: string;
    level: string;
    message: string;
    module: string;
  }>>> {
    return this.get('/system/logs', { level, limit });
  }

  // ==================== 工具方法 ====================
  
  // 设置认证token
  setAuthToken(token: string) {
    localStorage.setItem('auth_token', token);
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  // 清除认证token
  clearAuthToken() {
    localStorage.removeItem('auth_token');
    delete this.client.defaults.headers.common['Authorization'];
  }

  // 获取当前认证token
  getAuthToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  // 检查是否已认证
  isAuthenticated(): boolean {
    return !!this.getAuthToken();
  }

  // 更新基础URL
  updateBaseURL(baseURL: string) {
    this.baseURL = baseURL;
    this.client.defaults.baseURL = baseURL;
  }

  // 获取当前基础URL
  getBaseURL(): string {
    return this.baseURL;
  }
}

// 创建API服务实例
export const apiService = new ApiService();

// 导出类型
export type {
  ApiResponse,
  PaginatedResponse,
  TrainConfig,
  BacktestReport,
  ChatRequest,
  ChatResponse,
};

// 导出默认实例
export default apiService;