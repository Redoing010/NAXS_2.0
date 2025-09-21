// API服务层 - 统一管理所有API调用

import axios from 'axios';
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import type { MarketData, Strategy, StockRecommendation, ChatMessage } from '../store';

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

  constructor(baseURL: string = 'http://localhost:8001') {
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
      const response: AxiosResponse<ApiResponse<T>> = await this.client.request(requestConfig);
      return response.data;
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

  // 发送Agent消息
  async sendAgentMessage(data: {
    message: string;
    model?: string;
    provider?: string;
    context?: Array<{ role: string; content: string }>;
    tools_enabled?: boolean;
    knowledge_graph?: {
      enabled: boolean;
      focus?: string;
      entities?: any[];
      relations?: any[];
    };
    stream?: boolean;
  }): Promise<ApiResponse<any>> {
    return this.post('/agent/chat', data);
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
  async getAvailableModels(): Promise<ApiResponse<any>> {
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

  // ==================== 用户偏好相关 ====================
  
  async getUserPreferences(): Promise<ApiResponse<any>> {
    return this.get('/user/preferences');
  }

  async updateUserPreferences(preferences: any): Promise<ApiResponse<any>> {
    return this.put('/user/preferences', preferences);
  }

  async getUserPortfolio(): Promise<ApiResponse<{
    positions: Array<{
      symbol: string;
      quantity: number;
      avg_cost: number;
      current_price: number;
      unrealized_pnl: number;
    }>;
    cash: number;
    total_value: number;
  }>> {
    return this.get('/user/portfolio');
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