// 全局状态管理 - Zustand Store

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// 市场数据接口
export interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

// 聊天消息接口
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    tokens?: number;
    model?: string;
    tools_used?: string[];
    execution_time?: number;
    cost?: number;
    confidence?: number;
    provider?: string;
  };
}

// 大模型选项
export interface LLMModelOption {
  id: string;
  name: string;
  provider: string;
  model: string;
  context: string;
  description?: string;
  capabilities?: string[];
  streaming?: boolean;
  status?: 'online' | 'degraded' | 'offline';
}

// 知识图谱实体
export interface KnowledgeGraphEntity {
  id: string;
  name: string;
  type?: string;
  description?: string;
  attributes?: Record<string, any>;
  score?: number;
}

// 知识图谱关系
export interface KnowledgeGraphRelation {
  id: string;
  source: string;
  target: string;
  type: string;
  weight?: number;
  metadata?: Record<string, any>;
}

// 知识图谱查询历史
export interface KnowledgeGraphQueryHistory {
  id: string;
  query: string;
  relation?: string;
  timestamp: string;
  results: number;
}

// 知识图谱上下文
export interface KnowledgeGraphContextPayload {
  enabled: boolean;
  focus?: string;
  entities?: KnowledgeGraphEntity[];
  relations?: KnowledgeGraphRelation[];
}

// 知识图谱状态
export interface KnowledgeGraphState {
  enabled: boolean;
  loading: boolean;
  error?: string | null;
  searchResults: KnowledgeGraphEntity[];
  selectedEntity?: KnowledgeGraphEntity;
  relations: KnowledgeGraphRelation[];
  history: KnowledgeGraphQueryHistory[];
}

// 策略信息接口
export interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'momentum' | 'mean_reversion' | 'arbitrage' | 'multi_factor' | 'ml_based';
  status: 'active' | 'inactive' | 'backtesting';
  performance: {
    totalReturn: number;
    annualReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
  };
  config: {
    capital: number;
    max_position: number;
    stop_loss: number;
    take_profit: number;
    rebalance_frequency: string;
  };
  factors: Array<{
    name: string;
    weight: number;
  }>;
  created_at: string;
  updated_at: string;
  lastUpdated: string;
}

// 选股推荐接口
export interface StockRecommendation {
  symbol: string;
  name: string;
  score: number;
  reason: string;
  factors: {
    technical: number;
    fundamental: number;
    sentiment: number;
    momentum: number;
  };
  targetPrice: number;
  riskLevel: 'low' | 'medium' | 'high';
  timestamp: string;
}

// 用户偏好接口
export interface UserPreferences {
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentHorizon: 'short' | 'medium' | 'long';
  preferredSectors: string[];
  excludedSectors: string[];
  maxPositionSize: number;
  autoRebalance: boolean;
  notifications: {
    priceAlerts: boolean;
    strategyUpdates: boolean;
    marketNews: boolean;
  };
}

// WebSocket连接状态
export interface WebSocketState {
  connected: boolean;
  reconnecting: boolean;
  lastConnected?: string;
  subscriptions: string[];
}

// 主应用状态接口
interface AppState {
  // 用户状态
  user: {
    isAuthenticated: boolean;
    username?: string;
    preferences: UserPreferences;
  };
  
  // 市场数据
  marketData: {
    indices: MarketData[];
    watchlist: MarketData[];
    lastUpdated?: string;
  };
  
  // 聊天状态
  chat: {
    messages: ChatMessage[];
    isLoading: boolean;
    currentModel: string;
    currentProvider: string;
    availableModels: LLMModelOption[];
  };
  
  // 策略状态
  strategies: {
    list: Strategy[];
    active?: Strategy;
    backtesting: boolean;
  };
  
  // 选股推荐
  recommendations: {
    list: StockRecommendation[];
    filters: {
      minScore: number;
      maxRisk: 'low' | 'medium' | 'high';
      sectors: string[];
    };
    lastUpdated?: string;
  };
  
  // WebSocket连接
  websocket: WebSocketState;

  // 知识图谱状态
  knowledgeGraph: KnowledgeGraphState;
  
  // UI状态
  ui: {
    sidebarOpen: boolean;
    currentView: 'dashboard' | 'chat' | 'strategies' | 'recommendations' | 'settings' | 'testing';
    theme: 'light' | 'dark';
    notifications: Array<{
      id: string;
      type: 'info' | 'success' | 'warning' | 'error';
      title: string;
      message: string;
      timestamp: string;
      read: boolean;
    }>;
  };
}

// 状态操作接口
interface AppActions {
  // 用户操作
  setAuthenticated: (authenticated: boolean, username?: string) => void;
  updateUserPreferences: (preferences: Partial<UserPreferences>) => void;
  
  // 市场数据操作
  updateMarketData: (type: 'indices' | 'watchlist', data: MarketData[]) => void;
  addToWatchlist: (symbol: string) => void;
  removeFromWatchlist: (symbol: string) => void;
  
  // 聊天操作
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
  clearMessages: () => void;
  setLoading: (loading: boolean) => void;
  setCurrentModel: (model: string) => void;
  setAvailableModels: (models: LLMModelOption[]) => void;
  
  // 策略操作
  updateStrategies: (strategies: Strategy[]) => void;
  setActiveStrategy: (strategy: Strategy) => void;
  setBacktesting: (backtesting: boolean) => void;
  
  // 推荐操作
  updateRecommendations: (recommendations: StockRecommendation[]) => void;
  updateRecommendationFilters: (filters: Partial<AppState['recommendations']['filters']>) => void;
  
  // WebSocket操作
  setWebSocketConnected: (connected: boolean) => void;
  setWebSocketReconnecting: (reconnecting: boolean) => void;
  addSubscription: (subscription: string) => void;
  removeSubscription: (subscription: string) => void;
  
  // UI操作
  setSidebarOpen: (open: boolean) => void;
  setCurrentView: (view: AppState['ui']['currentView']) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  addNotification: (notification: Omit<AppState['ui']['notifications'][0], 'id' | 'timestamp' | 'read'>) => void;
  markNotificationRead: (id: string) => void;
  clearNotifications: () => void;

  // 知识图谱操作
  setKnowledgeGraphEnabled: (enabled: boolean) => void;
  setKnowledgeGraphLoading: (loading: boolean) => void;
  setKnowledgeGraphError: (error?: string | null) => void;
  setKnowledgeGraphResults: (results: KnowledgeGraphEntity[]) => void;
  setKnowledgeGraphSelectedEntity: (entity?: KnowledgeGraphEntity) => void;
  setKnowledgeGraphRelations: (relations: KnowledgeGraphRelation[]) => void;
  addKnowledgeGraphHistory: (entry: KnowledgeGraphQueryHistory) => void;
  clearKnowledgeGraph: () => void;
}

// 默认用户偏好
const defaultUserPreferences: UserPreferences = {
  riskTolerance: 'moderate',
  investmentHorizon: 'medium',
  preferredSectors: [],
  excludedSectors: [],
  maxPositionSize: 10,
  autoRebalance: true,
  notifications: {
    priceAlerts: true,
    strategyUpdates: true,
    marketNews: false,
  },
};

// 创建应用状态store
export const useAppStore = create<AppState & AppActions>()(
  devtools(
    persist(
      (set, get) => ({
        // 初始状态
        user: {
          isAuthenticated: false,
          preferences: defaultUserPreferences,
        },
        
        marketData: {
          indices: [],
          watchlist: [],
        },
        
        chat: {
          messages: [],
          isLoading: false,
          currentModel: 'gpt-4-turbo',
          currentProvider: 'openai',
          availableModels: [
            {
              id: 'openai:gpt-4-turbo',
              name: 'GPT-4 Turbo',
              provider: 'openai',
              model: 'gpt-4-turbo',
              context: '128K',
              description: 'OpenAI GPT-4 Turbo 模型',
              capabilities: ['general', 'analysis', 'reasoning'],
              streaming: true,
              status: 'online',
            },
            {
              id: 'qwen:qwen2.5-14b',
              name: 'Qwen2.5 14B',
              provider: 'qwen',
              model: 'qwen2.5-14b-instruct',
              context: '128K',
              description: '通义千问2.5 14B 指令微调模型（推荐）',
              capabilities: ['analysis', 'report', 'chinese'],
              streaming: true,
              status: 'online',
            },
            {
              id: 'qwen:qwen2.5-32b',
              name: 'Qwen2.5 32B',
              provider: 'qwen',
              model: 'qwen2.5-32b-instruct',
              context: '256K',
              description: '通义千问2.5 32B 长上下文版本',
              capabilities: ['long-context', 'reasoning'],
              streaming: true,
              status: 'online',
            },
            {
              id: 'anthropic:claude-3-sonnet',
              name: 'Claude 3 Sonnet',
              provider: 'anthropic',
              model: 'claude-3-sonnet',
              context: '200K',
              description: 'Anthropic Claude 3 Sonnet',
              capabilities: ['analysis', 'coding', 'english'],
              streaming: true,
              status: 'degraded',
            },
          ],
        },
        
        strategies: {
          list: [],
          backtesting: false,
        },
        
        recommendations: {
          list: [],
          filters: {
            minScore: 70,
            maxRisk: 'medium' as const,
            sectors: [],
          },
        },
        
        websocket: {
          connected: false,
          reconnecting: false,
          subscriptions: [],
        },
        
        ui: {
          sidebarOpen: true,
          currentView: 'dashboard' as const,
          theme: 'light' as const,
          notifications: [],
        },

        knowledgeGraph: {
          enabled: false,
          loading: false,
          error: null,
          searchResults: [],
          relations: [],
          history: [],
        },
        
        // 用户操作
        setAuthenticated: (authenticated: boolean, username?: string) => {
          set((state: AppState) => ({
            user: {
              ...state.user,
              isAuthenticated: authenticated,
              username,
            },
          }));
        },
        
        updateUserPreferences: (preferences: Partial<UserPreferences>) => {
          set((state: AppState) => ({
            user: {
              ...state.user,
              preferences: {
                ...state.user.preferences,
                ...preferences,
              },
            },
          }));
        },
        
        // 市场数据操作
        updateMarketData: (type: 'indices' | 'watchlist', data: MarketData[]) => {
          set((state: AppState) => ({
            marketData: {
              ...state.marketData,
              [type]: data,
              lastUpdated: new Date().toISOString(),
            },
          }));
        },
        
        addToWatchlist: (symbol: string) => {
          const { marketData } = get();
          const existingIndex = marketData.watchlist.findIndex(item => item.symbol === symbol);
          if (existingIndex === -1) {
            // 这里应该从API获取实际数据，暂时使用模拟数据
            const newItem: MarketData = {
              symbol,
              name: symbol,
              price: 0,
              change: 0,
              changePercent: 0,
              volume: 0,
              timestamp: new Date().toISOString(),
            };
            
            set((state: AppState) => ({
              marketData: {
                ...state.marketData,
                watchlist: [...state.marketData.watchlist, newItem],
              },
            }));
          }
        },
        
        removeFromWatchlist: (symbol: string) => {
          set((state: AppState) => ({
            marketData: {
              ...state.marketData,
              watchlist: state.marketData.watchlist.filter(item => item.symbol !== symbol),
            },
          }));
        },
        
        // 聊天操作
        addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => {
          const newMessage: ChatMessage = {
            ...message,
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
          };
          
          set((state: AppState) => ({
            chat: {
              ...state.chat,
              messages: [...state.chat.messages, newMessage],
            },
          }));
        },
        
        updateMessage: (id: string, updates: Partial<ChatMessage>) => {
          set((state: AppState) => ({
            chat: {
              ...state.chat,
              messages: state.chat.messages.map(msg => 
                msg.id === id ? { ...msg, ...updates } : msg
              ),
            },
          }));
        },
        
        clearMessages: () => {
          set((state: AppState) => ({
            chat: {
              ...state.chat,
              messages: [],
            },
          }));
        },
        
        setLoading: (loading: boolean) => {
          set((state: AppState) => ({
            chat: {
              ...state.chat,
              isLoading: loading,
            },
          }));
        },
        
        setCurrentModel: (model: string) => {
          set((state: AppState) => ({
            chat: {
              ...state.chat,
              currentModel: model,
              currentProvider:
                state.chat.availableModels.find((item) => item.model === model || item.id === model)?.provider ||
                state.chat.currentProvider,
            },
          }));
        },

        setAvailableModels: (models: LLMModelOption[]) => {
          set((state: AppState) => {
            const fallback = models[0];
            const matched = models.find((model) => model.model === state.chat.currentModel || model.id === state.chat.currentModel);
            return {
              chat: {
                ...state.chat,
                availableModels: models,
                currentModel: matched?.model ?? fallback?.model ?? state.chat.currentModel,
                currentProvider: matched?.provider ?? fallback?.provider ?? state.chat.currentProvider,
              },
            };
          });
        },
        
        // 策略操作
        updateStrategies: (strategies: Strategy[]) => {
          set((state: AppState) => ({
            strategies: {
              ...state.strategies,
              list: strategies,
            },
          }));
        },
        
        setActiveStrategy: (strategy: Strategy) => {
          set((state: AppState) => ({
            strategies: {
              ...state.strategies,
              active: strategy,
            },
          }));
        },
        
        setBacktesting: (backtesting: boolean) => {
          set((state: AppState) => ({
            strategies: {
              ...state.strategies,
              backtesting,
            },
          }));
        },
        
        // 推荐操作
        updateRecommendations: (recommendations: StockRecommendation[]) => {
          set((state: AppState) => ({
            recommendations: {
              ...state.recommendations,
              list: recommendations,
              lastUpdated: new Date().toISOString(),
            },
          }));
        },
        
        updateRecommendationFilters: (filters: Partial<AppState['recommendations']['filters']>) => {
          set((state: AppState) => ({
            recommendations: {
              ...state.recommendations,
              filters: {
                ...state.recommendations.filters,
                ...filters,
              },
            },
          }));
        },
        
        // WebSocket操作
        setWebSocketConnected: (connected: boolean) => {
          set((state: AppState) => ({
            websocket: {
              ...state.websocket,
              connected,
              lastConnected: connected ? new Date().toISOString() : state.websocket.lastConnected,
              reconnecting: false,
            },
          }));
        },
        
        setWebSocketReconnecting: (reconnecting: boolean) => {
          set((state: AppState) => ({
            websocket: {
              ...state.websocket,
              reconnecting,
            },
          }));
        },
        
        addSubscription: (subscription: string) => {
          set((state: AppState) => ({
            websocket: {
              ...state.websocket,
              subscriptions: [...new Set([...state.websocket.subscriptions, subscription])],
            },
          }));
        },
        
        removeSubscription: (subscription: string) => {
          set((state: AppState) => ({
            websocket: {
              ...state.websocket,
              subscriptions: state.websocket.subscriptions.filter(sub => sub !== subscription),
            },
          }));
        },
        
        // UI操作
        setSidebarOpen: (open: boolean) => {
          set((state: AppState) => ({
            ui: {
              ...state.ui,
              sidebarOpen: open,
            },
          }));
        },
        
        setCurrentView: (view: AppState['ui']['currentView']) => {
          set((state: AppState) => ({
            ui: {
              ...state.ui,
              currentView: view,
            },
          }));
        },
        
        setTheme: (theme: 'light' | 'dark') => {
          set((state: AppState) => ({
            ui: {
              ...state.ui,
              theme,
            },
          }));
        },
        
        addNotification: (notification: Omit<AppState['ui']['notifications'][0], 'id' | 'timestamp' | 'read'>) => {
          const newNotification = {
            ...notification,
            id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
            read: false,
          };
          
          set((state: AppState) => ({
            ui: {
              ...state.ui,
              notifications: [newNotification, ...state.ui.notifications].slice(0, 50), // 最多保留50条通知
            },
          }));
        },
        
        markNotificationRead: (id: string) => {
          set((state: AppState) => ({
            ui: {
              ...state.ui,
              notifications: state.ui.notifications.map(notif => 
                notif.id === id ? { ...notif, read: true } : notif
              ),
            },
          }));
        },
        
        clearNotifications: () => {
          set((state: AppState) => ({
            ui: {
              ...state.ui,
              notifications: [],
            },
          }));
        },

        setKnowledgeGraphEnabled: (enabled: boolean) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              enabled,
            },
          }));
        },

        setKnowledgeGraphLoading: (loading: boolean) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              loading,
            },
          }));
        },

        setKnowledgeGraphError: (error?: string | null) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              error: error ?? null,
            },
          }));
        },

        setKnowledgeGraphResults: (results: KnowledgeGraphEntity[]) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              searchResults: results,
            },
          }));
        },

        setKnowledgeGraphSelectedEntity: (entity?: KnowledgeGraphEntity) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              selectedEntity: entity,
            },
          }));
        },

        setKnowledgeGraphRelations: (relations: KnowledgeGraphRelation[]) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              relations,
            },
          }));
        },

        addKnowledgeGraphHistory: (entry: KnowledgeGraphQueryHistory) => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              history: [entry, ...state.knowledgeGraph.history].slice(0, 50),
            },
          }));
        },

        clearKnowledgeGraph: () => {
          set((state: AppState) => ({
            knowledgeGraph: {
              ...state.knowledgeGraph,
              searchResults: [],
              relations: [],
              selectedEntity: undefined,
              error: null,
            },
          }));
        },
      }),
      {
        name: 'naxs-app-store',
        partialize: (state: AppState & AppActions) => ({
          user: state.user,
          ui: {
            theme: state.ui.theme,
            sidebarOpen: state.ui.sidebarOpen,
          },
          recommendations: {
            filters: state.recommendations.filters,
          },
        }),
      }
    ),
    {
      name: 'NAXS App Store',
    }
  )
);

// 选择器hooks
export const useUser = () => useAppStore((state) => state.user);
export const useMarketData = () => useAppStore((state) => state.marketData);
export const useChat = () => useAppStore((state) => state.chat);
export const useKnowledgeGraph = () => useAppStore((state) => state.knowledgeGraph);
export const useStrategies = () => useAppStore((state) => state.strategies);
export const useRecommendations = () => useAppStore((state) => state.recommendations);
export const useWebSocket = () => useAppStore((state) => state.websocket);
export const useUI = () => useAppStore((state) => state.ui);

// 操作hooks
export const useUserActions = () => useAppStore((state) => ({
  setAuthenticated: state.setAuthenticated,
  updateUserPreferences: state.updateUserPreferences,
}));

export const useMarketActions = () => useAppStore((state) => ({
  updateMarketData: state.updateMarketData,
  addToWatchlist: state.addToWatchlist,
  removeFromWatchlist: state.removeFromWatchlist,
}));

export const useChatActions = () => useAppStore((state) => ({
  addMessage: state.addMessage,
  updateMessage: state.updateMessage,
  clearMessages: state.clearMessages,
  setLoading: state.setLoading,
  setCurrentModel: state.setCurrentModel,
  setAvailableModels: state.setAvailableModels,
}));

export const useStrategyActions = () => useAppStore((state) => ({
  updateStrategies: state.updateStrategies,
  setActiveStrategy: state.setActiveStrategy,
  setBacktesting: state.setBacktesting,
}));

export const useRecommendationActions = () => useAppStore((state) => ({
  updateRecommendations: state.updateRecommendations,
  updateRecommendationFilters: state.updateRecommendationFilters,
}));

export const useWebSocketActions = () => useAppStore((state) => ({
  setWebSocketConnected: state.setWebSocketConnected,
  setWebSocketReconnecting: state.setWebSocketReconnecting,
  addSubscription: state.addSubscription,
  removeSubscription: state.removeSubscription,
}));

export const useUIActions = () => useAppStore((state) => ({
  setSidebarOpen: state.setSidebarOpen,
  setCurrentView: state.setCurrentView,
  setTheme: state.setTheme,
  addNotification: state.addNotification,
  markNotificationRead: state.markNotificationRead,
  clearNotifications: state.clearNotifications,
}));

export const useKnowledgeGraphActions = () => useAppStore((state) => ({
  setKnowledgeGraphEnabled: state.setKnowledgeGraphEnabled,
  setKnowledgeGraphLoading: state.setKnowledgeGraphLoading,
  setKnowledgeGraphError: state.setKnowledgeGraphError,
  setKnowledgeGraphResults: state.setKnowledgeGraphResults,
  setKnowledgeGraphSelectedEntity: state.setKnowledgeGraphSelectedEntity,
  setKnowledgeGraphRelations: state.setKnowledgeGraphRelations,
  addKnowledgeGraphHistory: state.addKnowledgeGraphHistory,
  clearKnowledgeGraph: state.clearKnowledgeGraph,
}));