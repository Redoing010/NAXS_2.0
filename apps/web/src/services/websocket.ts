// WebSocket服务 - 实时数据连接

import { io, Socket } from 'socket.io-client';
import { useAppStore } from '../store';

// WebSocket消息类型
interface WebSocketMessage {
  type: 'subscribe' | 'unsubscribe' | 'data' | 'heartbeat' | 'auth' | 'error' | 'ack' | 'status';
  data: any;
  timestamp?: string;
  message_id?: string;
}

// 订阅配置
interface SubscriptionConfig {
  subscription_type: string;
  filters?: Record<string, any>;
}

// 连接状态
type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

class WebSocketService {
  private socket: Socket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private heartbeatInterval: number | null = null;
  private subscriptions = new Map<string, SubscriptionConfig>();
  private messageHandlers = new Map<string, (data: any) => void>();
  private status: ConnectionStatus = 'disconnected';
  
  // 统计信息
  private stats = {
    messagesReceived: 0,
    messagesSent: 0,
    reconnectCount: 0,
    lastConnected: null as Date | null,
    lastDisconnected: null as Date | null,
  };

  constructor(url: string = 'ws://localhost:8765') {
    this.url = url;
    this.setupStoreSubscription();
  }

  // 设置store订阅，监听状态变化
  private setupStoreSubscription() {
    // 监听WebSocket状态变化
    useAppStore.subscribe(
      (_state) => {
        // 可以在这里处理状态变化
      }
    );
  }

  // 连接WebSocket
  async connect(token?: string): Promise<void> {
    try {
      if (this.socket?.connected) {
        console.log('WebSocket already connected');
        return;
      }

      this.setStatus('connecting');
      console.log(`Connecting to WebSocket: ${this.url}`);

      // 创建Socket.IO连接
      this.socket = io(this.url, {
        transports: ['websocket'],
        timeout: 10000,
        reconnection: false, // 我们手动处理重连
        auth: token ? { token } : undefined,
      });

      // 设置事件监听器
      this.setupEventListeners();

      // 等待连接建立
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 10000);

        this.socket!.on('connect', () => {
          clearTimeout(timeout);
          resolve();
        });

        this.socket!.on('connect_error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });

    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.setStatus('error');
      throw error;
    }
  }

  // 断开连接
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    
    this.subscriptions.clear();
    this.setStatus('disconnected');
    this.stats.lastDisconnected = new Date();
    
    console.log('WebSocket disconnected');
  }

  // 设置事件监听器
  private setupEventListeners(): void {
    if (!this.socket) return;

    // 连接成功
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.setStatus('connected');
      this.reconnectAttempts = 0;
      this.stats.lastConnected = new Date();
      this.startHeartbeat();
      
      // 重新订阅之前的订阅
      this.resubscribeAll();
    });

    // 连接断开
    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.setStatus('disconnected');
      this.stats.lastDisconnected = new Date();
      
      if (this.heartbeatInterval) {
        clearInterval(this.heartbeatInterval);
        this.heartbeatInterval = null;
      }
      
      // 自动重连
      if (reason !== 'io client disconnect') {
        this.scheduleReconnect();
      }
    });

    // 连接错误
    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.setStatus('error');
      this.scheduleReconnect();
    });

    // 接收消息
    this.socket.on('message', (message: WebSocketMessage) => {
      this.handleMessage(message);
    });

    // 数据消息
    this.socket.on('data', (data: any) => {
      this.handleDataMessage(data);
    });

    // 错误消息
    this.socket.on('error', (error: any) => {
      console.error('WebSocket error:', error);
      this.handleError(error);
    });

    // 确认消息
    this.socket.on('ack', (ack: any) => {
      console.log('WebSocket ack:', ack);
    });

    // 状态消息
    this.socket.on('status', (status: any) => {
      console.log('WebSocket status:', status);
    });
  }

  // 处理消息
  private handleMessage(message: WebSocketMessage): void {
    this.stats.messagesReceived++;
    
    switch (message.type) {
      case 'data':
        this.handleDataMessage(message.data);
        break;
      case 'error':
        this.handleError(message.data);
        break;
      case 'heartbeat':
        // 心跳响应，不需要特殊处理
        break;
      default:
        console.log('Unknown message type:', message.type, message.data);
    }
  }

  // 处理数据消息
  private handleDataMessage(data: any): void {
    try {
      const { subscription_type, payload } = data;
      
      // 更新store中的数据
      this.updateStoreData(subscription_type, payload);
      
      // 调用注册的处理器
      const handler = this.messageHandlers.get(subscription_type);
      if (handler) {
        handler(payload);
      }
      
    } catch (error) {
      console.error('Error handling data message:', error);
    }
  }

  // 更新store数据
  private updateStoreData(subscriptionType: string, payload: any): void {
    const store = useAppStore.getState();
    
    switch (subscriptionType) {
      case 'market_data':
      case 'quotes':
        // 更新市场数据
        if (payload.data && Array.isArray(payload.data)) {
          store.updateMarketData('indices', payload.data);
        }
        break;
        
      case 'signals':
        // 更新交易信号
        // 这里可以添加信号处理逻辑
        break;
        
      case 'news':
        // 更新新闻数据
        store.addNotification({
          type: 'info',
          title: '市场新闻',
          message: payload.title || '收到新的市场新闻',
        });
        break;
        
      case 'alerts':
        // 系统告警
        store.addNotification({
          type: 'warning',
          title: '系统告警',
          message: payload.message || '收到系统告警',
        });
        break;
        
      default:
        console.log('Unknown subscription type:', subscriptionType);
    }
  }

  // 处理错误
  private handleError(error: any): void {
    console.error('WebSocket error:', error);
    
    const store = useAppStore.getState();
    store.addNotification({
      type: 'error',
      title: 'WebSocket错误',
      message: error.error || error.message || '未知错误',
    });
  }

  // 发送消息
  private sendMessage(message: WebSocketMessage): void {
    if (!this.socket?.connected) {
      console.warn('WebSocket not connected, cannot send message');
      return;
    }
    
    try {
      this.socket.emit('message', {
        ...message,
        timestamp: new Date().toISOString(),
        message_id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      });
      
      this.stats.messagesSent++;
    } catch (error) {
      console.error('Error sending message:', error);
    }
  }

  // 订阅数据
  subscribe(subscriptionType: string, filters?: Record<string, any>): string {
    const subscriptionId = `sub_${subscriptionType}_${Date.now()}`;
    
    const config: SubscriptionConfig = {
      subscription_type: subscriptionType,
      filters,
    };
    
    this.subscriptions.set(subscriptionId, config);
    
    // 发送订阅消息
    this.sendMessage({
      type: 'subscribe',
      data: config,
    });
    
    // 更新store
    const store = useAppStore.getState();
    store.addSubscription(subscriptionId);
    
    console.log(`Subscribed to ${subscriptionType}:`, subscriptionId);
    return subscriptionId;
  }

  // 取消订阅
  unsubscribe(subscriptionId: string): void {
    const config = this.subscriptions.get(subscriptionId);
    if (!config) {
      console.warn('Subscription not found:', subscriptionId);
      return;
    }
    
    // 发送取消订阅消息
    this.sendMessage({
      type: 'unsubscribe',
      data: { subscription_id: subscriptionId },
    });
    
    this.subscriptions.delete(subscriptionId);
    
    // 更新store
    const store = useAppStore.getState();
    store.removeSubscription(subscriptionId);
    
    console.log(`Unsubscribed from:`, subscriptionId);
  }

  // 重新订阅所有订阅
  private resubscribeAll(): void {
    for (const [_subscriptionId, config] of this.subscriptions) {
      this.sendMessage({
        type: 'subscribe',
        data: config,
      });
    }
    
    if (this.subscriptions.size > 0) {
      console.log(`Resubscribed to ${this.subscriptions.size} subscriptions`);
    }
  }

  // 注册消息处理器
  onMessage(subscriptionType: string, handler: (data: any) => void): void {
    this.messageHandlers.set(subscriptionType, handler);
  }

  // 移除消息处理器
  offMessage(subscriptionType: string): void {
    this.messageHandlers.delete(subscriptionType);
  }

  // 发送心跳
  private sendHeartbeat(): void {
    this.sendMessage({
      type: 'heartbeat',
      data: {
        client_time: new Date().toISOString(),
      },
    });
  }

  // 启动心跳
  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    this.heartbeatInterval = setInterval(() => {
      this.sendHeartbeat();
    }, 30000); // 30秒心跳
  }

  // 设置连接状态
  private setStatus(status: ConnectionStatus): void {
    this.status = status;
    
    const store = useAppStore.getState();
    store.setWebSocketConnected(status === 'connected');
    store.setWebSocketReconnecting(status === 'reconnecting');
  }

  // 计划重连
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      this.setStatus('error');
      return;
    }
    
    this.reconnectAttempts++;
    this.stats.reconnectCount++;
    this.setStatus('reconnecting');
    
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('Reconnect failed:', error);
        this.scheduleReconnect();
      });
    }, delay);
  }

  // 获取连接状态
  getStatus(): ConnectionStatus {
    return this.status;
  }

  // 检查是否已连接
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // 获取统计信息
  getStats() {
    return {
      ...this.stats,
      status: this.status,
      connected: this.isConnected(),
      subscriptions: this.subscriptions.size,
      reconnectAttempts: this.reconnectAttempts,
    };
  }

  // 获取订阅列表
  getSubscriptions(): Array<{ id: string; config: SubscriptionConfig }> {
    return Array.from(this.subscriptions.entries()).map(([id, config]) => ({
      id,
      config,
    }));
  }

  // 更新URL
  updateUrl(url: string): void {
    if (this.url !== url) {
      this.url = url;
      if (this.isConnected()) {
        this.disconnect();
        this.connect().catch(console.error);
      }
    }
  }
}

// 创建WebSocket服务实例
export const websocketService = new WebSocketService();

// 导出类型
export type { WebSocketMessage, SubscriptionConfig, ConnectionStatus };

// 导出默认实例
export default websocketService;