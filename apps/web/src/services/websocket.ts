// WebSocket服务 - 实时数据连接

import { WEBSOCKET_URL } from '../config/environment';
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

const HEARTBEAT_INTERVAL = 30000;
const CONNECTION_TIMEOUT = 10000;

class WebSocketService {
  private socket: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private manualDisconnect = false;
  private pendingToken?: string;
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

  constructor(url: string = WEBSOCKET_URL) {
    this.url = url;
    this.setupStoreSubscription();
  }

  // 设置store订阅，监听状态变化
  private setupStoreSubscription() {
    useAppStore.subscribe(() => undefined);
  }

  private clearReconnectTimer() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // 连接WebSocket
  async connect(token?: string): Promise<void> {
    if (token) {
      this.pendingToken = token;
    }

    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      if (token && this.socket.readyState === WebSocket.OPEN) {
        this.sendMessage({
          type: 'auth',
          data: { token },
        });
      }
      return;
    }

    this.manualDisconnect = false;
    this.setStatus('connecting');
    this.clearReconnectTimer();

    return new Promise((resolve, reject) => {
      try {
        const socket = new WebSocket(this.url);
        this.socket = socket;

        let hasOpened = false;
        const timeout = setTimeout(() => {
          if (socket.readyState !== WebSocket.OPEN) {
            socket.close();
            reject(new Error('Connection timeout'));
          }
        }, CONNECTION_TIMEOUT);

        socket.onopen = () => {
          clearTimeout(timeout);
          hasOpened = true;
          this.handleOpen();

          if (this.pendingToken) {
            this.sendMessage({
              type: 'auth',
              data: { token: this.pendingToken },
            });
          }

          resolve();
        };

        socket.onmessage = (event) => {
          this.handleIncomingMessage(event.data);
        };

        socket.onerror = (event) => {
          clearTimeout(timeout);
          const error = event instanceof ErrorEvent ? event.error : new Error('WebSocket connection error');
          console.error('WebSocket connection error:', error);
          this.handleError(error);
          if (!hasOpened) {
            reject(error instanceof Error ? error : new Error('WebSocket connection error'));
          }
        };

        socket.onclose = (event) => {
          clearTimeout(timeout);
          this.handleClose(event);
          if (!hasOpened) {
            reject(new Error(`WebSocket closed before ready: ${event.code}`));
          }
        };
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        this.setStatus('error');
        throw error;
      }
    });
  }

  // 处理连接成功
  private handleOpen() {
    console.log('WebSocket connected');
    this.setStatus('connected');
    this.reconnectAttempts = 0;
    this.stats.lastConnected = new Date();
    this.startHeartbeat();
    this.resubscribeAll();
  }

  // 处理关闭事件
  private handleClose(event: CloseEvent) {
    console.log('WebSocket disconnected:', event.code, event.reason);
    this.stopHeartbeat();
    this.socket = null;
    this.stats.lastDisconnected = new Date();

    if (this.manualDisconnect) {
      this.setStatus('disconnected');
      return;
    }

    this.scheduleReconnect();
  }

  // 处理错误
  private handleError(error: any) {
    this.setStatus('error');
    const store = useAppStore.getState();
    store.addNotification({
      type: 'error',
      title: 'WebSocket错误',
      message: error?.message || 'WebSocket connection error',
    });
  }

  // 处理消息
  private handleIncomingMessage(rawMessage: any) {
    this.stats.messagesReceived++;

    try {
      const message: WebSocketMessage = typeof rawMessage === 'string' ? JSON.parse(rawMessage) : rawMessage;

      switch (message.type) {
        case 'data':
          this.handleDataMessage(message.data);
          break;
        case 'error':
          this.handleError(message.data);
          break;
        case 'status':
          console.log('WebSocket status:', message.data);
          break;
        case 'ack':
          console.log('WebSocket ack:', message.data);
          break;
        default:
          break;
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  // 处理数据消息
  private handleDataMessage(data: any) {
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
  private updateStoreData(subscriptionType: string, payload: any) {
    const store = useAppStore.getState();

    switch (subscriptionType) {
      case 'market_data':
      case 'quotes':
        if (payload.data && Array.isArray(payload.data)) {
          store.updateMarketData('indices', payload.data);
        }
        break;
      case 'signals':
        break;
      case 'news':
        store.addNotification({
          type: 'info',
          title: '市场新闻',
          message: payload.title || '收到新的市场新闻',
        });
        break;
      case 'alerts':
        store.addNotification({
          type: 'warning',
          title: '系统告警',
          message: payload.message || '收到系统告警',
        });
        break;
      default:
        break;
    }
  }

  // 发送消息
  private sendMessage(message: WebSocketMessage): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, cannot send message');
      return;
    }

    try {
      const payload = {
        ...message,
        timestamp: new Date().toISOString(),
        message_id: message.message_id ?? `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      };
      this.socket.send(JSON.stringify(payload));
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

    this.sendMessage({
      type: 'subscribe',
      data: config,
    });

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

    this.sendMessage({
      type: 'unsubscribe',
      data: { subscription_id: subscriptionId },
    });

    this.subscriptions.delete(subscriptionId);

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

  // 启动心跳
  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatTimer = setInterval(() => {
      this.sendMessage({
        type: 'heartbeat',
        data: {
          client_time: new Date().toISOString(),
        },
      });
    }, HEARTBEAT_INTERVAL);
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
    if (this.manualDisconnect) {
      return;
    }

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

    this.clearReconnectTimer();
    this.reconnectTimer = setTimeout(() => {
      this.connect(this.pendingToken).catch((error) => {
        console.error('Reconnect failed:', error);
        this.scheduleReconnect();
      });
    }, delay);
  }

  // 断开连接
  disconnect(): void {
    this.manualDisconnect = true;
    this.clearReconnectTimer();
    this.stopHeartbeat();

    if (this.socket && this.socket.readyState !== WebSocket.CLOSED) {
      this.socket.close(1000, 'Manual disconnect');
    }

    this.socket = null;
    this.setStatus('disconnected');
  }

  // 获取连接状态
  getStatus(): ConnectionStatus {
    return this.status;
  }

  // 检查是否已连接
  isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
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
    if (this.url === url) {
      return;
    }

    this.url = url;

    if (this.isConnected()) {
      this.disconnect();
      this.connect(this.pendingToken).catch(console.error);
    }
  }
}

// 创建WebSocket服务实例
export const websocketService = new WebSocketService();

// 导出类型
export type { WebSocketMessage, SubscriptionConfig, ConnectionStatus };

// 导出默认实例
export default websocketService;
