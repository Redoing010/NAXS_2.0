// WebSocket连接Hook
// 提供可靠的WebSocket连接管理和自动重连功能

import { useState, useEffect, useRef, useCallback } from 'react';

// WebSocket配置
interface UseWebSocketConfig {
  url: string;
  protocols?: string | string[];
  enabled?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  onOpen?: (event: Event) => void;
  onMessage?: (data: any) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: Error) => void;
  onReconnect?: (attempt: number) => void;
}

// WebSocket状态
type WebSocketState = 'connecting' | 'connected' | 'disconnected' | 'error';

// Hook返回值
interface UseWebSocketReturn {
  isConnected: boolean;
  connectionState: WebSocketState;
  sendMessage: (data: any) => boolean;
  disconnect: () => void;
  reconnect: () => void;
  lastMessage?: any;
  error?: Error;
}

export const useWebSocket = (config: UseWebSocketConfig): UseWebSocketReturn => {
  const {
    url,
    protocols,
    enabled = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    heartbeatInterval = 30000,
    onOpen,
    onMessage,
    onClose,
    onError,
    onReconnect
  } = config;

  const [connectionState, setConnectionState] = useState<WebSocketState>('disconnected');
  const [lastMessage, setLastMessage] = useState<any>();
  const [error, setError] = useState<Error>();

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | undefined>(undefined);
  const heartbeatTimeoutRef = useRef<number | undefined>(undefined);
  const reconnectCountRef = useRef(0);
  const isManualDisconnectRef = useRef(false);

  // 清理定时器
  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = undefined;
    }
  }, []);

  // 发送心跳
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        
        // 设置下一次心跳
        heartbeatTimeoutRef.current = setTimeout(sendHeartbeat, heartbeatInterval);
      } catch (err) {
        console.error('Failed to send heartbeat:', err);
      }
    }
  }, [heartbeatInterval]);

  // 连接WebSocket
  const connect = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    try {
      setConnectionState('connecting');
      setError(undefined);
      
      // 关闭现有连接
      if (wsRef.current) {
        wsRef.current.close();
      }

      // 创建新连接
      wsRef.current = new WebSocket(url, protocols);

      wsRef.current.onopen = (event) => {
        console.log('WebSocket connected:', url);
        setConnectionState('connected');
        reconnectCountRef.current = 0;
        
        // 开始心跳
        if (heartbeatInterval > 0) {
          heartbeatTimeoutRef.current = setTimeout(sendHeartbeat, heartbeatInterval);
        }
        
        if (onOpen) onOpen(event);
      };

      wsRef.current.onmessage = (event) => {
        try {
          let data = event.data;
          
          // 尝试解析JSON
          if (typeof data === 'string') {
            try {
              data = JSON.parse(data);
            } catch {
              // 保持原始字符串
            }
          }
          
          // 处理心跳响应
          if (data?.type === 'pong') {
            return;
          }
          
          setLastMessage(data);
          if (onMessage) onMessage(data);
        } catch (err) {
          console.error('Failed to process WebSocket message:', err);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setConnectionState('disconnected');
        clearTimers();
        
        if (onClose) onClose(event);
        
        // 自动重连（如果不是手动断开）
        if (!isManualDisconnectRef.current && enabled && reconnectCountRef.current < reconnectAttempts) {
          const delay = reconnectInterval * Math.pow(1.5, reconnectCountRef.current);
          console.log(`WebSocket reconnecting in ${delay}ms (attempt ${reconnectCountRef.current + 1}/${reconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectCountRef.current++;
            if (onReconnect) onReconnect(reconnectCountRef.current);
            connect();
          }, delay);
        }
      };

      wsRef.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        const wsError = new Error('WebSocket connection error');
        setError(wsError);
        setConnectionState('error');
        
        if (onError) onError(wsError);
      };

    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      const connectionError = err as Error;
      setError(connectionError);
      setConnectionState('error');
      
      if (onError) onError(connectionError);
    }
  }, [enabled, url, protocols, reconnectAttempts, reconnectInterval, heartbeatInterval, sendHeartbeat, onOpen, onMessage, onClose, onError, onReconnect, clearTimers]);

  // 断开连接
  const disconnect = useCallback(() => {
    isManualDisconnectRef.current = true;
    clearTimers();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setConnectionState('disconnected');
  }, [clearTimers]);

  // 重新连接
  const reconnect = useCallback(() => {
    isManualDisconnectRef.current = false;
    reconnectCountRef.current = 0;
    connect();
  }, [connect]);

  // 发送消息
  const sendMessage = useCallback((data: any): boolean => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket is not connected');
      return false;
    }

    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      wsRef.current.send(message);
      return true;
    } catch (err) {
      console.error('Failed to send WebSocket message:', err);
      setError(err as Error);
      return false;
    }
  }, []);

  // 初始化连接
  useEffect(() => {
    if (enabled) {
      isManualDisconnectRef.current = false;
      connect();
    } else {
      disconnect();
    }

    return () => {
      isManualDisconnectRef.current = true;
      clearTimers();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, connect, disconnect, clearTimers]);

  // URL变化时重新连接
  useEffect(() => {
    if (enabled && wsRef.current) {
      reconnect();
    }
  }, [url, protocols, enabled, reconnect]);

  return {
    isConnected: connectionState === 'connected',
    connectionState,
    sendMessage,
    disconnect,
    reconnect,
    lastMessage,
    error
  };
};

export default useWebSocket;