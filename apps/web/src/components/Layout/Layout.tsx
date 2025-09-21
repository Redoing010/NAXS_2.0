// @ts-nocheck
// 主布局组件

import React, { useEffect } from 'react';
import { clsx } from 'clsx';
import Sidebar from './Sidebar';
import Header from './Header';
import { useUI } from '../../store';
import websocketService from '../../services/websocket';

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
}

export const Layout: React.FC<LayoutProps> = ({ children, className }) => {
  const { sidebarOpen, theme } = useUI();

  // 初始化WebSocket连接
  useEffect(() => {
    const initWebSocket = async () => {
      try {
        await websocketService.connect();
        console.log('WebSocket connected successfully');
        
        // 订阅基础数据流
        websocketService.subscribe('market_data');
        websocketService.subscribe('quotes');
        websocketService.subscribe('alerts');
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    };

    initWebSocket();

    // 清理函数
    return () => {
      websocketService.disconnect();
    };
  }, []);

  // 应用主题
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Cmd/Ctrl + K 打开搜索
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        // 触发搜索框聚焦
        const searchButton = document.querySelector('[data-search-trigger]') as HTMLElement;
        if (searchButton) {
          searchButton.click();
        }
      }
      
      // Cmd/Ctrl + B 切换侧边栏
      if ((event.metaKey || event.ctrlKey) && event.key === 'b') {
        event.preventDefault();
        // 这里可以添加切换侧边栏的逻辑
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <div className={clsx('relative flex h-screen overflow-hidden bg-gradient-to-br from-sky-50 via-white to-blue-50', className)}>
      {/* 背景装饰 */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-32 -left-40 h-96 w-96 rounded-full bg-sky-200/40 blur-3xl" />
        <div className="absolute top-1/3 -right-24 h-80 w-80 rounded-full bg-blue-200/30 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-sky-100/40 blur-3xl" />
      </div>

      {/* 侧边栏 */}
      <Sidebar />

      {/* 主内容区域 */}
      <div className="flex-1 flex flex-col min-w-0 relative z-10">
        {/* 头部 */}
        <Header />

        {/* 内容区域 */}
        <main className="flex-1 overflow-hidden px-6 pb-6">
          <div className="h-full overflow-auto">
            {children}
          </div>
        </main>
      </div>

      {/* 全局加载遮罩 */}
      <GlobalLoadingOverlay />

      {/* 全局通知 */}
      <GlobalNotifications />
    </div>
  );
};

// 全局加载遮罩组件
const GlobalLoadingOverlay: React.FC = () => {
  const [isLoading, setIsLoading] = React.useState(false);

  // 监听全局加载状态
  useEffect(() => {
    const handleLoadingStart = () => setIsLoading(true);
    const handleLoadingEnd = () => setIsLoading(false);

    // 这里可以监听API请求状态
    // 暂时使用简单的事件监听
    window.addEventListener('loading:start', handleLoadingStart);
    window.addEventListener('loading:end', handleLoadingEnd);

    return () => {
      window.removeEventListener('loading:start', handleLoadingStart);
      window.removeEventListener('loading:end', handleLoadingEnd);
    };
  }, []);

  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 flex items-center space-x-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="text-gray-700">加载中...</span>
      </div>
    </div>
  );
};

// 全局通知组件
const GlobalNotifications: React.FC = () => {
  const { notifications } = useUI();
  const [visibleNotifications, setVisibleNotifications] = React.useState<string[]>([]);

  // 显示最新的通知
  useEffect(() => {
    const latestNotifications = notifications
      .filter(n => !n.read)
      .slice(0, 3)
      .map(n => n.id);
    
    setVisibleNotifications(latestNotifications);
  }, [notifications]);

  // 自动隐藏通知
  useEffect(() => {
    if (visibleNotifications.length > 0) {
      const timer = setTimeout(() => {
        setVisibleNotifications([]);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [visibleNotifications]);

  if (visibleNotifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-40 space-y-2">
      {visibleNotifications.map((notificationId) => {
        const notification = notifications.find(n => n.id === notificationId);
        if (!notification) return null;

        return (
          <div
            key={notification.id}
            className={clsx(
              'max-w-sm bg-white border border-gray-200 rounded-lg shadow-lg p-4 animate-slide-up',
              {
                'border-l-4 border-l-blue-500': notification.type === 'info',
                'border-l-4 border-l-green-500': notification.type === 'success',
                'border-l-4 border-l-yellow-500': notification.type === 'warning',
                'border-l-4 border-l-red-500': notification.type === 'error',
              }
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-gray-900">
                  {notification.title}
                </h4>
                <p className="text-sm text-gray-600 mt-1">
                  {notification.message}
                </p>
              </div>
              <button
                onClick={() => {
                  setVisibleNotifications(prev => 
                    prev.filter(id => id !== notification.id)
                  );
                }}
                className="ml-2 text-gray-400 hover:text-gray-600"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default Layout;