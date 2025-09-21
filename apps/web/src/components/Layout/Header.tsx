// 头部组件

import React, { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import {
  Search,
  Bell,
  Settings,
  Moon,
  Sun,
  Wifi,
  WifiOff,
  AlertCircle,
  CheckCircle,
  Info,
  X,
} from 'lucide-react';
import { useUI, useUIActions, useWebSocket } from '../../store';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

interface HeaderProps {
  className?: string;
}

type PrimaryNavItem = {
  id: string;
  label: string;
  view: 'dashboard' | 'chat' | 'strategies' | 'recommendations' | 'settings' | 'testing';
  description?: string;
};

const primaryNavItems: PrimaryNavItem[] = [
  {
    id: 'chat',
    label: 'AI助手',
    view: 'chat',
    description: '智能投研对话入口',
  },
  {
    id: 'dashboard',
    label: '仪表板',
    view: 'dashboard',
    description: '实时运营与市场总览',
  },
  {
    id: 'strategies',
    label: '策略管理',
    view: 'strategies',
    description: '配置与监控自动化策略',
  },
  {
    id: 'recommendations',
    label: '选股推荐',
    view: 'recommendations',
    description: '个性化AI选股建议',
  },
  {
    id: 'testing',
    label: '系统监控',
    view: 'testing',
    description: '测试与运行健康状态',
  },
];

export const Header: React.FC<HeaderProps> = ({ className }) => {
  const { theme, notifications, currentView } = useUI();
  const { connected, reconnecting } = useWebSocket();
  const { setTheme, markNotificationRead, clearNotifications, setCurrentView } = useUIActions();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  
  const notificationRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLDivElement>(null);

  // 未读通知数量
  const unreadCount = notifications?.filter(n => !n.read).length || 0;

  // 点击外部关闭下拉菜单
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (notificationRef.current && !notificationRef.current.contains(event.target as Node)) {
        setShowNotifications(false);
      }
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowSearch(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // 切换主题
  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  // 处理搜索
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      console.log('Search:', searchQuery);
      // 这里可以添加搜索逻辑
      setSearchQuery('');
      setShowSearch(false);
    }
  };

  // 标记通知为已读
  const handleNotificationClick = (notificationId: string) => {
    markNotificationRead(notificationId);
  };

  const handlePrimaryNavClick = (view: PrimaryNavItem['view']) => {
    setCurrentView(view);
  };

  // 获取通知图标
  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-success-500" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-warning-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-danger-500" />;
      default:
        return <Info className="w-4 h-4 text-primary-500" />;
    }
  };

  // 获取连接状态
  const getConnectionStatus = () => {
    if (reconnecting) {
      return {
        icon: <WifiOff className="w-4 h-4 text-warning-500 animate-pulse" />,
        text: '重连中...',
        color: 'text-warning-600',
      };
    }
    
    if (connected) {
      return {
        icon: <Wifi className="w-4 h-4 text-success-500" />,
        text: '已连接',
        color: 'text-success-600',
      };
    }
    
    return {
      icon: <WifiOff className="w-4 h-4 text-danger-500" />,
      text: '未连接',
      color: 'text-danger-600',
    };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <header className={clsx('bg-white border-b border-gray-200 px-6 py-4', className)}>
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          {/* 左侧：搜索 */}
          <div className="flex items-center flex-1 max-w-xl w-full" ref={searchRef}>
            <div className="relative w-full">
              <button
                type="button"
                onClick={() => setShowSearch(!showSearch)}
                className="w-full flex items-center px-4 py-2 text-sm text-gray-500 bg-gray-50 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <Search className="w-4 h-4 mr-2" />
                <span>搜索股票、策略...</span>
                <kbd className="ml-auto px-2 py-1 text-xs text-gray-400 bg-white border border-gray-200 rounded">
                  ⌘K
                </kbd>
              </button>

              {showSearch && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                  <form onSubmit={handleSearch} className="p-4">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="输入搜索内容..."
                        className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        autoFocus
                      />
                    </div>
                    <div className="mt-3 text-xs text-gray-500">
                      <p>快速搜索：股票代码、策略名称、新闻关键词</p>
                    </div>
                  </form>
                </div>
              )}
            </div>
          </div>

          {/* 右侧：状态和操作 */}
          <div className="flex items-center space-x-4">
            {/* 连接状态 */}
            <div className="flex items-center space-x-2">
              {connectionStatus.icon}
              <span className={clsx('text-sm', connectionStatus.color)}>
                {connectionStatus.text}
              </span>
            </div>

            {/* 主题切换 */}
            <button
              type="button"
              onClick={toggleTheme}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title={theme === 'light' ? '切换到深色模式' : '切换到浅色模式'}
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5" />
              ) : (
                <Sun className="w-5 h-5" />
              )}
            </button>

            {/* 通知 */}
            <div className="relative" ref={notificationRef}>
              <button
                type="button"
                onClick={() => setShowNotifications(!showNotifications)}
                className="relative p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <Bell className="w-5 h-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white bg-red-500 rounded-full">
                    {unreadCount > 99 ? '99+' : unreadCount}
                  </span>
                )}
              </button>

              {showNotifications && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                  <div className="p-4 border-b border-gray-200">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-medium text-gray-900">通知</h3>
                      {(notifications?.length || 0) > 0 && (
                        <button
                          type="button"
                          onClick={clearNotifications}
                          className="text-xs text-gray-500 hover:text-gray-700"
                        >
                          清空全部
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="max-h-96 overflow-y-auto">
                    {(notifications?.length || 0) === 0 ? (
                      <div className="p-4 text-center text-gray-500">
                        <Bell className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                        <p className="text-sm">暂无通知</p>
                      </div>
                    ) : (
                      <div className="divide-y divide-gray-100">
                        {notifications?.slice(0, 10).map((notification) => (
                          <div
                            key={notification.id}
                            className={clsx(
                              'p-4 hover:bg-gray-50 cursor-pointer transition-colors',
                              !notification.read && 'bg-blue-50'
                            )}
                            onClick={() => handleNotificationClick(notification.id)}
                          >
                            <div className="flex items-start space-x-3">
                              {getNotificationIcon(notification.type)}
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-900">
                                  {notification.title}
                                </p>
                                <p className="text-sm text-gray-500 mt-1">
                                  {notification.message}
                                </p>
                                <p className="text-xs text-gray-400 mt-2">
                                  {format(new Date(notification.timestamp), 'MM-dd HH:mm', {
                                    locale: zhCN,
                                  })}
                                </p>
                              </div>
                              {!notification.read && (
                                <div className="w-2 h-2 bg-blue-500 rounded-full" />
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {(notifications?.length || 0) > 10 && (
                    <div className="p-3 border-t border-gray-200 text-center">
                      <button type="button" className="text-sm text-primary-600 hover:text-primary-700">
                        查看全部通知
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 设置 */}
            <button
              type="button"
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="设置"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        <nav className="flex items-stretch gap-2 overflow-x-auto pb-1" aria-label="Primary navigation">
          {primaryNavItems.map((item) => {
            const isActive = currentView === item.view;

            return (
              <button
                type="button"
                key={item.id}
                onClick={() => handlePrimaryNavClick(item.view)}
                className={clsx(
                  'flex-shrink-0 rounded-xl border px-4 py-2 text-left transition-all',
                  isActive
                    ? 'border-primary-600 bg-primary-600 text-white shadow-sm'
                    : 'border-transparent bg-gray-50 text-gray-600 hover:border-gray-200 hover:bg-white hover:text-gray-900'
                )}
              >
                <span className="block text-sm font-semibold">{item.label}</span>
                {item.description && (
                  <span className={clsx('mt-1 text-xs', isActive ? 'text-primary-100/90' : 'text-gray-500')}>
                    {item.description}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

export default Header;