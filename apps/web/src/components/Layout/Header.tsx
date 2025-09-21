// @ts-nocheck
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

export const Header: React.FC<HeaderProps> = ({ className }) => {
  const { theme, notifications } = useUI();
  const { connected, reconnecting } = useWebSocket();
  const { setTheme, markNotificationRead, clearNotifications } = useUIActions();
  
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
    <header className={clsx('relative z-20 bg-white/70 backdrop-blur-xl border-b border-white/40 px-6 py-4', className)}>
      <div className="flex items-center justify-between">
        {/* 左侧：搜索 */}
        <div className="flex items-center flex-1 max-w-md" ref={searchRef}>
          <div className="relative w-full">
            <button
              onClick={() => setShowSearch(!showSearch)}
              className="w-full flex items-center rounded-2xl border border-white/60 bg-white/80 px-4 py-3 text-sm text-slate-500 shadow-inner transition-all hover:border-sky-200 hover:bg-white"
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
            onClick={toggleTheme}
            className="rounded-xl bg-white/70 p-2.5 text-slate-500 shadow-inner transition-colors hover:text-slate-700"
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
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative rounded-xl bg-white/70 p-2.5 text-slate-500 shadow-inner transition-colors hover:text-slate-700"
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
                    <button className="text-sm text-primary-600 hover:text-primary-700">
                      查看全部通知
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 设置 */}
          <button
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            title="设置"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;