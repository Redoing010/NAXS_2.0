// 侧边栏组件

import React from 'react';
import { clsx } from 'clsx';
import {
  BarChart3,
  MessageSquare,
  TrendingUp,
  Target,
  Settings,
  Home,
  Activity,
  Bell,
  User,
  ChevronLeft,
  ChevronRight,
  Monitor,
} from 'lucide-react';
import { useUI, useUIActions } from '../../store';

// 导航项接口
interface NavItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  view: 'dashboard' | 'chat' | 'strategies' | 'recommendations' | 'settings' | 'testing';
  badge?: number;
}

// 导航项配置
const navItems: NavItem[] = [
  {
    id: 'dashboard',
    label: '仪表板',
    icon: Home,
    view: 'dashboard',
  },
  {
    id: 'chat',
    label: 'AI助手',
    icon: MessageSquare,
    view: 'chat',
  },
  {
    id: 'strategies',
    label: '策略管理',
    icon: TrendingUp,
    view: 'strategies',
  },
  {
    id: 'recommendations',
    label: '选股推荐',
    icon: Target,
    view: 'recommendations',
  },
  {
    id: 'testing',
    label: '系统监控',
    icon: Monitor,
    view: 'testing',
  },
  {
    id: 'settings',
    label: '设置',
    icon: Settings,
    view: 'settings',
  },
];

interface SidebarProps {
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ className }) => {
  const { sidebarOpen, currentView, notifications } = useUI();
  const { setSidebarOpen, setCurrentView } = useUIActions();

  // 计算未读通知数量
  const unreadNotifications = (notifications || []).filter(n => !n.read).length;

  // 处理导航项点击
  const handleNavClick = (view: NavItem['view']) => {
    setCurrentView(view);
  };

  // 切换侧边栏
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div
      className={clsx(
        'flex flex-col bg-white border-r border-gray-200 transition-all duration-300 ease-in-out',
        sidebarOpen ? 'w-64' : 'w-16',
        className
      )}
    >
      {/* 头部 */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className={clsx('flex items-center', !sidebarOpen && 'justify-center')}>
          <div className="flex items-center justify-center w-8 h-8 bg-primary-600 rounded-lg">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          {sidebarOpen && (
            <div className="ml-3">
              <h1 className="text-lg font-semibold text-gray-900">NAXS</h1>
              <p className="text-xs text-gray-500">智能投研系统</p>
            </div>
          )}
        </div>
        
        <button
          onClick={toggleSidebar}
          className="p-1 rounded-md hover:bg-gray-100 transition-colors"
        >
          {sidebarOpen ? (
            <ChevronLeft className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-500" />
          )}
        </button>
      </div>

      {/* 导航菜单 */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const isActive = currentView === item.view;
          const Icon = item.icon;
          
          // 特殊处理通知徽章
          let badge = item.badge;
          if (item.id === 'dashboard' && unreadNotifications > 0) {
            badge = unreadNotifications;
          }

          return (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.view)}
              className={clsx(
                'w-full flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary-100 text-primary-700 border border-primary-200'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
                !sidebarOpen && 'justify-center'
              )}
              title={!sidebarOpen ? item.label : undefined}
            >
              <Icon className={clsx('w-5 h-5', sidebarOpen && 'mr-3')} />
              {sidebarOpen && (
                <>
                  <span className="flex-1 text-left">{item.label}</span>
                  {badge && badge > 0 && (
                    <span className="inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white bg-red-500 rounded-full">
                      {badge > 99 ? '99+' : badge}
                    </span>
                  )}
                </>
              )}
              {!sidebarOpen && badge && badge > 0 && (
                <span className="absolute top-1 right-1 inline-flex items-center justify-center w-2 h-2 bg-red-500 rounded-full" />
              )}
            </button>
          );
        })}
      </nav>

      {/* 底部用户信息 */}
      <div className="p-4 border-t border-gray-200">
        <div className={clsx('flex items-center', !sidebarOpen && 'justify-center')}>
          <div className="flex items-center justify-center w-8 h-8 bg-gray-300 rounded-full">
            <User className="w-4 h-4 text-gray-600" />
          </div>
          {sidebarOpen && (
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium text-gray-900">用户</p>
              <p className="text-xs text-gray-500">投资者</p>
            </div>
          )}
        </div>
        
        {sidebarOpen && (
          <div className="mt-3 flex space-x-2">
            <button className="flex-1 px-3 py-1 text-xs text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
              个人资料
            </button>
            <button className="flex-1 px-3 py-1 text-xs text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
              退出
            </button>
          </div>
        )}
      </div>

      {/* 状态指示器 */}
      <div className={clsx('px-4 py-2 border-t border-gray-200', !sidebarOpen && 'px-2')}>
        <div className={clsx('flex items-center', !sidebarOpen && 'justify-center')}>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            {sidebarOpen && (
              <span className="ml-2 text-xs text-gray-500">系统正常</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;