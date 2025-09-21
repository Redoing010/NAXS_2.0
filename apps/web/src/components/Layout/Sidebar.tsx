// @ts-nocheck
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
        'relative z-20 flex flex-col h-full bg-white/75 backdrop-blur-xl border-r border-white/40 shadow-soft transition-all duration-300 ease-in-out',
        sidebarOpen ? 'w-64' : 'w-20',
        className
      )}
    >
      {/* 头部 */}
      <div className="flex items-center justify-between p-4 border-b border-white/40">
        <div className={clsx('flex items-center', !sidebarOpen && 'justify-center')}>
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-sky-400 text-white shadow-soft">
            <BarChart3 className="w-5 h-5" />
          </div>
          {sidebarOpen && (
            <div className="ml-3">
              <h1 className="text-lg font-semibold text-slate-900">NAXS</h1>
              <p className="text-xs text-slate-500">AI 智能投研</p>
            </div>
          )}
        </div>

        <button
          onClick={toggleSidebar}
          className="p-1.5 rounded-lg text-slate-500 hover:bg-white/60 transition-colors"
        >
          {sidebarOpen ? (
            <ChevronLeft className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
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
                'w-full flex items-center px-3 py-2 rounded-xl text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-gradient-to-r from-sky-500/90 to-blue-500/90 text-white shadow-soft'
                  : 'text-slate-600 hover:bg-white/60 hover:text-slate-900',
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
      <div className="p-4 border-t border-white/40 bg-white/50">
        <div className={clsx('flex items-center', !sidebarOpen && 'justify-center')}>
          <div className="flex items-center justify-center w-9 h-9 bg-gradient-to-br from-slate-200 to-slate-100 rounded-full">
            <User className="w-4 h-4 text-slate-700" />
          </div>
          {sidebarOpen && (
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium text-slate-900">用户</p>
              <p className="text-xs text-slate-500">投资者</p>
            </div>
          )}
        </div>

        {sidebarOpen && (
          <div className="mt-3 flex space-x-2">
            <button className="flex-1 px-3 py-1 text-xs text-slate-600 bg-white/70 border border-white/60 rounded-md hover:bg-white transition-colors">
              个人资料
            </button>
            <button className="flex-1 px-3 py-1 text-xs text-slate-600 bg-white/70 border border-white/60 rounded-md hover:bg-white transition-colors">
              退出
            </button>
          </div>
        )}
      </div>

      {/* 状态指示器 */}
      <div className={clsx('px-4 py-2 border-t border-white/40 bg-white/40', !sidebarOpen && 'px-2')}>
        <div className={clsx('flex items-center', !sidebarOpen && 'justify-center')}>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            {sidebarOpen && (
              <span className="ml-2 text-xs text-slate-500">系统稳定运行</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;