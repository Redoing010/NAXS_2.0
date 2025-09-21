// @ts-nocheck
// 用户偏好设置组件
// 提供个性化配置选项，包括界面主题、数据展示、通知设置等

import React, { useState, useEffect, useCallback } from 'react';
import { clsx } from 'clsx';
import {
  Settings,
  User,
  Bell,
  BellOff,
  Eye,
  EyeOff,
  Palette,
  Monitor,
  Sun,
  Moon,
  Volume2,
  VolumeX,
  Smartphone,
  Mail,
  MessageSquare,
  BarChart3,
  LineChart,
  PieChart,
  TrendingUp,
  Clock,
  RefreshCw,
  Database,
  Wifi,
  Shield,
  Key,
  Download,
  Upload,
  Save,
  RotateCcw,
  Check,
  X,
  Info,
  AlertTriangle,
} from 'lucide-react';
import { useSettings } from '../../hooks/useSettings';
import apiService from '../../services/api';

// 偏好设置类型定义
interface UserPreferences {
  // 界面设置
  theme: 'light' | 'dark' | 'auto';
  language: 'zh-CN' | 'en-US';
  fontSize: 'small' | 'medium' | 'large';
  compactMode: boolean;
  
  // 数据展示设置
  defaultTimeRange: '1h' | '4h' | '1d' | '1w' | '1m';
  refreshInterval: number; // 秒
  maxDataPoints: number;
  chartType: 'line' | 'candlestick' | 'bar';
  showVolume: boolean;
  showIndicators: boolean;
  
  // 通知设置
  enableNotifications: boolean;
  emailNotifications: boolean;
  pushNotifications: boolean;
  soundEnabled: boolean;
  notificationTypes: {
    alerts: boolean;
    factorUpdates: boolean;
    systemStatus: boolean;
    agentActivity: boolean;
    dataQuality: boolean;
  };
  
  // Agent设置
  defaultModel: string;
  enableAutoTools: boolean;
  maxTokens: number;
  temperature: number;
  
  // 仪表板设置
  dashboardLayout: 'grid' | 'list' | 'compact';
  visibleWidgets: string[];
  widgetOrder: string[];
  
  // 数据源设置
  preferredDataSources: string[];
  dataQualityThreshold: number;
  
  // 安全设置
  sessionTimeout: number; // 分钟
  requireReauth: boolean;
  enableTwoFactor: boolean;
}

// 默认偏好设置
const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'auto',
  language: 'zh-CN',
  fontSize: 'medium',
  compactMode: false,
  defaultTimeRange: '1d',
  refreshInterval: 30,
  maxDataPoints: 1000,
  chartType: 'line',
  showVolume: true,
  showIndicators: true,
  enableNotifications: true,
  emailNotifications: true,
  pushNotifications: true,
  soundEnabled: true,
  notificationTypes: {
    alerts: true,
    factorUpdates: true,
    systemStatus: true,
    agentActivity: false,
    dataQuality: true,
  },
  defaultModel: 'gpt-4',
  enableAutoTools: true,
  maxTokens: 4000,
  temperature: 0.7,
  dashboardLayout: 'grid',
  visibleWidgets: ['market_data', 'factor_monitor', 'agent_status', 'system_health'],
  widgetOrder: ['market_data', 'factor_monitor', 'agent_status', 'system_health'],
  preferredDataSources: ['akshare', 'tushare'],
  dataQualityThreshold: 0.8,
  sessionTimeout: 60,
  requireReauth: false,
  enableTwoFactor: false,
};

// 设置分组
interface SettingGroup {
  id: string;
  title: string;
  icon: React.ComponentType<any>;
  description: string;
}

const SETTING_GROUPS: SettingGroup[] = [
  {
    id: 'appearance',
    title: '外观设置',
    icon: Palette,
    description: '主题、字体大小和界面布局'
  },
  {
    id: 'data',
    title: '数据设置',
    icon: Database,
    description: '数据源、刷新频率和展示方式'
  },
  {
    id: 'notifications',
    title: '通知设置',
    icon: Bell,
    description: '告警通知和消息推送'
  },
  {
    id: 'agent',
    title: 'Agent设置',
    icon: MessageSquare,
    description: 'AI模型和工具配置'
  },
  {
    id: 'dashboard',
    title: '仪表板设置',
    icon: BarChart3,
    description: '组件布局和显示选项'
  },
  {
    id: 'security',
    title: '安全设置',
    icon: Shield,
    description: '认证和会话管理'
  },
];

interface UserPreferencesProps {
  className?: string;
}

export const UserPreferences: React.FC<UserPreferencesProps> = ({ className }) => {
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [activeGroup, setActiveGroup] = useState('appearance');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  
  const { updateSettings } = useSettings();

  // 加载用户偏好设置
  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.getUserPreferences();
      if (response.success) {
        setPreferences({ ...DEFAULT_PREFERENCES, ...response.data });
      }
    } catch (error) {
      console.error('Failed to load preferences:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 更新偏好设置
  const updatePreference = useCallback(<K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
    setSaveStatus('idle');
  }, []);

  // 更新嵌套偏好设置
  const updateNestedPreference = useCallback(<T extends keyof UserPreferences>(
    parentKey: T,
    childKey: keyof UserPreferences[T],
    value: any
  ) => {
    setPreferences(prev => ({
      ...prev,
      [parentKey]: {
        ...prev[parentKey],
        [childKey]: value
      }
    }));
    setHasChanges(true);
    setSaveStatus('idle');
  }, []);

  // 保存设置
  const savePreferences = async () => {
    setIsSaving(true);
    try {
      const response = await apiService.updateUserPreferences(preferences);
      if (response.success) {
        setHasChanges(false);
        setSaveStatus('success');
        
        // 应用设置到全局状态
        updateSettings({
          theme: preferences.theme,
          language: preferences.language,
          refreshInterval: preferences.refreshInterval,
        });
        
        setTimeout(() => setSaveStatus('idle'), 3000);
      } else {
        setSaveStatus('error');
      }
    } catch (error) {
      console.error('Failed to save preferences:', error);
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  };

  // 重置设置
  const resetPreferences = () => {
    setPreferences(DEFAULT_PREFERENCES);
    setHasChanges(true);
    setSaveStatus('idle');
  };

  // 导出设置
  const exportPreferences = () => {
    const dataStr = JSON.stringify(preferences, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'naxs-preferences.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  // 导入设置
  const importPreferences = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedPrefs = JSON.parse(e.target?.result as string);
        setPreferences({ ...DEFAULT_PREFERENCES, ...importedPrefs });
        setHasChanges(true);
        setSaveStatus('idle');
      } catch (error) {
        console.error('Failed to import preferences:', error);
        setSaveStatus('error');
      }
    };
    reader.readAsText(file);
  };

  // 渲染外观设置
  const renderAppearanceSettings = () => (
    <div className="space-y-6">
      {/* 主题设置 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">主题</label>
        <div className="grid grid-cols-3 gap-3">
          {[
            { value: 'light', label: '浅色', icon: Sun },
            { value: 'dark', label: '深色', icon: Moon },
            { value: 'auto', label: '自动', icon: Monitor },
          ].map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => updatePreference('theme', value as any)}
              className={clsx(
                'flex flex-col items-center p-4 border-2 rounded-lg transition-colors',
                preferences.theme === value
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 hover:border-gray-300'
              )}
            >
              <Icon className="w-6 h-6 mb-2" />
              <span className="text-sm font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 字体大小 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">字体大小</label>
        <select
          value={preferences.fontSize}
          onChange={(e) => updatePreference('fontSize', e.target.value as any)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="small">小</option>
          <option value="medium">中</option>
          <option value="large">大</option>
        </select>
      </div>

      {/* 紧凑模式 */}
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-gray-700">紧凑模式</div>
          <div className="text-sm text-gray-500">减少界面间距，显示更多内容</div>
        </div>
        <button
          onClick={() => updatePreference('compactMode', !preferences.compactMode)}
          className={clsx(
            'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
            preferences.compactMode ? 'bg-blue-600' : 'bg-gray-200'
          )}
        >
          <span
            className={clsx(
              'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
              preferences.compactMode ? 'translate-x-6' : 'translate-x-1'
            )}
          />
        </button>
      </div>

      {/* 语言设置 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">语言</label>
        <select
          value={preferences.language}
          onChange={(e) => updatePreference('language', e.target.value as any)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="zh-CN">简体中文</option>
          <option value="en-US">English</option>
        </select>
      </div>
    </div>
  );

  // 渲染数据设置
  const renderDataSettings = () => (
    <div className="space-y-6">
      {/* 默认时间范围 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">默认时间范围</label>
        <select
          value={preferences.defaultTimeRange}
          onChange={(e) => updatePreference('defaultTimeRange', e.target.value as any)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="1h">1小时</option>
          <option value="4h">4小时</option>
          <option value="1d">1天</option>
          <option value="1w">1周</option>
          <option value="1m">1个月</option>
        </select>
      </div>

      {/* 刷新间隔 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          刷新间隔: {preferences.refreshInterval}秒
        </label>
        <input
          type="range"
          min="5"
          max="300"
          step="5"
          value={preferences.refreshInterval}
          onChange={(e) => updatePreference('refreshInterval', parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>5秒</span>
          <span>5分钟</span>
        </div>
      </div>

      {/* 图表类型 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">默认图表类型</label>
        <div className="grid grid-cols-3 gap-3">
          {[
            { value: 'line', label: '折线图', icon: LineChart },
            { value: 'candlestick', label: 'K线图', icon: BarChart3 },
            { value: 'bar', label: '柱状图', icon: PieChart },
          ].map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => updatePreference('chartType', value as any)}
              className={clsx(
                'flex flex-col items-center p-3 border-2 rounded-lg transition-colors',
                preferences.chartType === value
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 hover:border-gray-300'
              )}
            >
              <Icon className="w-5 h-5 mb-1" />
              <span className="text-xs font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 显示选项 */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-700">显示成交量</div>
            <div className="text-sm text-gray-500">在图表中显示成交量信息</div>
          </div>
          <button
            onClick={() => updatePreference('showVolume', !preferences.showVolume)}
            className={clsx(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
              preferences.showVolume ? 'bg-blue-600' : 'bg-gray-200'
            )}
          >
            <span
              className={clsx(
                'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                preferences.showVolume ? 'translate-x-6' : 'translate-x-1'
              )}
            />
          </button>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-700">显示技术指标</div>
            <div className="text-sm text-gray-500">在图表中显示技术分析指标</div>
          </div>
          <button
            onClick={() => updatePreference('showIndicators', !preferences.showIndicators)}
            className={clsx(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
              preferences.showIndicators ? 'bg-blue-600' : 'bg-gray-200'
            )}
          >
            <span
              className={clsx(
                'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                preferences.showIndicators ? 'translate-x-6' : 'translate-x-1'
              )}
            />
          </button>
        </div>
      </div>

      {/* 数据质量阈值 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          数据质量阈值: {(preferences.dataQualityThreshold * 100).toFixed(0)}%
        </label>
        <input
          type="range"
          min="0.5"
          max="1"
          step="0.05"
          value={preferences.dataQualityThreshold}
          onChange={(e) => updatePreference('dataQualityThreshold', parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );

  // 渲染通知设置
  const renderNotificationSettings = () => (
    <div className="space-y-6">
      {/* 总开关 */}
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-gray-700">启用通知</div>
          <div className="text-sm text-gray-500">接收系统通知和告警</div>
        </div>
        <button
          onClick={() => updatePreference('enableNotifications', !preferences.enableNotifications)}
          className={clsx(
            'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
            preferences.enableNotifications ? 'bg-blue-600' : 'bg-gray-200'
          )}
        >
          <span
            className={clsx(
              'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
              preferences.enableNotifications ? 'translate-x-6' : 'translate-x-1'
            )}
          />
        </button>
      </div>

      {preferences.enableNotifications && (
        <>
          {/* 通知方式 */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-700">通知方式</h4>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Mail className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-700">邮件通知</span>
              </div>
              <button
                onClick={() => updatePreference('emailNotifications', !preferences.emailNotifications)}
                className={clsx(
                  'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                  preferences.emailNotifications ? 'bg-blue-600' : 'bg-gray-200'
                )}
              >
                <span
                  className={clsx(
                    'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                    preferences.emailNotifications ? 'translate-x-6' : 'translate-x-1'
                  )}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Smartphone className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-700">推送通知</span>
              </div>
              <button
                onClick={() => updatePreference('pushNotifications', !preferences.pushNotifications)}
                className={clsx(
                  'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                  preferences.pushNotifications ? 'bg-blue-600' : 'bg-gray-200'
                )}
              >
                <span
                  className={clsx(
                    'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                    preferences.pushNotifications ? 'translate-x-6' : 'translate-x-1'
                  )}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {preferences.soundEnabled ? (
                  <Volume2 className="w-4 h-4 text-gray-500" />
                ) : (
                  <VolumeX className="w-4 h-4 text-gray-500" />
                )}
                <span className="text-sm text-gray-700">声音提醒</span>
              </div>
              <button
                onClick={() => updatePreference('soundEnabled', !preferences.soundEnabled)}
                className={clsx(
                  'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                  preferences.soundEnabled ? 'bg-blue-600' : 'bg-gray-200'
                )}
              >
                <span
                  className={clsx(
                    'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                    preferences.soundEnabled ? 'translate-x-6' : 'translate-x-1'
                  )}
                />
              </button>
            </div>
          </div>

          {/* 通知类型 */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-700">通知类型</h4>
            
            {Object.entries({
              alerts: '系统告警',
              factorUpdates: '因子更新',
              systemStatus: '系统状态',
              agentActivity: 'Agent活动',
              dataQuality: '数据质量',
            }).map(([key, label]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-sm text-gray-700">{label}</span>
                <button
                  onClick={() => updateNestedPreference('notificationTypes', key as any, !preferences.notificationTypes[key as keyof typeof preferences.notificationTypes])}
                  className={clsx(
                    'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                    preferences.notificationTypes[key as keyof typeof preferences.notificationTypes] ? 'bg-blue-600' : 'bg-gray-200'
                  )}
                >
                  <span
                    className={clsx(
                      'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                      preferences.notificationTypes[key as keyof typeof preferences.notificationTypes] ? 'translate-x-6' : 'translate-x-1'
                    )}
                  />
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );

  // 渲染设置内容
  const renderSettingContent = () => {
    switch (activeGroup) {
      case 'appearance':
        return renderAppearanceSettings();
      case 'data':
        return renderDataSettings();
      case 'notifications':
        return renderNotificationSettings();
      default:
        return (
          <div className="flex items-center justify-center h-64 text-gray-500">
            <div className="text-center">
              <Settings className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p>该设置组正在开发中...</p>
            </div>
          </div>
        );
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-400" />
        <span className="ml-2 text-gray-500">加载设置中...</span>
      </div>
    );
  }

  return (
    <div className={clsx('bg-white rounded-lg shadow-sm border border-gray-200', className)}>
      {/* 头部 */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">用户偏好设置</h2>
            <p className="text-sm text-gray-600 mt-1">个性化您的使用体验</p>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* 导入/导出 */}
            <input
              type="file"
              accept=".json"
              onChange={importPreferences}
              className="hidden"
              id="import-preferences"
            />
            <label
              htmlFor="import-preferences"
              className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors cursor-pointer flex items-center space-x-1"
            >
              <Upload className="w-4 h-4" />
              <span>导入</span>
            </label>
            
            <button
              onClick={exportPreferences}
              className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors flex items-center space-x-1"
            >
              <Download className="w-4 h-4" />
              <span>导出</span>
            </button>
            
            <button
              onClick={resetPreferences}
              className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors flex items-center space-x-1"
            >
              <RotateCcw className="w-4 h-4" />
              <span>重置</span>
            </button>
          </div>
        </div>
      </div>

      <div className="flex">
        {/* 侧边栏 */}
        <div className="w-64 border-r border-gray-200 bg-gray-50">
          <nav className="p-4 space-y-2">
            {SETTING_GROUPS.map((group) => {
              const Icon = group.icon;
              return (
                <button
                  key={group.id}
                  onClick={() => setActiveGroup(group.id)}
                  className={clsx(
                    'w-full flex items-center space-x-3 px-3 py-2 text-left rounded-md transition-colors',
                    activeGroup === group.id
                      ? 'bg-blue-100 text-blue-700 border border-blue-200'
                      : 'text-gray-700 hover:bg-gray-100'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <div>
                    <div className="font-medium">{group.title}</div>
                    <div className="text-xs text-gray-500">{group.description}</div>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* 主内容区 */}
        <div className="flex-1">
          <div className="p-6">
            {renderSettingContent()}
          </div>
          
          {/* 底部操作栏 */}
          <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {saveStatus === 'success' && (
                  <div className="flex items-center space-x-1 text-green-600">
                    <Check className="w-4 h-4" />
                    <span className="text-sm">设置已保存</span>
                  </div>
                )}
                {saveStatus === 'error' && (
                  <div className="flex items-center space-x-1 text-red-600">
                    <X className="w-4 h-4" />
                    <span className="text-sm">保存失败</span>
                  </div>
                )}
                {hasChanges && saveStatus === 'idle' && (
                  <div className="flex items-center space-x-1 text-yellow-600">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-sm">有未保存的更改</span>
                  </div>
                )}
              </div>
              
              <button
                onClick={savePreferences}
                disabled={!hasChanges || isSaving}
                className={clsx(
                  'px-4 py-2 rounded-md font-medium transition-colors flex items-center space-x-2',
                  hasChanges && !isSaving
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                )}
              >
                {isSaving ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                <span>{isSaving ? '保存中...' : '保存设置'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPreferences;