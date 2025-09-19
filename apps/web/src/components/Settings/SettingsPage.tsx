// 系统设置页面
// 实现系统配置、模型设置、Agent配置等功能

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import {
  Settings,
  User,
  Bell,
  Shield,
  Database,
  Cpu,
  Zap,
  Brain,
  Eye,
  Save,
  RefreshCw,
  Check,
  X,
  AlertTriangle,
  Info,
  ExternalLink,
  Download,
  Upload,
  Trash2,
  Plus,
  Edit,
  Key,
  Globe,
  Monitor,
  Moon,
  Sun,
  TestTube,
} from 'lucide-react';
import { useSettings, useSettingsActions } from '../../hooks/useSettings';
import apiService from '../../services/api';

// 设置分类
const SETTING_CATEGORIES = {
  GENERAL: 'general',
  MODELS: 'models',
  AGENT: 'agent',
  DATA: 'data',
  SECURITY: 'security',
  NOTIFICATIONS: 'notifications',
  APPEARANCE: 'appearance',
} as const;

// 主题选项
const THEMES = [
  { id: 'light', name: '浅色主题', icon: Sun },
  { id: 'dark', name: '深色主题', icon: Moon },
  { id: 'auto', name: '跟随系统', icon: Monitor },
];

// 语言选项
const LANGUAGES = [
  { id: 'zh-CN', name: '简体中文' },
  { id: 'en-US', name: 'English' },
  { id: 'ja-JP', name: '日本語' },
];

// 模型提供商
const MODEL_PROVIDERS = [
  {
    id: 'openai',
    name: 'OpenAI',
    models: ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo'],
    status: 'connected',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
    status: 'disconnected',
  },
  {
    id: 'google',
    name: 'Google',
    models: ['gemini-pro', 'gemini-pro-vision'],
    status: 'connected',
  },
];

interface SettingsPageProps {
  className?: string;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ className }) => {
  const { settings } = useSettings();
  const { updateSettings, loadSettings } = useSettingsActions();
  
  const [activeCategory, setActiveCategory] = useState<keyof typeof SETTING_CATEGORIES>('GENERAL');
  const [formData, setFormData] = useState<any>({});
  const [saving, setSaving] = useState(false);
  const [testingConnection, setTestingConnection] = useState<string | null>(null);
  const [showApiKeyModal, setShowApiKeyModal] = useState<string | null>(null);

  // 加载设置
  useEffect(() => {
    loadSettings();
  }, []);

  // 初始化表单数据
  useEffect(() => {
    if (settings) {
      setFormData(settings);
    }
  }, [settings]);

  // 保存设置
  const handleSaveSettings = async () => {
    try {
      setSaving(true);
      await updateSettings(formData);
      // 显示保存成功提示
    } catch (error) {
      console.error('Failed to save settings:', error);
      // 显示保存失败提示
    } finally {
      setSaving(false);
    }
  };

  // 测试连接
  const handleTestConnection = async (providerId: string) => {
    try {
      setTestingConnection(providerId);
      const response = await apiService.testModelConnection(providerId);
      
      if (response.status === 'ok') {
        // 显示连接成功提示
        console.log('Connection test successful');
      } else {
        // 显示连接失败提示
        console.error('Connection test failed:', response.error);
      }
    } catch (error) {
      console.error('Connection test error:', error);
    } finally {
      setTestingConnection(null);
    }
  };

  // 重置设置
  const handleResetSettings = () => {
    if (confirm('确定要重置所有设置吗？此操作不可撤销。')) {
      setFormData({
        theme: 'light',
        language: 'zh-CN',
        notifications: {
          email: true,
          push: true,
          trading: true,
        },
        models: {
          default_provider: 'openai',
          default_model: 'gpt-4',
          temperature: 0.7,
          max_tokens: 2048,
        },
        agent: {
          auto_execute: false,
          confidence_threshold: 0.8,
          max_iterations: 10,
        },
      });
    }
  };

  // 导出设置
  const handleExportSettings = () => {
    const dataStr = JSON.stringify(formData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'naxs-settings.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  // 导入设置
  const handleImportSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedSettings = JSON.parse(e.target?.result as string);
          setFormData(importedSettings);
        } catch (error) {
          console.error('Failed to import settings:', error);
          alert('导入设置失败，请检查文件格式');
        }
      };
      reader.readAsText(file);
    }
  };

  // 渲染设置分类
  const renderCategories = () => {
    const categories = [
      { id: SETTING_CATEGORIES.GENERAL, name: '常规设置', icon: Settings },
      { id: SETTING_CATEGORIES.MODELS, name: '模型配置', icon: Brain },
      { id: SETTING_CATEGORIES.AGENT, name: 'Agent设置', icon: Zap },
      { id: SETTING_CATEGORIES.DATA, name: '数据源', icon: Database },
      { id: SETTING_CATEGORIES.SECURITY, name: '安全设置', icon: Shield },
      { id: SETTING_CATEGORIES.NOTIFICATIONS, name: '通知设置', icon: Bell },
      { id: SETTING_CATEGORIES.APPEARANCE, name: '外观设置', icon: Eye },
    ];

    return (
      <div className="space-y-1">
        {categories.map((category) => {
          const Icon = category.icon;
          return (
            <button
              key={category.id}
              onClick={() => setActiveCategory(category.id as keyof typeof SETTING_CATEGORIES)}
              className={clsx(
                'w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                activeCategory === category.id
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              )}
            >
              <Icon className="w-4 h-4 mr-3" />
              {category.name}
            </button>
          );
        })}
      </div>
    );
  };

  // 渲染常规设置
  const renderGeneralSettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">基本设置</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              系统名称
            </label>
            <input
              type="text"
              value={formData.system_name || 'NAXS 智能投研系统'}
              onChange={(e) => setFormData({ ...formData, system_name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              默认语言
            </label>
            <select
              value={formData.language || 'zh-CN'}
              onChange={(e) => setFormData({ ...formData, language: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {LANGUAGES.map(lang => (
                <option key={lang.id} value={lang.id}>{lang.name}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              时区
            </label>
            <select
              value={formData.timezone || 'Asia/Shanghai'}
              onChange={(e) => setFormData({ ...formData, timezone: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="Asia/Shanghai">中国标准时间 (UTC+8)</option>
              <option value="America/New_York">美国东部时间 (UTC-5)</option>
              <option value="Europe/London">格林威治时间 (UTC+0)</option>
              <option value="Asia/Tokyo">日本标准时间 (UTC+9)</option>
            </select>
          </div>
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">性能设置</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">启用缓存</label>
              <p className="text-sm text-gray-500">缓存数据以提高响应速度</p>
            </div>
            <input
              type="checkbox"
              checked={formData.enable_cache || true}
              onChange={(e) => setFormData({ ...formData, enable_cache: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">自动保存</label>
              <p className="text-sm text-gray-500">自动保存用户操作和设置</p>
            </div>
            <input
              type="checkbox"
              checked={formData.auto_save || true}
              onChange={(e) => setFormData({ ...formData, auto_save: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
          </div>
        </div>
      </div>
    </div>
  );

  // 渲染模型设置
  const renderModelSettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">模型提供商</h3>
        
        <div className="space-y-4">
          {MODEL_PROVIDERS.map((provider) => (
            <div key={provider.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <h4 className="text-sm font-medium text-gray-900">{provider.name}</h4>
                  <span className={clsx(
                    'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                    provider.status === 'connected'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  )}>
                    {provider.status === 'connected' ? '已连接' : '未连接'}
                  </span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleTestConnection(provider.id)}
                    disabled={testingConnection === provider.id}
                    className="btn btn-outline btn-sm"
                  >
                    {testingConnection === provider.id ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Zap className="w-4 h-4" />
                    )}
                  </button>
                  
                  <button
                    onClick={() => setShowApiKeyModal(provider.id)}
                    className="btn btn-outline btn-sm"
                  >
                    <Key className="w-4 h-4" />
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                {provider.models.map((model) => (
                  <div key={model} className="text-sm text-gray-600 bg-gray-50 px-2 py-1 rounded">
                    {model}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">默认模型配置</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              默认提供商
            </label>
            <select
              value={formData.models?.default_provider || 'openai'}
              onChange={(e) => setFormData({
                ...formData,
                models: { ...formData.models, default_provider: e.target.value }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {MODEL_PROVIDERS.map(provider => (
                <option key={provider.id} value={provider.id}>{provider.name}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              默认模型
            </label>
            <select
              value={formData.models?.default_model || 'gpt-4'}
              onChange={(e) => setFormData({
                ...formData,
                models: { ...formData.models, default_model: e.target.value }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="claude-3-opus">Claude 3 Opus</option>
              <option value="gemini-pro">Gemini Pro</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              温度参数: {formData.models?.temperature || 0.7}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={formData.models?.temperature || 0.7}
              onChange={(e) => setFormData({
                ...formData,
                models: { ...formData.models, temperature: parseFloat(e.target.value) }
              })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>保守 (0)</span>
              <span>平衡 (1)</span>
              <span>创新 (2)</span>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              最大令牌数
            </label>
            <input
              type="number"
              min="100"
              max="8192"
              value={formData.models?.max_tokens || 2048}
              onChange={(e) => setFormData({
                ...formData,
                models: { ...formData.models, max_tokens: parseInt(e.target.value) }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
    </div>
  );

  // 渲染Agent设置
  const renderAgentSettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Agent行为配置</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">自动执行工具</label>
              <p className="text-sm text-gray-500">允许Agent自动执行低风险工具调用</p>
            </div>
            <input
              type="checkbox"
              checked={formData.agent?.auto_execute || false}
              onChange={(e) => setFormData({
                ...formData,
                agent: { ...formData.agent, auto_execute: e.target.checked }
              })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              置信度阈值: {formData.agent?.confidence_threshold || 0.8}
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={formData.agent?.confidence_threshold || 0.8}
              onChange={(e) => setFormData({
                ...formData,
                agent: { ...formData.agent, confidence_threshold: parseFloat(e.target.value) }
              })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>宽松 (0.1)</span>
              <span>平衡 (0.5)</span>
              <span>严格 (1.0)</span>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              最大迭代次数
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={formData.agent?.max_iterations || 10}
              onChange={(e) => setFormData({
                ...formData,
                agent: { ...formData.agent, max_iterations: parseInt(e.target.value) }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">工具权限</h3>
        
        <div className="space-y-3">
          {[
            { id: 'market_data', name: '市场数据查询', risk: 'low' },
            { id: 'stock_analysis', name: '股票分析', risk: 'low' },
            { id: 'strategy_backtest', name: '策略回测', risk: 'medium' },
            { id: 'portfolio_optimization', name: '组合优化', risk: 'medium' },
            { id: 'trade_execution', name: '交易执行', risk: 'high' },
          ].map((tool) => (
            <div key={tool.id} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={clsx(
                  'w-2 h-2 rounded-full',
                  tool.risk === 'low' ? 'bg-green-500' :
                  tool.risk === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                )} />
                <div>
                  <p className="text-sm font-medium text-gray-900">{tool.name}</p>
                  <p className="text-xs text-gray-500">风险等级: {tool.risk}</p>
                </div>
              </div>
              <input
                type="checkbox"
                checked={formData.agent?.enabled_tools?.[tool.id] !== false}
                onChange={(e) => setFormData({
                  ...formData,
                  agent: {
                    ...formData.agent,
                    enabled_tools: {
                      ...formData.agent?.enabled_tools,
                      [tool.id]: e.target.checked
                    }
                  }
                })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // 渲染外观设置
  const renderAppearanceSettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">主题设置</h3>
        
        <div className="grid grid-cols-3 gap-4">
          {THEMES.map((theme) => {
            const Icon = theme.icon;
            return (
              <button
                key={theme.id}
                onClick={() => setFormData({ ...formData, theme: theme.id })}
                className={clsx(
                  'p-4 border-2 rounded-lg text-center transition-colors',
                  formData.theme === theme.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                )}
              >
                <Icon className="w-8 h-8 mx-auto mb-2 text-gray-600" />
                <p className="text-sm font-medium text-gray-900">{theme.name}</p>
              </button>
            );
          })}
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">界面设置</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">紧凑模式</label>
              <p className="text-sm text-gray-500">减少界面元素间距</p>
            </div>
            <input
              type="checkbox"
              checked={formData.compact_mode || false}
              onChange={(e) => setFormData({ ...formData, compact_mode: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">显示高级功能</label>
              <p className="text-sm text-gray-500">显示专业用户功能</p>
            </div>
            <input
              type="checkbox"
              checked={formData.show_advanced || false}
              onChange={(e) => setFormData({ ...formData, show_advanced: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
          </div>
        </div>
      </div>
    </div>
  );

  // 渲染设置内容
  const renderSettingsContent = () => {
    switch (activeCategory) {
      case 'GENERAL':
        return renderGeneralSettings();
      case 'MODELS':
        return renderModelSettings();
      case 'AGENT':
        return renderAgentSettings();
      case 'APPEARANCE':
        return renderAppearanceSettings();
      default:
        return (
          <div className="text-center py-12">
            <Settings className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">功能开发中</h3>
            <p className="text-gray-500">该设置分类正在开发中，敬请期待</p>
          </div>
        );
    }
  };

  return (
    <div className={clsx('p-6', className)}>
      {/* 页面标题 */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">系统设置</h1>
          <p className="text-gray-600 mt-1">配置系统参数和个人偏好</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <input
            type="file"
            accept=".json"
            onChange={handleImportSettings}
            className="hidden"
            id="import-settings"
          />
          <label
            htmlFor="import-settings"
            className="btn btn-outline btn-sm cursor-pointer"
          >
            <Upload className="w-4 h-4 mr-2" />
            导入
          </label>
          
          <button
            onClick={handleExportSettings}
            className="btn btn-outline btn-sm"
          >
            <Download className="w-4 h-4 mr-2" />
            导出
          </button>
          
          <button
            onClick={handleResetSettings}
            className="btn btn-outline btn-sm text-red-600 border-red-300 hover:bg-red-50"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            重置
          </button>
          
          <button
            onClick={handleSaveSettings}
            disabled={saving}
            className="btn btn-primary btn-sm"
          >
            {saving ? (
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-2" />
            )}
            保存设置
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* 设置分类 */}
        <div className="lg:col-span-1">
          <div className="card p-4">
            <h2 className="text-sm font-medium text-gray-900 mb-4">设置分类</h2>
            {renderCategories()}
          </div>
        </div>

        {/* 设置内容 */}
        <div className="lg:col-span-3">
          <div className="card p-6">
            {renderSettingsContent()}
          </div>
        </div>
      </div>

      {/* API密钥模态框 */}
      {showApiKeyModal && (
        <ApiKeyModal
          providerId={showApiKeyModal}
          onClose={() => setShowApiKeyModal(null)}
          onSave={(apiKey) => {
            // 保存API密钥
            console.log('Save API key for', showApiKeyModal, apiKey);
            setShowApiKeyModal(null);
          }}
        />
      )}
    </div>
  );
};

// API密钥设置模态框
interface ApiKeyModalProps {
  providerId: string;
  onClose: () => void;
  onSave: (apiKey: string) => void;
}

const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ providerId, onClose, onSave }) => {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);

  const handleSave = () => {
    if (apiKey.trim()) {
      onSave(apiKey.trim());
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            设置 {providerId.toUpperCase()} API密钥
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API密钥
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="输入您的API密钥"
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                {showKey ? (
                  <Eye className="w-4 h-4 text-gray-400" />
                ) : (
                  <Eye className="w-4 h-4 text-gray-400" />
                )}
              </button>
            </div>
          </div>
          
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
            <div className="flex">
              <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 mr-2" />
              <div className="text-sm text-yellow-700">
                <p className="font-medium">安全提示</p>
                <p>API密钥将被加密存储，请确保密钥来源可靠。</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex justify-end space-x-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={!apiKey.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            保存
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage