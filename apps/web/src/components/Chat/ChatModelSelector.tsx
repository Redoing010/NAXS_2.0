import React from 'react';
import { clsx } from 'clsx';
import { Activity, CheckCircle2, AlertCircle, RefreshCw, Server, TestTube2 } from 'lucide-react';
import type { LLMModelOption } from '../../store';
import type { ModelTestResult } from '../../hooks/useChatService';

interface ChatModelSelectorProps {
  models: LLMModelOption[];
  currentModel: string;
  onChange: (model: string) => void;
  onTestConnection?: (provider: string) => Promise<ModelTestResult>;
  tests?: Record<string, ModelTestResult>;
  isLoading?: boolean;
}

const providerLabel: Record<string, string> = {
  openai: 'OpenAI',
  qwen: '通义千问',
  anthropic: 'Claude',
  gemini: 'Google Gemini',
};

const statusStyles: Record<string, string> = {
  online: 'bg-green-100 text-green-700 border-green-200',
  degraded: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  offline: 'bg-red-100 text-red-700 border-red-200',
};

export const ChatModelSelector: React.FC<ChatModelSelectorProps> = ({
  models,
  currentModel,
  onChange,
  onTestConnection,
  tests,
  isLoading,
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">模型选择</h3>
          <p className="text-xs text-gray-500">支持通义千问 QWEN 与多家厂商模型</p>
        </div>
        {isLoading && (
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <RefreshCw className="w-3 h-3 animate-spin" />
            <span>同步模型列表</span>
          </div>
        )}
      </div>

      <div className="space-y-2">
        {models.map((model) => {
          const isActive = model.model === currentModel || model.id === currentModel;
          const status = model.status ?? 'online';
          const test = tests?.[model.provider];

          return (
            <div
              key={model.id}
              className={clsx(
                'border rounded-lg p-3 transition-all duration-200 bg-white shadow-sm',
                isActive ? 'border-primary-300 ring-2 ring-primary-100' : 'border-gray-200 hover:border-primary-200'
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <Server className={clsx('w-4 h-4', isActive ? 'text-primary-600' : 'text-gray-400')} />
                    <button
                      type="button"
                      onClick={() => onChange(model.model)}
                      className={clsx(
                        'text-sm font-medium transition-colors',
                        isActive ? 'text-primary-700' : 'text-gray-800 hover:text-primary-600'
                      )}
                    >
                      {model.name}
                    </button>
                    <span className="text-xs text-gray-500">
                      {providerLabel[model.provider] ?? model.provider}
                    </span>
                  </div>

                  <p className="mt-1 text-xs text-gray-500 leading-snug">
                    {model.description ?? '无详细描述'}
                  </p>

                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    <span
                      className={clsx(
                        'inline-flex items-center px-2 py-0.5 rounded-full border',
                        statusStyles[status] ?? statusStyles.online
                      )}
                    >
                      <Activity className="w-3 h-3 mr-1" />
                      {status === 'online'
                        ? '在线'
                        : status === 'degraded'
                        ? '性能下降'
                        : '离线'}
                    </span>
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                      上下文 {model.context}
                    </span>
                    {model.capabilities?.map((cap) => (
                      <span
                        key={cap}
                        className="inline-flex items-center px-2 py-0.5 rounded-full bg-primary-50 text-primary-600"
                      >
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>

                {onTestConnection && (
                  <div className="ml-4 flex flex-col items-end space-y-1">
                    <button
                      type="button"
                      onClick={() => onTestConnection(model.provider)}
                      className="inline-flex items-center px-2 py-1 text-xs font-medium text-primary-600 border border-primary-200 rounded hover:bg-primary-50"
                    >
                      <TestTube2 className="w-3 h-3 mr-1" /> 测试连接
                    </button>
                    {test && (
                      <div
                        className={clsx(
                          'inline-flex items-center px-2 py-0.5 text-[11px] rounded-full',
                          test.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        )}
                      >
                        {test.success ? <CheckCircle2 className="w-3 h-3 mr-1" /> : <AlertCircle className="w-3 h-3 mr-1" />}
                        {test.success ? '连接正常' : '连接失败'}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ChatModelSelector;
