// 增强的LLM聊天界面
// 引入 QWEN 模型联动与知识图谱控制

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { clsx } from 'clsx';
import {
  Bot,
  User,
  Send,
  Paperclip,
  Mic,
  Square,
  Settings,
  Trash2,
  Loader2,
  Sparkles,
  Workflow,
  ShieldCheck,
  Copy,
} from 'lucide-react';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import ChatModelSelector from './ChatModelSelector';
import KnowledgeGraphPanel from '../KnowledgeGraph/KnowledgeGraphPanel';
import { useChatService } from '../../hooks/useChatService';
import { useKnowledgeGraphService } from '../../hooks/useKnowledgeGraph';
import type { ChatMessage } from '../../store';

interface EnhancedChatInterfaceProps {
  className?: string;
}

interface MessageBubbleProps {
  message: ChatMessage;
  onCopy: (text: string) => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onCopy }) => {
  const isUser = message.type === 'user';
  const timestamp = useMemo(() => {
    try {
      return format(new Date(message.timestamp), 'HH:mm:ss', { locale: zhCN });
    } catch {
      return '';
    }
  }, [message.timestamp]);

  return (
    <div className={clsx('flex w-full', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={clsx(
          'max-w-2xl rounded-2xl px-4 py-3 shadow-sm border space-y-2 transition-all duration-150',
          isUser
            ? 'bg-primary-600 text-white border-primary-600'
            : 'bg-white text-gray-900 border-gray-200'
        )}
      >
        <div className="flex items-center justify-between space-x-3 text-xs">
          <div className="flex items-center space-x-2">
            {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            <span>{timestamp}</span>
            {!isUser && message.metadata?.model && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-primary-50 text-primary-700">
                {message.metadata.provider ?? 'model'} · {message.metadata.model}
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => onCopy(message.content)}
            className={clsx(
              'inline-flex items-center space-x-1',
              isUser ? 'text-primary-50 hover:text-white' : 'text-gray-400 hover:text-gray-600'
            )}
          >
            <Copy className="w-3 h-3" />
            <span>复制</span>
          </button>
        </div>

        <div className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>

        {!isUser && (
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-gray-500">
            {typeof message.metadata?.tokens === 'number' && (
              <span>Tokens: {message.metadata.tokens}</span>
            )}
            {typeof message.metadata?.execution_time === 'number' && (
              <span>耗时: {Math.round(message.metadata.execution_time)}ms</span>
            )}
            {typeof message.metadata?.confidence === 'number' && (
              <span>置信度: {Math.round(message.metadata.confidence * 100)}%</span>
            )}
            {Array.isArray(message.metadata?.tools_used) && message.metadata?.tools_used.length > 0 && (
              <span>工具: {message.metadata.tools_used.join(', ')}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export const EnhancedChatInterface: React.FC<EnhancedChatInterfaceProps> = ({ className }) => {
  const [input, setInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [agentMode, setAgentMode] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [isSyncingModels, setIsSyncingModels] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    chatState,
    currentModelOption,
    modelTests,
    loadAvailableModels,
    testModelConnection,
    sendAgentMessage,
    clearMessages,
    setCurrentModel,
  } = useChatService();
  const knowledgeGraphService = useKnowledgeGraphService();

  useEffect(() => {
    let isMounted = true;
    setIsSyncingModels(true);
    loadAvailableModels()
      .catch((error) => console.error('Load models error:', error))
      .finally(() => {
        if (isMounted) {
          setIsSyncingModels(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [loadAvailableModels]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatState.messages]);

  const handleSend = async () => {
    if (!input.trim() || chatState.isLoading) return;
    try {
      const knowledgePayload = knowledgeGraphService.getKnowledgeGraphContext();
      await sendAgentMessage(input, {
        agentMode,
        knowledgeGraph: knowledgePayload,
      });
      setInput('');
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleVoiceInput = () => {
    if (isRecording) {
      setIsRecording(false);
    } else {
      setIsRecording(true);
      setTimeout(() => {
        setIsRecording(false);
        setInput((value) => `${value}${value ? '\n' : ''}（语音转写示例）`);
      }, 2500);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      console.info('Selected files for upload:', Array.from(files).map((file) => file.name));
    }
  };

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content).catch((error) => console.error('Copy failed:', error));
  };

  const modelSummary = useMemo(() => {
    if (!currentModelOption) {
      return chatState.currentModel;
    }
    return `${currentModelOption.name} · ${currentModelOption.provider.toUpperCase()}`;
  }, [chatState.currentModel, currentModelOption]);

  const agentModeLabel = agentMode ? 'Agent编排模式' : '纯模型对话';

  return (
    <div className={clsx('h-full', className)}>
      <div className="grid h-full gap-4 xl:grid-cols-[minmax(0,2.15fr)_minmax(320px,1fr)]">
        <div className="flex flex-col h-full bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
          <header className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-primary-100 flex items-center justify-center">
                <Bot className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-gray-900">AI 智能投研助手</h2>
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  <span>{modelSummary}</span>
                  <span>•</span>
                  <button
                    type="button"
                    onClick={() => setAgentMode((prev) => !prev)}
                    className={clsx(
                      'inline-flex items-center px-2 py-0.5 rounded-full border transition-colors',
                      agentMode
                        ? 'bg-primary-100 border-primary-200 text-primary-700'
                        : 'bg-gray-100 border-gray-200 text-gray-600'
                    )}
                  >
                    <Workflow className="w-3 h-3 mr-1" />
                    {agentModeLabel}
                  </button>
                  <span>•</span>
                  <button
                    type="button"
                    onClick={() =>
                      knowledgeGraphService.toggleKnowledgeGraph(!knowledgeGraphService.knowledgeGraph.enabled)
                    }
                    className={clsx(
                      'inline-flex items-center px-2 py-0.5 rounded-full border transition-colors',
                      knowledgeGraphService.knowledgeGraph.enabled
                        ? 'bg-emerald-100 border-emerald-200 text-emerald-700'
                        : 'bg-gray-100 border-gray-200 text-gray-600'
                    )}
                  >
                    <Sparkles className="w-3 h-3 mr-1" />
                    {knowledgeGraphService.knowledgeGraph.enabled ? '知识图谱已联动' : '知识图谱关闭'}
                  </button>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => setShowSettings((prev) => !prev)}
                className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white"
                title="聊天设置"
              >
                <Settings className="w-5 h-5" />
              </button>
              <button
                type="button"
                onClick={clearMessages}
                className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white"
                title="清空对话"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          </header>

          {showSettings && (
            <div className="px-4 py-4 border-b border-gray-200 bg-white space-y-4">
              <ChatModelSelector
                models={chatState.availableModels}
                currentModel={chatState.currentModel}
                onChange={setCurrentModel}
                onTestConnection={testModelConnection}
                tests={modelTests}
                isLoading={isSyncingModels}
              />

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-700">Agent编排模式</p>
                  <p className="text-xs text-gray-500">
                    开启后支持任务拆解、工具调用与流程自动执行
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setAgentMode((prev) => !prev)}
                  className={clsx(
                    'inline-flex items-center px-3 py-1 rounded-full border text-sm transition-colors',
                    agentMode
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-gray-100 text-gray-600 border-gray-200'
                  )}
                >
                  <Workflow className="w-4 h-4 mr-1" />
                  {agentMode ? '已开启' : '已关闭'}
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-700">知识图谱辅助</p>
                  <p className="text-xs text-gray-500">同步实体与关系上下文，增强 QWEN 推理能力</p>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    knowledgeGraphService.toggleKnowledgeGraph(!knowledgeGraphService.knowledgeGraph.enabled)
                  }
                  className={clsx(
                    'inline-flex items-center px-3 py-1 rounded-full border text-sm transition-colors',
                    knowledgeGraphService.knowledgeGraph.enabled
                      ? 'bg-emerald-600 text-white border-emerald-600'
                      : 'bg-gray-100 text-gray-600 border-gray-200'
                  )}
                >
                  <Sparkles className="w-4 h-4 mr-1" />
                  {knowledgeGraphService.knowledgeGraph.enabled ? '已开启' : '已关闭'}
                </button>
              </div>

              <div className="xl:hidden">
                <KnowledgeGraphPanel />
              </div>
            </div>
          )}

          <main className="flex-1 overflow-y-auto bg-gray-50 px-4 py-6 space-y-4">
            {chatState.messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                <div className="w-16 h-16 rounded-full bg-primary-100 flex items-center justify-center">
                  <Bot className="w-7 h-7 text-primary-600" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">欢迎体验 QWEN 驱动的智能投研助手</h3>
                  <p className="text-sm text-gray-600 max-w-md">
                    连接通义千问与知识图谱，自动生成行情洞察、策略建议与风险提示。
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl text-left">
                  {[
                    '分析当前市场趋势并列出关键风险',
                    '结合知识图谱推理新能源车产业链的投资机会',
                    '请给出两条适合稳健型投资者的资产配置策略',
                    '总结美股科技龙头最新财报亮点和潜在风险',
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => setInput(suggestion)}
                      className="p-3 bg-white border border-gray-200 rounded-lg hover:border-primary-200 hover:bg-primary-50/50 transition-colors"
                    >
                      <div className="flex items-start space-x-2">
                        <ShieldCheck className="w-4 h-4 text-primary-500 mt-0.5" />
                        <span className="text-sm text-gray-700">{suggestion}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              chatState.messages.map((message) => (
                <MessageBubble key={message.id} message={message} onCopy={handleCopy} />
              ))
            )}

            {chatState.isLoading && (
              <div className="flex items-center space-x-3 text-sm text-gray-600">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>QWEN 正在思考并协调工具...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </main>

          <footer className="border-t border-gray-200 bg-white px-4 py-3">
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              multiple
              onChange={handleFileUpload}
            />
            <div className="flex items-end space-x-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                title="上传附件"
              >
                <Paperclip className="w-5 h-5" />
              </button>

              <button
                type="button"
                onClick={handleVoiceInput}
                className={clsx(
                  'p-2 rounded-lg transition-colors',
                  isRecording
                    ? 'bg-red-100 text-red-600 hover:bg-red-200'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                )}
                title={isRecording ? '停止录音' : '语音输入'}
              >
                {isRecording ? <Square className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </button>

              <div className="flex-1 relative">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="请输入问题，Enter 发送 / Shift+Enter 换行"
                  className="w-full resize-none rounded-lg border border-gray-300 px-4 py-3 pr-12 text-sm leading-relaxed focus:border-primary-400 focus:ring-2 focus:ring-primary-100"
                  rows={2}
                />
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={!input.trim() || chatState.isLoading}
                  className={clsx(
                    'absolute right-3 bottom-3 inline-flex items-center justify-center rounded-full p-2 text-white shadow-sm transition-colors',
                    !input.trim() || chatState.isLoading
                      ? 'bg-gray-300 cursor-not-allowed'
                      : 'bg-primary-600 hover:bg-primary-700'
                  )}
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
              <span>Enter 发送 · Shift+Enter 换行 · 支持文件和语音辅助</span>
              <span>
                {agentMode ? 'Agent 模式' : '纯对话'} · {currentModelOption?.name ?? chatState.currentModel}
              </span>
            </div>
          </footer>
        </div>

        <div className="hidden xl:flex">
          <KnowledgeGraphPanel className="w-full" />
        </div>
      </div>
    </div>
  );
};

export default EnhancedChatInterface;
