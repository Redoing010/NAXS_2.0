// @ts-nocheck
// LLM聊天界面组件

import React, { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import {
  Send,
  Bot,
  User,
  Loader2,
  Settings,
  Trash2,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Paperclip,
  Mic,
  Square,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useChat, useChatActions } from '../../store';
import apiService from '../../services/api';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

interface ChatInterfaceProps {
  className?: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ className }) => {
  const { messages, isLoading, currentModel, availableModels } = useChat();
  const { addMessage, updateMessage, clearMessages, setLoading, setCurrentModel } = useChatActions();
  
  const [inputValue, setInputValue] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 发送消息
  const handleSendMessage = async () => {
    const content = inputValue.trim();
    if (!content || isLoading) return;

    // 添加用户消息
    const userMessage = {
      type: 'user' as const,
      content,
      timestamp: new Date(),
    };
    addMessage(userMessage);
    setInputValue('');
    setLoading(true);

    try {
      // 调用Agent API发送消息
      const response = await apiService.sendAgentMessage({
        message: content,
        model: currentModel,
        context: (messages || []).slice(-10).map(m => ({
          role: m.type === 'user' ? 'user' : 'assistant',
          content: m.content
        })),
        tools_enabled: true,
        stream: false,
      });

      if (response.status === 'ok' && response.data) {
        // 添加助手回复
        const assistantMessage = {
          type: 'assistant' as const,
          content: response.data.message,
          timestamp: new Date(),
          metadata: {
            model: response.data.model,
            tokens: response.data.tokens_used,
            execution_time: response.data.execution_time,
            tools_used: response.data.tools_used || [],
            agent_actions: response.data.agent_actions || [],
            reasoning: response.data.reasoning,
            confidence: response.data.confidence,
          },
        };
        addMessage(assistantMessage);

        // 如果有工具调用结果，显示相关图表或数据
        if (response.data.tools_used && response.data.tools_used.length > 0) {
          handleToolResults(response.data.tools_used, response.data.tool_results);
        }
      } else {
        // 错误处理
        const errorMessage = {
          type: 'system' as const,
          content: `错误: ${response.error || '未知错误'}`,
          timestamp: new Date(),
        };
        addMessage(errorMessage);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage = {
        type: 'system' as const,
        content: '发送消息失败，请检查网络连接或Agent服务状态',
        timestamp: new Date(),
      };
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 处理工具调用结果
  const handleToolResults = (toolsUsed: string[], toolResults: any[]) => {
    // 根据工具类型显示不同的结果
    toolsUsed.forEach((tool, index) => {
      const result = toolResults[index];
      
      if (tool === 'market_data' && result) {
        // 显示市场数据图表
        const chartMessage = {
          type: 'chart' as const,
          content: '市场数据分析',
          timestamp: new Date(),
          chartData: result,
          chartType: 'line',
        };
        addMessage(chartMessage);
      } else if (tool === 'stock_analysis' && result) {
        // 显示股票分析结果
        const analysisMessage = {
          type: 'analysis' as const,
          content: '股票分析报告',
          timestamp: new Date(),
          analysisData: result,
        };
        addMessage(analysisMessage);
      } else if (tool === 'strategy_backtest' && result) {
        // 显示回测结果
        const backtestMessage = {
          type: 'backtest' as const,
          content: '策略回测结果',
          timestamp: new Date(),
          backtestData: result,
        };
        addMessage(backtestMessage);
      }
    });
  };

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 复制消息内容
  const handleCopyMessage = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      // 这里可以添加复制成功的提示
    } catch (error) {
      console.error('Failed to copy message:', error);
    }
  };

  // 重新生成回复
  const handleRegenerateResponse = async (messageIndex: number) => {
    if (messageIndex <= 0) return;
    
    const userMessage = messages[messageIndex - 1];
    if (userMessage.type !== 'user') return;

    setLoading(true);
    
    try {
      const response = await apiService.sendChatMessage({
        message: userMessage.content,
        model: currentModel,
        context: (messages || []).slice(0, messageIndex - 1).map(m => m.content),
      });

      if (response.status === 'ok' && response.data) {
        // 更新助手回复
        updateMessage(messages[messageIndex].id, {
          content: response.data.message,
          metadata: {
            model: response.data.model,
            tokens: response.data.tokens_used,
            execution_time: response.data.execution_time,
            tools_used: response.data.tools_used,
          },
        });
      }
    } catch (error) {
      console.error('Failed to regenerate response:', error);
    } finally {
      setLoading(false);
    }
  };

  // 语音输入（模拟）
  const handleVoiceInput = () => {
    if (isRecording) {
      setIsRecording(false);
      // 停止录音逻辑
    } else {
      setIsRecording(true);
      // 开始录音逻辑
      setTimeout(() => {
        setIsRecording(false);
        setInputValue('这是语音输入的模拟文本');
      }, 3000);
    }
  };

  // 文件上传
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      // 处理文件上传逻辑
      console.log('Files selected:', files);
    }
  };

  return (
    <div className={clsx('flex flex-col h-full bg-white', className)}>
      {/* 头部 */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-8 h-8 bg-primary-100 rounded-full">
            <Bot className="w-5 h-5 text-primary-600" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">AI投研助手</h2>
            <p className="text-sm text-gray-500">模型: {currentModel}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            title="设置"
          >
            <Settings className="w-5 h-5" />
          </button>
          <button
            onClick={clearMessages}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            title="清空对话"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* 设置面板 */}
      {showSettings && (
        <div className="p-4 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">模型选择:</label>
            <select
              value={currentModel}
              onChange={(e) => setCurrentModel(e.target.value)}
              className="select text-sm"
            >
              {(availableModels || []).map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-4">
              <Bot className="w-8 h-8 text-primary-600" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">欢迎使用AI投研助手</h3>
            <p className="text-gray-600 mb-6 max-w-md">
              我可以帮助您分析市场数据、制定投资策略、解答投资问题。请输入您的问题开始对话。
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
              <button
                onClick={() => setInputValue('分析一下当前市场趋势')}
                className="p-3 text-left bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="font-medium text-gray-900">市场分析</div>
                <div className="text-sm text-gray-600">分析当前市场趋势和机会</div>
              </button>
              <button
                onClick={() => setInputValue('推荐一些优质股票')}
                className="p-3 text-left bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="font-medium text-gray-900">选股推荐</div>
                <div className="text-sm text-gray-600">获取个性化选股建议</div>
              </button>
              <button
                onClick={() => setInputValue('如何构建投资组合')}
                className="p-3 text-left bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="font-medium text-gray-900">投资策略</div>
                <div className="text-sm text-gray-600">学习投资组合构建方法</div>
              </button>
              <button
                onClick={() => setInputValue('评估投资风险')}
                className="p-3 text-left bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="font-medium text-gray-900">风险评估</div>
                <div className="text-sm text-gray-600">了解投资风险管理</div>
              </button>
            </div>
          </div>
        ) : (
          (messages || []).map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              onCopy={handleCopyMessage}
              onRegenerate={() => handleRegenerateResponse(index)}
              showRegenerate={message.type === 'assistant' && index === messages.length - 1}
            />
          ))
        )}
        
        {/* 加载指示器 */}
        {isLoading && (
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-8 h-8 bg-primary-100 rounded-full">
              <Bot className="w-5 h-5 text-primary-600" />
            </div>
            <div className="flex items-center space-x-2 px-4 py-2 bg-gray-100 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-gray-600" />
              <span className="text-gray-600">AI正在思考...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex items-end space-x-3">
          {/* 附件按钮 */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            title="上传文件"
          >
            <Paperclip className="w-5 h-5" />
          </button>
          
          {/* 语音输入按钮 */}
          <button
            onClick={handleVoiceInput}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              isRecording
                ? 'text-red-600 bg-red-100 hover:bg-red-200'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
            )}
            title={isRecording ? '停止录音' : '语音输入'}
          >
            {isRecording ? (
              <Square className="w-5 h-5" />
            ) : (
              <Mic className="w-5 h-5" />
            )}
          </button>
          
          {/* 输入框 */}
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入您的问题...（Shift+Enter换行）"
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              rows={1}
              style={{
                minHeight: '48px',
                maxHeight: '120px',
                height: 'auto',
              }}
              disabled={isLoading}
            />
            
            {/* 发送按钮 */}
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="absolute right-2 bottom-2 p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        {/* 隐藏的文件输入 */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleFileUpload}
          accept=".txt,.pdf,.doc,.docx,.csv,.xlsx"
        />
      </div>
    </div>
  );
};

// 消息气泡组件
interface MessageBubbleProps {
  message: any;
  onCopy: (content: string) => void;
  onRegenerate: () => void;
  showRegenerate: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onCopy,
  onRegenerate,
  showRegenerate,
}) => {
  const [showActions, setShowActions] = useState(false);

  const isUser = message.type === 'user';
  const isSystem = message.type === 'system';

  return (
    <div
      className={clsx(
        'flex items-start space-x-3',
        isUser && 'flex-row-reverse space-x-reverse'
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* 头像 */}
      <div className={clsx(
        'flex items-center justify-center w-8 h-8 rounded-full flex-shrink-0',
        isUser ? 'bg-primary-600' : isSystem ? 'bg-warning-100' : 'bg-gray-100'
      )}>
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : isSystem ? (
          <Bot className="w-5 h-5 text-warning-600" />
        ) : (
          <Bot className="w-5 h-5 text-gray-600" />
        )}
      </div>

      {/* 消息内容 */}
      <div className={clsx(
        'flex-1 max-w-3xl',
        isUser && 'flex flex-col items-end'
      )}>
        <div className={clsx(
          'px-4 py-3 rounded-lg',
          isUser
            ? 'bg-primary-600 text-white'
            : isSystem
            ? 'bg-warning-100 text-warning-800 border border-warning-200'
            : 'bg-gray-100 text-gray-900'
        )}>
          {isUser || isSystem ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                      className="rounded-md"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className="bg-gray-200 px-1 py-0.5 rounded text-sm" {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* 消息元数据 */}
        <div className={clsx(
          'flex items-center mt-1 text-xs text-gray-500 space-x-2',
          isUser && 'justify-end'
        )}>
          <span>{format(new Date(message.timestamp), 'HH:mm', { locale: zhCN })}</span>
          {message.metadata?.model && (
            <span>• {message.metadata.model}</span>
          )}
          {message.metadata?.tokens && (
            <span>• {message.metadata.tokens} tokens</span>
          )}
          {message.metadata?.execution_time && (
            <span>• {message.metadata.execution_time.toFixed(2)}s</span>
          )}
        </div>

        {/* 操作按钮 */}
        {showActions && (
          <div className={clsx(
            'flex items-center mt-2 space-x-1',
            isUser && 'justify-end'
          )}>
            <button
              onClick={() => onCopy(message.content)}
              className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
              title="复制"
            >
              <Copy className="w-4 h-4" />
            </button>
            
            {!isUser && !isSystem && (
              <>
                <button
                  className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
                  title="点赞"
                >
                  <ThumbsUp className="w-4 h-4" />
                </button>
                <button
                  className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
                  title="点踩"
                >
                  <ThumbsDown className="w-4 h-4" />
                </button>
                {showRegenerate && (
                  <button
                    onClick={onRegenerate}
                    className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
                    title="重新生成"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;