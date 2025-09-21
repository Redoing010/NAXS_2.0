// @ts-nocheck
// 增强的LLM聊天界面
// 支持Agent工具调用可视化、状态监控和高级交互功能

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { clsx } from 'clsx';
import {
  Send,
  Mic,
  MicOff,
  Paperclip,
  MoreVertical,
  Copy,
  RefreshCw,
  Trash2,
  Download,
  Settings,
  Bot,
  User,
  Zap,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Eye,
  EyeOff,
  Play,
  Pause,
  Square,
  ChevronDown,
  ChevronRight,
  Code,
  BarChart3,
  TrendingUp,
  Database,
  Search,
  FileText,
  Brain,
  Target,
  Layers,
  Activity,
} from 'lucide-react';
import { useChat, useChatActions } from '../../store';
import apiService from '../../services/api';

// 消息类型
interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    model?: string;
    tokens?: number;
    cost?: number;
    tools_used?: ToolCall[];
    agent_status?: AgentStatus;
    execution_time?: number;
    confidence?: number;
  };
  status?: 'sending' | 'sent' | 'error' | 'streaming';
  error?: string;
}

// 工具调用
interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
  result?: any;
  status: 'pending' | 'running' | 'completed' | 'failed';
  execution_time?: number;
  error?: string;
}

// Agent状态
interface AgentStatus {
  current_task?: string;
  progress?: number;
  stage?: string;
  tools_available?: string[];
  active_tools?: string[];
  reasoning?: string;
  next_actions?: string[];
}

// 模型选项
const MODELS = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'openai', context: '8K' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'openai', context: '128K' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai', context: '16K' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'anthropic', context: '200K' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'anthropic', context: '200K' },
  { id: 'gemini-pro', name: 'Gemini Pro', provider: 'google', context: '32K' },
];

// 工具图标映射
const TOOL_ICONS: Record<string, React.ComponentType<any>> = {
  market_data: TrendingUp,
  stock_analysis: BarChart3,
  strategy_backtest: Activity,
  portfolio_optimization: Target,
  knowledge_graph: Layers,
  factor_analysis: Database,
  report_generation: FileText,
  web_search: Search,
  code_execution: Code,
  default: Zap,
};

interface EnhancedChatInterfaceProps {
  className?: string;
}

export const EnhancedChatInterface: React.FC<EnhancedChatInterfaceProps> = ({ className }) => {
  const { messages, currentModel, isLoading } = useChat();
  const { sendMessage, setModel, clearMessages } = useChatActions();
  
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showToolDetails, setShowToolDetails] = useState<Record<string, boolean>>({});
  const [agentMode, setAgentMode] = useState(true);
  const [streamingMessage, setStreamingMessage] = useState<string>('');
  const [currentAgentStatus, setCurrentAgentStatus] = useState<AgentStatus | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 自动滚动到底部
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage, scrollToBottom]);

  // 发送消息
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    
    try {
      if (agentMode) {
        // Agent模式：支持工具调用
        await sendAgentMessage(userMessage);
      } else {
        // 普通模式：直接LLM对话
        await sendMessage(userMessage);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  // 发送Agent消息
  const sendAgentMessage = async (content: string) => {
    try {
      const response = await apiService.sendAgentMessage({
        message: content,
        model: currentModel,
        enable_tools: true,
        context: messages.slice(-10).map(msg => ({
          role: msg.type === 'user' ? 'user' : 'assistant',
          content: msg.content,
        })),
      });

      // 处理流式响应
      if (response.stream) {
        handleStreamingResponse(response.stream);
      } else {
        // 处理完整响应
        handleCompleteResponse(response);
      }
    } catch (error) {
      console.error('Agent message error:', error);
    }
  };

  // 处理流式响应
  const handleStreamingResponse = async (stream: ReadableStream) => {
    const reader = stream.getReader();
    let accumulatedContent = '';
    let currentTools: ToolCall[] = [];
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'content') {
                accumulatedContent += data.content;
                setStreamingMessage(accumulatedContent);
              } else if (data.type === 'tool_call') {
                currentTools.push(data.tool);
                setCurrentAgentStatus(prev => ({
                  ...prev,
                  active_tools: currentTools.map(t => t.name),
                  current_task: data.tool.name,
                }));
              } else if (data.type === 'agent_status') {
                setCurrentAgentStatus(data.status);
              }
            } catch (e) {
              console.warn('Failed to parse streaming data:', e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
      setStreamingMessage('');
      setCurrentAgentStatus(null);
    }
  };

  // 处理完整响应
  const handleCompleteResponse = (response: any) => {
    // 添加助手消息到聊天记录
    const assistantMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: response.content,
      timestamp: new Date(),
      metadata: {
        model: response.model,
        tokens: response.usage?.total_tokens,
        cost: response.cost,
        tools_used: response.tools_used,
        execution_time: response.execution_time,
        confidence: response.confidence,
      },
      status: 'sent',
    };
    
    // 这里应该调用store的action来添加消息
    console.log('Complete response:', assistantMessage);
  };

  // 键盘事件处理
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 复制消息
  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    // 显示复制成功提示
  };

  // 重新生成回复
  const handleRegenerateResponse = (messageId: string) => {
    // 找到用户消息并重新发送
    const messageIndex = messages.findIndex(m => m.id === messageId);
    if (messageIndex > 0) {
      const userMessage = messages[messageIndex - 1];
      if (userMessage.type === 'user') {
        handleSendMessage();
      }
    }
  };

  // 语音输入
  const handleVoiceInput = () => {
    if (isRecording) {
      setIsRecording(false);
      // 停止录音并处理语音识别
    } else {
      setIsRecording(true);
      // 开始录音
    }
  };

  // 文件上传
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      // 处理文件上传
      console.log('Files uploaded:', files);
    }
  };

  // 渲染工具调用
  const renderToolCall = (tool: ToolCall, messageId: string) => {
    const Icon = TOOL_ICONS[tool.name] || TOOL_ICONS.default;
    const isExpanded = showToolDetails[`${messageId}-${tool.id}`];
    
    return (
      <div key={tool.id} className="border border-gray-200 rounded-lg p-3 mb-2">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowToolDetails({
            ...showToolDetails,
            [`${messageId}-${tool.id}`]: !isExpanded
          })}
        >
          <div className="flex items-center space-x-2">
            <Icon className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-gray-900">{tool.name}</span>
            <span className={clsx(
              'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
              tool.status === 'completed' ? 'bg-green-100 text-green-800' :
              tool.status === 'running' ? 'bg-blue-100 text-blue-800' :
              tool.status === 'failed' ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
            )}>
              {tool.status === 'completed' && <CheckCircle className="w-3 h-3 mr-1" />}
              {tool.status === 'running' && <RefreshCw className="w-3 h-3 mr-1 animate-spin" />}
              {tool.status === 'failed' && <XCircle className="w-3 h-3 mr-1" />}
              {tool.status === 'pending' && <Clock className="w-3 h-3 mr-1" />}
              {tool.status}
            </span>
            {tool.execution_time && (
              <span className="text-xs text-gray-500">
                {tool.execution_time}ms
              </span>
            )}
          </div>
          
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
        
        {isExpanded && (
          <div className="mt-3 space-y-2">
            {Object.keys(tool.arguments).length > 0 && (
              <div>
                <p className="text-xs font-medium text-gray-700 mb-1">参数:</p>
                <pre className="text-xs bg-gray-50 p-2 rounded overflow-x-auto">
                  {JSON.stringify(tool.arguments, null, 2)}
                </pre>
              </div>
            )}
            
            {tool.result && (
              <div>
                <p className="text-xs font-medium text-gray-700 mb-1">结果:</p>
                <pre className="text-xs bg-gray-50 p-2 rounded overflow-x-auto max-h-32">
                  {typeof tool.result === 'string' ? tool.result : JSON.stringify(tool.result, null, 2)}
                </pre>
              </div>
            )}
            
            {tool.error && (
              <div>
                <p className="text-xs font-medium text-red-700 mb-1">错误:</p>
                <p className="text-xs text-red-600 bg-red-50 p-2 rounded">{tool.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // 渲染Agent状态
  const renderAgentStatus = (status: AgentStatus) => (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
      <div className="flex items-center space-x-2 mb-2">
        <Bot className="w-4 h-4 text-blue-600" />
        <span className="text-sm font-medium text-blue-900">Agent状态</span>
      </div>
      
      {status.current_task && (
        <p className="text-sm text-blue-800 mb-1">
          当前任务: {status.current_task}
        </p>
      )}
      
      {status.progress !== undefined && (
        <div className="mb-2">
          <div className="flex justify-between text-xs text-blue-700 mb-1">
            <span>进度</span>
            <span>{Math.round(status.progress * 100)}%</span>
          </div>
          <div className="w-full bg-blue-200 rounded-full h-1.5">
            <div 
              className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${status.progress * 100}%` }}
            />
          </div>
        </div>
      )}
      
      {status.reasoning && (
        <p className="text-xs text-blue-700 mb-2">
          推理: {status.reasoning}
        </p>
      )}
      
      {status.active_tools && status.active_tools.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {status.active_tools.map((tool, index) => (
            <span key={index} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-blue-100 text-blue-800">
              <Zap className="w-3 h-3 mr-1" />
              {tool}
            </span>
          ))}
        </div>
      )}
    </div>
  );

  // 渲染消息
  const renderMessage = (message: Message) => {
    const isUser = message.type === 'user';
    
    return (
      <div key={message.id} className={clsx(
        'flex mb-4',
        isUser ? 'justify-end' : 'justify-start'
      )}>
        <div className={clsx(
          'max-w-3xl rounded-lg px-4 py-3',
          isUser 
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 text-gray-900'
        )}>
          {/* 消息头部 */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              {isUser ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
              <span className="text-sm font-medium">
                {isUser ? '用户' : (message.metadata?.model || 'Assistant')}
              </span>
              <span className="text-xs opacity-70">
                {message.timestamp.toLocaleTimeString()}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <button
                onClick={() => handleCopyMessage(message.content)}
                className={clsx(
                  'p-1 rounded hover:bg-opacity-20',
                  isUser ? 'hover:bg-white' : 'hover:bg-gray-300'
                )}
              >
                <Copy className="w-3 h-3" />
              </button>
              
              {!isUser && (
                <button
                  onClick={() => handleRegenerateResponse(message.id)}
                  className="p-1 rounded hover:bg-gray-300 hover:bg-opacity-20"
                >
                  <RefreshCw className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
          
          {/* 消息内容 */}
          <div className="prose prose-sm max-w-none">
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
          
          {/* 工具调用 */}
          {message.metadata?.tools_used && message.metadata.tools_used.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium mb-2 opacity-70">工具调用:</p>
              {message.metadata.tools_used.map(tool => 
                renderToolCall(tool, message.id)
              )}
            </div>
          )}
          
          {/* 消息元数据 */}
          {message.metadata && (
            <div className="mt-2 pt-2 border-t border-opacity-20 border-gray-300">
              <div className="flex items-center space-x-4 text-xs opacity-70">
                {message.metadata.tokens && (
                  <span>令牌: {message.metadata.tokens}</span>
                )}
                {message.metadata.cost && (
                  <span>成本: ${message.metadata.cost.toFixed(4)}</span>
                )}
                {message.metadata.execution_time && (
                  <span>耗时: {message.metadata.execution_time}ms</span>
                )}
                {message.metadata.confidence && (
                  <span>置信度: {Math.round(message.metadata.confidence * 100)}%</span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={clsx('flex flex-col h-full', className)}>
      {/* 聊天头部 */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Bot className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-900">
              {agentMode ? 'Agent助手' : 'LLM对话'}
            </h2>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setAgentMode(!agentMode)}
              className={clsx(
                'px-3 py-1 rounded-full text-sm font-medium transition-colors',
                agentMode
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-700'
              )}
            >
              {agentMode ? 'Agent模式' : '对话模式'}
            </button>
            
            <select
              value={currentModel}
              onChange={(e) => setModel(e.target.value)}
              className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {MODELS.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.context})
                </option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
          >
            <Settings className="w-4 h-4" />
          </button>
          
          <button
            onClick={clearMessages}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 聊天消息区域 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {agentMode ? '智能Agent助手' : 'LLM对话助手'}
            </h3>
            <p className="text-gray-500 mb-4">
              {agentMode 
                ? '我可以帮您分析市场、制定策略、执行量化研究任务'
                : '开始与AI助手对话，获取投资建议和分析'
              }
            </p>
            
            {agentMode && (
              <div className="grid grid-cols-2 gap-3 max-w-md mx-auto">
                {[
                  { icon: TrendingUp, text: '市场分析' },
                  { icon: BarChart3, text: '股票研究' },
                  { icon: Target, text: '策略优化' },
                  { icon: FileText, text: '报告生成' },
                ].map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={index}
                      onClick={() => setInput(item.text)}
                      className="flex items-center space-x-2 p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <Icon className="w-4 h-4 text-blue-600" />
                      <span className="text-sm text-gray-700">{item.text}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map(renderMessage)}
            
            {/* 当前Agent状态 */}
            {currentAgentStatus && renderAgentStatus(currentAgentStatus)}
            
            {/* 流式消息 */}
            {streamingMessage && (
              <div className="flex justify-start mb-4">
                <div className="max-w-3xl bg-gray-100 text-gray-900 rounded-lg px-4 py-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <Bot className="w-4 h-4" />
                    <span className="text-sm font-medium">Assistant</span>
                    <RefreshCw className="w-3 h-3 animate-spin" />
                  </div>
                  <div className="prose prose-sm max-w-none">
                    <p className="whitespace-pre-wrap">{streamingMessage}</p>
                  </div>
                </div>
              </div>
            )}
            
            {/* 加载指示器 */}
            {isLoading && !streamingMessage && (
              <div className="flex justify-start mb-4">
                <div className="bg-gray-100 rounded-lg px-4 py-3">
                  <div className="flex items-center space-x-2">
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span className="text-sm text-gray-600">正在思考...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={agentMode ? "描述您的投研需求，我将为您制定执行方案..." : "输入消息..."}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={input.split('\n').length}
              disabled={isLoading}
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              accept=".txt,.pdf,.doc,.docx,.csv,.xlsx"
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
              disabled={isLoading}
            >
              <Paperclip className="w-5 h-5" />
            </button>
            
            <button
              onClick={handleVoiceInput}
              className={clsx(
                'p-2 rounded-md transition-colors',
                isRecording
                  ? 'text-red-600 bg-red-100 hover:bg-red-200'
                  : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
              )}
              disabled={isLoading}
            >
              {isRecording ? (
                <MicOff className="w-5 h-5" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>
            
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading}
              className="p-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        {/* 输入提示 */}
        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <span>按 Enter 发送，Shift + Enter 换行</span>
          <span>{input.length}/4000</span>
        </div>
      </div>
    </div>
  );
};

export default EnhancedChatInterface;