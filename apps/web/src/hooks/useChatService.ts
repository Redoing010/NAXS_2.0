import { useCallback, useMemo, useState } from 'react';
import apiService from '../services/api';
import {
  useChat,
  useChatActions,
  type ChatMessage,
  type LLMModelOption,
  type KnowledgeGraphContextPayload,
} from '../store';

interface SendAgentMessageOptions {
  agentMode?: boolean;
  knowledgeGraph?: KnowledgeGraphContextPayload;
  stream?: boolean;
}

interface ModelTestResult {
  provider: string;
  success: boolean;
  timestamp: string;
  message?: string;
}

const FALLBACK_MODELS: LLMModelOption[] = [
  {
    id: 'qwen:qwen2.5-14b',
    name: 'Qwen2.5 14B',
    provider: 'qwen',
    model: 'qwen2.5-14b-instruct',
    context: '128K',
    description: '通义千问2.5 14B 指令微调模型',
    capabilities: ['analysis', 'reasoning', 'chinese'],
    streaming: true,
    status: 'online',
  },
  {
    id: 'qwen:qwen2.5-32b',
    name: 'Qwen2.5 32B',
    provider: 'qwen',
    model: 'qwen2.5-32b-instruct',
    context: '256K',
    description: '通义千问2.5 32B 长上下文版本',
    capabilities: ['long-context', 'analysis'],
    streaming: true,
    status: 'online',
  },
];

const normalizeModel = (item: unknown, index: number): LLMModelOption => {
  if (typeof item === 'string') {
    const [provider, model] = item.includes(':') ? item.split(':') : ['custom', item];
    return {
      id: `${provider}:${model}`,
      name: model,
      provider,
      model,
      context: '未知',
      description: '后端返回的模型',
      status: 'online',
    };
  }

  if (!item || typeof item !== 'object') {
    return {
      id: `fallback:${index}`,
      name: `模型${index + 1}`,
      provider: 'custom',
      model: `model_${index}`,
      context: '未知',
      description: '未识别的模型定义',
      status: 'online',
    };
  }

  const record = item as Record<string, unknown>;
  const provider = (record.provider as string | undefined) ?? (record.vendor as string | undefined) ?? 'custom';
  const model = (record.model as string | undefined) ?? (record.name as string | undefined) ?? `model_${index}`;

  return {
    id: (record.id as string | undefined) ?? `${provider}:${model}`,
    name: (record.name as string | undefined) ?? (model as string),
    provider,
    model: model as string,
    context: (record.context as string | undefined) ?? (record.max_context as string | undefined) ?? '未知',
    description: (record.description as string | undefined) ?? (record.summary as string | undefined),
    capabilities: (record.capabilities as string[] | undefined) ?? (record.features as string[] | undefined),
    streaming: Boolean(record.streaming ?? record.stream),
    status: (record.status as LLMModelOption['status']) ?? 'online',
  };
};

const normalizeModels = (data: unknown): LLMModelOption[] => {
  if (!data) return FALLBACK_MODELS;
  if (Array.isArray(data)) {
    const normalized = data.map((item, index) => normalizeModel(item, index));
    return normalized.length > 0 ? normalized : FALLBACK_MODELS;
  }
  if (typeof data === 'object') {
    const values = Object.values(data as Record<string, unknown>);
    if (values.length > 0) {
      return values.map((value, index) => normalizeModel(value, index));
    }
  }
  return FALLBACK_MODELS;
};

export const useChatService = () => {
  const chatState = useChat();
  const { addMessage, clearMessages, setLoading, setCurrentModel, setAvailableModels } = useChatActions();
  const [modelTests, setModelTests] = useState<Record<string, ModelTestResult>>({});

  const currentModelOption = useMemo(
    () =>
      chatState.availableModels.find(
        (model) => model.model === chatState.currentModel || model.id === chatState.currentModel
      ),
    [chatState.availableModels, chatState.currentModel]
  );

  const loadAvailableModels = useCallback(async () => {
    try {
      const response = await apiService.getAvailableModels();
      if (response.status === 'ok') {
        const models = normalizeModels(response.data);
        setAvailableModels(models);
        if (!models.find((model) => model.model === chatState.currentModel)) {
          setCurrentModel(models[0]?.model ?? chatState.currentModel);
        }
        return models;
      }

      const fallback = normalizeModels(undefined);
      setAvailableModels(fallback);
      return fallback;
    } catch (error: unknown) {
      console.error('Failed to load models, using fallback', error);
      const fallback = normalizeModels(undefined);
      setAvailableModels(fallback);
      if (!fallback.find((model) => model.model === chatState.currentModel)) {
        setCurrentModel(fallback[0]?.model ?? chatState.currentModel);
      }
      return fallback;
    }
  }, [chatState.currentModel, setAvailableModels, setCurrentModel]);

  const testModelConnection = useCallback(
    async (provider: string) => {
      const timestamp = new Date().toISOString();
      try {
        const response = await apiService.testModelConnection(provider);
        const success = response.status === 'ok';
        const result: ModelTestResult = {
          provider,
          success,
          timestamp,
          message: response.message || response.data?.message,
        };
        setModelTests((prev) => ({ ...prev, [provider]: result }));
        return result;
      } catch (error: unknown) {
        const result: ModelTestResult = {
          provider,
          success: false,
          timestamp,
          message: error instanceof Error ? error.message : '连接测试失败',
        };
        setModelTests((prev) => ({ ...prev, [provider]: result }));
        return result;
      }
    },
    []
  );

  const sendAgentMessage = useCallback(
    async (content: string, options?: SendAgentMessageOptions) => {
      const trimmedContent = content.trim();
      if (!trimmedContent) return;

      const userMessage: Omit<ChatMessage, 'id' | 'timestamp'> = {
        type: 'user',
        content: trimmedContent,
      };
      addMessage(userMessage);
      setLoading(true);

      try {
        const context = (chatState.messages || []).slice(-10).map((message) => ({
          role: message.type === 'user' ? 'user' : 'assistant',
          content: message.content,
        }));

        const response = await apiService.sendAgentMessage({
          message: trimmedContent,
          model: chatState.currentModel,
          provider: chatState.currentProvider,
          context,
          tools_enabled: options?.agentMode ?? true,
          knowledge_graph: options?.knowledgeGraph,
          stream: options?.stream ?? false,
        });

        if (response.status === 'ok' && response.data) {
          const assistantMessage: Omit<ChatMessage, 'id' | 'timestamp'> = {
            type: 'assistant',
            content: response.data.message ?? response.data.content ?? '',
            metadata: {
              model: response.data.model ?? chatState.currentModel,
              provider: response.data.provider ?? chatState.currentProvider,
              tokens: response.data.tokens_used ?? response.data.usage?.total_tokens,
              execution_time: response.data.execution_time ?? response.data.latency,
              tools_used: Array.isArray(response.data.tools_used)
                ? response.data.tools_used
                : undefined,
              cost: response.data.cost,
              confidence: response.data.confidence,
            },
          };
          addMessage(assistantMessage);
        } else {
          const errorMessage: Omit<ChatMessage, 'id' | 'timestamp'> = {
            type: 'system',
            content: `Agent返回错误：${response.error || response.message || '未知错误'}`,
          };
          addMessage(errorMessage);
        }
      } catch (error: unknown) {
        console.error('Failed to send agent message:', error);
        const errorMessage: Omit<ChatMessage, 'id' | 'timestamp'> = {
          type: 'system',
          content: `发送失败：${error instanceof Error ? error.message : '请检查网络或Agent服务状态'}`,
        };
        addMessage(errorMessage);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [addMessage, chatState.currentModel, chatState.currentProvider, chatState.messages, setLoading]
  );

  return {
    chatState,
    currentModelOption,
    modelTests,
    loadAvailableModels,
    testModelConnection,
    sendAgentMessage,
    clearMessages,
    setCurrentModel,
  };
};

export type { SendAgentMessageOptions, ModelTestResult };
