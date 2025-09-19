# LLM适配器模块
# 实现多种LLM提供商的统一接口和对话管理功能

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import aiohttp
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """LLM提供商"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    QWEN = "qwen"
    CUSTOM = "custom"

class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

@dataclass
class ChatMessage:
    """聊天消息"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.name:
            result['name'] = self.name
        if self.function_call:
            result['function_call'] = self.function_call
        if self.tool_calls:
            result['tool_calls'] = self.tool_calls
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """从字典创建消息"""
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            name=data.get('name'),
            function_call=data.get('function_call'),
            tool_calls=data.get('tool_calls'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        )

@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    api_key: str
    base_url: Optional[str] = None
    
    # 生成参数
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # 请求配置
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 功能配置
    supports_functions: bool = True
    supports_streaming: bool = True
    supports_system_message: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'provider': self.provider.value,
            'model': self.model,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'supports_functions': self.supports_functions,
            'supports_streaming': self.supports_streaming,
            'supports_system_message': self.supports_system_message
        }

@dataclass
class LLMRequest:
    """LLM请求"""
    messages: List[ChatMessage]
    functions: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream: bool = False
    
    # 覆盖配置参数
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'messages': [msg.to_dict() for msg in self.messages],
            'stream': self.stream
        }
        
        if self.functions:
            result['functions'] = self.functions
        if self.tools:
            result['tools'] = self.tools
        if self.function_call:
            result['function_call'] = self.function_call
        if self.tool_choice:
            result['tool_choice'] = self.tool_choice
        if self.temperature is not None:
            result['temperature'] = self.temperature
        if self.max_tokens is not None:
            result['max_tokens'] = self.max_tokens
        if self.top_p is not None:
            result['top_p'] = self.top_p
            
        return result

@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    role: MessageRole = MessageRole.ASSISTANT
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # 元数据
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    response_time: float = 0.0
    
    def to_message(self) -> ChatMessage:
        """转换为聊天消息"""
        return ChatMessage(
            role=self.role,
            content=self.content,
            function_call=self.function_call,
            tool_calls=self.tool_calls
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'content': self.content,
            'role': self.role.value,
            'model': self.model,
            'usage': self.usage,
            'finish_reason': self.finish_reason,
            'response_time': self.response_time
        }
        
        if self.function_call:
            result['function_call'] = self.function_call
        if self.tool_calls:
            result['tool_calls'] = self.tool_calls
            
        return result

class LLMAdapter(ABC):
    """LLM适配器基类
    
    定义统一的LLM接口
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.debug(f"LLM适配器初始化: {config.provider.value} - {config.model}")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动适配器"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        logger.debug(f"LLM适配器启动: {self.config.provider.value}")
    
    async def stop(self):
        """停止适配器"""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.debug(f"LLM适配器停止: {self.config.provider.value}")
    
    @abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """聊天接口
        
        Args:
            request: LLM请求
            
        Returns:
            LLM响应
        """
        pass
    
    @abstractmethod
    async def chat_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """流式聊天接口
        
        Args:
            request: LLM请求
            
        Yields:
            响应内容片段
        """
        pass
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """准备消息格式
        
        Args:
            messages: 聊天消息列表
            
        Returns:
            格式化的消息列表
        """
        formatted_messages = []
        
        for msg in messages:
            formatted_msg = {
                'role': msg.role.value,
                'content': msg.content
            }
            
            if msg.name:
                formatted_msg['name'] = msg.name
            if msg.function_call:
                formatted_msg['function_call'] = msg.function_call
            if msg.tool_calls:
                formatted_msg['tool_calls'] = msg.tool_calls
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    def _build_request_data(self, request: LLMRequest) -> Dict[str, Any]:
        """构建请求数据
        
        Args:
            request: LLM请求
            
        Returns:
            请求数据字典
        """
        data = {
            'model': self.config.model,
            'messages': self._prepare_messages(request.messages),
            'temperature': request.temperature or self.config.temperature,
            'max_tokens': request.max_tokens or self.config.max_tokens,
            'top_p': request.top_p or self.config.top_p,
            'frequency_penalty': self.config.frequency_penalty,
            'presence_penalty': self.config.presence_penalty,
            'stream': request.stream
        }
        
        if request.functions and self.config.supports_functions:
            data['functions'] = request.functions
            if request.function_call:
                data['function_call'] = request.function_call
        
        if request.tools and self.config.supports_functions:
            data['tools'] = request.tools
            if request.tool_choice:
                data['tool_choice'] = request.tool_choice
        
        return data

class OpenAIAdapter(LLMAdapter):
    """OpenAI适配器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        }
    
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """OpenAI聊天接口"""
        start_time = datetime.now()
        
        data = self._build_request_data(request)
        url = f"{self.base_url}/chat/completions"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, headers=self.headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        choice = result['choices'][0]
                        message = choice['message']
                        
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        return LLMResponse(
                            content=message.get('content', ''),
                            function_call=message.get('function_call'),
                            tool_calls=message.get('tool_calls'),
                            model=result.get('model'),
                            usage=result.get('usage'),
                            finish_reason=choice.get('finish_reason'),
                            response_time=response_time
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API错误: {response.status} - {error_text}")
                        
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise Exception(f"OpenAI API错误: {response.status} - {error_text}")
            
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(f"OpenAI请求失败，重试中: {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise e
    
    async def chat_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """OpenAI流式聊天接口"""
        data = self._build_request_data(request)
        data['stream'] = True
        
        url = f"{self.base_url}/chat/completions"
        
        async with self.session.post(url, headers=self.headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API错误: {response.status} - {error_text}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data: '):
                    data_str = line[6:]
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choice = data['choices'][0]
                        delta = choice.get('delta', {})
                        
                        if 'content' in delta and delta['content']:
                            yield delta['content']
                    
                    except json.JSONDecodeError:
                        continue

class ClaudeAdapter(LLMAdapter):
    """Claude适配器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.headers = {
            'x-api-key': config.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
    
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Claude聊天接口"""
        start_time = datetime.now()
        
        # Claude使用不同的消息格式
        messages = self._prepare_claude_messages(request.messages)
        
        data = {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': request.max_tokens or self.config.max_tokens,
            'temperature': request.temperature or self.config.temperature,
            'top_p': request.top_p or self.config.top_p
        }
        
        url = f"{self.base_url}/messages"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, headers=self.headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        content = result.get('content', [])
                        text_content = ''
                        
                        for item in content:
                            if item.get('type') == 'text':
                                text_content += item.get('text', '')
                        
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        return LLMResponse(
                            content=text_content,
                            model=result.get('model'),
                            usage=result.get('usage'),
                            finish_reason=result.get('stop_reason'),
                            response_time=response_time
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Claude API错误: {response.status} - {error_text}")
                        
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise Exception(f"Claude API错误: {response.status} - {error_text}")
            
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(f"Claude请求失败，重试中: {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise e
    
    async def chat_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Claude流式聊天接口"""
        messages = self._prepare_claude_messages(request.messages)
        
        data = {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': request.max_tokens or self.config.max_tokens,
            'temperature': request.temperature or self.config.temperature,
            'stream': True
        }
        
        url = f"{self.base_url}/messages"
        
        async with self.session.post(url, headers=self.headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Claude API错误: {response.status} - {error_text}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data: '):
                    data_str = line[6:]
                    
                    try:
                        data = json.loads(data_str)
                        
                        if data.get('type') == 'content_block_delta':
                            delta = data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                yield delta.get('text', '')
                    
                    except json.JSONDecodeError:
                        continue
    
    def _prepare_claude_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """准备Claude消息格式"""
        formatted_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Claude将system消息合并到第一个user消息中
                continue
            
            formatted_msg = {
                'role': 'user' if msg.role == MessageRole.USER else 'assistant',
                'content': msg.content
            }
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages

class GeminiAdapter(LLMAdapter):
    """Gemini适配器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
    
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Gemini聊天接口"""
        start_time = datetime.now()
        
        # Gemini使用不同的API格式
        contents = self._prepare_gemini_contents(request.messages)
        
        data = {
            'contents': contents,
            'generationConfig': {
                'temperature': request.temperature or self.config.temperature,
                'maxOutputTokens': request.max_tokens or self.config.max_tokens,
                'topP': request.top_p or self.config.top_p
            }
        }
        
        url = f"{self.base_url}/models/{self.config.model}:generateContent?key={self.config.api_key}"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        candidates = result.get('candidates', [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get('content', {})
                            parts = content.get('parts', [])
                            
                            text_content = ''
                            for part in parts:
                                text_content += part.get('text', '')
                            
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            return LLMResponse(
                                content=text_content,
                                model=self.config.model,
                                finish_reason=candidate.get('finishReason'),
                                response_time=response_time
                            )
                        else:
                            raise Exception("Gemini API返回空结果")
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API错误: {response.status} - {error_text}")
                        
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise Exception(f"Gemini API错误: {response.status} - {error_text}")
            
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(f"Gemini请求失败，重试中: {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise e
    
    async def chat_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Gemini流式聊天接口"""
        contents = self._prepare_gemini_contents(request.messages)
        
        data = {
            'contents': contents,
            'generationConfig': {
                'temperature': request.temperature or self.config.temperature,
                'maxOutputTokens': request.max_tokens or self.config.max_tokens,
                'topP': request.top_p or self.config.top_p
            }
        }
        
        url = f"{self.base_url}/models/{self.config.model}:streamGenerateContent?key={self.config.api_key}"
        
        async with self.session.post(url, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API错误: {response.status} - {error_text}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line:
                    try:
                        data = json.loads(line)
                        candidates = data.get('candidates', [])
                        
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get('content', {})
                            parts = content.get('parts', [])
                            
                            for part in parts:
                                if 'text' in part:
                                    yield part['text']
                    
                    except json.JSONDecodeError:
                        continue
    
    def _prepare_gemini_contents(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """准备Gemini内容格式"""
        contents = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Gemini将system消息作为第一个user消息
                role = 'user'
            elif msg.role == MessageRole.USER:
                role = 'user'
            else:
                role = 'model'
            
            content = {
                'role': role,
                'parts': [{'text': msg.content}]
            }
            
            contents.append(content)
        
        return contents

# 便捷函数
def create_llm_adapter(config: LLMConfig) -> LLMAdapter:
    """创建LLM适配器
    
    Args:
        config: LLM配置
        
    Returns:
        LLM适配器实例
    """
    if config.provider == LLMProvider.OPENAI:
        return OpenAIAdapter(config)
    elif config.provider == LLMProvider.CLAUDE:
        return ClaudeAdapter(config)
    elif config.provider == LLMProvider.GEMINI:
        return GeminiAdapter(config)
    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")

def create_openai_adapter(model: str = "gpt-3.5-turbo", api_key: str = None, **kwargs) -> OpenAIAdapter:
    """创建OpenAI适配器
    
    Args:
        model: 模型名称
        api_key: API密钥
        **kwargs: 其他配置参数
        
    Returns:
        OpenAI适配器实例
    """
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("需要提供OpenAI API密钥")
    
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return OpenAIAdapter(config)

def create_claude_adapter(model: str = "claude-3-sonnet-20240229", api_key: str = None, **kwargs) -> ClaudeAdapter:
    """创建Claude适配器
    
    Args:
        model: 模型名称
        api_key: API密钥
        **kwargs: 其他配置参数
        
    Returns:
        Claude适配器实例
    """
    api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("需要提供Anthropic API密钥")
    
    config = LLMConfig(
        provider=LLMProvider.CLAUDE,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return ClaudeAdapter(config)

def create_gemini_adapter(model: str = "gemini-pro", api_key: str = None, **kwargs) -> GeminiAdapter:
    """创建Gemini适配器
    
    Args:
        model: 模型名称
        api_key: API密钥
        **kwargs: 其他配置参数
        
    Returns:
        Gemini适配器实例
    """
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("需要提供Google API密钥")
    
    config = LLMConfig(
        provider=LLMProvider.GEMINI,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return GeminiAdapter(config)

def create_chat_message(role: MessageRole, content: str, **kwargs) -> ChatMessage:
    """创建聊天消息
    
    Args:
        role: 消息角色
        content: 消息内容
        **kwargs: 其他参数
        
    Returns:
        聊天消息实例
    """
    return ChatMessage(role=role, content=content, **kwargs)

def create_llm_request(messages: List[ChatMessage], **kwargs) -> LLMRequest:
    """创建LLM请求
    
    Args:
        messages: 聊天消息列表
        **kwargs: 其他参数
        
    Returns:
        LLM请求实例
    """
    return LLMRequest(messages=messages, **kwargs)