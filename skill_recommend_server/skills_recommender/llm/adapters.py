from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class LLMAdapter(ABC):
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass


class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[msg.model_dump() for msg in messages],
            temperature=kwargs.get("temperature", self.temperature),
            **kwargs
        )
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding


class AnthropicAdapter(LLMAdapter):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        system_message = ""
        filtered_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                filtered_messages.append(msg)
        
        response = self.client.messages.create(
            model=self.model,
            system=system_message,
            messages=[{"role": msg.role, "content": msg.content} for msg in filtered_messages],
            temperature=kwargs.get("temperature", self.temperature),
            **kwargs
        )
        
        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Anthropic does not provide embedding API")


class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5",
        temperature: float = 0.7,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": kwargs.get("temperature", self.temperature),
                **kwargs,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["message"]["content"],
            model=self.model,
        )

    def embed(self, text: str) -> List[float]:
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        data = response.json()
        
        return data["embedding"]


class ZhipuAdapter(LLMAdapter):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "glm-4",
        temperature: float = 0.7,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[msg.model_dump() for msg in messages],
            temperature=kwargs.get("temperature", self.temperature),
            **kwargs
        )
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text,
        )
        return response.data[0].embedding


class AliCloudAdapter(LLMAdapter):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        model: str = "qwen-turbo",
        temperature: float = 0.7,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[msg.model_dump() for msg in messages],
            temperature=kwargs.get("temperature", self.temperature),
            **kwargs
        )
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-vectora-1",
            input=text,
        )
        return response.data[0].embedding


class LLMFactory:
    _adapters: Dict[str, type] = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "ollama": OllamaAdapter,
        "zhipu": ZhipuAdapter,
        "alicloud": AliCloudAdapter,
    }

    @classmethod
    def create(cls, provider: str, **config) -> LLMAdapter:
        if provider not in cls._adapters:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._adapters.keys())}")
        
        return cls._adapters[provider](**config)

    @classmethod
    def register(cls, name: str, adapter_class: type):
        cls._adapters[name] = adapter_class
