"""
Model Interface

Unified interface for interacting with various LLM providers
including OpenAI, Anthropic, and custom endpoints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standardized model response."""
    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw_response: dict = field(default_factory=dict)


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def generate_full(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> ModelResponse:
        """Generate a full response with metadata."""
        pass


class AnthropicInterface(ModelInterface):
    """Interface for Anthropic Claude models."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com/v1",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            logger.warning("No API key provided for Anthropic")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate a response."""
        response = self.generate_full(prompt, max_tokens, temperature, **kwargs)
        return response.content
    
    def generate_full(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> ModelResponse:
        """Generate a full response with metadata."""
        import time
        
        try:
            import httpx
        except ImportError:
            # Fallback to basic implementation
            return ModelResponse(
                content="[API call would be made here]",
                model=self.model,
            )
        
        start = time.time()
        
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
            
            latency = (time.time() - start) * 1000
            
            return ModelResponse(
                content=data["content"][0]["text"] if data.get("content") else "",
                model=data.get("model", self.model),
                tokens_in=data.get("usage", {}).get("input_tokens", 0),
                tokens_out=data.get("usage", {}).get("output_tokens", 0),
                latency_ms=latency,
                finish_reason=data.get("stop_reason", ""),
                raw_response=data,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ModelResponse(
                content="",
                model=self.model,
                latency_ms=(time.time() - start) * 1000,
            )


class OpenAIInterface(ModelInterface):
    """Interface for OpenAI models."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            logger.warning("No API key provided for OpenAI")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate a response."""
        response = self.generate_full(prompt, max_tokens, temperature, **kwargs)
        return response.content
    
    def generate_full(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> ModelResponse:
        """Generate a full response with metadata."""
        import time
        
        try:
            import httpx
        except ImportError:
            return ModelResponse(
                content="[API call would be made here]",
                model=self.model,
            )
        
        start = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
            
            latency = (time.time() - start) * 1000
            
            choice = data["choices"][0] if data.get("choices") else {}
            
            return ModelResponse(
                content=choice.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                tokens_in=data.get("usage", {}).get("prompt_tokens", 0),
                tokens_out=data.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency,
                finish_reason=choice.get("finish_reason", ""),
                raw_response=data,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ModelResponse(
                content="",
                model=self.model,
                latency_ms=(time.time() - start) * 1000,
            )


class MockInterface(ModelInterface):
    """Mock interface for testing."""
    
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Return mock response."""
        self.call_count += 1
        return self.responses.get(prompt, "Mock response for testing.")
    
    def generate_full(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> ModelResponse:
        """Return mock response with metadata."""
        return ModelResponse(
            content=self.generate(prompt, max_tokens, temperature, **kwargs),
            model="mock",
            tokens_in=len(prompt.split()),
            tokens_out=10,
            latency_ms=50.0,
        )


def create_interface(
    provider: str,
    **kwargs
) -> ModelInterface:
    """
    Factory function to create model interfaces.
    
    Args:
        provider: Provider name ("anthropic", "openai", "mock")
        **kwargs: Provider-specific arguments
        
    Returns:
        Configured ModelInterface instance
    """
    providers = {
        "anthropic": AnthropicInterface,
        "openai": OpenAIInterface,
        "mock": MockInterface,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    
    return providers[provider](**kwargs)
