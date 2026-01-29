"""
LLM Client with token counting capabilities.
Supports OpenAI and Anthropic APIs.
"""

import os
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod
import tiktoken
from dotenv import load_dotenv

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()


class LLMClient(ABC):
    """Base class for LLM clients with token counting."""

    @abstractmethod
    def send_prompt(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Send prompt to LLM and return response with token counts.

        Returns:
            Tuple of (response_text, token_stats)
            token_stats contains: input_tokens, output_tokens, total_tokens
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client with token counting."""

    def __init__(self, model: str = None, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model = model or os.getenv("OPENAI_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.model:
            raise ValueError("OpenAI model not specified. Set OPENAI_MODEL environment variable.")

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.encoding_for_model(self.model)

    def send_prompt(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """Send prompt to OpenAI and return response with token counts."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )

        token_stats = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return response.choices[0].message.content, token_stats

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))


class AnthropicClient(LLMClient):
    """Anthropic API client with token counting."""

    def __init__(self, model: str = None, api_key: str = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.model = model or os.getenv("ANTHROPIC_MODEL")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.model:
            raise ValueError("Anthropic model not specified. Set ANTHROPIC_MODEL environment variable.")

        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")

        self.client = Anthropic(api_key=self.api_key)

    def send_prompt(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, Dict[str, int]]:
        """Send prompt to Anthropic and return response with token counts."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )

        token_stats = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

        return response.content[0].text, token_stats

    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's API."""
        response = self.client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens


def create_client(provider: str = "openai", **kwargs) -> LLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: 'openai' or 'anthropic'
        **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
