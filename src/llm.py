"""
LLM provider module for Digital Presenter.
"""

import openai
from google import genai
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str, model: str, max_tokens: int = 3000) -> str:
        """Generate text from the model."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, model: str, max_tokens: int = 3000) -> str:
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content.strip()


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    def generate(self, prompt: str, model: str, max_tokens: int = 3000) -> str:
        """Generate text using DeepSeek API."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = genai.Client(api_key=api_key)
    
    def generate(self, prompt: str, model: str, max_tokens: int = 3000) -> str:
        """Generate text using Gemini API."""
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text.strip()


def get_llm_provider(model: str, config: Dict[str, Any]) -> LLMProvider:
    """Get the appropriate LLM provider based on the model name."""
    if "deepseek" in model.lower():
        return DeepSeekProvider(config["deepseek_api_key"])
    elif "gemini" in model.lower():
        return GeminiProvider(config["gemini_api_key"])
    else:
        return OpenAIProvider(config["openai_api_key"]) 