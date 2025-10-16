"""
LLM provider module for Digital Presenter.
"""

import backoff
import logging
import threading
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import openai
from google import genai
from openai import OpenAI, AzureOpenAI


# Error handling for API requests
ERRORS = (
    requests.exceptions.RequestException,
    openai.APIError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.NotFoundError,
    openai.PermissionDeniedError,
    openai.AuthenticationError,
    openai.BadRequestError,
    openai.ConflictError,
    openai.UnprocessableEntityError,
)


def backoff_hdlr(details):
    """Handler for backoff retries."""
    logging.warning(
        f"Backing off {details['wait']} seconds after {details['tries']} tries. "
        f"Error: {details['exception']}"
    )


def giveup_hdlr(details):
    """Handler for when backoff gives up."""
    exception = details.get("exception")
    tries = details.get("tries", 0)

    if isinstance(exception, openai.RateLimitError):
        # Handle rate limit errors specially
        logging.error(
            f"API quota exceeded after {tries} tries. Error: {str(exception)}"
        )
    elif hasattr(exception, "status_code"):
        # Handle errors with status codes
        logging.error(
            f"API error (status: {exception.status_code}) after {tries} tries: {str(exception)}"
        )
    else:
        # Generic error handling
        logging.error(f"Giving up after {tries} tries. Error: {str(exception)}")


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.history = []

    @abstractmethod
    def generate(
        self, prompt: str, model: str, max_completion_tokens: int = 5000, **kwargs
    ) -> str:
        """Generate text from the model."""
        raise NotImplementedError

    def log_usage(self, response: Dict[str, Any]):
        """Log token usage from the API response."""
        usage_data = response.get("usage", {})
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self) -> Dict[str, Dict[str, int]]:
        """Get the total tokens used and reset the token usage."""
        usage = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def generate(
        self, prompt: str, model: str, max_completion_tokens: int = 5000, **kwargs
    ) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                stream=False,
                **kwargs,
            )

            # Log usage
            self.log_usage(
                {
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                }
            )

            # Log history
            self.history.append(
                {
                    "prompt": prompt,
                    "response": response.model_dump(),
                    "kwargs": kwargs,
                }
            )

            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            logging.error(f"OpenAI API quota exceeded: {str(e)}")
            # Return a fallback message when quota is exceeded
            return "I'm unable to process your request due to API quota limitations. Please try again later or use a different model."
        except Exception as e:
            logging.error(f"Error in OpenAI API request: {str(e)}")
            raise


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider."""

    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com"):
        super().__init__(api_key)
        self.api_base = api_base
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def generate(
        self, prompt: str, model: str, max_completion_tokens: int = 5000, **kwargs
    ) -> str:
        """Generate text using DeepSeek API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                stream=False,
                **kwargs,
            )

            # Log usage
            self.log_usage(
                {
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                }
            )

            # Log history
            self.history.append(
                {
                    "prompt": prompt,
                    "response": response.model_dump(),
                    "kwargs": kwargs,
                }
            )

            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            logging.error(f"DeepSeek API quota exceeded: {str(e)}")
            # Return a fallback message when quota is exceeded
            return "I'm unable to process your request due to API quota limitations. Please try again later or use a different model."
        except Exception as e:
            logging.error(f"Error in DeepSeek API request: {str(e)}")
            raise


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = genai.Client(api_key=api_key)

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def generate(
        self, prompt: str, model: str, max_completion_tokens: int = 5000, **kwargs
    ) -> str:
        """Generate text using Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                generation_config={"max_output_tokens": max_completion_tokens, **kwargs},
            )

            # Gemini API doesn't provide token usage directly
            # Log history
            self.history.append(
                {
                    "prompt": prompt,
                    "response": str(response),
                    "kwargs": kwargs,
                }
            )

            return response.text
        except Exception as e:
            # Check if it's a quota error
            error_str = str(e).lower()
            if (
                "quota" in error_str
                or "rate limit" in error_str
                or "exceeded" in error_str
            ):
                logging.error(f"Gemini API quota exceeded: {str(e)}")
                # Return a fallback message when quota is exceeded
                return "I'm unable to process your request due to API quota limitations. Please try again later or use a different model."
            else:
                logging.error(f"Error in Gemini API request: {str(e)}")
                raise


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider."""

    def __init__(self, azure_endpoint: str, api_version: str, api_key: str):
        super().__init__(api_key)
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def generate(
        self, prompt: str, model: str, max_completion_tokens: int = 5000, **kwargs
    ) -> str:
        """Generate text using Azure OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                stream=False,
                **kwargs,
            )

            # Log usage
            self.log_usage(
                {
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                }
            )

            # Log history
            self.history.append(
                {
                    "prompt": prompt,
                    "response": response.model_dump(),
                    "kwargs": kwargs,
                }
            )

            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            logging.error(f"Azure OpenAI API quota exceeded: {str(e)}")
            # Return a fallback message when quota is exceeded
            return "I'm unable to process your request due to API quota limitations. Please try again later or use a different model."
        except Exception as e:
            logging.error(f"Error in Azure OpenAI API request: {str(e)}")
            raise


def get_llm_provider(
    model: str, config: Dict[str, Any], provider: Optional[str] = None
) -> LLMProvider:
    """Get the appropriate LLM provider based on the model name.

    Args:
        model: The model name, optionally prefixed with provider (e.g., "openai/gpt-4")
        config: Configuration dictionary containing API keys and settings
        provider: Optional explicit provider to use (overrides model prefix)

    Returns:
        An instance of the appropriate LLMProvider

    Raises:
        ValueError: If required API keys are not found in config
    """

    if not provider:
        provider = "openai"

    # Get API key from config or environment
    if provider.lower() == "openai":
        api_key = config.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in config.toml")
        return OpenAIProvider(api_key=api_key)

    elif provider.lower() == "deepseek":
        api_key = config.get("deepseek_api_key")
        api_base = config.get("deepseek_api_base", "https://api.deepseek.com")
        if not api_key:
            raise ValueError("DeepSeek API key not found in config.toml")
        return DeepSeekProvider(api_key=api_key, api_base=api_base)

    elif provider.lower() == "gemini":
        api_key = config.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key not found in config.toml")
        return GeminiProvider(api_key=api_key)

    elif provider.lower() == "azure":
        azure_endpoint = config.get("azure_endpoint")
        api_key = config.get("azure_api_key")
        if not all([azure_endpoint, api_key]):
            raise ValueError("Azure OpenAI configuration incomplete in config.toml")
        return AzureOpenAIProvider(azure_endpoint=azure_endpoint, api_key=api_key)

    else:
        # Default to OpenAI for unknown providers
        api_key = config.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in config.toml")
        return OpenAIProvider(api_key=api_key)
