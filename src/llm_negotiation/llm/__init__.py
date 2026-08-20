"""Provider-independent LLM boundary with deterministic offline test support."""

from .client import (
    LLMClient,
    LLMClientError,
    LLMConfigurationError,
    LLMNonRetryableError,
    LLMRetryExhaustedError,
)
from .fake import FakeLLMClient, FakeResponse, FakeRule
from .models import (
    LLMErrorCode,
    LLMErrorInfo,
    LLMLatency,
    LLMMessage,
    LLMMessageRole,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ModelConfiguration,
    RetryConfiguration,
    aggregate_usage,
)
from .openai import OpenAILLMClient, load_openai_model_configuration

__all__ = [
    "FakeLLMClient",
    "FakeResponse",
    "FakeRule",
    "LLMClient",
    "LLMClientError",
    "LLMConfigurationError",
    "LLMErrorCode",
    "LLMErrorInfo",
    "LLMLatency",
    "LLMMessage",
    "LLMMessageRole",
    "LLMNonRetryableError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMRetryExhaustedError",
    "LLMUsage",
    "ModelConfiguration",
    "OpenAILLMClient",
    "RetryConfiguration",
    "aggregate_usage",
    "load_openai_model_configuration",
]
