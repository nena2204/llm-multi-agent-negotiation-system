from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LLMProvider(str, Enum):
    FAKE = "fake"
    OPENAI = "openai"


class LLMMessageRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class LLMErrorCode(str, Enum):
    BUDGET_EXCEEDED = "budget_exceeded"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    PROVIDER_ERROR = "provider_error"
    NO_RESPONSE = "no_response"
    RETRIES_EXHAUSTED = "retries_exhausted"


class RetryConfiguration(LLMModel):
    max_retries: int = Field(default=2, ge=0, le=10)
    initial_backoff_seconds: float = Field(default=0.25, ge=0.0, le=60.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    max_backoff_seconds: float = Field(default=4.0, ge=0.0, le=300.0)

    @model_validator(mode="after")
    def valid_backoff_bounds(self) -> RetryConfiguration:
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial_backoff_seconds must not exceed max_backoff_seconds")
        return self

    def delay_before_retry(self, retry_number: int) -> float:
        if retry_number < 1:
            raise ValueError("retry_number must be positive")
        return min(
            self.initial_backoff_seconds * (self.backoff_multiplier ** (retry_number - 1)),
            self.max_backoff_seconds,
        )


class ModelConfiguration(LLMModel):
    """Provider/model execution settings, intentionally separate from agent profiles."""

    provider: LLMProvider
    model: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    retry: RetryConfiguration = Field(default_factory=RetryConfiguration)

    @field_validator("model")
    @classmethod
    def model_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value


class LLMMessage(LLMModel):
    role: LLMMessageRole
    content: str = Field(min_length=1, max_length=1_000_000)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class LLMRequest(LLMModel):
    request_id: str = Field(min_length=1, max_length=200)
    messages: Tuple[LLMMessage, ...] = Field(min_length=1)
    model_configuration: ModelConfiguration

    @field_validator("request_id")
    @classmethod
    def request_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id must not be blank")
        return value


class LLMUsage(LLMModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_write_input_tokens: int = Field(default=0, ge=0)
    reasoning_output_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def consistent_counts(self) -> LLMUsage:
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must cover input_tokens plus output_tokens")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens must not exceed input_tokens")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens must not exceed output_tokens")
        return self

    def __add__(self, other: object) -> LLMUsage:
        if not isinstance(other, LLMUsage):
            return NotImplemented
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
        )


def aggregate_usage(items: Iterable[LLMUsage]) -> LLMUsage:
    total = LLMUsage()
    for item in items:
        total = total + item
    return total


class LLMErrorInfo(LLMModel):
    code: LLMErrorCode
    message: str = Field(min_length=1, max_length=1000)
    retryable: bool
    provider: Optional[LLMProvider] = None
    model: Optional[str] = Field(default=None, min_length=1, max_length=200)
    status_code: Optional[int] = Field(default=None, ge=100, le=599)
    attempts: int = Field(default=1, ge=1)


class LLMLatency(LLMModel):
    total_ms: float = Field(ge=0.0)


class LLMResponse(LLMModel):
    request_id: str = Field(min_length=1, max_length=200)
    response_id: str = Field(min_length=1, max_length=500)
    text: str
    provider: LLMProvider
    model: str = Field(min_length=1, max_length=200)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency: LLMLatency
    attempts: int = Field(default=1, ge=1)
    error: Optional[LLMErrorInfo] = None

    @property
    def latency_ms(self) -> float:
        return self.latency.total_ms

    @model_validator(mode="after")
    def successful_response_has_no_error(self) -> LLMResponse:
        if self.error is not None:
            raise ValueError("successful LLM responses cannot contain error information")
        return self
