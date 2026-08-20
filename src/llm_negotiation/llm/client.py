from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from .models import (
    LLMErrorCode,
    LLMErrorInfo,
    LLMLatency,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


class LLMClientError(RuntimeError):
    def __init__(self, info: LLMErrorInfo):
        super().__init__(info.message)
        self.info = info


class LLMConfigurationError(LLMClientError):
    """Raised before a provider call when required configuration is absent or invalid."""


class LLMNonRetryableError(LLMClientError):
    """Raised for a provider failure that retrying cannot correct."""


class LLMRetryExhaustedError(LLMClientError):
    """Raised once the configured retry budget has been consumed."""


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return one provider-independent response or raise a typed client error."""


@dataclass(frozen=True)
class ProviderResponse:
    response_id: str
    text: str
    model: str
    usage: LLMUsage


class RetryingLLMClient(ABC):
    provider: LLMProvider

    def __init__(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._sleep = sleep
        self._clock = clock

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.model_configuration.provider is not self.provider:
            raise LLMConfigurationError(
                LLMErrorInfo(
                    code=LLMErrorCode.CONFIGURATION,
                    message=(
                        f"request provider '{request.model_configuration.provider.value}' does not "
                        f"match client provider '{self.provider.value}'"
                    ),
                    retryable=False,
                    provider=self.provider,
                    model=request.model_configuration.model,
                )
            )

        started = self._clock()
        attempts = 0
        while True:
            attempts += 1
            try:
                provider_response = self._generate_once(request)
            except LLMClientError as error:
                info = error.info.model_copy(update={"attempts": attempts})
                if not info.retryable:
                    raise LLMNonRetryableError(info) from error
                if attempts > request.model_configuration.retry.max_retries:
                    exhausted = LLMErrorInfo(
                        code=LLMErrorCode.RETRIES_EXHAUSTED,
                        message=f"LLM request failed after {attempts} attempts: {info.message}",
                        retryable=False,
                        provider=info.provider or self.provider,
                        model=info.model or request.model_configuration.model,
                        status_code=info.status_code,
                        attempts=attempts,
                    )
                    raise LLMRetryExhaustedError(exhausted) from error
                delay = request.model_configuration.retry.delay_before_retry(attempts)
                self._sleep(delay)
                continue

            elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
            return LLMResponse(
                request_id=request.request_id,
                response_id=provider_response.response_id,
                text=provider_response.text,
                provider=self.provider,
                model=provider_response.model,
                usage=provider_response.usage,
                latency=LLMLatency(total_ms=elapsed_ms),
                attempts=attempts,
            )

    @abstractmethod
    def _generate_once(self, request: LLMRequest) -> ProviderResponse:
        raise NotImplementedError
