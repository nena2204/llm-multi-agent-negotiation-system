from __future__ import annotations

import os
from typing import Callable, Mapping, Optional

from .client import (
    LLMClientError,
    LLMConfigurationError,
    LLMNonRetryableError,
    ProviderResponse,
    RetryingLLMClient,
)
from .models import (
    LLMErrorCode,
    LLMErrorInfo,
    LLMProvider,
    LLMRequest,
    LLMUsage,
    ModelConfiguration,
    RetryConfiguration,
)


def _required_environment_value(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise LLMConfigurationError(
            LLMErrorInfo(
                code=LLMErrorCode.CONFIGURATION,
                message=f"{name} is required; set it in the process environment before using OpenAILLMClient",
                retryable=False,
                provider=LLMProvider.OPENAI,
            )
        )
    return value


def _environment_float(
    environment: Mapping[str, str], name: str, default: float
) -> float:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise LLMConfigurationError(
            LLMErrorInfo(
                code=LLMErrorCode.CONFIGURATION,
                message=f"{name} must be a number",
                retryable=False,
                provider=LLMProvider.OPENAI,
            )
        ) from error


def _environment_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise LLMConfigurationError(
            LLMErrorInfo(
                code=LLMErrorCode.CONFIGURATION,
                message=f"{name} must be an integer",
                retryable=False,
                provider=LLMProvider.OPENAI,
            )
        ) from error


def _optional_environment_int(
    environment: Mapping[str, str], name: str
) -> Optional[int]:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise LLMConfigurationError(
            LLMErrorInfo(
                code=LLMErrorCode.CONFIGURATION,
                message=f"{name} must be an integer",
                retryable=False,
                provider=LLMProvider.OPENAI,
            )
        ) from error


def load_openai_model_configuration(
    environment: Optional[Mapping[str, str]] = None,
) -> ModelConfiguration:
    """Load non-secret model execution settings independently from any agent profile."""

    values = os.environ if environment is None else environment
    model = _required_environment_value(values, "OPENAI_MODEL")
    try:
        retry = RetryConfiguration(
            max_retries=_environment_int(values, "OPENAI_MAX_RETRIES", 2),
            initial_backoff_seconds=_environment_float(
                values, "OPENAI_RETRY_INITIAL_BACKOFF_SECONDS", 0.25
            ),
            backoff_multiplier=_environment_float(
                values, "OPENAI_RETRY_BACKOFF_MULTIPLIER", 2.0
            ),
            max_backoff_seconds=_environment_float(
                values, "OPENAI_RETRY_MAX_BACKOFF_SECONDS", 4.0
            ),
        )
        return ModelConfiguration(
            provider=LLMProvider.OPENAI,
            model=model,
            timeout_seconds=_environment_float(values, "OPENAI_TIMEOUT_SECONDS", 30.0),
            max_output_tokens=_optional_environment_int(values, "OPENAI_MAX_OUTPUT_TOKENS"),
            retry=retry,
        )
    except ValueError as error:
        raise LLMConfigurationError(
            LLMErrorInfo(
                code=LLMErrorCode.CONFIGURATION,
                message=f"OpenAI model configuration is invalid: {error}",
                retryable=False,
                provider=LLMProvider.OPENAI,
                model=model,
            )
        ) from error


def _attribute(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer_attribute(value: object, name: str) -> int:
    raw = _attribute(value, name, 0)
    return int(raw) if raw is not None else 0


class OpenAIResponseAdapter:
    """Map the SDK response shape into the provider-independent application model."""

    @staticmethod
    def convert(response: object, fallback_model: str) -> ProviderResponse:
        provider_error = _attribute(response, "error")
        status = _attribute(response, "status")
        if provider_error is not None or status not in (None, "completed"):
            raise LLMNonRetryableError(
                LLMErrorInfo(
                    code=LLMErrorCode.PROVIDER_ERROR,
                    message="OpenAI returned an incomplete or failed response",
                    retryable=False,
                    provider=LLMProvider.OPENAI,
                    model=fallback_model,
                )
            )

        usage_value = _attribute(response, "usage")
        input_details = _attribute(usage_value, "input_tokens_details")
        output_details = _attribute(usage_value, "output_tokens_details")
        usage = LLMUsage(
            input_tokens=_integer_attribute(usage_value, "input_tokens"),
            output_tokens=_integer_attribute(usage_value, "output_tokens"),
            total_tokens=_integer_attribute(usage_value, "total_tokens"),
            cached_input_tokens=_integer_attribute(input_details, "cached_tokens"),
            cache_write_input_tokens=_integer_attribute(input_details, "cache_write_tokens"),
            reasoning_output_tokens=_integer_attribute(output_details, "reasoning_tokens"),
        )
        response_id = str(_attribute(response, "id", "")).strip()
        if not response_id:
            raise LLMNonRetryableError(
                LLMErrorInfo(
                    code=LLMErrorCode.PROVIDER_ERROR,
                    message="OpenAI response did not include a response id",
                    retryable=False,
                    provider=LLMProvider.OPENAI,
                    model=fallback_model,
                )
            )
        return ProviderResponse(
            response_id=response_id,
            text=str(_attribute(response, "output_text", "")),
            model=str(_attribute(response, "model", fallback_model)),
            usage=usage,
        )


class OpenAILLMClient(RetryingLLMClient):
    """OpenAI Responses API adapter; import and credentials are optional until constructed."""

    provider = LLMProvider.OPENAI

    def __init__(self, sdk_client: object, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._sdk_client = sdk_client

    @classmethod
    def from_env(
        cls,
        environment: Optional[Mapping[str, str]] = None,
        *,
        client_factory: Optional[Callable[..., object]] = None,
        **kwargs: object,
    ) -> OpenAILLMClient:
        values = os.environ if environment is None else environment
        api_key = _required_environment_value(values, "OPENAI_API_KEY")
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise LLMConfigurationError(
                    LLMErrorInfo(
                        code=LLMErrorCode.CONFIGURATION,
                        message=(
                            "The OpenAI SDK is not installed; install the optional provider with "
                            "python -m pip install -e '.[openai]'"
                        ),
                        retryable=False,
                        provider=LLMProvider.OPENAI,
                    )
                ) from error
            client_factory = OpenAI
        # SDK retries are disabled so the gateway's bounded retry policy is the only retry layer.
        return cls(client_factory(api_key=api_key, max_retries=0), **kwargs)

    def _generate_once(self, request: LLMRequest) -> ProviderResponse:
        configuration = request.model_configuration
        try:
            client = self._sdk_client.with_options(
                timeout=configuration.timeout_seconds,
                max_retries=0,
            )
            arguments = {
                "model": configuration.model,
                "input": [
                    {"role": message.role.value, "content": message.content}
                    for message in request.messages
                ],
                "store": False,
            }
            if configuration.max_output_tokens is not None:
                arguments["max_output_tokens"] = configuration.max_output_tokens
            response = client.responses.create(**arguments)
        except LLMClientError:
            raise
        except Exception as error:
            raise self._map_provider_error(error, configuration.model) from error
        return OpenAIResponseAdapter.convert(response, configuration.model)

    @staticmethod
    def _map_provider_error(error: Exception, model: str) -> LLMClientError:
        name = type(error).__name__
        status_code = getattr(error, "status_code", None)
        retryable_mapping = {
            "APITimeoutError": (LLMErrorCode.TIMEOUT, "OpenAI request timed out"),
            "APIConnectionError": (LLMErrorCode.CONNECTION, "Could not connect to OpenAI"),
            "RateLimitError": (LLMErrorCode.RATE_LIMIT, "OpenAI rate limit was reached"),
            "InternalServerError": (LLMErrorCode.SERVER_ERROR, "OpenAI returned a server error"),
        }
        non_retryable_mapping = {
            "AuthenticationError": (
                LLMErrorCode.AUTHENTICATION,
                "OpenAI authentication failed; verify OPENAI_API_KEY",
            ),
            "PermissionDeniedError": (
                LLMErrorCode.PERMISSION_DENIED,
                "OpenAI denied access to the requested model or project",
            ),
            "BadRequestError": (LLMErrorCode.INVALID_REQUEST, "OpenAI rejected the request"),
            "UnprocessableEntityError": (
                LLMErrorCode.INVALID_REQUEST,
                "OpenAI could not process the request",
            ),
            "NotFoundError": (LLMErrorCode.NOT_FOUND, "OpenAI model or resource was not found"),
        }
        if name in retryable_mapping:
            code, message = retryable_mapping[name]
            retryable = True
        elif name in non_retryable_mapping:
            code, message = non_retryable_mapping[name]
            retryable = False
        elif isinstance(status_code, int) and (status_code >= 500 or status_code in (408, 409, 429)):
            code, message, retryable = (
                LLMErrorCode.SERVER_ERROR,
                "OpenAI returned a temporary provider error",
                True,
            )
        else:
            code, message, retryable = (
                LLMErrorCode.PROVIDER_ERROR,
                "OpenAI request failed with a non-retryable provider error",
                False,
            )
        return LLMClientError(
            LLMErrorInfo(
                code=code,
                message=message,
                retryable=retryable,
                provider=LLMProvider.OPENAI,
                model=model,
                status_code=status_code if isinstance(status_code, int) else None,
            )
        )
