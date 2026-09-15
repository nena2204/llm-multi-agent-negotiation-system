from types import SimpleNamespace

import pytest

from llm_negotiation.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMErrorCode,
    LLMMessage,
    LLMMessageRole,
    LLMNonRetryableError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRetryExhaustedError,
    ModelConfiguration,
    OpenAILLMClient,
    RetryConfiguration,
    load_openai_model_configuration,
)


class MockResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MockSDKClient:
    def __init__(self, outcomes):
        self.responses = MockResponses(outcomes)
        self.options = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self


def sdk_response():
    return SimpleNamespace(
        id="response-1",
        output_text="typed result",
        model="provider-model-version",
        status="completed",
        error=None,
        usage=SimpleNamespace(
            input_tokens=9,
            output_tokens=5,
            total_tokens=14,
            input_tokens_details=SimpleNamespace(cached_tokens=2, cache_write_tokens=1),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )


def openai_request(max_retries=0, temperature=None):
    return LLMRequest(
        request_id="openai-request",
        messages=(
            LLMMessage(role=LLMMessageRole.DEVELOPER, content="public rules"),
            LLMMessage(role=LLMMessageRole.USER, content="participant-visible facts"),
        ),
        model_configuration=ModelConfiguration(
            provider=LLMProvider.OPENAI,
            model="configured-model",
            timeout_seconds=12.5,
            temperature=temperature,
            max_output_tokens=100,
            retry=RetryConfiguration(max_retries=max_retries),
        ),
    )


def test_mocked_openai_adapter_uses_responses_api_and_common_response_type():
    sdk = MockSDKClient((sdk_response(),))
    client = OpenAILLMClient(sdk, clock=lambda: 2.0)

    assert isinstance(client, LLMClient)
    result = client.generate(openai_request())

    assert isinstance(result, LLMResponse)
    assert result.provider is LLMProvider.OPENAI
    assert result.model == "provider-model-version"
    assert result.text == "typed result"
    assert result.usage.input_tokens == 9
    assert result.usage.cached_input_tokens == 2
    assert result.usage.cache_write_input_tokens == 1
    assert result.usage.reasoning_output_tokens == 3
    assert result.error is None
    assert sdk.options == [{"timeout": 12.5, "max_retries": 0}]
    assert sdk.responses.calls == [
        {
            "model": "configured-model",
            "input": [
                {"role": "developer", "content": "public rules"},
                {"role": "user", "content": "participant-visible facts"},
            ],
            "store": False,
            "max_output_tokens": 100,
        }
    ]


def test_openai_adapter_forwards_explicit_experiment_temperature():
    sdk = MockSDKClient((sdk_response(),))
    OpenAILLMClient(sdk, clock=lambda: 0.0).generate(
        openai_request(temperature=0.4)
    )
    assert sdk.responses.calls[0]["temperature"] == 0.4


def test_missing_credentials_are_actionable_without_importing_sdk():
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is required") as exc_info:
        OpenAILLMClient.from_env(environment={})

    assert exc_info.value.info.code is LLMErrorCode.CONFIGURATION
    assert exc_info.value.info.retryable is False


def test_model_configuration_loads_separately_from_agent_configuration():
    configuration = load_openai_model_configuration(
        {
            "OPENAI_MODEL": "configured-model",
            "OPENAI_TIMEOUT_SECONDS": "15",
            "OPENAI_TEMPERATURE": "0.3",
            "OPENAI_MAX_OUTPUT_TOKENS": "250",
            "OPENAI_MAX_RETRIES": "3",
            "OPENAI_RETRY_INITIAL_BACKOFF_SECONDS": "0.5",
            "OPENAI_RETRY_BACKOFF_MULTIPLIER": "1.5",
            "OPENAI_RETRY_MAX_BACKOFF_SECONDS": "2",
        }
    )

    assert configuration.provider is LLMProvider.OPENAI
    assert configuration.model == "configured-model"
    assert configuration.timeout_seconds == 15.0
    assert configuration.temperature == 0.3
    assert configuration.retry.max_retries == 3
    assert "persona" not in type(configuration).model_fields
    assert "preferences" not in type(configuration).model_fields


def test_missing_model_configuration_is_actionable():
    with pytest.raises(LLMConfigurationError, match="OPENAI_MODEL is required"):
        load_openai_model_configuration({})


def test_openai_timeout_is_retried_but_authentication_is_not():
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    timeout_sdk = MockSDKClient((APITimeoutError(), sdk_response()))
    timeout_client = OpenAILLMClient(timeout_sdk, sleep=lambda _delay: None, clock=lambda: 0.0)

    retried = timeout_client.generate(openai_request(max_retries=1))
    assert retried.attempts == 2
    assert len(timeout_sdk.responses.calls) == 2

    AuthenticationError = type("AuthenticationError", (Exception,), {})
    auth_sdk = MockSDKClient((AuthenticationError(), sdk_response()))
    auth_client = OpenAILLMClient(auth_sdk, sleep=lambda _delay: None, clock=lambda: 0.0)

    with pytest.raises(LLMNonRetryableError) as exc_info:
        auth_client.generate(openai_request(max_retries=2))
    assert exc_info.value.info.code is LLMErrorCode.AUTHENTICATION
    assert len(auth_sdk.responses.calls) == 1


def test_openai_retry_exhaustion_is_explicit():
    RateLimitError = type("RateLimitError", (Exception,), {})
    sdk = MockSDKClient((RateLimitError(), RateLimitError()))
    client = OpenAILLMClient(sdk, sleep=lambda _delay: None, clock=lambda: 0.0)

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        client.generate(openai_request(max_retries=1))

    assert exc_info.value.info.attempts == 2
    assert exc_info.value.info.status_code is None


def test_provider_exception_details_and_prompts_are_not_exposed(caplog):
    class BadRequestError(Exception):
        status_code = 400

    sdk = MockSDKClient((BadRequestError("private prompt and credential material"),))
    client = OpenAILLMClient(sdk, clock=lambda: 0.0)

    with pytest.raises(LLMNonRetryableError) as exc_info:
        client.generate(openai_request())

    assert exc_info.value.info.message == "OpenAI rejected the request"
    assert "private prompt" not in str(exc_info.value)
    assert caplog.records == []
