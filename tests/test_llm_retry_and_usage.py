import pytest
from pydantic import ValidationError

from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    LLMClientError,
    LLMErrorCode,
    LLMErrorInfo,
    LLMMessage,
    LLMMessageRole,
    LLMNonRetryableError,
    LLMProvider,
    LLMRequest,
    LLMRetryExhaustedError,
    LLMUsage,
    ModelConfiguration,
    RetryConfiguration,
    aggregate_usage,
)


def request_with_retry(max_retries=2):
    return LLMRequest(
        request_id="retry-request",
        messages=(LLMMessage(role=LLMMessageRole.USER, content="test"),),
        model_configuration=ModelConfiguration(
            provider=LLMProvider.FAKE,
            model="offline-model",
            retry=RetryConfiguration(
                max_retries=max_retries,
                initial_backoff_seconds=0.25,
                backoff_multiplier=2.0,
                max_backoff_seconds=1.0,
            ),
        ),
    )


def retryable_error(code=LLMErrorCode.TIMEOUT):
    return LLMClientError(
        LLMErrorInfo(
            code=code,
            message="temporary failure",
            retryable=True,
            provider=LLMProvider.FAKE,
        )
    )


def test_retryable_error_retries_with_bounded_backoff_then_succeeds():
    sleeps = []
    clock_values = iter((1.0, 1.5))
    client = FakeLLMClient(
        queued=(retryable_error(), retryable_error(), FakeResponse(text="ok")),
        sleep=sleeps.append,
        clock=lambda: next(clock_values),
    )

    response = client.generate(request_with_retry(max_retries=2))

    assert response.attempts == 3
    assert response.latency_ms == 500.0
    assert sleeps == [0.25, 0.5]
    assert client.call_count == 3


def test_non_retryable_error_is_not_retried():
    sleeps = []
    client = FakeLLMClient(
        queued=(
            LLMClientError(
                LLMErrorInfo(
                    code=LLMErrorCode.INVALID_REQUEST,
                    message="invalid",
                    retryable=False,
                    provider=LLMProvider.FAKE,
                )
            ),
            FakeResponse(text="must remain queued"),
        ),
        sleep=sleeps.append,
    )

    with pytest.raises(LLMNonRetryableError) as exc_info:
        client.generate(request_with_retry())

    assert exc_info.value.info.attempts == 1
    assert client.call_count == 1
    assert sleeps == []


def test_retry_budget_is_bounded_and_reports_attempts():
    client = FakeLLMClient(queued=(retryable_error(), retryable_error()))

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        client.generate(request_with_retry(max_retries=1))

    assert exc_info.value.info.code is LLMErrorCode.RETRIES_EXHAUSTED
    assert exc_info.value.info.attempts == 2
    assert client.call_count == 2


def test_usage_aggregation_includes_provider_details():
    total = aggregate_usage(
        (
            LLMUsage(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                cached_input_tokens=3,
                cache_write_input_tokens=2,
                reasoning_output_tokens=2,
            ),
            LLMUsage(
                input_tokens=5,
                output_tokens=7,
                total_tokens=12,
                cached_input_tokens=1,
                cache_write_input_tokens=3,
                reasoning_output_tokens=4,
            ),
        )
    )

    assert total == LLMUsage(
        input_tokens=15,
        output_tokens=11,
        total_tokens=26,
        cached_input_tokens=4,
        cache_write_input_tokens=5,
        reasoning_output_tokens=6,
    )


def test_usage_rejects_inconsistent_counts():
    with pytest.raises(ValidationError, match="total_tokens must cover"):
        LLMUsage(input_tokens=2, output_tokens=2, total_tokens=3)
