import pytest

from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    FakeRule,
    LLMClient,
    LLMErrorCode,
    LLMMessage,
    LLMMessageRole,
    LLMNonRetryableError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ModelConfiguration,
)


def fake_request(request_id="request-1", content="offer context"):
    return LLMRequest(
        request_id=request_id,
        messages=(LLMMessage(role=LLMMessageRole.USER, content=content),),
        model_configuration=ModelConfiguration(
            provider=LLMProvider.FAKE,
            model="offline-model",
        ),
    )


def test_fake_client_satisfies_contract_and_returns_queued_responses():
    usage = LLMUsage(input_tokens=4, output_tokens=2, total_tokens=6)
    client = FakeLLMClient(
        queued=(FakeResponse(text="counter", usage=usage, response_id="fake-1"),)
    )

    assert isinstance(client, LLMClient)
    response = client.generate(fake_request())

    assert isinstance(response, LLMResponse)
    assert response.text == "counter"
    assert response.response_id == "fake-1"
    assert response.provider is LLMProvider.FAKE
    assert response.usage == usage
    assert response.latency_ms == 0.0
    assert response.attempts == 1


def test_fake_client_rules_are_deterministic_and_queue_takes_precedence():
    rule = FakeRule(
        predicate=lambda request: request.messages[-1].content.startswith("price"),
        responder=lambda request: FakeResponse(text=f"rule:{request.request_id}"),
    )
    client = FakeLLMClient(
        queued=(FakeResponse(text="queued"),),
        rules=(rule,),
    )

    assert client.generate(fake_request(content="price 10")).text == "queued"
    assert client.generate(fake_request(request_id="request-2", content="price 11")).text == (
        "rule:request-2"
    )


def test_fake_client_does_not_retain_private_prompts_by_default():
    client = FakeLLMClient(queued=(FakeResponse(text="ok"),))
    client.generate(fake_request(content="private negotiation context"))

    assert client.call_count == 1
    assert client.recorded_requests == ()


def test_fake_client_can_explicitly_record_requests_for_tests():
    client = FakeLLMClient(
        queued=(FakeResponse(text="ok"),),
        record_requests=True,
    )
    request = fake_request()
    client.generate(request)

    assert client.recorded_requests == (request,)


def test_fake_client_without_response_fails_explicitly():
    client = FakeLLMClient()

    with pytest.raises(LLMNonRetryableError) as exc_info:
        client.generate(fake_request())

    assert exc_info.value.info.code is LLMErrorCode.NO_RESPONSE
    assert exc_info.value.info.retryable is False

