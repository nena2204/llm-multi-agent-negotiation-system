from types import SimpleNamespace

from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    LLMClient,
    LLMMessage,
    LLMMessageRole,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfiguration,
    OpenAILLMClient,
)


class ContractSDKClient:
    def __init__(self):
        self.responses = self

    def with_options(self, **_kwargs):
        return self

    def create(self, **kwargs):
        return SimpleNamespace(
            id="contract-response",
            output_text="contract-result",
            model=kwargs["model"],
            status="completed",
            error=None,
            usage=None,
        )


def contract_request(provider):
    return LLMRequest(
        request_id=f"{provider.value}-contract",
        messages=(LLMMessage(role=LLMMessageRole.USER, content="contract input"),),
        model_configuration=ModelConfiguration(provider=provider, model="contract-model"),
    )


def assert_client_contract(client, request):
    assert isinstance(client, LLMClient)
    response = client.generate(request)
    assert type(response) is LLMResponse
    assert response.request_id == request.request_id
    assert response.provider is request.model_configuration.provider
    assert response.model
    assert response.response_id
    assert response.latency.total_ms >= 0
    assert response.error is None


def test_fake_client_application_contract():
    assert_client_contract(
        FakeLLMClient(queued=(FakeResponse(text="contract-result"),)),
        contract_request(LLMProvider.FAKE),
    )


def test_mocked_openai_client_application_contract():
    assert_client_contract(
        OpenAILLMClient(ContractSDKClient(), clock=lambda: 0.0),
        contract_request(LLMProvider.OPENAI),
    )
