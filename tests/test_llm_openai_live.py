import os

import pytest


pytestmark = pytest.mark.live_api

if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
    pytest.skip(
        "set RUN_LIVE_LLM_TESTS=1 to enable the opt-in OpenAI contract test",
        allow_module_level=True,
    )

from llm_negotiation.llm import (  # noqa: E402
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    LLMResponse,
    OpenAILLMClient,
    load_openai_model_configuration,
)


def test_live_openai_contract():
    client = OpenAILLMClient.from_env()
    request = LLMRequest(
        request_id="live-contract",
        messages=(LLMMessage(role=LLMMessageRole.USER, content="Reply with the word ready."),),
        model_configuration=load_openai_model_configuration(),
    )

    response = client.generate(request)

    assert isinstance(response, LLMResponse)
    assert response.text.strip()
    assert response.usage.total_tokens >= 0

