from __future__ import annotations

import json
from typing import Mapping, Tuple

from ..llm import LLMMessage, LLMMessageRole
from .security_v1 import UNTRUSTED_DATA_NOTICE


PROMPT_VERSION = "opponent-model-v1"


def opponent_model_messages(
    schema: Mapping[str, object], context: Mapping[str, object]
) -> Tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role=LLMMessageRole.DEVELOPER,
            content=(
                f"Prompt version {PROMPT_VERSION}. Update an uncertain opponent belief using only "
                + UNTRUSTED_DATA_NOTICE
                +
                "the supplied observable actions and authorized messages. Distinguish facts, "
                "inferences, and unknowns. Never claim an inference is a fact or request hidden "
                "preferences or chain-of-thought. Return strict JSON only."
            ),
        ),
        LLMMessage(
            role=LLMMessageRole.USER,
            content=json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "stage": "opponent_belief_update",
                    "response_schema": schema,
                    "participant_visible_context": context,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )
