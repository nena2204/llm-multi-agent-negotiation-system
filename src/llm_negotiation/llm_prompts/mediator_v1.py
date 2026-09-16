from __future__ import annotations

import json
from typing import Mapping, Tuple

from ..llm import LLMMessage, LLMMessageRole
from .security_v1 import UNTRUSTED_DATA_NOTICE


PROMPT_VERSION = "mediator-v1"


def mediator_messages(
    schema: Mapping[str, object], public_context: Mapping[str, object]
) -> Tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role=LLMMessageRole.DEVELOPER,
            content=(
                f"Prompt version {PROMPT_VERSION}. Act as a non-binding negotiation mediator. "
                + UNTRUSTED_DATA_NOTICE
                +
                "Using only the supplied public context, provide a concise public disagreement "
                "summary, then either ask one clarifying question, propose one complete typed offer, "
                "or refuse intervention. Never accept or reject on "
                "behalf of a participant. Do not request hidden reasoning or reveal, infer, or quote "
                "confidential participant preferences. Return strict JSON only."
            ),
        ),
        LLMMessage(
            role=LLMMessageRole.USER,
            content=json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "response_schema": schema,
                    "public_mediation_context": public_context,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )
