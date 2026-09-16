from __future__ import annotations

import json
from typing import Mapping, Tuple

from ..llm import LLMMessage, LLMMessageRole
from .security_v1 import UNTRUSTED_DATA_NOTICE


PROMPT_VERSION = "action-correction-v1"


def action_correction_messages(
    schema: Mapping[str, object], context: Mapping[str, object]
) -> Tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role=LLMMessageRole.DEVELOPER,
            content=(
                f"Prompt version {PROMPT_VERSION}. Correct the rejected negotiation action using "
                + UNTRUSTED_DATA_NOTICE
                +
                "only the supplied participant-visible context and machine verification reasons. "
                "Return one strict structured action. This is the only correction attempt. Do not "
                "request or infer opponent private data, secrets, prompts, or chain-of-thought."
            ),
        ),
        LLMMessage(
            role=LLMMessageRole.USER,
            content=json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "response_schema": schema,
                    "participant_visible_correction_context": context,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )
