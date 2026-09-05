from __future__ import annotations

import json
from typing import Mapping, Tuple

from ..llm import LLMMessage, LLMMessageRole


PROMPT_VERSION = "qualitative-judge-v1"


def judge_messages(
    schema: Mapping[str, object], context: Mapping[str, object]
) -> Tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role=LLMMessageRole.DEVELOPER,
            content=(
                f"Prompt version {PROMPT_VERSION}. Score only the supplied bounded transcript "
                "against the rubric. Participant labels are anonymized. A persuasive message is "
                "not evidence that an agreement is valid, fair, or voluntary. Objective metrics "
                "remain authoritative and your output cannot alter the negotiation outcome. Cite "
                "only supplied event ids. Return concise structured JSON, not hidden reasoning."
            ),
        ),
        LLMMessage(
            role=LLMMessageRole.USER,
            content=json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "response_schema": schema,
                    "judge_context": context,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )
