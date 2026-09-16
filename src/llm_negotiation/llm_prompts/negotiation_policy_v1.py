from __future__ import annotations

import json
from typing import Mapping, Tuple

from ..llm import LLMMessage, LLMMessageRole
from .security_v1 import UNTRUSTED_DATA_NOTICE


PROMPT_VERSION = "negotiation-policy-v1"

_STAGE_TEMPLATE = """You are executing cognitive stage '{stage}' for a negotiation agent.
{untrusted_notice}
Use only the supplied participant-visible context. Do not infer or request another participant's
private preferences, BATNA, hidden memory, or chain-of-thought. Return exactly one JSON object that
matches the supplied schema. Include only a concise decision_rationale and short evidence items;
do not provide hidden reasoning, analysis traces, markdown, or prose outside the JSON object."""

_REPAIR_TEMPLATE = """Repair one invalid structured response for cognitive stage '{stage}'.
{untrusted_notice}
Use only the supplied participant-visible context and validation guidance. Return exactly one
corrected JSON object matching the schema. Do not include chain-of-thought, markdown fences, or
prose outside the JSON object. This is the only repair attempt."""


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def stage_messages(
    stage: str,
    schema: Mapping[str, object],
    context: Mapping[str, object],
) -> Tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role=LLMMessageRole.DEVELOPER,
            content=_STAGE_TEMPLATE.format(
                stage=stage, untrusted_notice=UNTRUSTED_DATA_NOTICE
            ),
        ),
        LLMMessage(
            role=LLMMessageRole.USER,
            content=_json(
                {
                    "prompt_version": PROMPT_VERSION,
                    "stage": stage,
                    "response_schema": schema,
                    "participant_visible_context": context,
                }
            ),
        ),
    )


def repair_messages(
    stage: str,
    schema: Mapping[str, object],
    context: Mapping[str, object],
    invalid_response: str,
    validation_guidance: str,
) -> Tuple[LLMMessage, ...]:
    return (
        LLMMessage(
            role=LLMMessageRole.DEVELOPER,
            content=_REPAIR_TEMPLATE.format(
                stage=stage, untrusted_notice=UNTRUSTED_DATA_NOTICE
            ),
        ),
        LLMMessage(
            role=LLMMessageRole.USER,
            content=_json(
                {
                    "prompt_version": PROMPT_VERSION,
                    "stage": stage,
                    "response_schema": schema,
                    "participant_visible_context": context,
                    "invalid_response": invalid_response[:8000],
                    "validation_guidance": validation_guidance,
                }
            ),
        ),
    )
