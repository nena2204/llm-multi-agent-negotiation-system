"""Unambiguous component-level evaluation cases separated from episode outcomes."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Optional, Tuple

from pydantic import Field, field_validator

from .domain import ActionType, IssueKind
from .domain.models import DomainModel
from .experiment_dataset import scenario_dataset


COMPONENT_DATASET_VERSION = "1.0"


class ComponentSet(str, Enum):
    ENVIRONMENT_COMPREHENSION = "environment_comprehension"
    OPPONENT_INFERENCE = "opponent_inference"
    OPPONENT_CALIBRATION = "opponent_calibration"
    NEXT_ACTION = "next_action"
    JOINT_PLANNING = "joint_planning"


class EnvironmentComprehensionCase(DomainModel):
    case_id: str
    scenario_id: str
    expected_participant_count: int = Field(ge=2)
    expected_issue_kinds: Tuple[IssueKind, ...] = Field(min_length=1)
    expected_feasible: bool

    @field_validator("expected_issue_kinds", mode="before")
    @classmethod
    def tuple_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class OpponentInferenceCase(DomainModel):
    case_id: str
    normalized_offer_values: Tuple[float, ...] = Field(min_length=2)
    expected_goal: str = Field(pattern="^(maximize|minimize|unknown)$")
    expected_strategy: str = Field(pattern="^(fixed|linear|boulware|conceder)$")
    labelled_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("normalized_offer_values", mode="before")
    @classmethod
    def values_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PlanningAccuracyCase(DomainModel):
    case_id: str
    legal_actions: Tuple[ActionType, ...] = Field(min_length=1)
    offered_utility: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reservation_utility: float = Field(ge=0.0, le=1.0)
    expected_next_action: ActionType
    expected_joint_plan: str = Field(
        pattern=(
            "^(accept_current_offer|counter_then_seek_acceptance|"
            "propose_then_evaluate_response|withdraw_without_safe_deal)$"
        )
    )

    @field_validator("legal_actions", mode="before")
    @classmethod
    def actions_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ComponentEvaluationRecord(DomainModel):
    schema_version: Literal["1.0"] = COMPONENT_DATASET_VERSION
    evaluation_set: ComponentSet
    case_id: str
    label: str
    prediction: str
    correct: bool
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    brier_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_visible: bool
    evidence: Tuple[str, ...] = ()

    @field_validator("evidence", mode="before")
    @classmethod
    def evidence_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def environment_comprehension_cases() -> Tuple[EnvironmentComprehensionCase, ...]:
    """Return labels fixed independently of the scenario parser under evaluation."""

    return (
        EnvironmentComprehensionCase(
            case_id="environment-price-feasible-symmetric",
            scenario_id="price-feasible-symmetric",
            expected_participant_count=2,
            expected_issue_kinds=(IssueKind.NUMERIC,),
            expected_feasible=True,
        ),
        EnvironmentComprehensionCase(
            case_id="environment-price-infeasible",
            scenario_id="price-infeasible",
            expected_participant_count=2,
            expected_issue_kinds=(IssueKind.NUMERIC,),
            expected_feasible=False,
        ),
        EnvironmentComprehensionCase(
            case_id="environment-price-feasible-asymmetric",
            scenario_id="price-feasible-asymmetric",
            expected_participant_count=2,
            expected_issue_kinds=(IssueKind.NUMERIC,),
            expected_feasible=True,
        ),
        EnvironmentComprehensionCase(
            case_id="environment-multi-issue-asymmetric",
            scenario_id="multi-issue-asymmetric",
            expected_participant_count=2,
            expected_issue_kinds=(IssueKind.NUMERIC, IssueKind.CATEGORICAL),
            expected_feasible=True,
        ),
        EnvironmentComprehensionCase(
            case_id="environment-three-principal-resource-allocation",
            scenario_id="three-principal-resource-allocation",
            expected_participant_count=3,
            expected_issue_kinds=(IssueKind.NUMERIC, IssueKind.CATEGORICAL),
            expected_feasible=True,
        ),
    )


def opponent_inference_cases() -> Tuple[OpponentInferenceCase, ...]:
    """Offers are public normalized observations; labels are fixed before evaluation."""

    return (
        OpponentInferenceCase(
            case_id="fixed-maximizer",
            normalized_offer_values=(0.9, 0.9, 0.9),
            expected_goal="unknown",
            expected_strategy="fixed",
            labelled_confidence=0.90,
        ),
        OpponentInferenceCase(
            case_id="linear-maximizer-concession",
            normalized_offer_values=(0.9, 0.7, 0.5),
            expected_goal="maximize",
            expected_strategy="linear",
            labelled_confidence=0.80,
        ),
        OpponentInferenceCase(
            case_id="late-concession-maximizer",
            normalized_offer_values=(0.95, 0.90, 0.50),
            expected_goal="maximize",
            expected_strategy="boulware",
            labelled_confidence=0.75,
        ),
        OpponentInferenceCase(
            case_id="early-concession-minimizer",
            normalized_offer_values=(0.05, 0.45, 0.55),
            expected_goal="minimize",
            expected_strategy="conceder",
            labelled_confidence=0.75,
        ),
    )


def planning_accuracy_cases() -> Tuple[PlanningAccuracyCase, ...]:
    return (
        PlanningAccuracyCase(
            case_id="accept-rational-offer",
            legal_actions=(ActionType.ACCEPT, ActionType.COUNTER, ActionType.REJECT),
            offered_utility=0.75,
            reservation_utility=0.50,
            expected_next_action=ActionType.ACCEPT,
            expected_joint_plan="accept_current_offer",
        ),
        PlanningAccuracyCase(
            case_id="counter-below-reservation",
            legal_actions=(ActionType.ACCEPT, ActionType.COUNTER, ActionType.REJECT),
            offered_utility=0.30,
            reservation_utility=0.55,
            expected_next_action=ActionType.COUNTER,
            expected_joint_plan="counter_then_seek_acceptance",
        ),
        PlanningAccuracyCase(
            case_id="open-with-proposal",
            legal_actions=(ActionType.PROPOSE, ActionType.WITHDRAW),
            reservation_utility=0.40,
            expected_next_action=ActionType.PROPOSE,
            expected_joint_plan="propose_then_evaluate_response",
        ),
        PlanningAccuracyCase(
            case_id="withdraw-only-safe-action",
            legal_actions=(ActionType.WITHDRAW,),
            reservation_utility=0.90,
            expected_next_action=ActionType.WITHDRAW,
            expected_joint_plan="withdraw_without_safe_deal",
        ),
    )


def _infer_opponent(case: OpponentInferenceCase) -> tuple[str, str, float]:
    values = case.normalized_offer_values
    deltas = tuple(abs(right - left) for left, right in zip(values, values[1:]))
    net = values[-1] - values[0]
    goal = "unknown" if math.isclose(net, 0.0, abs_tol=1e-12) else (
        "maximize" if net < 0 else "minimize"
    )
    if not any(deltas):
        strategy = "fixed"
    elif len(deltas) >= 2 and deltas[-1] > deltas[0] * 1.5:
        strategy = "boulware"
    elif len(deltas) >= 2 and deltas[0] > deltas[-1] * 1.5:
        strategy = "conceder"
    else:
        strategy = "linear"
    confidence = min(0.95, 0.55 + 0.1 * len(deltas))
    return goal, strategy, confidence


def _next_action(case: PlanningAccuracyCase) -> ActionType:
    if (
        case.offered_utility is not None
        and ActionType.ACCEPT in case.legal_actions
        and case.offered_utility >= case.reservation_utility
    ):
        return ActionType.ACCEPT
    for candidate in (ActionType.COUNTER, ActionType.PROPOSE, ActionType.WITHDRAW):
        if candidate in case.legal_actions:
            return candidate
    return case.legal_actions[0]


def run_component_evaluations(components: object) -> Tuple[ComponentEvaluationRecord, ...]:
    """Run separate labelled capability tests; no outcome metric is used as a proxy."""

    confidence_visible = bool(getattr(components, "confidence_visibility", True))
    records = []
    cases_by_id = {case.case_id: case for case in scenario_dataset()}
    for case in environment_comprehension_cases():
        source = cases_by_id[case.scenario_id]
        prediction = (
            len(source.scenario.participants),
            tuple(issue.kind for issue in source.scenario.issues),
            "infeasible" not in source.tags,
        )
        label = (
            case.expected_participant_count,
            case.expected_issue_kinds,
            case.expected_feasible,
        )
        records.append(
            ComponentEvaluationRecord(
                evaluation_set=ComponentSet.ENVIRONMENT_COMPREHENSION,
                case_id=case.case_id,
                label=str(label),
                prediction=str(prediction),
                correct=prediction == label,
                confidence=(1.0 if confidence_visible else None),
                confidence_visible=confidence_visible,
                evidence=(case.scenario_id,),
            )
        )
    theory_enabled = bool(getattr(components, "theory_of_mind", True))
    for case in opponent_inference_cases():
        goal, strategy, confidence = (
            _infer_opponent(case) if theory_enabled else ("unknown", "fixed", 0.0)
        )
        for evaluation_set, prediction, label in (
            (ComponentSet.OPPONENT_INFERENCE, f"{goal}:{strategy}", f"{case.expected_goal}:{case.expected_strategy}"),
            (ComponentSet.OPPONENT_CALIBRATION, strategy, case.expected_strategy),
        ):
            probability = confidence if prediction == label or strategy == case.expected_strategy else 1.0 - confidence
            records.append(
                ComponentEvaluationRecord(
                    evaluation_set=evaluation_set,
                    case_id=case.case_id,
                    label=label,
                    prediction=prediction,
                    correct=prediction == label,
                    confidence=(confidence if confidence_visible else None),
                    brier_score=(probability - 1.0) ** 2,
                    confidence_visible=confidence_visible,
                    evidence=tuple(f"offer-{index + 1}" for index in range(len(case.normalized_offer_values))),
                )
            )
    for case in planning_accuracy_cases():
        prediction = _next_action(case)
        joint = {
            ActionType.ACCEPT: "accept_current_offer",
            ActionType.COUNTER: "counter_then_seek_acceptance",
            ActionType.PROPOSE: "propose_then_evaluate_response",
            ActionType.WITHDRAW: "withdraw_without_safe_deal",
        }.get(prediction, prediction.value)
        for evaluation_set, predicted, label in (
            (ComponentSet.NEXT_ACTION, prediction.value, case.expected_next_action.value),
            (ComponentSet.JOINT_PLANNING, joint, case.expected_joint_plan),
        ):
            records.append(
                ComponentEvaluationRecord(
                    evaluation_set=evaluation_set,
                    case_id=case.case_id,
                    label=label,
                    prediction=predicted,
                    correct=predicted == label,
                    confidence=(1.0 if confidence_visible else None),
                    confidence_visible=confidence_visible,
                    evidence=tuple(action.value for action in case.legal_actions),
                )
            )
    return tuple(records)
