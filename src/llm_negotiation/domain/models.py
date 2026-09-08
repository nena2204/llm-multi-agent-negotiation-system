from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from .errors import (
    IncompleteOfferError,
    InvalidWeightsError,
    IssueValueError,
    MalformedActionError,
    OutcomeValidationError,
    PreferenceIssueMismatchError,
    UnknownIssueError,
    UnknownParticipantError,
)
from .identifiers import AgreementId, IssueId, MediatorInterventionId, OfferId, ParticipantId


WEIGHT_SUM_TOLERANCE = 1e-6


def _json_array_to_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ParticipantRole(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"
    NEGOTIATOR = "negotiator"
    MEDIATOR = "mediator"


class IssueKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class PreferenceDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ActionType(str, Enum):
    PROPOSE = "propose"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    MESSAGE = "message"
    REQUEST_MEDIATION = "request_mediation"
    WITHDRAW = "withdraw"


class OutcomeStatus(str, Enum):
    AGREEMENT = "agreement"
    NO_AGREEMENT = "no_agreement"
    WITHDRAWN = "withdrawn"


class MediationAccessMode(str, Enum):
    PUBLIC_ONLY = "public_only"
    CONFIDENTIAL_SUMMARY = "confidential_summary"
    SIMULATION_ORACLE = "simulation_oracle"


class MediationTrigger(str, Enum):
    EXPLICIT_REQUEST = "explicit_request"
    DEADLOCK = "deadlock"
    DEADLINE = "deadline"


class MediatorInterventionKind(str, Enum):
    PROPOSAL = "proposal"
    CLARIFYING_QUESTION = "clarifying_question"
    REFUSAL = "refusal"


class Participant(DomainModel):
    participant_id: ParticipantId
    display_name: str = Field(min_length=1, max_length=100)
    role: ParticipantRole = ParticipantRole.NEGOTIATOR

    @field_validator("display_name")
    @classmethod
    def display_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("display_name must not be blank")
        return value


class NumericIssue(DomainModel):
    kind: Literal[IssueKind.NUMERIC] = IssueKind.NUMERIC
    issue_id: IssueId
    name: str = Field(min_length=1, max_length=100)
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    unit: Optional[str] = None

    @model_validator(mode="after")
    def ordered_range(self) -> NumericIssue:
        if self.maximum <= self.minimum:
            raise ValueError("numeric issue maximum must be greater than minimum")
        if not math.isfinite(self.maximum - self.minimum):
            raise ValueError("numeric issue range must have a finite span")
        return self


class CategoricalIssue(DomainModel):
    kind: Literal[IssueKind.CATEGORICAL] = IssueKind.CATEGORICAL
    issue_id: IssueId
    name: str = Field(min_length=1, max_length=100)
    choices: Tuple[str, ...] = Field(min_length=1)

    @field_validator("choices", mode="before")
    @classmethod
    def choices_from_json_array(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @field_validator("choices")
    @classmethod
    def unique_non_blank_choices(cls, choices: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not choice.strip() for choice in choices):
            raise ValueError("categorical choices must not be blank")
        if len(set(choices)) != len(choices):
            raise ValueError("categorical choices must be unique")
        return choices


Issue = Annotated[Union[NumericIssue, CategoricalIssue], Field(discriminator="kind")]


class NumericIssueValue(DomainModel):
    kind: Literal[IssueKind.NUMERIC] = IssueKind.NUMERIC
    issue_id: IssueId
    value: float = Field(allow_inf_nan=False)


class CategoricalIssueValue(DomainModel):
    kind: Literal[IssueKind.CATEGORICAL] = IssueKind.CATEGORICAL
    issue_id: IssueId
    value: str = Field(min_length=1)


IssueValue = Annotated[Union[NumericIssueValue, CategoricalIssueValue], Field(discriminator="kind")]


class Offer(DomainModel):
    offer_id: OfferId
    values: Tuple[IssueValue, ...] = Field(min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def values_from_json_array(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @model_validator(mode="after")
    def unique_issue_values(self) -> Offer:
        issue_ids = [value.issue_id for value in self.values]
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError("an offer may contain only one value per issue")
        return self

    def value_for(self, issue_id: IssueId) -> IssueValue:
        for issue_value in self.values:
            if issue_value.issue_id == issue_id:
                return issue_value
        raise UnknownIssueError(f"offer has no value for issue '{issue_id}'")


class NumericPreference(DomainModel):
    kind: Literal[IssueKind.NUMERIC] = IssueKind.NUMERIC
    issue_id: IssueId
    weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    direction: PreferenceDirection


class CategoryUtility(DomainModel):
    category: str = Field(min_length=1)
    utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class CategoricalPreference(DomainModel):
    kind: Literal[IssueKind.CATEGORICAL] = IssueKind.CATEGORICAL
    issue_id: IssueId
    weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    category_utilities: Tuple[CategoryUtility, ...] = Field(min_length=1)

    @field_validator("category_utilities", mode="before")
    @classmethod
    def utilities_from_json_array(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @model_validator(mode="after")
    def unique_categories(self) -> CategoricalPreference:
        categories = [item.category for item in self.category_utilities]
        if len(set(categories)) != len(categories):
            raise ValueError("categorical preference utilities must have unique categories")
        return self


IssuePreference = Annotated[Union[NumericPreference, CategoricalPreference], Field(discriminator="kind")]


class ReservationPolicy(DomainModel):
    reservation_utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    batna_utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    batna_description: Optional[str] = Field(default=None, max_length=500)


class ParticipantPreferences(DomainModel):
    """Private utility information belonging to exactly one participant.

    Weights must sum to 1.0 within ``WEIGHT_SUM_TOLERANCE`` (1e-6).
    This model must never be embedded in public scenario or action payloads.
    """

    participant_id: ParticipantId
    issue_preferences: Tuple[IssuePreference, ...] = Field(min_length=1)
    reservation: ReservationPolicy

    @field_validator("issue_preferences", mode="before")
    @classmethod
    def preferences_from_json_array(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @model_validator(mode="after")
    def valid_preference_set(self) -> ParticipantPreferences:
        issue_ids = [preference.issue_id for preference in self.issue_preferences]
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError("participant preferences may contain only one entry per issue")
        weight_sum = math.fsum(preference.weight for preference in self.issue_preferences)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise InvalidWeightsError(
                f"issue weights must sum to 1 within {WEIGHT_SUM_TOLERANCE}; got {weight_sum}"
            )
        return self


class NegotiationActionBase(DomainModel):
    actor_id: ParticipantId
    recipients: Tuple[ParticipantId, ...] = ()
    round_number: int = Field(ge=0)

    @field_validator("recipients", mode="before")
    @classmethod
    def recipients_from_json_array(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @model_validator(mode="after")
    def valid_recipients(self) -> NegotiationActionBase:
        if self.actor_id in self.recipients:
            raise ValueError("an action actor cannot also be a recipient")
        if len(set(self.recipients)) != len(self.recipients):
            raise ValueError("action recipients must be unique")
        return self


class ProposeAction(NegotiationActionBase):
    action: Literal[ActionType.PROPOSE] = ActionType.PROPOSE
    offer: Offer


class CounterAction(NegotiationActionBase):
    action: Literal[ActionType.COUNTER] = ActionType.COUNTER
    offer: Offer
    responds_to: OfferId


class AcceptAction(NegotiationActionBase):
    action: Literal[ActionType.ACCEPT] = ActionType.ACCEPT
    offer_id: OfferId


class RejectAction(NegotiationActionBase):
    action: Literal[ActionType.REJECT] = ActionType.REJECT
    offer_id: OfferId
    reason: Optional[str] = Field(default=None, max_length=500)


class MessageAction(NegotiationActionBase):
    action: Literal[ActionType.MESSAGE] = ActionType.MESSAGE
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class RequestMediationAction(NegotiationActionBase):
    action: Literal[ActionType.REQUEST_MEDIATION] = ActionType.REQUEST_MEDIATION
    reason: Optional[str] = Field(default=None, max_length=500)


class WithdrawAction(NegotiationActionBase):
    action: Literal[ActionType.WITHDRAW] = ActionType.WITHDRAW
    reason: Optional[str] = Field(default=None, max_length=500)


class MediatorIntervention(DomainModel):
    """A non-binding mediator output, deliberately separate from participant actions."""

    intervention_id: MediatorInterventionId
    mediator_id: ParticipantId
    negotiation_id: str = Field(min_length=1, max_length=100)
    round_number: int = Field(ge=1)
    trigger: MediationTrigger
    access_mode: MediationAccessMode = MediationAccessMode.PUBLIC_ONLY
    kind: MediatorInterventionKind
    public_disagreement_summary: Optional[str] = Field(
        default=None, min_length=1, max_length=1000
    )
    public_explanation: str = Field(min_length=1, max_length=1000)
    offer: Optional[Offer] = None
    question: Optional[str] = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def payload_matches_kind(self) -> MediatorIntervention:
        if self.kind is MediatorInterventionKind.PROPOSAL:
            if self.offer is None or self.question is not None:
                raise ValueError("a mediator proposal requires one offer and no question")
        elif self.kind is MediatorInterventionKind.CLARIFYING_QUESTION:
            if self.question is None or self.offer is not None:
                raise ValueError("a clarifying intervention requires one question and no offer")
        elif self.offer is not None or self.question is not None:
            raise ValueError("a mediation refusal cannot contain an offer or question")
        return self


NegotiationAction = Annotated[
    Union[
        ProposeAction,
        CounterAction,
        AcceptAction,
        RejectAction,
        MessageAction,
        RequestMediationAction,
        WithdrawAction,
    ],
    Field(discriminator="action"),
]
_ACTION_ADAPTER = TypeAdapter(NegotiationAction)


def parse_action(data: Any) -> NegotiationAction:
    """Validate untrusted external or future LLM action data."""

    try:
        return _ACTION_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise MalformedActionError(f"malformed negotiation action: {exc}") from exc


class Agreement(DomainModel):
    agreement_id: AgreementId
    offer: Offer
    accepted_by: Tuple[ParticipantId, ...] = Field(min_length=2)
    round_number: int = Field(ge=0)

    @field_validator("accepted_by", mode="before")
    @classmethod
    def acceptors_from_json_array(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @field_validator("accepted_by")
    @classmethod
    def unique_acceptors(cls, participant_ids: Tuple[ParticipantId, ...]) -> Tuple[ParticipantId, ...]:
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("agreement acceptors must be unique")
        return participant_ids


class TerminalOutcome(DomainModel):
    status: OutcomeStatus
    final_round: int = Field(ge=0)
    agreement: Optional[Agreement] = None
    reason: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def agreement_matches_status(self) -> TerminalOutcome:
        if self.status is OutcomeStatus.AGREEMENT and self.agreement is None:
            raise ValueError("an agreement outcome must contain an agreement")
        if self.status is not OutcomeStatus.AGREEMENT and self.agreement is not None:
            raise ValueError("only an agreement outcome may contain an agreement")
        return self


class NegotiationScenario(DomainModel):
    scenario_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    participants: Tuple[Participant, ...] = Field(min_length=2)
    issues: Tuple[Issue, ...] = Field(min_length=1)

    @field_validator("participants", "issues", mode="before")
    @classmethod
    def collections_from_json_arrays(cls, value: Any) -> Any:
        return _json_array_to_tuple(value)

    @model_validator(mode="after")
    def unique_public_entities(self) -> NegotiationScenario:
        participant_ids = [participant.participant_id for participant in self.participants]
        issue_ids = [issue.issue_id for issue in self.issues]
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("scenario participant identifiers must be unique")
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError("scenario issue identifiers must be unique")
        return self

    def participant(self, participant_id: ParticipantId) -> Participant:
        for participant in self.participants:
            if participant.participant_id == participant_id:
                return participant
        raise UnknownParticipantError(f"unknown participant '{participant_id}'")

    def issue(self, issue_id: IssueId) -> Issue:
        for issue in self.issues:
            if issue.issue_id == issue_id:
                return issue
        raise UnknownIssueError(f"unknown issue '{issue_id}'")

    def validate_offer(self, offer: Offer) -> Offer:
        expected = {issue.issue_id for issue in self.issues}
        supplied = {value.issue_id for value in offer.values}
        unknown = supplied - expected
        missing = expected - supplied
        if unknown:
            names = ", ".join(sorted(str(issue_id) for issue_id in unknown))
            raise UnknownIssueError(f"offer contains unknown issues: {names}")
        if missing:
            names = ", ".join(sorted(str(issue_id) for issue_id in missing))
            raise IncompleteOfferError(f"offer is missing required issues: {names}")

        for issue_value in offer.values:
            issue = self.issue(issue_value.issue_id)
            if isinstance(issue, NumericIssue):
                if not isinstance(issue_value, NumericIssueValue):
                    raise IssueValueError(f"issue '{issue.issue_id}' requires a numeric value")
                if not issue.minimum <= issue_value.value <= issue.maximum:
                    raise IssueValueError(
                        f"value {issue_value.value} for issue '{issue.issue_id}' is outside "
                        f"[{issue.minimum}, {issue.maximum}]"
                    )
            else:
                if not isinstance(issue_value, CategoricalIssueValue):
                    raise IssueValueError(f"issue '{issue.issue_id}' requires a categorical value")
                if issue_value.value not in issue.choices:
                    raise IssueValueError(
                        f"value '{issue_value.value}' for issue '{issue.issue_id}' is not one of {issue.choices}"
                    )
        return offer

    def validate_action(self, action: NegotiationAction) -> NegotiationAction:
        self.participant(action.actor_id)
        for recipient in action.recipients:
            self.participant(recipient)
        if isinstance(action, (ProposeAction, CounterAction)):
            self.validate_offer(action.offer)
        return action

    def validate_agreement(self, agreement: Agreement, *, require_all: bool = True) -> Agreement:
        self.validate_offer(agreement.offer)
        scenario_participants = {participant.participant_id for participant in self.participants}
        unknown = set(agreement.accepted_by) - scenario_participants
        if unknown:
            names = ", ".join(sorted(str(participant_id) for participant_id in unknown))
            raise OutcomeValidationError(f"agreement contains unknown participants: {names}")
        if require_all and set(agreement.accepted_by) != scenario_participants:
            raise OutcomeValidationError("agreement must be accepted by every scenario participant")
        return agreement


def calculate_utility(
    scenario: NegotiationScenario,
    offer: Offer,
    preferences: ParticipantPreferences,
) -> float:
    """Return weighted normalized utility in [0, 1] using only one participant's private data."""

    scenario.participant(preferences.participant_id)
    scenario.validate_offer(offer)
    expected = {issue.issue_id for issue in scenario.issues}
    supplied = {preference.issue_id for preference in preferences.issue_preferences}
    if supplied != expected:
        unknown = supplied - expected
        missing = expected - supplied
        details = []
        if unknown:
            details.append("unknown: " + ", ".join(sorted(str(item) for item in unknown)))
        if missing:
            details.append("missing: " + ", ".join(sorted(str(item) for item in missing)))
        raise PreferenceIssueMismatchError("preference issues must match scenario issues (" + "; ".join(details) + ")")

    preferences_by_issue = {preference.issue_id: preference for preference in preferences.issue_preferences}
    total = 0.0
    for issue in scenario.issues:
        issue_value = offer.value_for(issue.issue_id)
        preference = preferences_by_issue[issue.issue_id]
        if isinstance(issue, NumericIssue):
            if not isinstance(preference, NumericPreference) or not isinstance(issue_value, NumericIssueValue):
                raise PreferenceIssueMismatchError(f"numeric issue '{issue.issue_id}' requires numeric preferences")
            position = (issue_value.value - issue.minimum) / (issue.maximum - issue.minimum)
            issue_utility = position if preference.direction is PreferenceDirection.MAXIMIZE else 1.0 - position
        else:
            if not isinstance(preference, CategoricalPreference) or not isinstance(
                issue_value, CategoricalIssueValue
            ):
                raise PreferenceIssueMismatchError(
                    f"categorical issue '{issue.issue_id}' requires categorical preferences"
                )
            category_utilities: Dict[str, float] = {
                item.category: item.utility for item in preference.category_utilities
            }
            if set(category_utilities) != set(issue.choices):
                raise PreferenceIssueMismatchError(
                    f"categorical utilities for issue '{issue.issue_id}' must cover exactly {issue.choices}"
                )
            issue_utility = category_utilities[issue_value.value]
        total += preference.weight * issue_utility

    return min(1.0, max(0.0, total))
