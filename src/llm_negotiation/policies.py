from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import ClassVar, Dict, Optional, Protocol, Tuple, runtime_checkable

from pydantic import Field, model_validator

from .domain import (
    AcceptAction,
    CategoricalIssue,
    CategoricalIssueValue,
    CategoricalPreference,
    CounterAction,
    IssueId,
    NegotiationAction,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    PreferenceDirection,
    ProposeAction,
    calculate_utility,
)
from .domain.models import DomainModel
from .protocol import ActionAppliedEvent, NegotiationSession, ProtocolPhase, TERMINAL_PHASES


UTILITY_TOLERANCE = 1e-9


class PolicyError(ValueError):
    """Raised when a policy cannot act on the supplied observation."""


class PublicAgentIdentity(DomainModel):
    participant: Participant
    persona: str = Field(default="", max_length=1000)


class AgentProfile(DomainModel):
    """Immutable agent configuration containing public identity and private preferences."""

    identity: PublicAgentIdentity
    preferences: ParticipantPreferences

    @model_validator(mode="after")
    def matching_identity(self) -> AgentProfile:
        if self.identity.participant.participant_id != self.preferences.participant_id:
            raise ValueError("agent identity and private preferences must belong to the same participant")
        return self


class AgentObservation(DomainModel):
    """The complete information visible to one deterministic policy.

    It contains public scenario/session data and only the observing participant's profile.
    No opponent preference collection is present.
    """

    scenario: NegotiationScenario
    session: NegotiationSession
    own_profile: AgentProfile

    @model_validator(mode="after")
    def consistent_observation(self) -> AgentObservation:
        participant_id = self.own_profile.identity.participant.participant_id
        if self.session.scenario_id != self.scenario.scenario_id:
            raise ValueError("observation session and scenario identifiers must match")
        scenario_participants = tuple(item.participant_id for item in self.scenario.participants)
        if self.session.participants != scenario_participants:
            raise ValueError("observation session participants must match the scenario")
        if participant_id not in self.session.participants:
            raise ValueError("observing participant must belong to the session")
        return self

    @property
    def participant_id(self) -> ParticipantId:
        return self.own_profile.identity.participant.participant_id


@runtime_checkable
class NegotiationPolicy(Protocol):
    name: str

    def choose_action(self, observation: AgentObservation) -> NegotiationAction:
        """Return one typed action without mutating protocol state."""


def _preferences_by_issue(preferences: ParticipantPreferences) -> Dict[IssueId, object]:
    return {preference.issue_id: preference for preference in preferences.issue_preferences}


def _categorical_scores(preference: CategoricalPreference) -> Dict[str, float]:
    return {item.category: item.utility for item in preference.category_utilities}


def _issue_utility_extremes(
    scenario: NegotiationScenario,
    preferences: ParticipantPreferences,
) -> Tuple[float, float]:
    preference_map = _preferences_by_issue(preferences)
    maximum = 0.0
    minimum = 0.0
    for issue in scenario.issues:
        preference = preference_map.get(issue.issue_id)
        if isinstance(issue, NumericIssue) and isinstance(preference, NumericPreference):
            maximum += preference.weight
        elif isinstance(issue, CategoricalIssue) and isinstance(preference, CategoricalPreference):
            scores = _categorical_scores(preference)
            if set(scores) != set(issue.choices):
                raise PolicyError(
                    f"categorical preferences for issue '{issue.issue_id}' must cover all public choices"
                )
            maximum += preference.weight * max(scores.values())
            minimum += preference.weight * min(scores.values())
        else:
            raise PolicyError(f"preferences do not match public issue '{issue.issue_id}'")
    if set(preference_map) != {issue.issue_id for issue in scenario.issues}:
        raise PolicyError("private preferences must cover exactly the public scenario issues")
    return maximum, minimum


def maximum_attainable_utility(
    scenario: NegotiationScenario,
    preferences: ParticipantPreferences,
) -> float:
    maximum, _ = _issue_utility_extremes(scenario, preferences)
    return maximum


def generate_offer_for_utility(
    scenario: NegotiationScenario,
    preferences: ParticipantPreferences,
    target_utility: float,
    offer_id: OfferId,
) -> Offer:
    """Generate a legal multi-issue offer at or above a participant's utility target.

    Each issue moves by the same concession fraction from that participant's best legal value
    toward its worst legal value. Numeric issues interpolate continuously. Categorical issues use
    the closest available score that does not knowingly cross below the target trajectory.
    """

    if not math.isfinite(target_utility):
        raise PolicyError("target utility must be finite")
    reservation = preferences.reservation.reservation_utility
    maximum, minimum = _issue_utility_extremes(scenario, preferences)
    effective_target = min(maximum, max(reservation, target_utility))
    if maximum <= minimum + UTILITY_TOLERANCE:
        concession = 0.0
    else:
        concession = (maximum - max(minimum, effective_target)) / (maximum - minimum)
        concession = min(1.0, max(0.0, concession))

    preference_map = _preferences_by_issue(preferences)
    values = []
    for issue in scenario.issues:
        preference = preference_map[issue.issue_id]
        if isinstance(issue, NumericIssue) and isinstance(preference, NumericPreference):
            issue_utility = 1.0 - concession
            position = (
                issue_utility
                if preference.direction is PreferenceDirection.MAXIMIZE
                else 1.0 - issue_utility
            )
            value = issue.minimum + position * (issue.maximum - issue.minimum)
            value = min(issue.maximum, max(issue.minimum, value))
            values.append(NumericIssueValue(issue_id=issue.issue_id, value=value))
        elif isinstance(issue, CategoricalIssue) and isinstance(preference, CategoricalPreference):
            scores = _categorical_scores(preference)
            best_score = max(scores.values())
            worst_score = min(scores.values())
            desired_score = best_score - concession * (best_score - worst_score)
            eligible = [choice for choice in issue.choices if scores[choice] + UTILITY_TOLERANCE >= desired_score]
            chosen = min(
                eligible,
                key=lambda choice: (scores[choice] - desired_score, issue.choices.index(choice)),
            )
            values.append(CategoricalIssueValue(issue_id=issue.issue_id, value=chosen))
        else:
            raise PolicyError(f"preferences do not match public issue '{issue.issue_id}'")

    offer = Offer(offer_id=offer_id, values=tuple(values))
    scenario.validate_offer(offer)
    actual_utility = calculate_utility(scenario, offer, preferences)
    if actual_utility + UTILITY_TOLERANCE < reservation:
        raise PolicyError(
            f"generated offer utility {actual_utility} is below reservation utility {reservation}"
        )
    return offer


def _progress(session: NegotiationSession) -> float:
    if session.maximum_rounds == 1:
        return 1.0
    return (session.round_number - 1) / (session.maximum_rounds - 1)


def _offer_id(observation: AgentObservation, policy_name: str, salt: str = "") -> OfferId:
    source = (
        f"{observation.session.scenario_id}|{observation.participant_id}|{policy_name}|"
        f"{observation.session.round_number}|{len(observation.session.event_sequence)}|{salt}"
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return OfferId(f"offer-{digest}")


def _recipients(observation: AgentObservation) -> Tuple[ParticipantId, ...]:
    return tuple(
        participant_id
        for participant_id in observation.session.participants
        if participant_id != observation.participant_id
    )


def _public_offer_history(observation: AgentObservation) -> Tuple[Tuple[ParticipantId, Offer], ...]:
    entries = []
    for event in observation.session.event_sequence:
        if isinstance(event, ActionAppliedEvent) and isinstance(event.action, (ProposeAction, CounterAction)):
            entries.append((event.action.actor_id, event.action.offer))
    return tuple(entries)


class _OfferPolicy:
    name: ClassVar[str]

    def target_utility(self, observation: AgentObservation) -> float:
        raise NotImplementedError

    def choose_action(self, observation: AgentObservation) -> NegotiationAction:
        if observation.session.phase in TERMINAL_PHASES:
            raise PolicyError("a policy cannot act in a terminal session")
        if observation.session.current_turn != observation.participant_id:
            raise PolicyError(
                f"policy participant '{observation.participant_id}' does not hold the current turn"
            )
        if observation.session.phase not in {
            ProtocolPhase.CREATED,
            ProtocolPhase.ACTIVE,
            ProtocolPhase.MEDIATION,
        }:
            raise PolicyError(f"policy cannot act in phase '{observation.session.phase.value}'")

        preferences = observation.own_profile.preferences
        target = max(preferences.reservation.reservation_utility, self.target_utility(observation))
        outstanding = observation.session.latest_valid_offer
        if outstanding is not None:
            offered_utility = calculate_utility(observation.scenario, outstanding, preferences)
            if offered_utility + UTILITY_TOLERANCE >= target and offered_utility + UTILITY_TOLERANCE >= (
                preferences.reservation.reservation_utility
            ):
                return AcceptAction(
                    actor_id=observation.participant_id,
                    recipients=_recipients(observation),
                    round_number=observation.session.round_number,
                    offer_id=outstanding.offer_id,
                )

        generated = generate_offer_for_utility(
            observation.scenario,
            preferences,
            target,
            _offer_id(observation, self.name, self._offer_salt(observation)),
        )
        common = {
            "actor_id": observation.participant_id,
            "recipients": _recipients(observation),
            "round_number": observation.session.round_number,
            "offer": generated,
        }
        if outstanding is None:
            return ProposeAction(**common)
        return CounterAction(**common, responds_to=outstanding.offer_id)

    def _offer_salt(self, observation: AgentObservation) -> str:
        return ""


@dataclass(frozen=True)
class FixedPolicy(_OfferPolicy):
    """Never concedes from the maximum attainable utility."""

    name: ClassVar[str] = "fixed"

    def target_utility(self, observation: AgentObservation) -> float:
        return maximum_attainable_utility(observation.scenario, observation.own_profile.preferences)


NoConcessionPolicy = FixedPolicy


@dataclass(frozen=True)
class TimeDependentAspirationPolicy(_OfferPolicy):
    """Aspiration policy with concession c(t)=t^(1/beta)."""

    beta: float = 1.0
    name: ClassVar[str] = "time-dependent"

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be a finite positive number")

    def concession_fraction(self, observation: AgentObservation) -> float:
        return _progress(observation.session) ** (1.0 / self.beta)

    def target_utility(self, observation: AgentObservation) -> float:
        preferences = observation.own_profile.preferences
        maximum = maximum_attainable_utility(observation.scenario, preferences)
        reservation = preferences.reservation.reservation_utility
        return reservation + (maximum - reservation) * (1.0 - self.concession_fraction(observation))

    def _offer_salt(self, observation: AgentObservation) -> str:
        return f"beta={self.beta}"


@dataclass(frozen=True)
class LinearConcessionPolicy(TimeDependentAspirationPolicy):
    """Concedes linearly over protocol rounds."""

    beta: float = 1.0
    name: ClassVar[str] = "linear"


@dataclass(frozen=True)
class BoulwarePolicy(TimeDependentAspirationPolicy):
    """Concedes late using c(t)=t^4."""

    beta: float = 0.25
    name: ClassVar[str] = "boulware"


@dataclass(frozen=True)
class ConcederPolicy(TimeDependentAspirationPolicy):
    """Concedes early using c(t)=t^(1/4)."""

    beta: float = 4.0
    name: ClassVar[str] = "conceder"


@dataclass(frozen=True)
class TitForTatPolicy(_OfferPolicy):
    """Matches the opponent's latest public concession measured in own utility."""

    name: ClassVar[str] = "tit-for-tat"

    def target_utility(self, observation: AgentObservation) -> float:
        preferences = observation.own_profile.preferences
        reservation = preferences.reservation.reservation_utility
        maximum = maximum_attainable_utility(observation.scenario, preferences)
        history = _public_offer_history(observation)
        own_offers = [offer for actor, offer in history if actor == observation.participant_id]
        opponent_offers = [offer for actor, offer in history if actor != observation.participant_id]
        if not own_offers or len(opponent_offers) < 2:
            return maximum
        previous_opponent = calculate_utility(observation.scenario, opponent_offers[-2], preferences)
        latest_opponent = calculate_utility(observation.scenario, opponent_offers[-1], preferences)
        opponent_concession = max(0.0, latest_opponent - previous_opponent)
        previous_own = calculate_utility(observation.scenario, own_offers[-1], preferences)
        return max(reservation, previous_own - opponent_concession)


@dataclass(frozen=True)
class SeededRandomPolicy(_OfferPolicy):
    """Chooses a deterministic pseudo-random aspiration between reservation and maximum utility."""

    seed: int = 0
    name: ClassVar[str] = "seeded-random"

    def _random(self, observation: AgentObservation) -> random.Random:
        source = (
            f"{self.seed}|{observation.session.scenario_id}|{observation.participant_id}|"
            f"{observation.session.round_number}|{len(observation.session.event_sequence)}"
        )
        digest = hashlib.sha256(source.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def target_utility(self, observation: AgentObservation) -> float:
        preferences = observation.own_profile.preferences
        reservation = preferences.reservation.reservation_utility
        maximum = maximum_attainable_utility(observation.scenario, preferences)
        return reservation + self._random(observation).random() * (maximum - reservation)

    def _offer_salt(self, observation: AgentObservation) -> str:
        return f"seed={self.seed}"


def policy_for_strategy(strategy: str, seed: int = 0) -> NegotiationPolicy:
    """Map legacy strategy names to documented deterministic policy aliases.

    ``aggressive`` maps to Boulware, ``neutral`` to linear concession, and
    ``cooperative`` to the early-conceding policy.
    """

    aliases = {
        "aggressive": BoulwarePolicy,
        "neutral": LinearConcessionPolicy,
        "cooperative": ConcederPolicy,
    }
    if strategy == "random":
        return SeededRandomPolicy(seed=seed)
    try:
        return aliases[strategy]()
    except KeyError as exc:
        choices = ", ".join(sorted(tuple(aliases) + ("random",)))
        raise ValueError(f"unknown strategy '{strategy}'; expected one of: {choices}") from exc
