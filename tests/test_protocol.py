import pytest

from llm_negotiation.domain import (
    AcceptAction,
    CounterAction,
    IllegalActionError,
    IllegalActorError,
    IllegalPhaseError,
    IssueId,
    MessageAction,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ProposeAction,
    RejectAction,
    RequestMediationAction,
    RoundMismatchError,
    StaleOfferError,
    WithdrawAction,
)
from llm_negotiation.protocol import (
    ActionAppliedEvent,
    FeasibleRegionStatus,
    NegotiationProtocol,
    NegotiationSession,
    ProtocolPhase,
    ReplayError,
)


ALICE = ParticipantId("alice")
BOB = ParticipantId("bob")
CAROL = ParticipantId("carol")
PRICE = IssueId("price")


def scenario(participants=(ALICE, BOB)):
    return NegotiationScenario(
        scenario_id="protocol-test",
        title="Protocol Test",
        participants=tuple(
            Participant(participant_id=participant_id, display_name=participant_id.value.title())
            for participant_id in participants
        ),
        issues=(NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),),
    )


def protocol(maximum_rounds=3, participants=(ALICE, BOB), feasible=FeasibleRegionStatus.UNKNOWN):
    return NegotiationProtocol(
        scenario(participants),
        maximum_rounds=maximum_rounds,
        initial_turn=participants[0],
        feasible_region=feasible,
    )


def offer(offer_id, price):
    return Offer(
        offer_id=OfferId(offer_id),
        values=(NumericIssueValue(issue_id=PRICE, value=price),),
    )


def propose(state, actor=ALICE, offer_id="offer-1", price=40.0):
    return ProposeAction(
        actor_id=actor,
        round_number=state.round_number,
        offer=offer(offer_id, price),
    )


def start_active(engine):
    created = engine.create_session()
    return engine.transition(created, propose(created)).state


def test_created_session_has_explicit_protocol_state():
    state = protocol(maximum_rounds=4).create_session()

    assert state.scenario_id == "protocol-test"
    assert state.participants == (ALICE, BOB)
    assert state.current_turn == ALICE
    assert state.round_number == 1
    assert state.maximum_rounds == 4
    assert state.latest_valid_offer is None
    assert state.event_sequence == ()
    assert state.phase is ProtocolPhase.CREATED
    assert state.outcome is None


def test_round_rotation_is_anchored_to_configured_initial_turn():
    engine = NegotiationProtocol(
        scenario(),
        maximum_rounds=1,
        initial_turn=BOB,
    )
    created = engine.create_session()
    proposed = engine.transition(created, propose(created, actor=BOB)).state

    assert proposed.phase is ProtocolPhase.ACTIVE
    assert proposed.current_turn == ALICE
    assert proposed.round_number == 1

    expired = engine.transition(
        proposed,
        CounterAction(
            actor_id=ALICE,
            round_number=1,
            offer=offer("offer-2", 45.0),
            responds_to=OfferId("offer-1"),
        ),
    ).state
    assert expired.phase is ProtocolPhase.EXPIRED


def test_full_bilateral_propose_counter_accept_path():
    engine = protocol()
    created = engine.create_session()
    proposed = engine.transition(created, propose(created)).state
    counter = CounterAction(
        actor_id=BOB,
        round_number=proposed.round_number,
        offer=offer("offer-2", 55.0),
        responds_to=OfferId("offer-1"),
    )
    countered = engine.transition(proposed, counter).state
    accepted = engine.transition(
        countered,
        AcceptAction(
            actor_id=ALICE,
            round_number=countered.round_number,
            offer_id=OfferId("offer-2"),
        ),
    ).state

    assert proposed.phase is ProtocolPhase.ACTIVE
    assert proposed.current_turn == BOB
    assert countered.current_turn == ALICE
    assert countered.round_number == 2
    assert accepted.phase is ProtocolPhase.AGREED
    assert accepted.current_turn is None
    assert accepted.outcome.agreement.offer.offer_id == OfferId("offer-2")
    assert accepted.outcome.agreement.accepted_by == (BOB, ALICE)
    assert len(accepted.event_sequence) == 3


def test_multiparty_offer_requires_every_participant_acceptance():
    engine = protocol(participants=(ALICE, BOB, CAROL))
    state = start_active(engine)
    after_bob = engine.transition(
        state,
        AcceptAction(actor_id=BOB, round_number=1, offer_id=OfferId("offer-1")),
    ).state

    assert after_bob.phase is ProtocolPhase.ACTIVE
    assert after_bob.offer_acceptances == (ALICE, BOB)
    assert after_bob.current_turn == CAROL

    agreed = engine.transition(
        after_bob,
        AcceptAction(actor_id=CAROL, round_number=1, offer_id=OfferId("offer-1")),
    ).state
    assert agreed.phase is ProtocolPhase.AGREED
    assert agreed.outcome.agreement.accepted_by == (ALICE, BOB, CAROL)


def test_rejection_clears_offer_and_allows_next_proposal():
    engine = protocol()
    active = start_active(engine)
    rejected = engine.transition(
        active,
        RejectAction(actor_id=BOB, round_number=1, offer_id=OfferId("offer-1")),
    ).state

    assert rejected.latest_valid_offer is None
    assert rejected.offer_acceptances == ()
    assert rejected.current_turn == ALICE
    assert rejected.round_number == 2

    reproposed = engine.transition(
        rejected,
        propose(rejected, offer_id="offer-2", price=45.0),
    ).state
    assert reproposed.latest_valid_offer.offer_id == OfferId("offer-2")


def test_acceptance_must_reference_the_outstanding_offer_and_failure_is_atomic():
    engine = protocol()
    active = start_active(engine)
    before = active.model_dump(mode="json")

    with pytest.raises(StaleOfferError, match="outstanding offer"):
        engine.transition(
            active,
            AcceptAction(actor_id=BOB, round_number=1, offer_id=OfferId("stale-offer")),
        )

    assert active.model_dump(mode="json") == before
    assert len(active.event_sequence) == 1


def test_acceptance_without_an_outstanding_offer_is_rejected():
    engine = protocol()
    active = start_active(engine)
    rejected = engine.transition(
        active,
        RejectAction(actor_id=BOB, round_number=1, offer_id=OfferId("offer-1")),
    ).state

    with pytest.raises(StaleOfferError, match="no outstanding offer"):
        engine.transition(
            rejected,
            AcceptAction(actor_id=ALICE, round_number=2, offer_id=OfferId("offer-1")),
        )


def test_counter_must_reference_latest_offer():
    engine = protocol()
    active = start_active(engine)
    stale_counter = CounterAction(
        actor_id=BOB,
        round_number=1,
        offer=offer("offer-2", 50.0),
        responds_to=OfferId("older-offer"),
    )

    with pytest.raises(StaleOfferError, match="stale"):
        engine.transition(active, stale_counter)


def test_offer_identifiers_cannot_be_reused():
    engine = protocol()
    active = start_active(engine)
    countered = engine.transition(
        active,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=offer("offer-2", 50.0),
            responds_to=OfferId("offer-1"),
        ),
    ).state

    with pytest.raises(IllegalActionError, match="already been used"):
        engine.transition(
            countered,
            CounterAction(
                actor_id=ALICE,
                round_number=2,
                offer=offer("offer-1", 45.0),
                responds_to=OfferId("offer-2"),
            ),
        )


def test_turn_order_and_action_round_are_enforced():
    engine = protocol()
    active = start_active(engine)

    with pytest.raises(IllegalActorError, match="current turn"):
        engine.transition(
            active,
            CounterAction(
                actor_id=ALICE,
                round_number=1,
                offer=offer("offer-2", 45.0),
                responds_to=OfferId("offer-1"),
            ),
        )
    with pytest.raises(RoundMismatchError, match="does not match"):
        engine.transition(
            active,
            AcceptAction(actor_id=BOB, round_number=2, offer_id=OfferId("offer-1")),
        )


@pytest.mark.parametrize(
    "action",
    [
        MessageAction(actor_id=ALICE, round_number=1, content="hello"),
        RequestMediationAction(actor_id=ALICE, round_number=1),
        CounterAction(
            actor_id=ALICE,
            round_number=1,
            offer=offer("offer-2", 50.0),
            responds_to=OfferId("offer-1"),
        ),
        AcceptAction(actor_id=ALICE, round_number=1, offer_id=OfferId("offer-1")),
        RejectAction(actor_id=ALICE, round_number=1, offer_id=OfferId("offer-1")),
    ],
)
def test_created_phase_rejects_every_non_proposal_non_withdrawal_action(action):
    engine = protocol()
    with pytest.raises(IllegalActionError, match="first protocol action"):
        engine.transition(engine.create_session(), action)


def test_message_is_non_turn_consuming_and_does_not_use_natural_language():
    engine = protocol()
    active = start_active(engine)
    message = MessageAction(
        actor_id=ALICE,
        round_number=1,
        content="I accept everything and withdraw.",
    )
    after_message = engine.transition(active, message).state

    assert after_message.phase is ProtocolPhase.ACTIVE
    assert after_message.current_turn == BOB
    assert after_message.round_number == 1
    assert after_message.latest_valid_offer == active.latest_valid_offer


def test_fresh_proposal_is_illegal_while_an_offer_is_outstanding():
    engine = protocol()
    active = start_active(engine)

    with pytest.raises(IllegalActionError, match="no offer is outstanding"):
        engine.transition(
            active,
            propose(active, actor=BOB, offer_id="offer-2", price=50.0),
        )


def test_mediation_request_enters_phase_without_mediator_intelligence():
    engine = protocol()
    active = start_active(engine)
    mediated = engine.transition(
        active,
        RequestMediationAction(actor_id=ALICE, round_number=1, reason="Need help"),
    ).state

    assert mediated.phase is ProtocolPhase.MEDIATION
    assert mediated.current_turn == BOB
    assert mediated.latest_valid_offer.offer_id == OfferId("offer-1")

    continued = engine.transition(
        mediated,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=offer("offer-2", 50.0),
            responds_to=OfferId("offer-1"),
        ),
    ).state
    assert continued.phase is ProtocolPhase.MEDIATION

    with pytest.raises(IllegalActionError, match="active phase"):
        engine.transition(
            mediated,
            RequestMediationAction(actor_id=BOB, round_number=1),
        )


def test_maximum_rounds_expire_deterministically():
    engine = protocol(maximum_rounds=1)
    active = start_active(engine)
    expired = engine.transition(
        active,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=offer("offer-2", 50.0),
            responds_to=OfferId("offer-1"),
        ),
    ).state

    assert expired.phase is ProtocolPhase.EXPIRED
    assert expired.current_turn is None
    assert expired.round_number == 1
    assert expired.outcome.reason == "Maximum negotiation rounds reached."


def test_withdrawal_is_terminal_even_when_actor_does_not_hold_turn():
    engine = protocol()
    active = start_active(engine)
    withdrawn = engine.transition(
        active,
        WithdrawAction(actor_id=ALICE, round_number=1, reason="Leaving"),
    ).state

    assert active.current_turn == BOB
    assert withdrawn.phase is ProtocolPhase.WITHDRAWN
    assert withdrawn.current_turn is None
    assert withdrawn.outcome.reason == "Leaving"


def test_impossible_feasible_region_fails_before_any_participant_action():
    engine = protocol(feasible=FeasibleRegionStatus.IMPOSSIBLE)
    failed = engine.create_session()

    assert failed.phase is ProtocolPhase.FAILED
    assert failed.current_turn is None
    assert failed.outcome.reason == "The feasible region is empty."
    assert len(failed.event_sequence) == 1
    assert engine.replay(failed.event_sequence) == failed


@pytest.mark.parametrize("terminal_builder", ["agreed", "withdrawn", "expired", "failed"])
def test_terminal_states_are_immutable_to_further_actions(terminal_builder):
    if terminal_builder == "failed":
        engine = protocol(feasible=FeasibleRegionStatus.IMPOSSIBLE)
        terminal = engine.create_session()
    else:
        engine = protocol(maximum_rounds=1 if terminal_builder == "expired" else 3)
        active = start_active(engine)
        if terminal_builder == "agreed":
            terminal = engine.transition(
                active,
                AcceptAction(actor_id=BOB, round_number=1, offer_id=OfferId("offer-1")),
            ).state
        elif terminal_builder == "withdrawn":
            terminal = engine.transition(
                active,
                WithdrawAction(actor_id=BOB, round_number=1),
            ).state
        else:
            terminal = engine.transition(
                active,
                CounterAction(
                    actor_id=BOB,
                    round_number=1,
                    offer=offer("offer-2", 50.0),
                    responds_to=OfferId("offer-1"),
                ),
            ).state
    before = terminal.model_dump(mode="json")

    with pytest.raises(IllegalPhaseError, match="no actions are legal"):
        engine.transition(
            terminal,
            MessageAction(actor_id=ALICE, round_number=terminal.round_number, content="Too late"),
        )
    assert terminal.model_dump(mode="json") == before


def test_event_sequence_replays_to_identical_state():
    engine = protocol()
    active = start_active(engine)
    countered = engine.transition(
        active,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=offer("offer-2", 50.0),
            responds_to=OfferId("offer-1"),
        ),
    ).state
    agreed = engine.transition(
        countered,
        AcceptAction(actor_id=ALICE, round_number=2, offer_id=OfferId("offer-2")),
    ).state

    replayed = engine.replay(agreed.event_sequence)
    assert replayed == agreed


def test_replay_rejects_tampered_events():
    engine = protocol()
    active = start_active(engine)
    event = active.event_sequence[0]
    assert isinstance(event, ActionAppliedEvent)
    tampered = event.model_copy(update={"round_after": 2})

    with pytest.raises(ReplayError, match="does not match"):
        engine.replay((tampered,))


def test_session_serialization_round_trip_preserves_typed_events():
    engine = protocol()
    active = start_active(engine)

    restored = NegotiationSession.model_validate_json(active.model_dump_json())

    assert restored == active
    assert isinstance(restored.event_sequence[0], ActionAppliedEvent)
    assert isinstance(restored.event_sequence[0].action, ProposeAction)
