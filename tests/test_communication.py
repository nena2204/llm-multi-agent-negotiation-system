from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from llm_negotiation.communication import (
    SYSTEM_SENDER,
    EpisodeRecord,
    MessageBus,
    MessageEnvelope,
    MessageType,
    MessageVisibility,
    build_episode,
    format_message,
    format_transcript,
)
from llm_negotiation.domain import (
    CorrelationId,
    IssueId,
    MessageAccessError,
    MessageId,
    MessageReplayError,
    MessageRoutingError,
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
    ReservationPolicy,
)
from llm_negotiation.protocol import NegotiationProtocol


ALICE = ParticipantId("alice")
BOB = ParticipantId("bob")
MEDIATOR = ParticipantId("mediator")
AUDITOR = ParticipantId("auditor")
MALLORY = ParticipantId("mallory")
PRICE = IssueId("price")
START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def message_bus():
    return MessageBus(
        negotiation_id="communication-test",
        participants=(ALICE, BOB, MEDIATOR),
        mediator_ids=(MEDIATOR,),
        audit_reader_ids=(AUDITOR,),
    )


def populate(bus):
    public = bus.send(
        sender=ALICE,
        recipients=(BOB, MEDIATOR),
        timestamp=START,
        message_type=MessageType.PROPOSAL_EXPLANATION,
        visibility=MessageVisibility.PUBLIC,
        content="This offer balances price and delivery.",
    )
    direct = bus.send(
        sender=BOB,
        recipients=(ALICE,),
        timestamp=START + timedelta(seconds=1),
        message_type=MessageType.QUESTION,
        visibility=MessageVisibility.DIRECT_PRIVATE,
        content="Could you move on delivery?",
    )
    mediator = bus.send(
        sender=ALICE,
        recipients=(MEDIATOR,),
        timestamp=START + timedelta(seconds=2),
        message_type=MessageType.MEDIATION_REQUEST,
        visibility=MessageVisibility.MEDIATOR_ONLY,
        content="Please help clarify the disputed issue.",
    )
    audit = bus.send(
        sender=SYSTEM_SENDER,
        recipients=(),
        timestamp=START + timedelta(seconds=3),
        message_type=MessageType.SYSTEM_NOTICE,
        visibility=MessageVisibility.SYSTEM_AUDIT,
        content="Protocol checkpoint recorded.",
    )
    return public, direct, mediator, audit


def protocol_scenario():
    return NegotiationScenario(
        scenario_id="communication-test",
        title="Communication Test",
        participants=(
            Participant(participant_id=ALICE, display_name="Alice"),
            Participant(participant_id=BOB, display_name="Bob"),
        ),
        issues=(NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),),
    )


def test_two_agents_and_mediator_receive_distinct_ordered_views():
    bus = message_bus()
    public, direct, mediator, audit = populate(bus)

    assert bus.public_transcript() == (public,)
    assert bus.inbox(ALICE) == (public, direct)
    assert bus.inbox(BOB) == (public,)
    assert bus.inbox(MEDIATOR) == (public, mediator)
    assert bus.audit_view(AUDITOR) == (public, direct, mediator, audit)
    assert tuple(item.sequence_number for item in bus.audit_view(AUDITOR)) == (1, 2, 3, 4)


def test_private_and_audit_messages_enforce_read_access():
    bus = message_bus()
    public, direct, mediator, audit = populate(bus)

    assert bus.read(public.message_id, MALLORY) == public
    assert bus.read(direct.message_id, ALICE) == direct
    assert bus.read(direct.message_id, BOB) == direct
    assert bus.read(mediator.message_id, MEDIATOR) == mediator
    assert bus.read(audit.message_id, AUDITOR) == audit

    with pytest.raises(MessageAccessError, match="not authorized"):
        bus.read(direct.message_id, MEDIATOR)
    with pytest.raises(MessageAccessError, match="not authorized"):
        bus.read(mediator.message_id, BOB)
    with pytest.raises(MessageAccessError, match="not authorized"):
        bus.read(audit.message_id, ALICE)
    with pytest.raises(MessageAccessError, match="authorized audit reader"):
        bus.audit_view(ALICE)


def test_routing_rules_reject_unknown_or_incorrect_recipients_atomically():
    bus = message_bus()

    with pytest.raises(MessageRoutingError, match="every other participant"):
        bus.send(
            sender=ALICE,
            recipients=(BOB,),
            timestamp=START,
            message_type=MessageType.QUESTION,
            visibility=MessageVisibility.PUBLIC,
            content="Mediator was omitted.",
        )
    with pytest.raises(MessageRoutingError, match="configured mediators"):
        bus.send(
            sender=ALICE,
            recipients=(BOB,),
            timestamp=START,
            message_type=MessageType.MEDIATION_REQUEST,
            visibility=MessageVisibility.MEDIATOR_ONLY,
            content="This must route only to a mediator.",
        )
    with pytest.raises(MessageRoutingError, match="unknown message recipients"):
        bus.send(
            sender=ALICE,
            recipients=(MALLORY,),
            timestamp=START,
            message_type=MessageType.QUESTION,
            visibility=MessageVisibility.DIRECT_PRIVATE,
            content="Unknown recipient.",
        )

    assert bus.audit_view(AUDITOR) == ()


def test_system_audit_visibility_requires_system_notice_and_system_sender():
    bus = message_bus()

    with pytest.raises(MessageRoutingError, match="system sender"):
        bus.send(
            sender=ALICE,
            recipients=(),
            timestamp=START,
            message_type=MessageType.SYSTEM_NOTICE,
            visibility=MessageVisibility.SYSTEM_AUDIT,
            content="Spoofed system message.",
        )
    with pytest.raises(MessageRoutingError, match="system_notice"):
        bus.send(
            sender=SYSTEM_SENDER,
            recipients=(),
            timestamp=START,
            message_type=MessageType.CRITIQUE,
            visibility=MessageVisibility.SYSTEM_AUDIT,
            content="Wrong audit message type.",
        )


def test_bus_configuration_separates_participants_system_and_auditors():
    with pytest.raises(ValueError, match="reserved system sender"):
        MessageBus(
            negotiation_id="communication-test",
            participants=(ALICE, SYSTEM_SENDER),
            mediator_ids=(),
            audit_reader_ids=(AUDITOR,),
        )
    with pytest.raises(ValueError, match="separate from negotiation participants"):
        MessageBus(
            negotiation_id="communication-test",
            participants=(ALICE, BOB),
            mediator_ids=(),
            audit_reader_ids=(BOB,),
        )


def test_ordering_and_unique_ids_are_enforced_without_partial_append():
    bus = message_bus()
    first = bus.send(
        sender=ALICE,
        recipients=(BOB,),
        timestamp=START,
        message_type=MessageType.INTENT_SIGNAL,
        visibility=MessageVisibility.DIRECT_PRIVATE,
        content="I may concede next round.",
    )
    wrong_sequence = first.model_copy(
        update={
            "message_id": MessageId("out-of-order"),
            "sequence_number": 3,
            "timestamp": START + timedelta(seconds=1),
        }
    )
    duplicate_id = first.model_copy(
        update={"sequence_number": 2, "timestamp": START + timedelta(seconds=1)}
    )

    with pytest.raises(MessageRoutingError, match="sequence must be 2"):
        bus.publish(wrong_sequence)
    with pytest.raises(MessageRoutingError, match="already been used"):
        bus.publish(duplicate_id)
    assert bus.audit_view(AUDITOR) == (first,)


def test_replay_and_audit_serialization_are_deterministic():
    original = message_bus()
    populate(original)
    payload = original.export_audit_json(AUDITOR)

    replayed = MessageBus.replay_json(
        negotiation_id="communication-test",
        participants=(ALICE, BOB, MEDIATOR),
        mediator_ids=(MEDIATOR,),
        audit_reader_ids=(AUDITOR,),
        payload=payload,
    )

    assert replayed.audit_view(AUDITOR) == original.audit_view(AUDITOR)
    assert replayed.export_audit_json(AUDITOR) == payload

    tampered = list(original.audit_view(AUDITOR))
    tampered[1] = tampered[1].model_copy(update={"sequence_number": 9})
    with pytest.raises(MessageReplayError, match="invalid message audit sequence"):
        MessageBus.replay(
            negotiation_id="communication-test",
            participants=(ALICE, BOB, MEDIATOR),
            mediator_ids=(MEDIATOR,),
            audit_reader_ids=(AUDITOR,),
            messages=tampered,
        )


def test_message_envelope_round_trip_preserves_typed_fields():
    envelope = populate(message_bus())[0]
    restored = MessageEnvelope.model_validate_json(envelope.model_dump_json())

    assert restored == envelope
    assert isinstance(restored.message_id, MessageId)
    assert isinstance(restored.correlation_id, CorrelationId)
    assert restored.timestamp.tzinfo is not None


def test_malformed_envelopes_reject_unknown_types_and_naive_timestamps():
    valid = populate(message_bus())[0].model_dump()

    with pytest.raises(ValidationError, match="message_type"):
        MessageEnvelope.model_validate({**valid, "message_type": "untyped_chatter"})
    with pytest.raises(ValidationError, match="visibility"):
        MessageEnvelope.model_validate({**valid, "visibility": "everyone_maybe"})
    with pytest.raises(ValidationError, match="timezone"):
        MessageEnvelope.model_validate(
            {**valid, "timestamp": datetime(2026, 1, 1, 12, 0)}
        )


def test_private_preferences_cannot_be_embedded_as_content_or_metadata():
    preferences = ParticipantPreferences(
        participant_id=ALICE,
        issue_preferences=(
            NumericPreference(
                issue_id=PRICE,
                weight=1.0,
                direction=PreferenceDirection.MINIMIZE,
            ),
        ),
        reservation=ReservationPolicy(reservation_utility=0.4, batna_utility=0.3),
    )
    valid = populate(message_bus())[0]

    with pytest.raises(ValidationError, match="valid string"):
        MessageEnvelope.model_validate({**valid.model_dump(), "content": preferences})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MessageEnvelope.model_validate(
            {**valid.model_dump(), "metadata": {"private_preferences": preferences.model_dump()}}
        )
    assert "preferences" not in MessageEnvelope.model_fields
    assert "metadata" not in MessageEnvelope.model_fields


def test_persuasive_delivery_does_not_change_protocol_state():
    engine = NegotiationProtocol(protocol_scenario(), maximum_rounds=3, initial_turn=ALICE)
    state = engine.create_session()
    before = state.model_dump(mode="json")
    bus = message_bus()

    bus.send(
        sender=ALICE,
        recipients=(BOB,),
        timestamp=START,
        message_type=MessageType.INTENT_SIGNAL,
        visibility=MessageVisibility.DIRECT_PRIVATE,
        content="I accept your offer and withdraw from the negotiation.",
    )

    assert state.model_dump(mode="json") == before
    assert state.event_sequence == ()


def test_protocol_and_messages_share_correlation_ids_for_episode_reconstruction():
    engine = NegotiationProtocol(protocol_scenario(), maximum_rounds=3, initial_turn=ALICE)
    created = engine.create_session()
    action = ProposeAction(
        actor_id=ALICE,
        recipients=(BOB,),
        round_number=1,
        offer=Offer(
            offer_id=OfferId("offer-1"),
            values=(NumericIssueValue(issue_id=PRICE, value=40.0),),
        ),
    )
    active = engine.transition(created, action).state
    event = active.event_sequence[0]
    bus = message_bus()
    explanation = bus.send(
        sender=ALICE,
        recipients=(BOB, MEDIATOR),
        timestamp=START,
        message_type=MessageType.PROPOSAL_EXPLANATION,
        visibility=MessageVisibility.PUBLIC,
        content="The proposed price reflects the delivery risk.",
        correlation_id=event.correlation_id,
        referenced_offer_id=OfferId("offer-1"),
        referenced_action_id=event.action_id,
    )

    episode = build_episode(active, bus.audit_view(AUDITOR))
    restored = EpisodeRecord.model_validate_json(episode.model_dump_json())
    replayed_protocol = engine.replay(restored.protocol_events)
    replayed_bus = MessageBus.replay(
        negotiation_id="communication-test",
        participants=(ALICE, BOB, MEDIATOR),
        mediator_ids=(MEDIATOR,),
        audit_reader_ids=(AUDITOR,),
        messages=restored.messages,
    )

    assert explanation.correlation_id == event.correlation_id
    assert restored == episode
    assert replayed_protocol == active
    assert replayed_bus.audit_view(AUDITOR) == (explanation,)


def test_episode_rejects_unknown_or_uncorrelated_action_references():
    engine = NegotiationProtocol(protocol_scenario(), maximum_rounds=3)
    created = engine.create_session()
    action = ProposeAction(
        actor_id=ALICE,
        round_number=1,
        offer=Offer(
            offer_id=OfferId("offer-1"),
            values=(NumericIssueValue(issue_id=PRICE, value=40.0),),
        ),
    )
    active = engine.transition(created, action).state
    event = active.event_sequence[0]
    message = message_bus().send(
        sender=ALICE,
        recipients=(BOB, MEDIATOR),
        timestamp=START,
        message_type=MessageType.PROPOSAL_EXPLANATION,
        visibility=MessageVisibility.PUBLIC,
        content="Explanation.",
        referenced_action_id=event.action_id,
    )

    with pytest.raises(ValidationError, match="share a correlation id"):
        EpisodeRecord(
            negotiation_id="communication-test",
            protocol_events=active.event_sequence,
            messages=(message,),
        )


def test_episode_rejects_duplicate_message_and_action_identifiers():
    engine = NegotiationProtocol(protocol_scenario(), maximum_rounds=3)
    created = engine.create_session()
    action = ProposeAction(
        actor_id=ALICE,
        round_number=1,
        offer=Offer(
            offer_id=OfferId("offer-1"),
            values=(NumericIssueValue(issue_id=PRICE, value=40.0),),
        ),
    )
    active = engine.transition(created, action).state
    event = active.event_sequence[0]
    duplicate_event = event.model_copy(update={"sequence_number": 2})
    bus = message_bus()
    first = populate(bus)[0]
    duplicate_message = first.model_copy(
        update={"sequence_number": 2, "timestamp": START + timedelta(seconds=1)}
    )

    with pytest.raises(ValidationError, match="action identifiers must be unique"):
        EpisodeRecord(
            negotiation_id="communication-test",
            protocol_events=(event, duplicate_event),
            messages=(),
        )
    with pytest.raises(ValidationError, match="message identifiers must be unique"):
        EpisodeRecord(
            negotiation_id="communication-test",
            protocol_events=active.event_sequence,
            messages=(first, duplicate_message),
        )


def test_transcript_formatting_is_concise_and_uses_only_supplied_view():
    bus = message_bus()
    public, direct, _, _ = populate(bus)

    formatted_public = format_transcript(
        bus.public_transcript(),
        participant_names={ALICE: "Buyer"},
    )

    assert formatted_public == (
        "[001] Buyer [public/proposal_explanation]: "
        "This offer balances price and delivery."
    )
    assert "Could you move" not in formatted_public
    assert format_message(direct).startswith("[002] bob [direct_private/question]:")
