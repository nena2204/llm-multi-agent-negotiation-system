import pytest
from pydantic import ValidationError

from llm_negotiation.benchmark import run_policy_session
from llm_negotiation.domain import (
    AcceptAction,
    CategoryUtility,
    CategoricalIssue,
    CategoricalIssueValue,
    CategoricalPreference,
    CounterAction,
    IssueId,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ProposeAction,
    ReservationPolicy,
    calculate_utility,
)
from llm_negotiation.policies import (
    AgentObservation,
    AgentProfile,
    BoulwarePolicy,
    ConcederPolicy,
    FixedPolicy,
    LinearConcessionPolicy,
    NoConcessionPolicy,
    PolicyError,
    PublicAgentIdentity,
    SeededRandomPolicy,
    TimeDependentAspirationPolicy,
    TitForTatPolicy,
    generate_offer_for_utility,
    policy_for_strategy,
)
from llm_negotiation.protocol import (
    ActionAppliedEvent,
    NegotiationProtocol,
    NegotiationSession,
    ProtocolPhase,
)


ALICE = ParticipantId("alice")
BOB = ParticipantId("bob")
PRICE = IssueId("price")
DELIVERY = IssueId("delivery")


def policy_scenario():
    return NegotiationScenario(
        scenario_id="policy-test",
        title="Policy Test",
        participants=(
            Participant(participant_id=ALICE, display_name="Alice", role=ParticipantRole.BUYER),
            Participant(participant_id=BOB, display_name="Bob", role=ParticipantRole.SELLER),
        ),
        issues=(
            NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),
            CategoricalIssue(
                issue_id=DELIVERY,
                name="Delivery",
                choices=("standard", "express"),
            ),
        ),
    )


def profile(participant_id, direction, batna_description):
    participant = policy_scenario().participant(participant_id)
    return AgentProfile(
        identity=PublicAgentIdentity(
            participant=participant,
            persona=f"Deterministic {participant.display_name} baseline",
        ),
        preferences=ParticipantPreferences(
            participant_id=participant_id,
            issue_preferences=(
                NumericPreference(
                    issue_id=PRICE,
                    weight=0.7,
                    direction=direction,
                ),
                CategoricalPreference(
                    issue_id=DELIVERY,
                    weight=0.3,
                    category_utilities=(
                        CategoryUtility(category="standard", utility=0.0),
                        CategoryUtility(category="express", utility=1.0),
                    ),
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=0.3,
                batna_utility=0.25,
                batna_description=batna_description,
            ),
        ),
    )


def alice_profile():
    return profile(ALICE, PreferenceDirection.MINIMIZE, "ALICE PRIVATE BATNA")


def bob_profile():
    return profile(BOB, PreferenceDirection.MAXIMIZE, "BOB PRIVATE BATNA")


def empty_observation(round_number, maximum_rounds=5, participant_id=ALICE):
    scenario = policy_scenario()
    own_profile = alice_profile() if participant_id == ALICE else bob_profile()
    return AgentObservation(
        scenario=scenario,
        session=NegotiationSession(
            scenario_id=scenario.scenario_id,
            participants=(ALICE, BOB),
            current_turn=participant_id,
            round_number=round_number,
            maximum_rounds=maximum_rounds,
            phase=ProtocolPhase.ACTIVE,
        ),
        own_profile=own_profile,
    )


def action_utility(action, observation):
    assert isinstance(action, (ProposeAction, CounterAction))
    return calculate_utility(
        observation.scenario,
        action.offer,
        observation.own_profile.preferences,
    )


def test_agent_profile_is_immutable_and_identity_matches_private_preferences():
    agent_profile = alice_profile()
    with pytest.raises(ValidationError, match="frozen"):
        agent_profile.preferences = bob_profile().preferences

    with pytest.raises(ValidationError, match="same participant"):
        AgentProfile(
            identity=agent_profile.identity,
            preferences=bob_profile().preferences,
        )


def test_observation_exposes_only_own_private_preferences():
    observation = empty_observation(round_number=1)
    payload = observation.model_dump_json()

    assert set(AgentObservation.model_fields) == {"scenario", "session", "own_profile"}
    assert observation.own_profile.preferences.participant_id == ALICE
    assert "ALICE PRIVATE BATNA" in payload
    assert "BOB PRIVATE BATNA" not in payload
    assert not hasattr(observation, "opponent_preferences")


def test_multi_issue_offer_generation_is_legal_and_reservation_safe():
    scenario = policy_scenario()
    preferences = alice_profile().preferences
    generated = generate_offer_for_utility(
        scenario,
        preferences,
        target_utility=0.55,
        offer_id=OfferId("multi-issue"),
    )

    assert scenario.validate_offer(generated) is generated
    assert {value.issue_id for value in generated.values} == {PRICE, DELIVERY}
    assert calculate_utility(scenario, generated, preferences) >= 0.55 - 1e-9


def test_time_based_concessions_are_monotonic_and_strategically_distinct():
    policies = {
        "fixed": FixedPolicy(),
        "boulware": BoulwarePolicy(),
        "linear": LinearConcessionPolicy(),
        "time": TimeDependentAspirationPolicy(beta=2.0),
        "conceder": ConcederPolicy(),
    }
    trajectories = {}
    for name, policy in policies.items():
        trajectories[name] = tuple(
            action_utility(policy.choose_action(empty_observation(round_number)), empty_observation(round_number))
            for round_number in range(1, 6)
        )
        assert all(
            later <= earlier + 1e-9
            for earlier, later in zip(trajectories[name], trajectories[name][1:])
        )

    assert len(set(trajectories["fixed"])) == 1
    assert trajectories["boulware"][2] > trajectories["linear"][2]
    assert trajectories["linear"][2] > trajectories["time"][2]
    assert trajectories["time"][2] > trajectories["conceder"][2]
    assert len({trajectory for trajectory in trajectories.values()}) == len(trajectories)


def test_policy_never_accepts_below_reservation_and_counter_is_safe():
    scenario = policy_scenario()
    low_offer = Offer(
        offer_id=OfferId("low-for-alice"),
        values=(
            NumericIssueValue(issue_id=PRICE, value=100.0),
            # Alice assigns zero utility to standard delivery.
            CategoricalIssueValue(issue_id=DELIVERY, value="standard"),
        ),
    )
    session = NegotiationSession(
        scenario_id=scenario.scenario_id,
        participants=(ALICE, BOB),
        current_turn=ALICE,
        round_number=5,
        maximum_rounds=5,
        latest_valid_offer=low_offer,
        offer_acceptances=(BOB,),
        phase=ProtocolPhase.ACTIVE,
    )
    observation = AgentObservation(scenario=scenario, session=session, own_profile=alice_profile())

    action = ConcederPolicy().choose_action(observation)

    assert isinstance(action, CounterAction)
    assert calculate_utility(scenario, low_offer, alice_profile().preferences) < 0.3
    assert action_utility(action, observation) >= 0.3 - 1e-9


def test_policy_accepts_offer_meeting_aspiration_and_reservation():
    scenario = policy_scenario()
    excellent_offer = Offer(
        offer_id=OfferId("excellent"),
        values=(
            NumericIssueValue(issue_id=PRICE, value=0.0),
            CategoricalIssueValue(issue_id=DELIVERY, value="express"),
        ),
    )
    observation = AgentObservation(
        scenario=scenario,
        session=NegotiationSession(
            scenario_id=scenario.scenario_id,
            participants=(ALICE, BOB),
            current_turn=ALICE,
            round_number=1,
            maximum_rounds=5,
            latest_valid_offer=excellent_offer,
            offer_acceptances=(BOB,),
            phase=ProtocolPhase.ACTIVE,
        ),
        own_profile=alice_profile(),
    )

    action = FixedPolicy().choose_action(observation)
    assert isinstance(action, AcceptAction)
    assert action.offer_id == excellent_offer.offer_id


def test_seeded_random_policy_is_reproducible_and_seed_sensitive():
    observation = empty_observation(round_number=3)

    first = SeededRandomPolicy(seed=7).choose_action(observation)
    repeated = SeededRandomPolicy(seed=7).choose_action(observation)
    different_seed = SeededRandomPolicy(seed=8).choose_action(observation)

    assert first == repeated
    assert first != different_seed
    assert action_utility(first, observation) >= alice_profile().preferences.reservation.reservation_utility
    assert action_utility(first, observation) != pytest.approx(action_utility(different_seed, observation))


def test_tit_for_tat_matches_observed_opponent_concession_without_private_data():
    scenario = policy_scenario()
    engine = NegotiationProtocol(scenario, maximum_rounds=5, initial_turn=ALICE)
    state = engine.create_session()

    def typed_offer(offer_id, price):
        return Offer(
            offer_id=OfferId(offer_id),
            values=(
                NumericIssueValue(issue_id=PRICE, value=price),
                CategoricalIssueValue(issue_id=DELIVERY, value="express"),
            ),
        )

    actions = (
        ProposeAction(actor_id=ALICE, round_number=1, offer=typed_offer("a1", 20.0)),
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=typed_offer("b1", 90.0),
            responds_to=OfferId("a1"),
        ),
        CounterAction(
            actor_id=ALICE,
            round_number=2,
            offer=typed_offer("a2", 30.0),
            responds_to=OfferId("b1"),
        ),
        CounterAction(
            actor_id=BOB,
            round_number=2,
            offer=typed_offer("b2", 70.0),
            responds_to=OfferId("a2"),
        ),
    )
    for action in actions:
        state = engine.transition(state, action).state

    observation = AgentObservation(scenario=scenario, session=state, own_profile=alice_profile())
    policy = TitForTatPolicy()

    assert policy.target_utility(observation) == pytest.approx(0.65)
    action = policy.choose_action(observation)
    assert isinstance(action, CounterAction)
    assert action_utility(action, observation) >= 0.65 - 1e-9
    assert "BOB PRIVATE BATNA" not in observation.model_dump_json()


def test_legacy_strategy_names_are_documented_policy_aliases():
    assert NoConcessionPolicy is FixedPolicy
    assert isinstance(policy_for_strategy("aggressive"), BoulwarePolicy)
    assert isinstance(policy_for_strategy("neutral"), LinearConcessionPolicy)
    assert isinstance(policy_for_strategy("cooperative"), ConcederPolicy)
    assert isinstance(policy_for_strategy("random", seed=11), SeededRandomPolicy)

    with pytest.raises(ValueError, match="unknown strategy"):
        policy_for_strategy("mystery")


def test_policy_rejects_observation_when_it_does_not_hold_turn():
    observation = empty_observation(round_number=1, participant_id=ALICE)
    wrong_turn = observation.session.model_copy(update={"current_turn": BOB})
    observation = AgentObservation(
        scenario=observation.scenario,
        session=wrong_turn,
        own_profile=observation.own_profile,
    )

    with pytest.raises(PolicyError, match="does not hold"):
        LinearConcessionPolicy().choose_action(observation)


@pytest.mark.parametrize(
    "policy",
    [
        FixedPolicy(),
        LinearConcessionPolicy(),
        BoulwarePolicy(),
        ConcederPolicy(),
        TimeDependentAspirationPolicy(beta=2.0),
        TitForTatPolicy(),
        SeededRandomPolicy(seed=42),
    ],
)
def test_every_baseline_produces_a_typed_scenario_valid_action(policy):
    scenario = policy_scenario()
    protocol = NegotiationProtocol(scenario, maximum_rounds=3, initial_turn=ALICE)
    state = protocol.create_session()
    observation = AgentObservation(
        scenario=scenario,
        session=state,
        own_profile=alice_profile(),
    )
    action = policy.choose_action(observation)

    assert observation.scenario.validate_action(action) is action
    assert isinstance(action, ProposeAction)
    transitioned = protocol.transition(state, action).state
    assert transitioned.latest_valid_offer == action.offer

    completed = run_policy_session(
        scenario,
        {ALICE: alice_profile(), BOB: bob_profile()},
        {ALICE: policy, BOB: FixedPolicy()},
        maximum_rounds=3,
    )
    assert completed.session.phase in {
        ProtocolPhase.AGREED,
        ProtocolPhase.FAILED,
        ProtocolPhase.WITHDRAWN,
        ProtocolPhase.EXPIRED,
    }


def test_small_benchmark_is_deterministic_and_shows_different_trajectories():
    scenario = policy_scenario()
    profiles = {ALICE: alice_profile(), BOB: bob_profile()}

    fixed = run_policy_session(
        scenario,
        profiles,
        {ALICE: FixedPolicy(), BOB: FixedPolicy()},
        maximum_rounds=4,
    )
    linear = run_policy_session(
        scenario,
        profiles,
        {ALICE: LinearConcessionPolicy(), BOB: LinearConcessionPolicy()},
        maximum_rounds=4,
    )
    repeated_linear = run_policy_session(
        scenario,
        profiles,
        {ALICE: LinearConcessionPolicy(), BOB: LinearConcessionPolicy()},
        maximum_rounds=4,
    )
    mixed = run_policy_session(
        scenario,
        profiles,
        {ALICE: BoulwarePolicy(), BOB: ConcederPolicy()},
        maximum_rounds=4,
    )

    assert fixed.session.phase is ProtocolPhase.EXPIRED
    assert linear.session == repeated_linear.session
    assert linear.offer_trajectory == repeated_linear.offer_trajectory
    assert fixed.offer_trajectory != linear.offer_trajectory
    assert linear.offer_trajectory != mixed.offer_trajectory

    for result in (fixed, linear, mixed):
        for event in result.session.event_sequence:
            if isinstance(event, ActionAppliedEvent):
                assert scenario.validate_action(event.action) is event.action
