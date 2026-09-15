from __future__ import annotations

import json

import pytest

from llm_negotiation.domain.legacy import BUYER_ID
from llm_negotiation.learning import (
    BanditAlgorithm,
    CompletedEpisodeOutcome,
    ContextualBanditLearner,
    EpsilonGreedyLearner,
    LearnerError,
    LearnerFrozenError,
    LearnerMode,
    LearnerPersistenceError,
    RewardConfiguration,
    StrategyLearner,
    UCBLearner,
    build_strategy_context,
    compose_reward,
)
from llm_negotiation.learning_experiment import (
    EXPERIMENT_STRATEGIES,
    _run_episode,
    _scenario_and_profiles,
    run_learning_curve_experiment,
)
from llm_negotiation.protocol import ProtocolPhase


@pytest.fixture(scope="module")
def learning_setup():
    scenario, profiles = _scenario_and_profiles()
    context = build_strategy_context(scenario, profiles[BUYER_ID])
    outcomes = {}
    for strategy in EXPERIMENT_STRATEGIES:
        result = _run_episode(strategy, scenario, profiles)
        outcomes[strategy] = CompletedEpisodeOutcome(
            terminal_phase=result.final_session.phase,
            evaluation=result.metrics,
        )
    return scenario, profiles, context, outcomes


def train_selection(learner, context, outcomes):
    strategy = learner.select_strategy(context)
    reward = learner.observe_outcome(context, strategy, outcomes[strategy])
    return strategy, reward


def test_context_uses_public_metadata_and_only_the_owners_profile(learning_setup):
    _scenario, profiles, context, _outcomes = learning_setup
    payload = context.model_dump(mode="json")
    assert payload["owner_id"] == "buyer"
    assert payload["participant_count"] == 2
    assert set(payload) == {
        "owner_id",
        "owner_role",
        "participant_count",
        "numeric_issue_count",
        "categorical_issue_count",
        "reservation_bucket",
        "batna_bucket",
        "persona_present",
        "issue_features",
    }
    serialized = context.model_dump_json()
    assert "seller" not in serialized
    assert "category_utilities" not in serialized
    assert profiles[BUYER_ID].preferences.participant_id == context.owner_id


def test_reward_formula_uses_objective_evaluation_components(learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    config = RewardConfiguration(
        own_utility_weight=0.4,
        agreement_weight=0.3,
        efficiency_weight=0.2,
        fairness_weight=0.1,
        cost_weight=0.05,
    )
    result = compose_reward(outcomes["linear"].evaluation, context.owner_id, config)
    expected = (
        0.4 * result.own_utility
        + 0.3 * result.agreement
        + 0.2 * result.efficiency
        + 0.1 * result.fairness
        - 0.05 * result.cost
    )
    assert result.reward == pytest.approx(expected)


def test_seeded_epsilon_greedy_exploration_is_reproducible(learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    left = EpsilonGreedyLearner(EXPERIMENT_STRATEGIES, epsilon=1.0, seed=91)
    right = EpsilonGreedyLearner(EXPERIMENT_STRATEGIES, epsilon=1.0, seed=91)
    left_choices = tuple(train_selection(left, context, outcomes)[0] for _ in range(12))
    right_choices = tuple(train_selection(right, context, outcomes)[0] for _ in range(12))
    assert left_choices == right_choices
    assert len(set(left_choices)) > 1


def test_ucb_explores_then_exploits_observed_rewards(learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    learner = UCBLearner(EXPERIMENT_STRATEGIES, exploration=0.0, seed=3)
    initial = tuple(train_selection(learner, context, outcomes)[0] for _ in range(4))
    assert initial == EXPERIMENT_STRATEGIES
    assert learner.select_strategy(context) == "boulware"
    learner.observe_outcome(context, "boulware", outcomes["boulware"])
    statistics = learner.state.contexts[0]
    assert next(item for item in statistics.arms if item.strategy == "boulware").observations == 2


def test_observation_changes_policy_state_and_must_match_a_selection(learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    learner = EpsilonGreedyLearner(EXPERIMENT_STRATEGIES, epsilon=0.0)
    with pytest.raises(LearnerError, match="not selected"):
        learner.observe_outcome(context, "fixed", outcomes["fixed"])
    selected = learner.select_strategy(context)
    with pytest.raises(LearnerError, match="completed outcome"):
        learner.select_strategy(context)
    with pytest.raises(LearnerError, match="unresolved training episode"):
        learner.set_mode(LearnerMode.EVALUATION)
    before = learner.state.model_dump_json()
    learner.observe_outcome(context, selected, outcomes[selected])
    assert learner.state.model_dump_json() != before
    assert learner.state.contexts[0].arms[0].observations == 1


def test_saved_and_reloaded_learner_has_identical_future_behavior(tmp_path, learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    learner = EpsilonGreedyLearner(EXPERIMENT_STRATEGIES, epsilon=0.35, seed=44)
    for _ in range(8):
        train_selection(learner, context, outcomes)
    path = tmp_path / "learner.json"
    learner.save(path)
    loaded = ContextualBanditLearner.load(path)
    assert isinstance(loaded, EpsilonGreedyLearner)
    assert loaded.state == learner.state
    assert loaded.select_strategy(context) == learner.select_strategy(context)

    ucb = UCBLearner(EXPERIMENT_STRATEGIES, exploration=0.75, seed=45)
    for _ in EXPERIMENT_STRATEGIES:
        train_selection(ucb, context, outcomes)
    ucb_path = tmp_path / "ucb.json"
    ucb.save(ucb_path)
    loaded_ucb = ContextualBanditLearner.load(ucb_path)
    assert isinstance(loaded_ucb, UCBLearner)
    assert loaded_ucb.state == ucb.state
    assert loaded_ucb.select_strategy(context) == ucb.select_strategy(context)
    assert not tuple(tmp_path.glob("*.tmp"))


def test_corrupted_or_wrong_algorithm_state_fails_clearly(tmp_path, learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text("{not-json", encoding="utf-8")
    with pytest.raises(LearnerPersistenceError, match="could not load valid"):
        ContextualBanditLearner.load(corrupted)

    learner = EpsilonGreedyLearner(EXPERIMENT_STRATEGIES)
    train_selection(learner, context, outcomes)
    valid = tmp_path / "epsilon.json"
    learner.save(valid)
    with pytest.raises(LearnerPersistenceError, match="cannot load"):
        UCBLearner.load(valid)

    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["schema_version"] = "999"
    valid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LearnerPersistenceError, match="could not load valid"):
        ContextualBanditLearner.load(valid)


def test_evaluation_mode_is_frozen_and_cannot_update(learning_setup):
    _scenario, _profiles, context, outcomes = learning_setup
    learner = UCBLearner(EXPERIMENT_STRATEGIES, exploration=100.0)
    for _ in range(len(EXPERIMENT_STRATEGIES)):
        train_selection(learner, context, outcomes)
    # One extra UCB selection gives a successful arm more observations. A very large
    # exploration bonus would select an under-observed arm if evaluation were not greedy.
    train_selection(learner, context, outcomes)
    learner.set_mode(LearnerMode.EVALUATION)
    before = learner.state.model_dump_json()
    selected = learner.select_strategy(context)
    best_mean = max(item.mean_reward for item in learner.state.contexts[0].arms)
    assert next(
        item.mean_reward
        for item in learner.state.contexts[0].arms
        if item.strategy == selected
    ) == best_mean
    assert learner.state.model_dump_json() == before
    with pytest.raises(LearnerFrozenError, match="cannot observe"):
        learner.observe_outcome(context, selected, outcomes[selected])
    assert learner.state.model_dump_json() == before


def test_nonterminal_evidence_cannot_be_used_for_learning(learning_setup):
    _scenario, _profiles, _context, outcomes = learning_setup
    with pytest.raises(ValueError, match="terminal episode"):
        CompletedEpisodeOutcome(
            terminal_phase=ProtocolPhase.ACTIVE,
            evaluation=outcomes["fixed"].evaluation,
        )


def test_learning_curve_is_seeded_and_reports_real_state_changes():
    first = run_learning_curve_experiment(episodes=12, seed=17)
    second = run_learning_curve_experiment(episodes=12, seed=17)
    assert first == second
    assert first.curve[0].selected_strategy == "fixed"
    best_mean = max(item.mean_reward for item in first.arms)
    best_strategies = {
        item.strategy for item in first.arms if item.mean_reward == best_mean
    }
    assert first.frozen_strategy in best_strategies
    assert first.frozen_strategy != first.curve[0].selected_strategy
    assert first.curve[-1].cumulative_mean_reward > first.curve[0].reward
    assert sum(item.observations for item in first.arms) == 12
    assert any(item.agreement for item in first.curve)
    assert any(not item.agreement for item in first.curve)
    assert isinstance(
        EpsilonGreedyLearner(EXPERIMENT_STRATEGIES), StrategyLearner
    )
    assert first.context.owner_id == BUYER_ID
    assert first.fixed_opponent_strategy == "linear"
