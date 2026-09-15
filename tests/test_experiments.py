from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

import llm_negotiation.experiments as experiment_module
from llm_negotiation.domain import IssueKind
from llm_negotiation.experiment_components import (
    ComponentEvaluationRecord,
    ComponentSet,
    environment_comprehension_cases,
    run_component_evaluations,
)
from llm_negotiation.experiment_dataset import scenario_dataset
from llm_negotiation.experiment_cli import main as experiment_main
from llm_negotiation.experiments import (
    BaselineKind,
    EnabledComponents,
    EpisodeRecordStatus,
    ExperimentConfiguration,
    ExperimentEpisodeRecord,
    ExperimentModelSpec,
    aggregate_records,
    one_at_a_time_ablation_configurations,
    paired_comparisons,
    rebuild_ablation_table,
    rebuild_aggregates,
    records_from_jsonl,
    run_experiment,
)
from llm_negotiation.llm_policy import OpponentHypothesis
from llm_negotiation.llm import LLMProvider
from llm_negotiation.orchestration import EpisodeResult


FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def experiment_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment-harness")
    configuration = ExperimentConfiguration(
        experiment_id="test-offline-harness",
        scenario_ids=(
            "price-feasible-symmetric",
            "three-principal-resource-allocation",
        ),
        baselines=tuple(BaselineKind),
        seeds=(11,),
        rounds=6,
    )
    first = run_experiment(configuration, root / "first", timestamp=FIXED_TIME)
    second = run_experiment(configuration, root / "second", timestamp=FIXED_TIME)
    return root, configuration, first, second


def test_versioned_configuration_hash_is_stable_and_live_models_are_explicit():
    configuration = ExperimentConfiguration(
        experiment_id="hash-test",
        scenario_ids=("price-feasible-symmetric",),
        baselines=(BaselineKind.DETERMINISTIC,),
        seeds=(1, 2),
        rounds=5,
    )
    restored = ExperimentConfiguration.model_validate_json(
        configuration.model_dump_json()
    )
    assert restored == configuration
    assert restored.configuration_hash == configuration.configuration_hash
    assert len(configuration.configuration_hash) == 64
    live = ExperimentConfiguration(
        experiment_id="live-must-be-explicit",
        scenario_ids=("price-feasible-symmetric",),
        baselines=(BaselineKind.SINGLE_LLM,),
        models=(
            ExperimentModelSpec(
                provider=LLMProvider.OPENAI,
                model="live-model",
                temperature=0.2,
            ),
        ),
    )
    with pytest.raises(
        experiment_module.ExperimentError, match="explicit live command"
    ):
        run_experiment(
            live,
            Path("unused-live-artifacts"),
            timestamp=FIXED_TIME,
        )


def test_unknown_configuration_and_artifact_versions_are_rejected(experiment_run):
    _root, configuration, result, _second = experiment_run
    config_payload = configuration.model_dump(mode="python")
    config_payload["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        ExperimentConfiguration.model_validate(config_payload)
    config_payload = configuration.model_dump(mode="python")
    config_payload["dataset_version"] = "2.0"
    with pytest.raises(ValueError, match="dataset_version"):
        ExperimentConfiguration.model_validate(config_payload)

    episode_payload = result.records[0].model_dump(mode="python")
    episode_payload["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        ExperimentEpisodeRecord.model_validate(episode_payload)
    component_payload = run_component_evaluations(EnabledComponents())[0].model_dump(
        mode="python"
    )
    component_payload["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        ComponentEvaluationRecord.model_validate(component_payload)


def test_multiple_model_settings_remain_distinct_in_statistics(tmp_path):
    configuration = ExperimentConfiguration(
        experiment_id="multi-model-settings",
        scenario_ids=("price-feasible-symmetric",),
        baselines=(BaselineKind.SINGLE_LLM,),
        models=(
            ExperimentModelSpec(model="fake-a", temperature=0.0),
            ExperimentModelSpec(model="fake-b", temperature=0.7),
        ),
        seeds=(4,),
        rounds=4,
    )
    result = run_experiment(configuration, tmp_path, timestamp=FIXED_TIME)
    assert len(result.records) == 2
    assert {(item.model, item.temperature) for item in result.records} == {
        ("fake-a", 0.0),
        ("fake-b", 0.7),
    }
    aggregates = aggregate_records(result.records)
    assert len(aggregates) == 2
    assert {(item.model, item.temperature) for item in aggregates} == {
        ("fake-a", 0.0),
        ("fake-b", 0.7),
    }
    comparisons = paired_comparisons(result.records)
    assert len(comparisons) == 5
    assert all(item.paired_seeded_cases == 1 for item in comparisons)

    with pytest.raises(ValueError, match="models must be unique"):
        ExperimentConfiguration(
            experiment_id="duplicate-model-settings",
            scenario_ids=("price-feasible-symmetric",),
            baselines=(BaselineKind.SINGLE_LLM,),
            models=(ExperimentModelSpec(), ExperimentModelSpec()),
        )


def test_scenario_dataset_covers_every_required_case_type():
    cases = scenario_dataset()
    tags = {tag for case in cases for tag in case.tags}
    assert {"feasible", "infeasible", "symmetric", "asymmetric"} <= tags
    assert {"price_only", "multi_issue", "bilateral", "multiparty"} <= tags
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.profiles for case in cases)


def test_component_sets_are_labelled_separate_and_confidence_can_be_hidden():
    visible = run_component_evaluations(EnabledComponents())
    hidden = run_component_evaluations(
        EnabledComponents(confidence_visibility=False)
    )
    assert {item.evaluation_set for item in visible} == set(ComponentSet)
    assert all(item.label and item.prediction for item in visible)
    assert all(item.correct for item in visible)
    assert all(item.confidence is None for item in hidden)
    calibration = [
        item for item in visible
        if item.evaluation_set is ComponentSet.OPPONENT_CALIBRATION
    ]
    assert calibration and all(item.brier_score is not None for item in calibration)

    spec = ExperimentModelSpec()
    policy = experiment_module.ExperimentLLMPolicy(
        experiment_module._offline_fake_client(spec),
        experiment_module._model_configuration(spec),
        confidence_visible=False,
    )
    hypothesis = OpponentHypothesis(
        decision_rationale="labelled test",
        hypothesis="uncertain concession",
        confidence=0.91,
    )
    assert policy._visible_beliefs((hypothesis,))[0].confidence == 0.5


def test_environment_comprehension_labels_are_fixed_and_unambiguous():
    labels = {
        case.scenario_id: (
            case.expected_participant_count,
            case.expected_issue_kinds,
            case.expected_feasible,
        )
        for case in environment_comprehension_cases()
    }
    assert labels == {
        "price-feasible-symmetric": (2, (IssueKind.NUMERIC,), True),
        "price-infeasible": (2, (IssueKind.NUMERIC,), False),
        "price-feasible-asymmetric": (2, (IssueKind.NUMERIC,), True),
        "multi-issue-asymmetric": (
            2,
            (
                IssueKind.NUMERIC,
                IssueKind.CATEGORICAL,
            ),
            True,
        ),
        "three-principal-resource-allocation": (
            3,
            (
                IssueKind.NUMERIC,
                IssueKind.CATEGORICAL,
            ),
            True,
        ),
    }


def test_all_baselines_run_offline_and_raw_results_are_preserved(experiment_run):
    _root, configuration, result, second = experiment_run
    records = records_from_jsonl(Path(result.episode_jsonl))
    assert records == result.records
    assert len(records) == 10
    assert {item.baseline for item in records} == set(BaselineKind)
    assert sum(item.status is EpisodeRecordStatus.SUCCESS for item in records) == 9
    assert sum(item.status is EpisodeRecordStatus.NOT_APPLICABLE for item in records) == 1
    assert not any(item.status is EpisodeRecordStatus.FAILED for item in records)
    for item in records:
        assert item.configuration_hash == configuration.configuration_hash
        assert item.git_commit
        assert item.timestamp_utc == FIXED_TIME
        assert item.partition == "evaluation"
        if item.status is EpisodeRecordStatus.SUCCESS:
            assert EpisodeResult.model_validate_json(
                item.episode_result.model_dump_json()
            ) == item.episode_result

    # Fixed timestamp, seed, fake model and clock make raw and aggregate artifacts byte stable.
    assert Path(result.episode_jsonl).read_bytes() == Path(second.episode_jsonl).read_bytes()
    assert Path(result.aggregate_csv).read_bytes() == Path(second.aggregate_csv).read_bytes()


def test_aggregates_ablation_tables_statistics_and_plots_rebuild_from_raw(experiment_run):
    root, _configuration, result, _second = experiment_run
    rebuilt_aggregate = root / "rebuilt-aggregate.csv"
    rebuilt_ablation = root / "rebuilt-ablation.csv"
    rebuild_aggregates(Path(result.episode_jsonl), rebuilt_aggregate)
    rebuild_ablation_table(Path(result.episode_jsonl), rebuilt_ablation)
    assert rebuilt_aggregate.read_bytes() == Path(result.aggregate_csv).read_bytes()
    assert rebuilt_ablation.read_bytes() == Path(result.ablation_csv).read_bytes()
    with rebuilt_aggregate.open(newline="", encoding="utf-8") as handle:
        aggregate_row = next(csv.DictReader(handle))
    assert aggregate_row["configuration_hash"]
    assert aggregate_row["git_commit"]
    assert aggregate_row["timestamp_utc"] == FIXED_TIME.isoformat()
    assert aggregate_row["seeds"] == "11"

    aggregates = aggregate_records(result.records)
    assert aggregates
    assert all(
        interval.method == "normal_approximation_95_percent"
        for aggregate in aggregates
        for interval in aggregate.intervals.values()
    )
    paired = paired_comparisons(result.records)
    assert paired
    assert all(item.paired_seeded_cases >= 1 for item in paired)
    assert {Path(item).name for item in result.plot_files} == {
        "agreement.svg",
        "utility_welfare.svg",
        "fairness.svg",
        "rounds.svg",
        "invalid_actions.svg",
        "latency.svg",
        "usage_cost.svg",
        "learning_curves.svg",
    }
    assert all(Path(item).read_text(encoding="utf-8").startswith("<svg") for item in result.plot_files)


def test_statistics_never_mix_distinct_ablation_configurations(experiment_run):
    _root, _configuration, result, _second = experiment_run
    alternate_hash = "a" * 64
    alternate_records = tuple(
        ExperimentEpisodeRecord.model_validate(
            {**item.model_dump(mode="python"), "configuration_hash": alternate_hash}
        )
        for item in result.records
    )
    combined = result.records + alternate_records

    aggregates = aggregate_records(combined)
    assert {item.configuration_hash for item in aggregates} == {
        result.records[0].configuration_hash,
        alternate_hash,
    }
    assert len(aggregates) == 2 * len(aggregate_records(result.records))

    original_pairs = paired_comparisons(result.records)
    combined_pairs = paired_comparisons(combined)
    assert len(combined_pairs) == 2 * len(original_pairs)
    assert {item.configuration_hash for item in combined_pairs} == {
        result.records[0].configuration_hash,
        alternate_hash,
    }


def test_learning_training_partition_never_enters_evaluation_jsonl(experiment_run):
    _root, _configuration, result, _second = experiment_run
    records = records_from_jsonl(Path(result.episode_jsonl))
    assert {item.partition for item in records} == {"evaluation"}
    learning = Path(result.learning_curve_json).read_text(encoding="utf-8")
    assert '"curve"' in learning
    assert "learning_curve_training.json" not in Path(result.episode_jsonl).read_text(
        encoding="utf-8"
    )


def test_every_requested_ablation_has_an_explicit_configuration():
    base = ExperimentConfiguration(
        experiment_id="ablation-base",
        scenario_ids=("three-principal-resource-allocation",),
        baselines=(BaselineKind.HETEROGENEOUS_TEAM,),
    )
    configurations = one_at_a_time_ablation_configurations(base)
    assert len({item.configuration_hash for item in configurations}) == len(configurations)
    by_id = {item.experiment_id: item for item in configurations}
    for name in (
        "memory",
        "theory_of_mind",
        "verifier",
        "mediator",
        "communication",
        "confidence_visibility",
    ):
        assert f"ablation-base-ablate-{name}" in by_id
    assert {
        item.enabled_components.team_size for item in configurations
        if item.enabled_components.team_size is not None
    } == {2, 3, 4}
    assert {item.enabled_components.deliberation_depth for item in configurations} >= {1, 2}


def test_component_toggles_are_wired_into_real_episode_dependencies(tmp_path):
    components = EnabledComponents(
        memory=False,
        theory_of_mind=False,
        verifier=False,
        mediator=True,
        communication=False,
        team_size=3,
        deliberation_depth=2,
        confidence_visibility=False,
    )
    configuration = ExperimentConfiguration(
        experiment_id="wired-ablation",
        scenario_ids=("three-principal-resource-allocation",),
        baselines=(BaselineKind.HETEROGENEOUS_TEAM,),
        seeds=(5,),
        rounds=8,
        enabled_components=components,
    )
    result = run_experiment(configuration, tmp_path, timestamp=FIXED_TIME)
    assert len(result.records) == 1
    record = result.records[0]
    assert record.status is EpisodeRecordStatus.SUCCESS
    assert record.components == components
    assert record.episode_result.verification_log
    assert all(
        entry.results[0].verdict.value == "skipped"
        for entry in record.episode_result.verification_log
    )


def test_failed_episode_is_recorded_and_not_silently_dropped(tmp_path, monkeypatch):
    configuration = ExperimentConfiguration(
        experiment_id="failure-recording",
        scenario_ids=("price-feasible-symmetric",),
        baselines=(BaselineKind.DETERMINISTIC,),
        seeds=(9,),
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("private provider detail must not be persisted")

    monkeypatch.setattr(experiment_module, "_run_episode", fail)
    result = run_experiment(configuration, tmp_path, timestamp=FIXED_TIME)
    record = result.records[0]
    assert record.status is EpisodeRecordStatus.FAILED
    assert record.failure_code == "RuntimeError"
    assert "private provider detail" not in record.failure_message
    assert len(records_from_jsonl(Path(result.episode_jsonl))) == 1
    with Path(result.aggregate_csv).open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["failed_episodes"] == "1"


def test_one_offline_command_creates_artifacts_and_rebuild_command_works(
    tmp_path, capsys
):
    configuration = ExperimentConfiguration(
        experiment_id="cli-offline",
        scenario_ids=("price-feasible-symmetric",),
        baselines=(BaselineKind.DETERMINISTIC,),
        seeds=(3,),
        rounds=4,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(configuration.model_dump_json(indent=2), encoding="utf-8")
    output = tmp_path / "run"
    assert experiment_main(
        (
            "offline",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--timestamp",
            FIXED_TIME.isoformat(),
        )
    ) == 0
    assert (output / "episodes.jsonl").is_file()
    rebuilt_aggregate = tmp_path / "aggregate-rebuilt.csv"
    rebuilt_ablation = tmp_path / "ablation-rebuilt.csv"
    assert experiment_main(
        (
            "rebuild",
            str(output / "episodes.jsonl"),
            "--aggregate",
            str(rebuilt_aggregate),
            "--ablation",
            str(rebuilt_ablation),
        )
    ) == 0
    assert rebuilt_aggregate.is_file() and rebuilt_ablation.is_file()
    assert "Configuration hash:" in capsys.readouterr().out
