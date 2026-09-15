"""Command-line entry point for offline experiment execution and artifact rebuilding."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .experiments import (
    ExperimentConfiguration,
    rebuild_ablation_table,
    rebuild_aggregates,
    run_ablation_suite,
    run_experiment,
    small_offline_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-negotiation-experiment",
        description="Run deterministic offline negotiation experiments.",
    )
    subparsers = parser.add_subparsers(dest="command")
    offline = subparsers.add_parser("offline", help="run the small network-free benchmark")
    offline.add_argument("--output", type=Path, default=Path("artifacts/offline-benchmark"))
    offline.add_argument("--config", type=Path, help="optional versioned JSON configuration")
    offline.add_argument(
        "--timestamp",
        help="timezone-aware ISO timestamp for byte-reproducible provenance",
    )

    live = subparsers.add_parser(
        "live", help="explicitly run a provider-backed configuration (may incur cost)"
    )
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--config", type=Path, required=True)
    live.add_argument("--timestamp", help="timezone-aware ISO provenance timestamp")

    ablate = subparsers.add_parser(
        "ablate", help="explicitly run the one-factor offline ablation matrix"
    )
    ablate.add_argument("--output", type=Path, required=True)
    ablate.add_argument("--config", type=Path, required=True)
    ablate.add_argument("--timestamp", help="timezone-aware ISO provenance timestamp")

    rebuild = subparsers.add_parser(
        "rebuild", help="rebuild aggregate and ablation CSV files from raw JSONL"
    )
    rebuild.add_argument("episodes", type=Path)
    rebuild.add_argument("--aggregate", type=Path, required=True)
    rebuild.add_argument("--ablation", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        arguments = parser.parse_args(["offline"])
    command = arguments.command
    if command == "rebuild":
        rebuild_aggregates(arguments.episodes, arguments.aggregate)
        rebuild_ablation_table(arguments.episodes, arguments.ablation)
        print(f"Rebuilt aggregates: {arguments.aggregate}")
        print(f"Rebuilt ablations: {arguments.ablation}")
        return 0
    configuration = (
        ExperimentConfiguration.model_validate_json(
            arguments.config.read_text(encoding="utf-8")
        )
        if getattr(arguments, "config", None)
        else small_offline_configuration()
    )
    timestamp = (
        datetime.fromisoformat(arguments.timestamp)
        if getattr(arguments, "timestamp", None)
        else None
    )
    if command == "ablate":
        result = run_ablation_suite(
            configuration,
            arguments.output,
            timestamp=timestamp,
        )
        print(f"Ablation episodes: {result.episode_jsonl}")
        print(f"Ablation table: {result.ablation_csv}")
        print(f"Configurations: {len(result.configuration_hashes)}")
        return 0
    result = run_experiment(
        configuration,
        arguments.output,
        timestamp=timestamp,
        allow_live=command == "live",
    )
    print(f"Configuration hash: {configuration.configuration_hash}")
    print(f"Episode records: {result.episode_jsonl}")
    print(f"Aggregate metrics: {result.aggregate_csv}")
    print(f"Ablation table: {result.ablation_csv}")
    print(f"Plots: {len(result.plot_files)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
