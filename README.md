# LLM-Based Multi-Agent Negotiation System

This project is a deterministic, rule-based simulation of negotiation between buyer, seller,
mediator, judge, and learning roles. It does not currently call an LLM or an external API.

## Features

- Buyer and seller strategies: aggressive, neutral, and cooperative
- Multi-round negotiation with a mediator fallback
- Outcome validity and fairness evaluation
- Deterministic rewards and repeated simulations
- Reproducible non-LLM baseline policies for multi-issue experiments
- Command-line interface and optional Streamlit UI

## Domain model

The `llm_negotiation.domain` package defines validated participants, issues, offers, private utility
profiles, actions, agreements, and terminal outcomes. Public scenarios contain participants and
issue definitions but never private preferences. Numeric utilities are linearly normalized, while
categorical utilities use participant-specific scores in `[0, 1]`. Issue weights must sum to `1.0`
within an absolute tolerance of `1e-6`.

The current deterministic price negotiation remains available through a compatibility adapter while
the orchestration layer is migrated incrementally.

The `llm_negotiation.protocol` module defines the authoritative negotiation lifecycle. It validates
typed actions against the current phase, participant turn, outstanding offer, and round deadline;
every successful transition emits a replayable event. Natural-language messages never change
protocol meaning.

Protocol phases are `created`, `active`, `mediation`, `agreed`, `failed`, `withdrawn`, and `expired`.
A round is one complete rotation through the configured participant order, anchored to the initial
participant. Mediation is entered by an explicit `request_mediation` action; no mediator policy is
implemented yet. A trusted coordinator may mark the feasible region `impossible` during session
creation, allowing deterministic failure without adding private preference data to public state.

## Baseline negotiation policies

Policies implement the typed `NegotiationPolicy` interface. They receive an immutable, bounded
`MemorySnapshot`, not the full protocol session. The snapshot contains public scenario facts and
only that agent's own preferences; opponent preferences and BATNA data are never included.

The following deterministic baselines are available in `llm_negotiation.policies`:

- `FixedPolicy` / `NoConcessionPolicy`: holds aspiration at the maximum attainable utility.
- `LinearConcessionPolicy`: concedes linearly over the available rounds.
- `BoulwarePolicy`: concedes late with concession fraction `t^4`.
- `ConcederPolicy`: concedes early with concession fraction `t^(1/4)`.
- `TimeDependentAspirationPolicy(beta)`: uses `c(t) = t^(1/beta)` and aspiration
  `r + (u_max - r) * (1 - c(t))`, where `r` is reservation utility.
- `TitForTatPolicy`: reduces its previous aspiration by the opponent's latest positive concession,
  measured using only its own utility model and public offers.
- `SeededRandomPolicy(seed)`: chooses a reproducible pseudo-random aspiration between reservation
  and maximum utility for each memory snapshot.

Multi-issue proposals move numeric and categorical issue values toward a utility target while
remaining inside public issue bounds. A policy never knowingly accepts or proposes below its own
reservation utility. The compatibility factory `policy_for_strategy()` maps `aggressive` to
Boulware, `neutral` to linear concession, and `cooperative` to the early conceder policy; the legacy
CLI remains available during migration.

`llm_negotiation.benchmark.run_policy_session()` provides a small deterministic harness for strategy
comparisons through the authoritative protocol. Its trajectories and outcomes are baselines for
experiments, not evidence that any strategy is universally best.

## Typed communication

`llm_negotiation.communication` provides immutable `MessageEnvelope` records and an in-memory,
append-only `MessageBus`. Envelopes use a fixed message-type enum, typed participant and reference
identifiers, timezone-aware timestamps, deterministic sequence numbers, and correlation identifiers.
They cannot contain arbitrary metadata or domain objects.

The bus supports four visibility routes:

- `public`: delivered to every participant and included in the public transcript.
- `direct_private`: readable only by its sender, explicit recipients, and authorized auditors.
- `mediator_only`: routed only to configured mediators, with sender and audit access retained.
- `system_audit`: emitted only by the system and visible only to authorized audit readers.

An agent inbox combines its public and addressed messages; it never includes another agent's private
traffic. Audit access must be explicitly configured. Protocol events and related messages can share
a correlation identifier, and `EpisodeRecord` preserves both ordered streams for deterministic
serialization and replay. Sending persuasive content or a `mediation_request` message does not
execute a protocol action, accept an offer, withdraw, or enter mediation.

## Agent memory

`llm_negotiation.memory` owns one isolated memory stack per participant. A full `AgentObservation`
and that participant's authorized message inbox are ingestion inputs only; neither the full
`NegotiationSession` nor the message-bus audit log appears in the policy-facing snapshot.

- `LongTermMemory` stores stable public identity, role, persona, goals, protocol rules, public issue
  descriptions, strategy guidance, learned strategy facts, and only the owner's private preferences.
- `WorkingMemory` stores authoritative legal actions, the outstanding offer, current round and
  deadline, bounded visible-message digests, recent public offers, owner-specific utility estimates,
  terminal outcome, ingestion cursors, and a concise active plan.
- `EpisodicMemory` stores a bounded sequence of observations, owner actions, observed public actions,
  received messages, outcomes, and deterministic factual summaries.

`MemoryLimits` independently bounds episodic events, episodic characters/tokens, working messages,
recent offers, and learned strategy facts. `DeterministicMemorySummarizer` performs non-LLM
compaction; current offer, recent offer utilities, phase, legal actions, round/deadline, and outcome
remain in working memory even when older episodes are summarized. Snapshots persist their limits,
history digest, and ingestion cursors under schema version `1.0`, so they can be saved, loaded, and
resumed without re-ingesting old events or messages.
Starting a new negotiation clears working and episodic memory while optionally retaining learned
long-term strategy data. Raw protocol events and the append-only communication audit remain external
for exact research replay.

## Setup

Python 3.9 or newer is required. From the repository root, create a virtual environment and install
the package in editable mode with its test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run the CLI

Use the installed console command:

```powershell
llm-negotiation
```

Or run the package module directly:

```powershell
python -m llm_negotiation.cli
```

For all options, run `llm-negotiation --help`.

## Verify

```powershell
python -c "import llm_negotiation; print(llm_negotiation.__file__)"
pytest -q
llm-negotiation --rounds 2
python -m llm_negotiation.cli --rounds 2
```

## Optional Streamlit UI

Streamlit is deliberately separate from the core runtime and test dependencies:

```powershell
python -m pip install -e ".[ui]"
streamlit run src/llm_negotiation/ui_streamlit.py
```

