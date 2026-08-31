# LLM-Based Multi-Agent Negotiation System

This project is a deterministic, rule-based simulation of negotiation between buyer, seller,
mediator, judge, and learning roles. It includes an optional provider-independent LLM boundary, but
no negotiation policy calls an LLM yet and the default test suite never accesses the network.

## Features

- Buyer and seller strategies: aggressive, neutral, and cooperative
- Multi-round negotiation with a mediator fallback
- Outcome validity and fairness evaluation
- Deterministic rewards and repeated simulations
- Reproducible non-LLM baseline policies for multi-issue experiments
- Command-line interface and optional Streamlit UI
- Typed LLM client contract with deterministic fake and optional OpenAI Responses API adapter

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

## Provider-independent LLM boundary

`llm_negotiation.llm` defines immutable request, response, usage, error, retry, and model
configuration types plus the `LLMClient` protocol. Model selection, timeout, and retry settings live
in `ModelConfiguration`; they are deliberately separate from agent strategy, persona, memory, and
private preferences.

`FakeLLMClient` supports queued results and deterministic request rules. It is the default testing
client and does not retain request content unless a test explicitly enables `record_requests`.
`OpenAILLMClient` is an optional adapter using the current Responses API. The OpenAI SDK is lazily
imported, SDK-level retries are disabled, and the gateway applies the configured bounded exponential
backoff. Provider exception bodies and prompts are not logged. The adapter sends `store=False` and
returns the same `LLMResponse` application type as the fake.

Install the optional provider only when needed:

```powershell
python -m pip install -e ".[openai]"
```

If a local environment file matches your development workflow, create the ignored template copy
with `Copy-Item .env.example .env`, then populate it through a trusted editor or secret manager and
configure your shell or IDE to export those variables. This project does not load dotenv files
automatically. Never commit the populated file and never paste a credential into source, tests,
documentation, command history, or logs.

The supported variables are:

- `OPENAI_API_KEY` (required to construct the real adapter)
- `OPENAI_MODEL` (required by `load_openai_model_configuration()`)
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_MAX_OUTPUT_TOKENS`
- `OPENAI_MAX_RETRIES`
- `OPENAI_RETRY_INITIAL_BACKOFF_SECONDS`
- `OPENAI_RETRY_BACKOFF_MULTIPLIER`
- `OPENAI_RETRY_MAX_BACKOFF_SECONDS`
- `RUN_LIVE_LLM_TESTS` (test-only opt-in; leave unset for normal offline verification)

Missing credentials and model selection raise actionable `LLMConfigurationError` exceptions before
any request. The ordinary suite uses mock SDK objects and has an automatic network guard:

```powershell
pytest -q
pytest -q -m "not live_api"
```

The real-provider contract test is intentionally opt-in. Only after installing `.[openai]` and
supplying the required variables from a secure environment, enable it with:

```powershell
$env:RUN_LIVE_LLM_TESTS = "1"
pytest -q -m live_api
```

If the flag is absent, the live module is skipped before a client is constructed. No live API call
is part of normal installation or verification.

## Staged LLM negotiation policy

`llm_negotiation.llm_policy.LLMNegotiationPolicy` implements the same `NegotiationPolicy`
interface as the deterministic baselines. It consumes only a bounded, participant-owned
`MemorySnapshot`; it never receives or mutates `NegotiationSession`. Its cognitive pipeline is:

1. Build a typed agent-visible observation from public scenario data, the owner's preferences,
   legal actions, bounded memory, and messages already authorized by `MessageBus`.
2. Summarize the objective and constraints.
3. Form or update a concise, uncertain opponent hypothesis using visible behavior only.
4. Choose a negotiation plan.
5. Generate exactly one structured action with concise decision rationale and evidence.
6. Ground the action through the domain schema and visible protocol constraints.

Prompts live in the versioned `llm_negotiation.llm_prompts.negotiation_policy_v1` module. Stage
responses reject extra fields and prose outside strict JSON. The action response contains a typed
`NegotiationAction` plus bounded `decision_rationale` and `evidence`; no stage requests or persists
hidden chain-of-thought.

Grounding revalidates the actor, round, recipients, legal action kind, outstanding-offer reference,
required unique offer identifier, issue completeness and bounds, and the owner's reservation
utility. The policy cannot transition protocol state itself. Malformed or context-illegal output is
given at most one structured repair attempt for the entire decision. A provider failure or failed
repair invokes the deterministic linear-concession baseline; if that cannot act safely, the policy
withdraws when withdrawal is legal. Per-stage call attempts, latency, token usage, repair status,
and concise audit artifacts are exposed through `last_trace` and `cumulative_telemetry` without
retaining prompts.

## Opponent modelling

`llm_negotiation.opponent` treats Theory of Mind as an explicit optional component. An
`OpponentBeliefState` keeps observable facts separate from uncertain goal, reservation-range,
strategy, next-action, and information-need hypotheses. Every snapshot records its confidence,
supporting observable event identifiers, unknowns, and last update round. It never contains an
opponent's private preference object.

`OpponentModelConfiguration` supports `disabled`, deterministic `heuristic`, and optional `llm`
modes. The heuristic baseline updates numeric issue directions with deterministic Bayesian-style
likelihoods, tracks legal public bounds, observes categorical choices, and classifies concession
trajectories. The LLM modeller uses the provider-independent gateway and a strict versioned JSON
schema; invalid or out-of-bounds output falls back to the heuristic baseline. Planning receives
beliefs explicitly as defeasible inferences and may ignore them below the configured confidence
threshold.

Policy-produced beliefs are stored as typed episodic-memory events. Simulation ground truth is
accepted only by the separate post-episode `calibrate_belief` evaluator, which requires a matching
terminal `NegotiationSession` and reports reservation
interval coverage/width, probability assigned to the actual strategy, goal Brier score, and
confidence error. Ground truth is never passed to either modeller or the policy decision path.

## Action verification

`llm_negotiation.verification` places a side-effect-free verification gate before protocol
transition. `DeterministicActionVerifier` parses the typed schema and preflights the action through
the authoritative pure protocol transition, then checks actor ownership, turn/phase and deadline,
offer completeness and public issue bounds, stale or fabricated offer references, the proposer's
own reservation utility, authorized message recipients, and conservative prompt-injection patterns.
It never evaluates against an opponent's private profile.

`VerificationResult` records a pass, rejection, or ablation skip with machine-readable reasons,
severity, a deterministic checked-action identifier, and verifier/provider metadata. The optional
`LLMActionVerifier` receives identity-blinded public rules, public issue definitions, declared
strategy, visible evidence, and the candidate action. It may add a qualitative rejection for
strategy, evidence, or safety concerns, but it is never called after deterministic rejection and
can never turn that rejection into a pass. Provider or structured-output failure fails closed.

`VerificationCoordinator` permits zero or one correction attempt. Feedback contains only concise
machine reasons; `LLMActionCorrector` uses the proposing participant's bounded view and never sees
opponent preferences. A rejected or failed correction produces a deterministic withdrawal fallback,
which is itself deterministically verified. Outcomes and correction counts are retained in an
append-only audit log and exposed by policy benchmark results. Configure `VerificationMode` as
`deterministic`, `deterministic_and_llm`, or `disabled` for explicit ablation; even in disabled mode,
the protocol remains the final legality authority.

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

