# LLM-Based Multi-Agent Negotiation System

This project is a deterministic, rule-based simulation of negotiation between buyer, seller,
mediator, judge, and learning roles. It includes an optional provider-independent LLM policy, while
the default test suite remains fully offline and never accesses the network.

## Features

- Buyer and seller strategies: aggressive, neutral, and cooperative
- Multi-round negotiation with a mediator fallback
- Outcome validity and fairness evaluation
- Deterministic rewards and repeated simulations
- Reproducible non-LLM baseline policies for multi-issue experiments
- Command-line interface and optional Streamlit UI
- Typed LLM client contract with deterministic fake and optional OpenAI Responses API adapter
- Dependency-injected orchestration with replayable events, budgets, mediation, and evaluation
- Explicit multiparty proposal, deliberation, revision, and voting semantics

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
participant. Mediation is entered by an explicit `request_mediation` action or a replayable
deadlock/deadline trigger. Mediator interventions are separate protocol events and their offers are
non-binding: participants must explicitly accept, counter, or reject them. A trusted coordinator
may mark the feasible region `impossible` during session
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

## Mediation

`llm_negotiation.mediation` provides configurable detection for repeated offers, low concessions,
offer cycles, repeated invalid actions, and an approaching deadline. Explicit participant requests
remain typed protocol actions. Automatic deadlock/deadline entry and every mediator intervention
are replayable protocol events, separate from participant actions. Correlated public mediator
messages can be emitted through the existing `MessageBus`.

The default `public_only` access mode uses public offers and messages. `confidential_summary` accepts
mediator-only utility summaries that exclude BATNA prose. The research-only `simulation_oracle`
mode requires both an explicit mode selection and `allow_simulation_oracle=True`; it is never the
production default. Public explanations are generated only from public context and do not disclose
private reservations or preferences.

For a single numeric price, deterministic mediation proposes the midpoint of the permitted
bargaining information. Multi-issue mediation enumerates a bounded deterministic candidate grid and
selects by normalized Nash product or max-min welfare. If no candidate satisfies every available
reservation threshold, the mediator records a refusal rather than forcing a proposal. The optional
`LLMMediator` uses the provider-independent gateway and strict structured output to ask a clarifying
question, provide a public disagreement summary, make a typed proposal, or refuse; it cannot accept
for any participant.

## Evaluation

`llm_negotiation.evaluation` keeps authoritative deterministic metrics separate from optional,
model-dependent qualitative judgement. Evaluation receives the terminal session and explicitly
supplied post-episode participant preferences; those preferences are not added to public protocol
state. All utilities use the domain's normalized `[0, 1]` utility function.

The deterministic formulas are:

- Agreement is reached exactly when the protocol phase is `agreed`; validity means the resulting
  agreement passes the public scenario's complete-offer and all-participant acceptance rules.
- Rounds to agreement is the terminal session round. Elapsed time is an optional measured duration
  supplied by the episode runner and is reported only for a valid agreement.
- Agreement utility `u_i` is participant `i`'s normalized utility for the final offer. Outcome
  utility is `u_i` for a valid agreement and that participant's BATNA utility otherwise. Named
  buyer/seller fields report this outcome utility; agreement-only utility remains explicit in each
  participant record.
- Individual rationality is `u_i >= reservation_i`, evaluated separately from public validity.
- Social welfare is `sum(outcome_utility_i)`.
- Bargaining gain is `max(0, outcome_utility_i - BATNA_i)`. Nash product is the product of these
  gains for a valid agreement and zero when there is no agreement.
- Utility balance is `1 - (max(u_i) - min(u_i))` for a valid agreement. It measures equality of
  normalized utility, not validity or Pareto efficiency.
- Pareto efficiency and distance use the participant-utility vectors of a deterministic candidate
  frontier. Categorical domains are exhaustive; numeric issues use a documented configurable grid
  that always includes the final offer. Distance is Euclidean distance in normalized utility space,
  and reports explicitly identify numeric frontiers as approximations or mark oversized searches
  `intractable`.
- Invalid-action rate is rejected initial submissions divided by actions submitted through the
  verification gate. Correction rate is correction attempts divided by those submissions. Both are
  zero when no verification submissions were logged.
- Message totals, public-message totals, model calls, total latency, and provider token usage are
  summed from typed episode telemetry records without logging prompts.

The old price midpoint score remains available only as `legacy_midpoint_fairness`: it computes
`max(0, 100 - 100 * abs(price - midpoint) / bargaining_range)` and preserves the historical score
of `100` for a zero-width range. It is not used as the authoritative fairness metric.

`LLMJudge` receives a size-bounded public transcript with stable `participant-N` labels, public
issues and outcome, and no deterministic score to imitate. It returns strict rubric scores for
process fairness, communication quality, justification quality, and coercion safety. Every rubric
score carries evidence citations restricted to supplied event ids. Private preferences are
unavailable by default; the separately
labelled `research_private` mode requires explicit opt-in and remains anonymized. Multiple models
and samples are retained individually and summarized with descriptive statistics—not majority vote
or ground truth. `EvaluationBundle` stores deterministic and qualitative reports in separate fields
and `export_evaluation_json()` provides a versioned JSON export.

## Episode orchestration

`llm_negotiation.orchestration.NegotiationOrchestrator` is the single execution lifecycle used by
the policy benchmark and legacy CLI adapter. Policies, message bus, verification coordinator,
mediator, optional model gateway, participant-owned memory store, clock, and random seed are all
injected explicitly; no global mutable episode state is used. The default factory remains the
backward-compatible bilateral alternating-offer protocol. An explicitly injected
`MultipartyProtocolFactory` enables staged group negotiation.

Each iteration builds only the current participant's authorized memory snapshot, obtains one typed
action, applies bounded correction and deterministic verification, transitions the protocol,
delivers correlated messages/events, and refreshes participant-specific memories. Requested or
automatic mediation remains non-binding and is recorded separately. The returned `EpisodeResult`
contains the outcome, public transcript, complete event/message audit streams, content-addressed
memory references, authoritative metrics, and provider/model-labelled usage records. The public
transcript is validated as exactly the public subset of the append-only audit stream, and correlated
message/action references are revalidated during result deserialization.

`OrchestratorConfiguration` bounds rounds, episode/action elapsed time, total actions, and model
calls. An episode-scoped gateway applies the model-call limit across policies, opponent modelling,
verification, correction, and mediation, and reduces retry allowance to the remaining hard budget.
Its explicit failure policy either uses an injected deterministic fallback, records a
verified withdrawal, or raises `OrchestrationError`. `NegotiationOrchestrator.replay()` reconstructs
the final state from protocol events and recomputes metrics from the stored episode artifacts.

## Multiparty research protocol

`llm_negotiation.multiparty` defines explicit semantics for groups of three or more principals.
`MultipartyProtocolConfiguration` names the complete speaking order, eligible proposal owners and
voters, required participants, coalition rules, and one of three acceptance rules: unanimity,
strict majority (`floor(n/2) + 1`), or threshold (`ceil(threshold * n)`). Required participants must
personally approve an offer regardless of the numerical quorum. Their own reservation utility is
enforced by the authoritative verifier, so majority pressure cannot manufacture consent.
The resolved voter set, quorum, rule name, and required participants are also embedded in the
serialized session, allowing agreement validity to be checked without relying on undocumented
factory state.

An episode has four bounded stages:

1. Every eligible proposer creates an independent initial proposal from an empty participant view;
   no earlier proposal, reasoning, or message is visible until all initial proposals are sealed.
2. Each principal sends a typed critique in configured speaking order for the configured number of
   deliberation passes. Private caucuses are denied unless their exact membership is allow-listed.
3. Eligible owners receive bounded revision turns in speaking order, with every offer attributed to
   its proposer. Each valid revision replaces the outstanding offer, so the last scheduled revision
   is the unambiguous ballot proposal.
4. Eligible voters explicitly accept or reject the final outstanding offer. Votes always reference
   that offer, the quorum is deterministic, and required-participant rejection ends without a deal.

Proposals, revisions, and votes are public protocol acts and must address every other principal.
Only deliberation messages may use an explicitly allow-listed private coalition.

The concrete `three_principal_scenario()` negotiates research-budget allocation and launch timing
between Operations, Research, and Community. Their personas, utility weights, preference directions,
categorical values, reservation utilities, and baseline strategy assignments differ. Optional model
configuration is stored in `TeamMemberConfiguration`, separate from `AgentProfile` and its private
preferences. Policies still receive only their own bounded memory snapshot, and public critiques do
not contain private utility functions.

`summarize_transcript()` deterministically compacts large visible transcripts while retaining one
source action/correlation id for every input message. `run_team_comparison()` runs homogeneous and
heterogeneous configurations for two, three, and four agents. It reports outcomes, welfare,
mean outcome utility (for a size-aware comparison), and independent-proposal diversity as
experimental observations and deliberately makes no claim that diversity or team size is always
beneficial.


## Between-episode strategy learning

`llm_negotiation.learning` provides transparent contextual-bandit learners over a registered set
of negotiation strategies. Epsilon-greedy selects the highest empirical mean reward, with seeded
probability epsilon selecting a registered arm uniformly. UCB selects every unobserved arm once,
then maximizes `mean_reward + c * sqrt(log(total_observations) / arm_observations)`. Hash-derived
seeded choices make behavior reproducible across save/reload without serializing an opaque random
generator.

The context contains only public scenario structure and the learner's own profile: participant and
issue counts, public issue identifiers and types, the learner's role/persona presence, and bucketed
own issue weights, preference directions, reservation utility, and BATNA. Scenario IDs, opponent
identities, messages, behavior, and opponent preferences are excluded, so equivalent contexts can
share statistics without leaking private opponent state.

The configurable reward is
`R = 0.50 * own_utility + 0.20 * valid_agreement + 0.15 * efficiency + 0.10 * fairness - 0.05 * cost`
by default. Own utility is the authoritative participant outcome utility; efficiency is
`max(0, 1 - Pareto_distance / sqrt(participant_count))` when a valid agreement and tractable Pareto
frontier are available; fairness is utility balance; and cost is the mean of separately capped
message-count/100, model-call-count/20, and latency/60000-ms ratios. All five weights and three cost
scales are explicit `RewardConfiguration` fields and need not sum to one.

Only a completed terminal episode can update arm counts and rewards. Training selections create a
pending observation that must be resolved exactly once. Evaluation mode performs deterministic
greedy selection without exploration or state mutation, and rejects outcome updates. Learner files
are versioned JSON written atomically through a same-directory temporary file and `os.replace`;
invalid JSON, incompatible versions, and wrong algorithm subclasses fail explicitly.

`run_learning_curve_experiment()` supplies a small seeded demonstration against a fixed linear
seller. With 12 episodes, seed 23, and epsilon 0.25, the fixed buyer arm received reward 0.440 in
two no-deal episodes; Boulware, linear, and conceder arms each produced agreements and reward 0.814,
and the frozen tie-break selected Boulware. This is a fixture-specific learning curve that exposes
state changes and reproducibility, not evidence of general strategy superiority or generalization.

## Reproducible experiment harness

The versioned harness in `llm_negotiation.experiments` separates final negotiation outcomes from
component capability tests and training-only learning curves. The built-in scenario dataset covers
feasible and infeasible bargaining regions, symmetric and asymmetric preferences, price-only and
multi-issue negotiations, and a three-principal multiparty case. Every configuration records the
scenario set, policy/model baselines, seed set, maximum rounds, each model and temperature, execution and
token budgets, dataset version, and enabled components. Its SHA-256 configuration hash identifies
the complete configuration.

The supported baselines are deterministic policies, one structured LLM policy, independent
sample-and-vote, homogeneous teams, and heterogeneous teams. The default LLM baseline uses the
provider-independent structured fake and remains fully offline. Sample-and-vote independently asks
Boulware, linear, and conceder policies for typed actions and uses deterministic action-type majority
voting. A single-LLM policy is explicitly marked not applicable for multiparty cases rather than
being silently omitted.

`EnabledComponents` provides one-factor ablations for memory, Theory of Mind, deterministic
pre-verification, mediation, substantive communication, team size (2/3/4), deliberation depth, and
belief-confidence visibility. `one_at_a_time_ablation_configurations()` produces the complete matrix.
Memory-off retains only protocol-critical bounded working state; communication-off emits a neutral
protocol critique without sharing substantive reasoning; verifier-off still leaves the protocol
state machine authoritative. ToM is updated only from observable bilateral evidence, while its
separate labelled calibration set also supports multiparty research without exposing private truth
during an episode.

Run the full one-factor matrix explicitly (it is intentionally excluded from default CI):

```powershell
llm-negotiation-experiment ablate --config experiment_configs/offline-small.json --output artifacts/ablations
```

Run the checked-in small configuration without network access:

```powershell
llm-negotiation-experiment offline --config experiment_configs/offline-small.json --output artifacts/offline-small --timestamp 2026-01-01T00:00:00+00:00
```

The explicit timestamp makes complete artifacts byte-reproducible for the same commit, Python
environment, configuration, and seed. Omitting it records the current UTC run time. Larger offline
runs require an explicit configuration passed to the same command. Provider-backed runs are never
selected by default and require both the optional provider installation/configuration and the
explicit command below; missing credentials fail before an API request:

```powershell
llm-negotiation-experiment live --config path\to\live-config.json --output artifacts/live-run
```

Each run produces:

- `episodes.jsonl`: every successful, failed, or explicitly non-applicable cell. Successful rows
  preserve the full serializable `EpisodeResult`; failure messages are sanitized.
- `aggregate.csv`: configuration hash, Git commit, timestamp, seeds, model/temperature, counts,
  metric means, and 95% confidence intervals.
- `paired_comparisons.csv`: paired mean differences for baselines evaluated on identical
  scenario/seed cells.
- `ablation_table.csv`: component settings and outcome metrics, reproducible from raw JSONL.
- `component_evaluations.jsonl`: separately labelled environment-comprehension, opponent-inference,
  calibration, next-action, and joint-planning cases.
- `learning_curve_training.json`: an explicitly separate training partition that is never included
  in final evaluation aggregates.
- `plots/*.svg`: agreement, utility/welfare, fairness, rounds, invalid actions, latency,
  usage/estimated cost, and learning curves.
- `manifest.json`: the full configuration, configuration hash, commit, timestamp, partition names,
  statistical method, and artifact index.

Rebuild aggregate and ablation tables solely from preserved raw results:

```powershell
llm-negotiation-experiment rebuild artifacts/offline-small/episodes.jsonl `
  --aggregate artifacts/offline-small/aggregate-rebuilt.csv `
  --ablation artifacts/offline-small/ablation-rebuilt.csv
```

Confidence intervals use a two-sided 95% normal approximation around the arithmetic mean. A single
observation receives a zero-width descriptive interval; paired comparisons calculate within-cell
differences before their interval. These methods are transparent and dependency-free, but very
small samples do not justify population-level conclusions. In particular, do not conclude that an
agentic feature, larger team, or model is superior from the smoke benchmark, and never treat an LLM
judge score, persuasive transcript, or majority vote as a substitute for authoritative validity,
utility, Pareto, fairness, and calibration metrics.

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

