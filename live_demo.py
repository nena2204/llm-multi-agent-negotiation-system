"""One readable negotiation with a REAL OpenAI model, printed turn by turn.

Setup (once):
    pip install -e ".[openai]"

Every new terminal:
    $env:OPENAI_API_KEY = "sk-..."          # your key, never commit it
    $env:OPENAI_MODEL   = "<model name>"    # a model your OpenAI account can use

Run:
    python live_demo.py                 # GPT buyer (with budget rule) vs. rule-based seller
    python live_demo.py --both-llm      # GPT buyer vs. GPT seller
    python live_demo.py --offline       # no internet / no key: built-in fake model (wiring check)
    python live_demo.py --price 350 --rounds 6
"""

import argparse
import sys

from llm_negotiation.benchmark import run_policy_session
from llm_negotiation.domain import (
    AcceptAction,
    AttributeRequirement,
    BudgetPolicy,
    CategoricalIssue,
    CategoricalPreference,
    CategoryUtility,
    CounterAction,
    IssueId,
    NegotiationScenario,
    NumericIssue,
    NumericPreference,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ProposeAction,
    ReservationPolicy,
    calculate_utility,
)
from llm_negotiation.llm import LLMConfigurationError, LLMProvider, ModelConfiguration, RetryConfiguration
from llm_negotiation.llm.openai import OpenAILLMClient, load_openai_model_configuration
from llm_negotiation.llm_policy import LLMNegotiationPolicy
from llm_negotiation.memory import AgentProfile, PublicAgentIdentity
from llm_negotiation.policies import LinearConcessionPolicy

BUYER, SELLER = ParticipantId("buyer"), ParticipantId("seller")
PRICE, WARRANTY, DAYS = IssueId("price"), IssueId("warranty"), IssueId("delivery_days")


def scenario():
    return NegotiationScenario(
        scenario_id="live-laptop",
        title="Laptop purchase",
        participants=(
            Participant(participant_id=BUYER, display_name="Buyer", role=ParticipantRole.BUYER),
            Participant(participant_id=SELLER, display_name="Seller", role=ParticipantRole.SELLER),
        ),
        issues=(
            NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=1000.0, unit="EUR"),
            CategoricalIssue(issue_id=WARRANTY, name="Warranty", choices=("none", "1y", "3y")),
            NumericIssue(issue_id=DAYS, name="Delivery days", minimum=1.0, maximum=30.0, unit="days"),
        ),
    )


def buyer_preferences():
    return ParticipantPreferences(
        participant_id=BUYER,
        issue_preferences=(
            NumericPreference(issue_id=PRICE, weight=0.5, direction=PreferenceDirection.MINIMIZE),
            CategoricalPreference(issue_id=WARRANTY, weight=0.3, category_utilities=(
                CategoryUtility(category="none", utility=0.0),
                CategoryUtility(category="1y", utility=0.5),
                CategoryUtility(category="3y", utility=1.0),
            )),
            NumericPreference(issue_id=DAYS, weight=0.2, direction=PreferenceDirection.MINIMIZE),
        ),
        reservation=ReservationPolicy(reservation_utility=0.45, batna_utility=0.4,
                                      batna_description="Buy a refurbished laptop elsewhere"),
        budget=BudgetPolicy(
            price_issue_id=PRICE, limit_price=300.0, maximum_concession_fraction=0.30,
            required_attributes=(
                AttributeRequirement(issue_id=WARRANTY, acceptable_values=("3y",)),
                AttributeRequirement(issue_id=DAYS, maximum=7.0),
            ),
        ),
    )


def seller_preferences():
    return ParticipantPreferences(
        participant_id=SELLER,
        issue_preferences=(
            NumericPreference(issue_id=PRICE, weight=0.6, direction=PreferenceDirection.MAXIMIZE),
            CategoricalPreference(issue_id=WARRANTY, weight=0.25, category_utilities=(
                CategoryUtility(category="none", utility=1.0),
                CategoryUtility(category="1y", utility=0.6),
                CategoryUtility(category="3y", utility=0.0),
            )),
            NumericPreference(issue_id=DAYS, weight=0.15, direction=PreferenceDirection.MAXIMIZE),
        ),
        reservation=ReservationPolicy(reservation_utility=0.25, batna_utility=0.2,
                                      batna_description="Sell to another customer next week"),
    )


def profile(pid, prefs, persona):
    return AgentProfile(
        identity=PublicAgentIdentity(participant=scenario().participant(pid), persona=persona),
        preferences=prefs,
    )


class LoggingClient:
    """Passes requests to the real client and keeps every raw model answer for diagnosis."""

    def __init__(self, inner):
        self.inner, self.records = inner, []

    def generate(self, request):
        response = self.inner.generate(request)
        self.records.append({"request_id": request.request_id, "answer": response.text})
        return response


class DiagnosticLLMPolicy(LLMNegotiationPolicy):
    """Same policy, but remembers WHY a model action or answer was rejected."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problems = []

    def ground_action(self, memory, decision):
        try:
            return super().ground_action(memory, decision)
        except Exception as error:  # recorded, then handled exactly as before
            action = decision.action
            offer = getattr(action, "offer", None)
            summary = action.action.value
            if offer is not None:
                summary += " " + fmt(offer)
            self.problems.append(f"illegal action ({summary}): {error}")
            raise

    def _invoke_structured(self, stage, output_type, context, telemetry, recovery, response_schema=None):
        try:
            return super()._invoke_structured(stage, output_type, context, telemetry, recovery, response_schema)
        except Exception as error:
            if type(error).__name__ != "_StageFailure":  # already explained by the repair step
                self.problems.append(f"{stage.value}: {type(error).__name__} {str(error)[:160]}")
            raise

    def _repair_structured(self, stage, output_type, context, invalid_response, guidance, telemetry, response_schema=None):
        try:
            output_type.model_validate_json(invalid_response)
        except Exception as error:
            first = str(error).splitlines()
            self.problems.append(f"{stage.value}: invalid JSON from model -> {' | '.join(first[:3])[:220]}")
        return super()._repair_structured(stage, output_type, context, invalid_response, guidance, telemetry, response_schema)


class Recorder:
    """Wraps an LLM policy and keeps the trace of every decision for printing."""

    def __init__(self, inner, label):
        self.inner, self.label, self.traces = inner, label, []
        self.name = inner.name

    def choose_action(self, memory):
        start = len(self.inner.problems)
        action = self.inner.choose_action(memory)
        self.traces.append((self.inner.last_trace, self.inner.problems[start:]))
        return action


def make_client(offline):
    if offline:
        from llm_negotiation.experiments import ExperimentModelSpec, _offline_fake_client
        spec = ExperimentModelSpec()
        config = ModelConfiguration(provider=LLMProvider.FAKE, model=spec.model, timeout_seconds=5.0,
                                    retry=RetryConfiguration(max_retries=0))
        return _offline_fake_client(spec), config
    try:
        return OpenAILLMClient.from_env(), load_openai_model_configuration()
    except LLMConfigurationError as error:
        sys.exit(f"\nOpenAI is not configured: {error}\n"
                 "Set it first, e.g.:\n"
                 '  $env:OPENAI_API_KEY = "sk-..."\n'
                 '  $env:OPENAI_MODEL   = "<model name>"\n'
                 'and install the adapter once: pip install -e ".[openai]"\n')


def fmt(offer):
    return (f"price {offer.value_for(PRICE).value:7.2f} EUR | warranty {offer.value_for(WARRANTY).value:<4} "
            f"| delivery {offer.value_for(DAYS).value:4.1f} days")


def main():
    # Model text may contain symbols such as "≤"; never crash on Windows consoles/pipes (cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--both-llm", action="store_true", help="the seller is also a GPT agent")
    parser.add_argument("--offline", action="store_true", help="use the built-in fake model (no key)")
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    client, config = make_client(args.offline)
    client = LoggingClient(client)
    buyer = Recorder(DiagnosticLLMPolicy(client, config), "Buyer (GPT)")
    if args.both_llm:
        seller = Recorder(DiagnosticLLMPolicy(client, config), "Seller (GPT)")
    else:
        seller = LinearConcessionPolicy()

    model_name = "offline fake" if args.offline else config.model
    print("=" * 78)
    print(f" LIVE NEGOTIATION  |  model: {model_name}  |  rounds: {args.rounds}")
    print(" Buyer : GPT agent, budget 300 EUR, may go up to 390 ONLY with 3y warranty + delivery <= 7 days")
    print(f" Seller: {'GPT agent' if args.both_llm else 'rule-based linear strategy'}")
    print("=" * 78)

    result = run_policy_session(
        scenario(),
        {BUYER: profile(BUYER, buyer_preferences(), "Student buying a laptop for university"),
         SELLER: profile(SELLER, seller_preferences(), "Local electronics shop")},
        {BUYER: buyer, SELLER: seller},
        maximum_rounds=args.rounds,
    )

    traces = {BUYER: list(buyer.traces), SELLER: list(getattr(seller, "traces", []))}
    for event in result.session.event_sequence:
        action = getattr(event, "action", None)
        if action is None:
            continue
        who = "BUYER " if action.actor_id == BUYER else "SELLER"
        if isinstance(action, (ProposeAction, CounterAction)):
            print(f"\n[round {action.round_number}] {who} {action.action.value.upper():<8} {fmt(action.offer)}")
        elif isinstance(action, AcceptAction):
            print(f"\n[round {action.round_number}] {who} ACCEPTS")
        else:
            print(f"\n[round {action.round_number}] {who} {action.action.value.upper()}")
        queue = traces[action.actor_id]
        while queue:
            trace, problems = queue.pop(0)
            if trace is not None and trace.selected_action == action:
                if trace.used_fallback:
                    print(f"      (model output was not usable -> safe fallback strategy; reason: {trace.fallback_reason})")
                for problem in problems:
                    print(f"      why      : {problem}")
                if trace.plan is not None:
                    print("      plan     : " + "; ".join(trace.plan.plan)[:300])
                if trace.decision_rationale:
                    print("      reasoning: " + trace.decision_rationale[:400])
                break

    print("\n" + "-" * 78)
    outcome = result.session.outcome
    print(f" RESULT: {result.session.phase.value.upper()}")
    if outcome is not None and outcome.agreement is not None:
        offer = outcome.agreement.offer
        print(" Deal  : " + fmt(offer))
        u_b = calculate_utility(scenario(), offer, buyer_preferences())
        u_s = calculate_utility(scenario(), offer, seller_preferences())
        print(f" Utility buyer {u_b:.2f} | seller {u_s:.2f}")
        violation = buyer_preferences().budget_violation(offer)
        print(" Budget rule: " + ("respected" if violation is None else violation))
    elif outcome is not None and outcome.reason:
        print(" Reason: " + outcome.reason)
    calls = sum(stage.model_calls for stage in buyer.inner.cumulative_telemetry)
    tokens = sum(stage.usage.total_tokens for stage in buyer.inner.cumulative_telemetry)
    if args.both_llm:
        calls += sum(stage.model_calls for stage in seller.inner.cumulative_telemetry)
        tokens += sum(stage.usage.total_tokens for stage in seller.inner.cumulative_telemetry)
    print(f" Model calls: {calls} | tokens: {tokens}")
    fallbacks = sum(1 for t, _ in buyer.traces if t is not None and t.used_fallback)
    print(f" Buyer fallbacks: {fallbacks}/{len(buyer.traces)}", end="")
    if args.both_llm:
        s_fb = sum(1 for t, _ in seller.traces if t is not None and t.used_fallback)
        print(f" | Seller fallbacks: {s_fb}/{len(seller.traces)}", end="")
    print()
    try:
        import json
        with open("live_log.jsonl", "w", encoding="utf-8") as handle:
            for record in client.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(" Raw model answers saved to live_log.jsonl")
    except OSError:
        pass
    print("-" * 78)


if __name__ == "__main__":
    main()
