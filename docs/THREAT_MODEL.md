# Communication Safety Threat Model

All participant-, mediator-, and model-authored natural language is untrusted. Protocol actions,
typed offers, deterministic verification, and objective evaluation remain authoritative. The
controls below reduce the chance that communication becomes an unmonitored control channel.

| Threat | Protected boundary | Deterministic controls | Residual risk |
|---|---|---|---|
| Private-preference leakage | Participant-specific preferences and BATNA data | Participant-specific observations, prohibited-field monitoring, protected quarantine | Undeclared facts may still be described indirectly. |
| Unauthorized direct messaging | Visibility and recipient authorization | Role-based channel allowlists and typed recipients | Authorization does not prove that an allowed message is benign. |
| Prompt injection | Model instruction hierarchy | Pattern monitoring, strict structured outputs, and explicit untrusted-data framing in every model prompt | Novel or indirect instructions may evade pattern rules. |
| Encoded or covert content | Observable communication policy | Size/recipient limits, ASCII protocol profile, known encoding and invisible-character rules, coordinated-anomaly monitoring | Semantic steganography can use ordinary-looking language. |
| Judge manipulation | Separation of qualitative and objective evaluation | Persuasion monitoring, anonymized bounded judge context, objective metrics remain authoritative | Qualitative judges can still exhibit model or framing bias. |
| Mediator manipulation | Non-binding and privacy-scoped mediation | Manipulation monitoring, pre-transition message checks, typed interventions, participants retain acceptance rights | A subtle message may influence a mediator without matching a rule. |
| Reward hacking | Evaluation and learner integrity | Reward/metric manipulation rules, post-episode metrics, frozen evaluation learners | A policy can exploit an unknown proxy weakness. |

## Control layers

1. Typed envelopes restrict identifiers, visibility, message types, recipients, references, and
   serialized fields. Arbitrary metadata and object references are not accepted.
2. Role capabilities define which principal, mediator, or system role may use each message type
   and visibility. Optional directional sender-to-recipient grants further restrict direct/private
   and mediator-only channels; an explicitly empty rule or grant list means deny all.
3. Configurable content limits bound characters, recipient counts, and the selected protocol's
   character set.
4. The deterministic monitor records known prompt-injection, encoding, private-field, evaluator,
   mediator, reward, fake-system, and repeated cross-sender anomaly patterns.
5. Configurable quarantine or rejection blocks delivery. Raw suspicious content is retained only in
   the protected in-memory safety audit; normal audit messages, memories, public transcripts,
   incidents, and `EpisodeResult` contain hashes and machine reasons rather than the payload. The
   protected audit can verify its raw payloads against those hash commitments.
6. Participant and mediator messages are checked before protocol mutation. A blocked participant
   action uses a deterministic withdrawal fallback; it cannot become a protocol message event.
7. Episode protocol events, delivered messages, and sanitized incidents are linked by a SHA-256
   audit chain. Replay validation recomputes the complete chain and rejects modification.
8. Safety metrics report attempted, delivered, quarantined and rejected messages plus incident
   counts by threat category.

## Limitations

Monitoring is risk reduction, not proof of safety. It cannot prove that semantic steganography,
implicit coordination, novel encodings, coded use of normal negotiation terms, or emergent
collusion is absent. ASCII restrictions and known-pattern matching may produce false positives and
false negatives. Paraphrasing is not used as a security boundary. Protected audits also require
operational access controls if replaced by durable storage in a future deployment.
