# Shared State Schema

Status: Phase 1, Task 1. Implemented in `app/state/`. The Supervisor
that uses it is documented in `docs/supervisor-graph.md`.

Tests: `app/test_state_models.py`, `app/test_platform_state.py`,
`app/test_state_serde.py` and `app/test_platform_lifecycle.py` (a stub
graph that exercises every state field). The Supervisor tests are listed
in `docs/supervisor-graph.md`.

## Purpose

The shared state is the common language of the platform. The Supervisor,
the specialist agents, the approval flow and the audit trail exchange
information only as the typed records defined here. Specialists never
call each other; they read and write shared state through the Supervisor.

## Design rules

1. **Records validate themselves.** Safety rules live in the models, so an
   invalid record cannot be constructed.
2. **Single writer for decisions.** Only the Supervisor writes the
   last-write-wins fields. Specialists write only the append-only fields,
   so parallel specialists cannot overwrite one another.
3. **Propose-only in Phase 1.** `ExecutionRecord.simulated` is
   `Literal[True]`, so real execution cannot be represented yet.
4. **Everything survives a checkpoint.** Every record round-trips through
   JSON and through the SQLite checkpointer.
5. **Reuse existing contracts.** Models extend `AgentContract` and reuse
   `AgentEvidence` from `app/contracts/agent_messages.py`.

## Records (`app/state/models.py`)

| Record | Purpose | Enforced rules |
|---|---|---|
| `IncidentContext` | The alert that started the incident | Non-empty ID, description, environment; unknown fields rejected; optional `namespace`, `pod_name` and `deployment` (a remediation proposal needs `namespace` and `deployment`) |
| `RoutingDecision` | Supervisor's choice of specialists | At least one agent; only `kubernetes`, `observability`, `security`; reason required |
| `Finding` | One specialist's conclusion and its evidence | Evidence quality above `insufficient` requires evidence; `attempt >= 1`; `contradicts` links conflicting findings |
| `ProposedAction` | A remediation proposal | At least one supporting finding ID; action type from a closed list; risk level required |
| `ApprovalRecord` | The human decision | Pending cannot carry a decision; approved or rejected requires `decided_by`; `decided_at` is set automatically |
| `ExecutionRecord` | Outcome of an executed action | `simulated` must be `True` in Phase 1 |
| `UsageRecord` | One LLM call, attributed | Token counts >= 0; carries incident ID, agent, model, optional cost |
| `CallLimits` | Safety ceilings per incident | Per-agent (default 5, matches `MAX_TOOL_CALLS`), per-incident (15), Supervisor hops (6) |
| `AuditEntry` | One line of the audit trail | Incident, actor, event required |
| `IncidentReport` | Supervisor's final report | Outcome from a closed list |

## Top-level state (`app/state/graph_state.py`)

`PlatformState` is the LangGraph state. Merge behaviour is declared per field.

| Field | Written by | Merge rule |
|---|---|---|
| `incident` | Initial state | Set once |
| `status` | Supervisor | Last write wins |
| `limits` | Initial state | Set once |
| `routing` | Supervisor | Last write wins |
| `proposed_action` | Supervisor | Last write wins |
| `approval` | Approval node | Last write wins |
| `execution` | Supervisor | Last write wins |
| `report` | Supervisor | Last write wins |
| `findings` | Specialists | Append |
| `usage` | Any agent | Append |
| `audit_log` | Any actor | Append |
| `agent_call_counts` | Supervisor, per specialist call | Per-key sum |
| `supervisor_hops` | Supervisor | Sum |

Helpers: `new_platform_state`, `check_call_budget` (per-agent and
per-incident limits, used by the Supervisor before every specialist
call), `supervisor_hops_exhausted`, `summarize_usage` (tokens and cost
per agent), `merge_counts`.

## Adapters (`app/state/adapters.py`)

`finding_from_kubernetes_response` converts the existing
`KubernetesInvestigationResponse` into a `Finding`.

- It downgrades `evidence_quality` to `insufficient` when no diagnostic
  evidence exists.
- The Kubernetes agent records failed tool runs as observations carrying
  `"execution_failed": true`. These stay in the finding for the audit
  trail, but they never count as diagnostic evidence
  (`is_execution_failure`).
- Summaries longer than 500 characters are truncated.

Observability and Security adapters will be added with those agents.

## Incident lifecycle: CrashLoopBackOff walkthrough

Validated two ways: `app/test_platform_lifecycle.py` runs a stub graph
that touches every state field, and the Supervisor tests run the same
lifecycle through the real graph (see `docs/supervisor-graph.md`). Both
include a simulated process restart between pause and resume.

| # | Step | State written | Audit event |
|---|---|---|---|
| 1 | Alert received | `incident`, `status=received`, `supervisor_hops` | `alert_received` |
| 2 | Supervisor routes to Kubernetes | `routing`, `status=investigating` | `routed` |
| 3 | Kubernetes agent investigates | `agent_call_counts` (and `usage`, once agents make LLM calls) | `kubernetes_investigated` |
| 4 | Findings and evidence recorded | `findings` (with `AgentEvidence`) | |
| 5 | Supervisor evaluates | Reads `status`, `evidence_quality`, `needs_follow_up` | |
| 6 | Supervisor proposes rollout restart | `proposed_action`, `status=awaiting_approval` | `action_proposed` |
| 7 | Graph pauses at the interrupt | Checkpoint saved to SQLite | |
| 8 | Engineer approves or rejects | `approval` | `approval_granted` / `approval_rejected` |
| 9 | Graph resumes from checkpoint | Restored state equals state at pause | |
| 10 | Simulated execution (approved only) | `execution`, `status=executing` | `execution_simulated` |
| 11 | Final report | `report`, `status=completed` | `report_generated` |

On rejection, step 10 is skipped and the report outcome is
`action_rejected`.

## Checkpointing

- Phase 1 uses `SqliteSaver`; Phase 2 swaps in PostgreSQL.
- Every checkpointer must be created with
  `serde=platform_serializer()` (`app/state/serde.py`). Without it,
  LangGraph warns that it will block deserialization of custom types in
  a future version.
- The allowlist is built from `app/state/models.py`, plus `AgentEvidence`,
  and contains only `app.*` modules. New records are covered
  automatically.
- A node that calls `interrupt()` re-runs from its top on resume, so
  nothing before the `interrupt()` call may have side effects.
- A graph that runs inside a Supervisor node must be compiled with
  `checkpointer=False`. Otherwise it inherits the Supervisor's
  checkpointer and overwrites the Supervisor's thread state, which loses
  the approval pause. The async Kubernetes agent graph is compiled this
  way.

## Known limitations

- Only a Kubernetes adapter exists.
- `UsageRecord` is never populated: the current Kubernetes agent run
  makes no LLM calls.
- `evidence_quality` from the Kubernetes agent is only `insufficient` or
  `limited`. `corroborated` is defined but nothing produces it yet.
- An empty tool result (for example no pod events) still counts as
  evidence, so a failed investigation of an unknown pod reports quality
  `limited`. The Supervisor's evidence gate still rejects it, because the
  finding is not `completed`.
- `Finding.contradicts` is not used yet; there is no evidence correlation
  logic.
- `IncidentReport.root_cause` is never populated.
- Execution is simulated only.

Supervisor-level limitations are listed in `docs/supervisor-graph.md`.

## Extending the schema

1. Add the record to `app/state/models.py`, extending `AgentContract`.
2. Add validation tests and include it in the JSON round-trip test.
3. Add a field to `PlatformState` with an explicit reducer if more than
   one node can write it.
4. Update the lifecycle table in this document.