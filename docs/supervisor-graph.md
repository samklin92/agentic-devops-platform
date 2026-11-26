# Supervisor Graph

Status: Phase 1. Implemented in `app/supervisor/`. State records are
documented in `docs/shared-state-schema.md`.

## Purpose

The Supervisor receives an incident, routes it to specialist agents,
records how their findings relate to each other, evaluates them, proposes
a remediation only when the evidence supports one, pauses for human approval, simulates the approved action
and produces the final report. Phase 1 is propose-only: nothing real is
executed.

## Flow (`app/supervisor/graph.py`)

| Node | What it does | Next |
|---|---|---|
| `receive_alert` | Records the alert | `route` |
| `route` | Rule-based routing; no match means no specialist | `investigate`, or `report` if nothing matched |
| `investigate` | Calls each selected specialist within its call budget | `correlate` |
| `correlate` | Records how the findings relate (Supervisor step, counts as a hop) | `evaluate` |
| `evaluate` | Builds a proposal from usable evidence, or none; adds correlation notes to an existing proposal | `approval`, or `report` if no proposal |
| `approval` | Pauses with `interrupt()`; records the human decision | `execute` if approved, otherwise `report` |
| `execute` | Simulated execution only | `report` |
| `report` | Final `IncidentReport` and status | End |

Report outcomes: `action_executed_simulated`, `action_rejected`,
`no_action_proposed` (nothing routed, or evidence gathered but no
remediation applies) and `investigation_failed` (status `failed`).

## Routing (`app/supervisor/routing.py`)

A deterministic baseline: regex signal rules per specialist. An incident
goes to every specialist whose rules match, in the fixed order
kubernetes, observability, security. A named `pod_name` is itself a
Kubernetes signal. No match returns `None`, and the incident is
escalated to a human in the report. Every decision lists the signals
that fired.

Measured by `python -m app.evaluate_routing` on 43 labelled incidents
(`evals/routing_questions.json`). At the time of writing the baseline
gets 40 of 43 exactly right, with precision 1.00 for every specialist,
recall 0.93 for Kubernetes and 0.94 for Observability and Security, and
all 5 out-of-scope cases handled. The three misses are deliberately
keyword-free cases marked `hard` (an exit code 137 restart loop, a vague
customer-impact report and a sign-in from an unfamiliar country). They
mark the limit of rules and the bar an LLM router would have to beat.
The set was extended when the Observability specialist was added (mixed
crash-loop and HTTP 500 cases, metrics-only cases, and a pod with no
metrics signal that must not route to Observability) and again for
Security (failed logins, IAM access denied, open ingress, root login,
mixed Observability and Kubernetes cases, and password reset emails as
a false-positive guard). Kubernetes precision did not change either
time.

`memory pressure` routes to Observability, as a resource metric. The
Kubernetes node condition `MemoryPressure` is a different thing: written
as one word it does not match, and a pod or other Kubernetes signal in
the same description still routes to Kubernetes.

## Evidence gate and proposals (`app/supervisor/proposals.py`)

- A finding is usable only if its status is `completed`, its evidence
  quality is above `insufficient`, and it needs no follow-up.
- A proposal needs a usable Kubernetes finding whose text shows a crash
  loop, and an incident that names a namespace and a deployment.
- The only proposal today is `rollout_restart` on
  `deployment/<name>`. Risk is `medium` in production and `low`
  elsewhere.
- No usable evidence means no proposal and no approval request; the
  report says human review is needed.

## Correlation (`app/supervisor/correlation.py`)

The separation of roles is the point of this step. Specialists produce
observations and findings. The Supervisor identifies relationships
between findings. The proposal gate independently decides whether an
action is eligible. A human makes the approval decision. Correlation
annotates evidence and never authorizes action.

**Signals.** A finding carries structured `signals` from a closed
vocabulary, and each specialist may emit only its own:

| Specialist | Signals |
|---|---|
| Kubernetes | `crash_loop` |
| Observability | `errors_elevated`, `latency_elevated`, `cpu_saturated`, `memory_near_limit` |
| Security | `failed_auth_elevated`, `iam_denied_elevated`, `credential_stale`, `publicly_exposed`, `open_ingress` |

A signal is emitted only when the check that supports it succeeded. A
failed, malformed or missing check produces no signal, and its absence is
never read as a clean result. A failed or blocked finding cannot carry
signals at all (the `Finding` model rejects it). Observability and
Security build their summary text from the same signals, so text and
signals cannot disagree. Kubernetes gets `crash_loop` only from a
successful diagnostic observation and the agent's findings, never from an
error message or a failed tool run, using the same pattern as the
proposal gate (`app/state/signals.py`), so there is one definition of a
crash loop.

**Rules.** Only usable findings take part, and every relation needs two
different specialists:

| Rule | Condition | Kind |
|---|---|---|
| R1 `shared_incident_impact` | Kubernetes `crash_loop` and Observability `errors_elevated` or `latency_elevated` | `corroborates` |
| C1 `impact_not_observed` | Kubernetes `crash_loop` and an Observability finding with no signals whose evidence covers exactly the three required checks (error rate, latency, resource usage) with no failed-run observations | `contradicts` |
| R2 `load_with_failed_auth` | Observability `latency_elevated` or `cpu_saturated` and Security `failed_auth_elevated` | `co_occurs` |

R1 says independent specialists support a shared incident-impact
hypothesis and states that this does not establish the cause of the crash
loop. C1 never fires on a partial, failed or unfinished Observability
investigation, or on a finding that only claims to be complete without
evidence from all three checks. R2 says co-occurrence does not establish causation and
implies no remediation, and it is never a corroboration. Explanations are
built from signal names and fixed caveats, and tests reject causal and
remediation wording.

**How the result is computed.** `correlate(findings)` is pure and
deterministic. It looks only at the latest attempt per specialist, leaves
out a specialist whose finding ID is ambiguous (the same ID on two
different records), and gives the same correlations, with the same IDs,
in any input order. The `correlate` node recomputes the whole list from
the current findings each time and replaces `PlatformState.correlations`,
so a repeated investigation cannot leave stale relationships. It writes
one `correlation_recorded` audit entry per correlation.

**What it may do.** Add explanatory text to the rationale of a proposal
that already exists (`annotate_action`), add a structured `correlations`
list to the approval payload, and add an "Evidence relationships" clause
to the report summary. `effective_quality(finding, correlations)` gives a
derived reading of "corroborated" for a usable finding that another
specialist corroborates. It never changes the finding, and the proposal
gate never uses it.

**What it may not do.** Create a proposal, remove one, change its risk
level or its supporting finding IDs, authorize execution or override the
human decision. `propose_action(incident, findings)` is unchanged and
takes no correlations. Tests compare proposals with and without
correlations across many finding combinations, and the only permitted
difference is the rationale text.

The correlation step counts as a Supervisor hop, so an investigated
incident uses 5 of the 6 hops (receive, route, correlate, evaluate,
report).

## Approval

`approval` pauses the graph with a payload describing the action. The
resume value must be a dict with a boolean `approved` and a non-empty
`decided_by`; anything else raises an error, so a malformed payload can
never count as approval (`parse_decision`). The payload also lists the
action's correlations, so the approver sees how the evidence relates.

## Failure containment

The `investigate` node turns each of these into a finding and an audit
event, and the incident continues to the report:

- a specialist that raises becomes a `failed` finding;
- an unregistered specialist becomes a `blocked` finding;
- an exhausted per-agent or per-incident call budget becomes a `blocked`
  finding.

## Kubernetes agent bridge (`app/supervisor/kubernetes_bridge.py`)

The Supervisor graph is synchronous and the Kubernetes agent is async.
`build_kubernetes_investigator(tools_provider)` runs one investigation
per call with `asyncio.run`. The tools provider is an async context
manager, so a resource such as an MCP session opens and closes on the
same event loop as the agent.

Only `inspect_pod_events` and `gather_additional_diagnostics` may reach
the agent (`KUBERNETES_TOOL_ALLOWLIST`). Any other tool name raises an
error before the agent runs. The agent also gets `approved_actions=set()`
as defence in depth.

## MCP tools provider (`app/supervisor/mcp_tools.py`)

`stdio_tools_provider()` starts the simulated Kubernetes MCP server over
stdio for each investigation, with a 20 second timeout on
`initialize()`, builds the agent's tools with the existing adapter
(`build_mcp_tools`) and closes the session afterwards. The server is
started with `sys.executable`, never a bare `python`. A server that
cannot start becomes a failed finding.
`stdio_observability_investigator()` and `stdio_security_investigator()`
do the same for the two MCP-backed specialists, which open one session
per investigation.

## Shared specialist helpers (`app/supervisor/specialist_common.py`)

The Observability and Security investigators share one pattern, so the
identical parts live here: the target resolver (`service`, falling back
to `deployment`, with a `namespace` required), the allowlist guard, the
runner that contains each tool failure separately, field validation and
the MCP session wrapper. The module knows no allowlist. Each specialist
passes its own, so the allowlists stay independent. Observability still
exports the same names, so its tests did not change.

## Observability investigator (`app/supervisor/observability.py`)

`make_observability_investigator(session_factory)` opens one MCP session
per investigation and calls exactly three read-only tools in a fixed
order: `get_error_rate`, `get_latency_percentiles` and
`get_resource_usage`, over a fixed window. Each call is wrapped
separately, so one failure never stops the other two. A failed call, an
`error` result or a malformed result is recorded as an `execution_failed`
observation, which never counts as evidence.

The target is `incident.service`, falling back to `incident.deployment`,
and a `namespace` is required. A blank value counts as missing, and a
missing target returns a `blocked` finding without opening a session.

The investigator reports symptoms only and never states a cause. For
`inventory-api` it says "Errors elevated, latency elevated, resources
normal." Relating that to a Kubernetes crash loop is the Supervisor's
job, not this specialist's. Thresholds are named constants and
inclusive: error rate 5.0%, p95 1000 ms, CPU 90% and memory 90% of
limits.

| Metrics collected | Status | Evidence quality |
|---|---|---|
| 3 of 3 | `completed` | `limited` |
| 1 or 2 of 3 | `partial` | `limited` |
| 0 of 3 | `failed` | `insufficient` |
| No namespace or service | `blocked` | `insufficient` |

A `partial` finding is deliberately not usable evidence. The evidence
gate requires status `completed`, so an incomplete metrics picture does
not pass even though the metrics that were collected stay in the finding
and the audit trail. An Observability-only incident with a partial
finding therefore ends as `investigation_failed`. This is conservative
for Phase 1.

Architectural rule: only an eligible Kubernetes finding can currently
support a rollout restart proposal. Observability findings can support
investigation and evidence correlation, but cannot independently
authorize remediation.

The tools are allowlisted in `OBSERVABILITY_TOOL_ALLOWLIST`, separate
from the Kubernetes allowlist. The guard rejects every other tool name,
including the four Kubernetes MCP tools and the Security tools, and
that rejection is not swallowed as an ordinary tool failure. The server
itself exposes all ten tools to any client, so the allowlists are the
only separation between the specialists.

## Security investigator (`app/supervisor/security.py`)

`make_security_investigator(session_factory)` opens one MCP session per
investigation and calls exactly three read-only tools in a fixed order:
`get_failed_auth_attempts` and `get_iam_activity` (both over a 15 minute
window) and `check_network_exposure` (no window). Failure handling is the
same as Observability: each call is wrapped separately, and a failed,
`error` or malformed result becomes an `execution_failed` observation
that never counts as evidence. An unknown service returns an error from
the server, never zeros, because missing data must not look like a clean
result.

The target is resolved exactly as for Observability (`service`, falling
back to `deployment`, with a `namespace` required). An incident that
names no workload, such as a leaked key in a repository, a public bucket
or a root login, returns a `blocked` finding without opening a session.

The investigator reports signals only and never states a cause, an
intent or a remedy. Thresholds are named constants and inclusive:
failed authentications 100 per window, access-denied calls 10, credential
age 90 days, and any exposure flag. For `checkout-api` it says "Failed
authentication elevated, IAM access-denied calls elevated, publicly
exposed, ingress open to 0.0.0.0/0." The clean wording is "No elevated
security signals in the checks run." It never says the service is
secure, because the absence of a signal in three checks is not a claim of
safety. Evidence text is built only from validated numbers and booleans,
so free text returned by a tool never reaches a finding.

| Checks collected | Status | Evidence quality |
|---|---|---|
| 3 of 3 | `completed` | `limited` |
| 1 or 2 of 3 | `partial` | `limited` |
| 0 of 3 | `failed` | `insufficient` |
| No namespace or service | `blocked` | `insufficient` |

A `partial` finding is deliberately not usable evidence, for the same
reason as Observability, and it matters more here: a missing exposure
check could hide a hole. Security never produces `corroborated` quality
and never sets `needs_follow_up`.

Architectural rule: only an eligible Kubernetes finding can currently
support a rollout restart proposal. Security findings can support
investigation and evidence correlation, but cannot independently
authorize remediation. There is no Security action type, and a Security
finding is never listed as supporting a proposal, even beside a
Kubernetes finding.

The tools are allowlisted in `SECURITY_TOOL_ALLOWLIST`, separate from the
Kubernetes and Observability allowlists. The Kubernetes allowlist holds
the agent-facing tool names the adapter builds, not MCP tool names, so
the boundaries are tested pairwise rather than as a union: Security
rejects the Observability tools and the two Kubernetes agent tools,
Observability rejects the Security tools, Kubernetes rejects the Security
and Observability tools, and no tool is allowed for more than one
specialist. The partition of the ten server tools into Kubernetes,
metrics and security groups is tested in `test_mcp_server`.

## Safety properties

| Property | Enforced in | Tested by |
|---|---|---|
| No proposal without usable evidence | `proposals.py` | `test_supervisor_components`, `test_supervisor_graph`, `test_supervisor_stdio_integration` (healthy and unknown pod) |
| No execution without approval | Graph edges: `execute` is reachable only through `approval` | `test_supervisor_graph` (rejected path) |
| Malformed approval fails closed | `parse_decision` | `test_supervisor_components`, `test_supervisor_graph` |
| Execution is simulated only | `ExecutionRecord.simulated` is `Literal[True]` | `test_state_models` |
| Kubernetes agent gets only allowlisted read-only tools | `validate_tool_allowlist` | `test_supervisor_kubernetes_integration` |
| A specialist failure does not crash the incident | `investigate` node | `test_supervisor_graph`, `test_supervisor_kubernetes_integration`, `test_supervisor_stdio_integration` |
| Failed tool runs never count as evidence | `adapters.is_execution_failure` | `test_platform_state`, `test_supervisor_mcp_tools` |
| The approval pause survives a process restart | SQLite checkpoint and serde allowlist | `test_platform_lifecycle`, `test_supervisor_graph`, `test_supervisor_stdio_integration` |
| The agent graph does not overwrite the Supervisor's checkpoint | `compile(checkpointer=False)` | `test_supervisor_kubernetes_integration` (approved path) |
| Observability agent gets only allowlisted metrics tools | `guard_call_tool` in `observability.py` | `test_supervisor_observability`, `test_supervisor_kubernetes_integration` (rejects metrics tools) |
| Observability findings never authorize remediation | `propose_action` filters on the Kubernetes agent | `test_supervisor_observability`, `test_supervisor_stdio_integration` (mixed incident) |
| Security agent gets only allowlisted security tools | `guard_call_tool` in `security.py` | `test_supervisor_security`, `test_supervisor_kubernetes_integration` (rejects security tools), `test_mcp_server` (tool partition) |
| Security findings never authorize or support remediation | `propose_action` filters on the Kubernetes agent | `test_supervisor_security` (gate tests), `test_supervisor_stdio_integration` (mixed incident) |
| Tool output cannot inject text into a Security finding | Evidence built from validated numbers and booleans | `test_supervisor_security` |
| A missing target is an error or `blocked`, never a clean result | Server returns `error`; `blocked` and `failed` findings | `test_mcp_server`, `test_supervisor_security`, `test_supervisor_stdio_integration` |
| Correlation never changes whether a proposal exists, its risk or its supporting findings | `propose_action` is unchanged; `annotate_action` only extends rationale text and returns `None` for `None` | `test_supervisor_correlation` (combination test), `test_supervisor_graph`, `test_supervisor_stdio_integration` |
| Only usable findings from two different specialists correlate | `correlate` | `test_supervisor_correlation` |
| Signals come only from checks that succeeded | Specialist `_signals`, `Finding` validator, Kubernetes adapter | `test_supervisor_observability`, `test_supervisor_security`, `test_platform_state`, `test_state_models` |
| Co-occurrence is never read as corroboration or causation | R2 kind and caveat | `test_supervisor_correlation`, `test_supervisor_graph` |
| Correlations are recomputed, never accumulated | `correlate` node | `test_supervisor_graph`, `test_supervisor_correlation` |
| Checkpoints from before correlations existed still resume | `current_correlations` tolerates a missing field | `test_supervisor_graph` |
| One definition of a crash loop | `app/state/signals.py` | `test_state_signals`, `test_platform_state` |
| An incomplete metrics picture is not usable evidence | `is_usable` requires `completed` | `test_supervisor_observability`, `test_supervisor_stdio_integration` (unknown service) |
| A specialist does not run again on resume | Checkpoint written after the `investigate` node | `test_supervisor_graph`, `test_supervisor_stdio_integration` |

## Tests

`test_supervisor_routing`, `test_routing_eval`,
`test_supervisor_components`, `test_supervisor_graph`,
`test_supervisor_correlation`, `test_state_signals`,
`test_supervisor_kubernetes_integration`, `test_supervisor_mcp_tools`, `test_supervisor_observability`,
`test_supervisor_security`, `test_mcp_server` and
`test_supervisor_stdio_integration`, all under `app/`.

## Running

- Run from the repository root. The stdio server is started as
  `python -m app.mcp_server.server`, which needs the root as its working
  directory.
- Use the project venv interpreter explicitly
  (`.venv-rag312/Scripts/python.exe`); a bare `python` can resolve to a
  different install without the project's packages.
- Each fresh Python process takes 25 to 50 seconds to import, because
  `app.retriever` loads sentence-transformers even when no retrieval
  runs. The tests themselves are fast.

## Known limitations

- Only the Kubernetes, Observability and Security specialists have
  investigators. A specialist with no registered investigator produces a
  `blocked` finding ("no investigator registered").
- Security needs a workload. An incident with no namespace and service,
  such as a leaked key in a repository, a public bucket or a root login,
  is routed to Security but produces a `blocked` finding. A field for
  other resources (a repository, bucket or account) would be needed to
  investigate those.
- `build_supervisor_graph` is called only from tests: no production entry
  point builds the investigators mapping yet. That arrives with the
  FastAPI phase.
- Correlation is a first version: three pairwise rules, and nothing
  else is compared. Security findings are not compared with Kubernetes
  findings, because Security makes no claims about pod state. All
  findings in an incident are assumed to share its target, which holds
  while an incident is a single workload.
- A contradiction (C1) is the weakest rule, since normal metrics can
  coexist with one crash-looping replica. It is shown to the approver and
  never blocks a proposal. There is no veto, and no loop yet that would
  ask a specialist for more evidence when findings disagree.
- The proposal gate searches all evidence text, including failed-run
  observations, while the Kubernetes `crash_loop` signal ignores them. A
  failed run whose error text contained "crash loop" could therefore
  satisfy the gate without producing the signal. The effect is safe (no
  annotation, never an extra proposal), and both could be tightened
  together later.
- Only a crash loop produces a proposal. Other failure types (image pull
  errors, OOM kills) investigate but propose nothing.
- Deployment name: the MCP adapter passes the pod name as the deployment
  name to `check_deployment_status`, and `KubernetesInvestigationRequest`
  has no deployment field. This works against the simulated server only
  because pod and deployment names are equal there. Real pods carry
  suffixes.
- `asyncio.run` fails if the calling thread already has a running event
  loop. The Supervisor contains that as a failed finding. In Phase 2 the
  graph must run in a worker thread, or move to `ainvoke` with async
  nodes.
- Routing is rule-based and misses incidents without known keywords.
- There is no loop yet for requesting more evidence, so
  `supervisor_hops_exhausted` is defined but not used by the graph.
- No LLM call is made anywhere in the Supervisor path, so usage and cost
  attribution are not exercised.
- `app/mcp_server/client.py` is a stale demo: it starts the server with a
  bare `python` and omits the `namespace` argument the tools now require.
  The Supervisor does not use it.

## Adding a specialist

1. Add an adapter from its response contract to `Finding` in
   `app/state/adapters.py`.
2. Add an investigator (as in `app/supervisor/investigators.py`) and,
   if it is async, a bridge with its own tool allowlist. An MCP-backed
   specialist can reuse `app/supervisor/specialist_common.py`.
3. Register it in the investigators mapping passed to
   `build_supervisor_graph`.
4. Extend the routing rules and the eval set and re-run the evaluation.
   Add its signals to the closed vocabulary (emitted only from checks
   that succeeded) and any correlation rules, with tests that the rules
   cannot change a proposal.
5. Add a proposal rule only for evidence that justifies one.