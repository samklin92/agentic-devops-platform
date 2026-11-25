# Supervisor Graph

Status: Phase 1. Implemented in `app/supervisor/`. State records are
documented in `docs/shared-state-schema.md`.

## Purpose

The Supervisor receives an incident, routes it to specialist agents,
evaluates their findings, proposes a remediation only when the evidence
supports one, pauses for human approval, simulates the approved action
and produces the final report. Phase 1 is propose-only: nothing real is
executed.

## Flow (`app/supervisor/graph.py`)

| Node | What it does | Next |
|---|---|---|
| `receive_alert` | Records the alert | `route` |
| `route` | Rule-based routing; no match means no specialist | `investigate`, or `report` if nothing matched |
| `investigate` | Calls each selected specialist within its call budget | `evaluate` |
| `evaluate` | Builds a proposal from usable evidence, or none | `approval`, or `report` if no proposal |
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

Measured by `python -m app.evaluate_routing` on 35 labelled incidents
(`evals/routing_questions.json`). At the time of writing the baseline
gets 33 of 35 exactly right, with precision 1.00 for every specialist
and recall 0.93 for Kubernetes and Observability. The two misses are
deliberately keyword-free cases marked `hard` (an exit code 137 restart
loop and a vague customer-impact report). They mark the limit of rules
and the bar an LLM router would have to beat. The set was extended when
the Observability specialist was added (mixed crash-loop and HTTP 500
cases, metrics-only cases, and a pod with no metrics signal that must
not route to Observability), and Kubernetes precision did not change.

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

## Approval

`approval` pauses the graph with a payload describing the action. The
resume value must be a dict with a boolean `approved` and a non-empty
`decided_by`; anything else raises an error, so a malformed payload can
never count as approval (`parse_decision`).

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
including the four Kubernetes MCP tools, and that rejection is not
swallowed as an ordinary tool failure. The server itself exposes all
seven tools to any client, so the allowlists are the only separation
between the specialists.

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
| The agent graph does not overwrite the Supervisor's checkpoint | `compile(checkpointer=False)` | `test_supervisor_kubernetes_integration` (approved path) | Observability agent gets only allowlisted metrics tools | `guard_call_tool` in `observability.py` | `test_supervisor_observability` |
| Observability findings never authorize remediation | `propose_action` filters on the Kubernetes agent | `test_supervisor_observability`, `test_supervisor_stdio_integration` (mixed incident) |
| An incomplete metrics picture is not usable evidence | `is_usable` requires `completed` | `test_supervisor_observability`, `test_supervisor_stdio_integration` (unknown service) |
| A specialist does not run again on resume | Checkpoint written after the `investigate` node | `test_supervisor_graph`, `test_supervisor_stdio_integration` |

## Tests

`test_supervisor_routing`, `test_routing_eval`,
`test_supervisor_components`, `test_supervisor_graph`,
`test_supervisor_kubernetes_integration`, `test_supervisor_mcp_tools` and
`test_supervisor_observability` and `test_supervisor_stdio_integration`,
all under `app/`.

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

- Only the Kubernetes and Observability specialists have investigators.
  Incidents routed to Security produce a `blocked` finding ("no
  investigator registered").
- `build_supervisor_graph` is called only from tests: no production entry
  point builds the investigators mapping yet. That arrives with the
  FastAPI phase.
- There is no evidence correlation yet. Observability and Kubernetes
  findings about the same workload are reported side by side, and
  `Finding.contradicts` and `corroborated` quality are unused.
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
   if it is async, a bridge with its own tool allowlist.
3. Register it in the investigators mapping passed to
   `build_supervisor_graph`.
4. Extend the signal rules and the routing eval set, and re-run the
   evaluation.
5. Add a proposal rule only for evidence that justifies one.