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

Measured by `python -m app.evaluate_routing` on 28 labelled incidents
(`evals/routing_questions.json`). At the time of writing the baseline
gets 26 of 28 exactly right, with precision 1.00 for every specialist.
The two misses are deliberately keyword-free cases marked `hard` (an exit
code 137 restart loop and a vague customer-impact report). They mark the
limit of rules and the bar an LLM router would have to beat.

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
| A specialist does not run again on resume | Checkpoint written after the `investigate` node | `test_supervisor_graph`, `test_supervisor_stdio_integration` |

## Tests

`test_supervisor_routing`, `test_routing_eval`,
`test_supervisor_components`, `test_supervisor_graph`,
`test_supervisor_kubernetes_integration`, `test_supervisor_mcp_tools` and
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

- Only the Kubernetes specialist has an investigator. Incidents routed to
  Observability or Security produce a `blocked` finding ("no
  investigator registered").
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