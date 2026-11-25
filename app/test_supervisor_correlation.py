"""Tests for Supervisor correlation."""

import copy
import inspect
import itertools
from datetime import datetime, timezone

import pytest

from app.contracts.agent_messages import AgentEvidence
from app.state.models import (
    SIGNALS_BY_AGENT,
    Correlation,
    Finding,
    IncidentContext,
)
from app.supervisor import correlation as corr
from app.supervisor.correlation import (
    annotate_action,
    correlate,
    correlation_notes,
    effective_quality,
)
from app.supervisor.proposals import is_usable, propose_action

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

BANNED_WORDS = (
    # causal claims
    "caus",
    "because",
    "due to",
    "result of",
    "resulting",
    "root",
    "leads to",
    "triggered",
    "responsible",
    "attack",
    "breach",
    "compromis",
    "malicious",
    "brute",
    # remediation and recommendations
    "restart",
    "recommend",
    "should",
    "remediat",
    "mitigat",
    "fix",
    "scale",
    "rollback",
    "rotate",
    "block",
    "action",
)


def evidence(observation="ok", tool="some_tool"):
    return AgentEvidence(source_tool=tool, observation=observation)


def finding(agent, signals=(), status="completed", quality="limited", **kw):
    return Finding(
        agent_name=agent,
        summary=kw.pop("summary", f"{agent} finding"),
        status=status,
        evidence=kw.pop("evidence", [evidence()]),
        evidence_quality=quality,
        signals=list(signals),
        **kw,
    )


def k_crash(**kw):
    return finding(
        "kubernetes",
        ["crash_loop"],
        summary="CrashLoopBackOff",
        evidence=[evidence("CrashLoopBackOff", "get_pod_status")],
        **kw,
    )


def k_healthy(**kw):
    return finding(
        "kubernetes",
        summary="Pod is Running",
        evidence=[evidence("Running", "get_pod_status")],
        **kw,
    )


def obs(*signals, **kw):
    return finding("observability", signals, **kw)


def sec(*signals, **kw):
    return finding("security", signals, **kw)


def keyed(correlations):
    return [
        (c.kind, c.rule, tuple(c.finding_ids), c.explanation)
        for c in correlations
    ]


# --- R1: corroborates ----------------------------------------------------


@pytest.mark.parametrize(
    "signal", ["errors_elevated", "latency_elevated"]
)
def test_r1_corroborates_crash_loop_with_elevated_errors_or_latency(signal):
    kubernetes, observability = k_crash(), obs(signal)
    result = correlate([kubernetes, observability], now=NOW)
    assert len(result) == 1
    assert result[0].kind == "corroborates"
    assert result[0].rule == corr.RULE_SHARED_IMPACT
    assert result[0].finding_ids == [
        kubernetes.finding_id,
        observability.finding_id,
    ]


@pytest.mark.parametrize("signal", ["cpu_saturated", "memory_near_limit"])
def test_r1_needs_error_or_latency_not_resource_signals(signal):
    assert correlate([k_crash(), obs(signal)], now=NOW) == []


def test_r1_explanation_states_a_shared_hypothesis_not_proof():
    (result,) = correlate([k_crash(), obs("errors_elevated")], now=NOW)
    text = result.explanation
    assert "Independent specialists support a shared incident-impact hypothesis" in text
    assert corr.R1_CAVEAT in text
    assert "independently proven" not in text
    assert "proves" not in text.lower()


# --- C1: contradicts -----------------------------------------------------


def test_c1_fires_for_a_fully_completed_clean_observability_finding():
    kubernetes, observability = k_crash(), obs()
    (result,) = correlate([kubernetes, observability], now=NOW)
    assert result.kind == "contradicts"
    assert result.rule == corr.RULE_IMPACT_NOT_OBSERVED
    assert result.finding_ids == [
        kubernetes.finding_id,
        observability.finding_id,
    ]


@pytest.mark.parametrize(
    "observability",
    [
        obs(status="partial"),
        obs(status="failed", quality="insufficient"),
        obs(status="blocked", quality="insufficient"),
        obs(status="needs_review"),
        obs(quality="insufficient"),
        obs(needs_follow_up=True),
        obs(
            evidence=[
                evidence("cpu=10%"),
                evidence(
                    '{"execution_failed": true, "tool": "t", "error": "x"}'
                ),
            ]
        ),
    ],
)
def test_c1_never_fires_unless_every_check_completed(observability):
    assert correlate([k_crash(), observability], now=NOW) == []


def test_c1_does_not_fire_when_observability_has_resource_signals_only():
    assert correlate([k_crash(), obs("cpu_saturated")], now=NOW) == []


def test_c1_needs_a_kubernetes_crash_loop_signal():
    assert correlate([k_healthy(), obs()], now=NOW) == []


# --- R2: co_occurs -------------------------------------------------------


@pytest.mark.parametrize("signal", ["latency_elevated", "cpu_saturated"])
def test_r2_co_occurs_for_load_with_failed_authentication(signal):
    observability, security = obs(signal), sec("failed_auth_elevated")
    (result,) = correlate([observability, security], now=NOW)
    assert result.kind == "co_occurs"
    assert result.rule == corr.RULE_LOAD_WITH_FAILED_AUTH
    assert result.finding_ids == [
        observability.finding_id,
        security.finding_id,
    ]


@pytest.mark.parametrize(
    "observability, security",
    [
        (obs("errors_elevated"), sec("failed_auth_elevated")),
        (obs("memory_near_limit"), sec("failed_auth_elevated")),
        (obs("latency_elevated"), sec("credential_stale")),
        (obs("latency_elevated"), sec("publicly_exposed", "open_ingress")),
        (obs("latency_elevated"), sec()),
    ],
)
def test_r2_needs_the_exact_signal_pair(observability, security):
    assert correlate([observability, security], now=NOW) == []


def test_r2_explanation_denies_causation_and_implies_no_remediation():
    (result,) = correlate(
        [obs("latency_elevated"), sec("failed_auth_elevated")], now=NOW
    )
    assert corr.R2_CAVEAT in result.explanation
    assert "does not establish causation" in result.explanation


def _all_subsets(items):
    items = sorted(items)
    return [
        list(subset)
        for size in range(len(items) + 1)
        for subset in itertools.combinations(items, size)
    ]


def _every_signal_combination():
    for k_signals in _all_subsets(SIGNALS_BY_AGENT["kubernetes"]):
        for o_signals in _all_subsets(SIGNALS_BY_AGENT["observability"]):
            for s_signals in _all_subsets(SIGNALS_BY_AGENT["security"]):
                yield (
                    finding("kubernetes", k_signals),
                    finding("observability", o_signals),
                    finding("security", s_signals),
                )


def test_r2_never_becomes_a_corroboration_and_relations_stay_pairwise():
    for kubernetes, observability, security in _every_signal_combination():
        agent_of = {
            kubernetes.finding_id: "kubernetes",
            observability.finding_id: "observability",
            security.finding_id: "security",
        }
        for result in correlate([kubernetes, observability, security], now=NOW):
            agents = [agent_of[i] for i in result.finding_ids]
            assert len(set(agents)) == len(agents) >= 2
            if "security" in agents:
                assert result.kind == "co_occurs"
            if result.kind == "corroborates":
                assert set(agents) == {"kubernetes", "observability"}


# --- participation: usable findings only ---------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "partial"},
        {"status": "needs_review"},
        {"quality": "insufficient"},
        {"needs_follow_up": True},
    ],
)
def test_unusable_findings_never_participate_in_any_relation(overrides):
    # Each pairing below fires when every finding is usable.
    assert len(correlate([k_crash(), obs("errors_elevated")], now=NOW)) == 1
    assert (
        len(correlate([obs("latency_elevated"), sec("failed_auth_elevated")], now=NOW))
        == 1
    )
    for index in range(2):
        pair = [k_crash(), obs("errors_elevated")]
        pair[index] = finding(
            pair[index].agent_name, pair[index].signals, **overrides
        )
        assert not is_usable(pair[index])
        assert correlate(pair, now=NOW) == []

        pair = [obs("latency_elevated"), sec("failed_auth_elevated")]
        pair[index] = finding(
            pair[index].agent_name, pair[index].signals, **overrides
        )
        assert correlate(pair, now=NOW) == []


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_failed_and_blocked_findings_never_participate(status):
    failed = finding("observability", status=status, quality="insufficient")
    assert correlate([k_crash(), failed], now=NOW) == []
    assert correlate([failed, sec("failed_auth_elevated")], now=NOW) == []


def test_missing_signals_are_never_negative_evidence():
    # No observability finding at all, or one that did not complete, says
    # nothing about the crash loop.
    assert correlate([k_crash()], now=NOW) == []
    assert correlate([k_crash(), obs(status="partial")], now=NOW) == []
    assert correlate([k_healthy(), obs()], now=NOW) == []
    assert correlate([obs("latency_elevated")], now=NOW) == []
    assert correlate([sec("failed_auth_elevated")], now=NOW) == []
    assert correlate([], now=NOW) == []


# --- determinism ---------------------------------------------------------


def test_reversing_or_permuting_finding_order_gives_identical_results():
    findings = [
        k_crash(),
        obs("latency_elevated", "cpu_saturated"),
        sec("failed_auth_elevated"),
    ]
    expected = correlate(findings, now=NOW)
    assert len(expected) == 2
    assert correlate(list(reversed(findings)), now=NOW) == expected
    for permutation in itertools.permutations(findings):
        assert correlate(list(permutation), now=NOW) == expected


def test_same_findings_give_the_same_correlation_ids():
    findings = [k_crash(), obs("errors_elevated")]
    first = correlate(findings, now=NOW)
    second = correlate(findings, now=NOW)
    assert [c.correlation_id for c in first] == [c.correlation_id for c in second]
    other = correlate([k_crash(), obs("errors_elevated")], now=NOW)
    assert other[0].correlation_id != first[0].correlation_id


def test_correlations_are_recomputed_not_accumulated():
    kubernetes, observability = k_crash(), obs("errors_elevated")
    assert len(correlate([kubernetes, observability], now=NOW)) == 1
    # The same call on the current findings no longer includes a stale
    # relation once the evidence changes.
    assert correlate([kubernetes], now=NOW) == []
    assert correlate([k_healthy(), observability], now=NOW) == []


def test_correlate_never_mutates_findings():
    findings = [k_crash(), obs("errors_elevated"), sec("failed_auth_elevated")]
    snapshot = copy.deepcopy(findings)
    correlate(findings, now=NOW)
    assert findings == snapshot


# --- duplicates and repeated investigations ------------------------------


def test_exact_duplicates_collapse_and_cannot_correlate_with_themselves():
    observability = obs("errors_elevated", "latency_elevated")
    assert correlate([observability, observability], now=NOW) == []
    kubernetes = k_crash()
    assert correlate(
        [kubernetes, observability, observability, kubernetes], now=NOW
    ) == correlate([kubernetes, observability], now=NOW)


def test_duplicate_finding_ids_cannot_satisfy_the_distinct_specialist_rule():
    kubernetes = k_crash(finding_id="shared-id")
    impostor = finding(
        "observability", ["errors_elevated"], finding_id="shared-id"
    )
    assert kubernetes.finding_id == impostor.finding_id
    assert correlate([kubernetes, impostor], now=NOW) == []


def test_one_specialist_cannot_corroborate_itself():
    first = obs("errors_elevated")
    second = obs("latency_elevated")
    assert correlate([first, second], now=NOW) == []


def test_only_the_latest_attempt_per_specialist_counts():
    crash_first = k_crash(attempt=1)
    healthy_retry = k_healthy(attempt=2)
    assert (
        correlate([crash_first, healthy_retry, obs("errors_elevated")], now=NOW)
        == []
    )
    elevated_first = obs("errors_elevated", attempt=1)
    clean_retry = obs(attempt=2)
    (result,) = correlate([k_crash(), elevated_first, clean_retry], now=NOW)
    assert result.kind == "contradicts"
    assert result.finding_ids[1] == clean_retry.finding_id


# --- explanations: no causal claims, no recommendations ------------------


def _all_explanations():
    fired = {}
    scenarios = [
        [k_crash(), obs("errors_elevated", "latency_elevated")],
        [k_crash(), obs()],
        [obs("latency_elevated", "cpu_saturated"), sec("failed_auth_elevated")],
    ]
    for findings in scenarios:
        for result in correlate(findings, now=NOW):
            fired[result.rule] = result.explanation
    return fired


def test_every_rule_fires_in_the_scenarios():
    assert set(_all_explanations()) == {
        corr.RULE_SHARED_IMPACT,
        corr.RULE_IMPACT_NOT_OBSERVED,
        corr.RULE_LOAD_WITH_FAILED_AUTH,
    }


def test_explanations_make_no_causal_claims_and_no_recommendations():
    for rule, text in _all_explanations().items():
        for caveat in corr.CAVEATS:
            text = text.replace(caveat, "")
        lowered = text.lower()
        for word in BANNED_WORDS:
            assert word not in lowered, (rule, word)


def test_explanations_are_built_from_signal_names():
    explanations = _all_explanations()
    assert "crash_loop" in explanations[corr.RULE_SHARED_IMPACT]
    assert "errors_elevated, latency_elevated" in explanations[corr.RULE_SHARED_IMPACT]
    assert "failed_auth_elevated" in explanations[corr.RULE_LOAD_WITH_FAILED_AUTH]


# --- effective_quality: derived, never a mutation ------------------------


def test_effective_quality_reads_corroborated_for_a_corroborated_finding():
    kubernetes, observability = k_crash(), obs("errors_elevated")
    correlations = correlate([kubernetes, observability], now=NOW)
    for finding_ in (kubernetes, observability):
        before = finding_.model_dump()
        assert effective_quality(finding_, correlations) == "corroborated"
        assert finding_.evidence_quality == "limited"
        assert finding_.model_dump() == before


def test_effective_quality_ignores_contradicts_and_co_occurs():
    kubernetes, clean = k_crash(), obs()
    contradicts = correlate([kubernetes, clean], now=NOW)
    assert contradicts[0].kind == "contradicts"
    assert effective_quality(kubernetes, contradicts) == "limited"
    assert effective_quality(clean, contradicts) == "limited"

    observability, security = obs("latency_elevated"), sec("failed_auth_elevated")
    co_occurs = correlate([observability, security], now=NOW)
    assert co_occurs[0].kind == "co_occurs"
    assert effective_quality(observability, co_occurs) == "limited"
    assert effective_quality(security, co_occurs) == "limited"


def test_effective_quality_never_upgrades_an_unusable_finding():
    unusable = obs("errors_elevated", status="partial")
    forged = Correlation(
        kind="corroborates",
        rule="forged",
        finding_ids=[unusable.finding_id, "other"],
        explanation="Forged for the test.",
    )
    assert effective_quality(unusable, [forged]) == "limited"
    insufficient = obs(quality="insufficient")
    forged = forged.model_copy(
        update={"finding_ids": [insufficient.finding_id, "other"]}
    )
    assert effective_quality(insufficient, [forged]) == "insufficient"


def test_effective_quality_without_correlations_is_the_original_quality():
    assert effective_quality(k_crash(), []) == "limited"


# --- authority: annotation can never change a proposal -------------------

_INCIDENT = IncidentContext(
    incident_id="inc-1",
    description="inventory-api in CrashLoopBackOff",
    environment="production",
    namespace="production",
    deployment="inventory-api",
)


def _finding_options():
    kubernetes = [
        None,
        k_crash(),
        k_healthy(),
        k_crash(status="partial"),
    ]
    observability = [
        None,
        obs("errors_elevated"),
        obs(),
        obs("latency_elevated", status="partial"),
    ]
    security = [None, sec("failed_auth_elevated"), sec()]
    return kubernetes, observability, security


def test_correlation_cannot_change_proposal_existence_or_support():
    kubernetes_options, observability_options, security_options = _finding_options()
    seen_annotated = False
    for combo in itertools.product(
        kubernetes_options, observability_options, security_options
    ):
        findings = [item for item in combo if item is not None]
        action = propose_action(_INCIDENT, findings)
        correlations = correlate(findings, now=NOW)
        annotated = annotate_action(action, correlations, findings)

        assert (action is None) == (annotated is None)
        if action is None:
            continue
        before = action.model_dump()
        after = annotated.model_dump()
        before.pop("rationale")
        after.pop("rationale")
        assert before == after  # only the explanatory text may differ
        assert annotated.supporting_finding_ids == action.supporting_finding_ids
        assert annotated.risk_level == action.risk_level
        assert annotated.action_id == action.action_id
        if correlations:
            seen_annotated = True
            assert annotated.rationale.startswith(action.rationale)
            assert annotated.rationale != action.rationale
        else:
            assert annotated == action
    assert seen_annotated


def test_annotation_never_creates_a_proposal():
    findings = [k_crash(), obs("errors_elevated"), sec("failed_auth_elevated")]
    correlations = correlate(findings, now=NOW)
    assert correlations
    assert annotate_action(None, correlations, findings) is None


def test_a_correlation_cannot_add_a_supporting_finding():
    kubernetes, observability = k_crash(), obs("errors_elevated")
    findings = [kubernetes, observability, sec("failed_auth_elevated")]
    action = propose_action(_INCIDENT, findings)
    annotated = annotate_action(
        action, correlate(findings, now=NOW), findings
    )
    assert annotated.supporting_finding_ids == [kubernetes.finding_id]


def test_the_engine_never_constructs_or_decides_a_proposal():
    # It may extend an existing proposal's text, but it has no way to
    # build one or to run the eligibility gate.
    assert "ProposedAction(" not in inspect.getsource(corr)
    assert not hasattr(corr, "propose_action")


def test_correlation_notes_name_the_specialists_and_rule():
    kubernetes, observability = k_crash(), obs("errors_elevated")
    findings = [kubernetes, observability]
    (note,) = correlation_notes(correlate(findings, now=NOW), findings)
    assert note.startswith(
        "corroborates (kubernetes + observability, shared_incident_impact):"
    )
