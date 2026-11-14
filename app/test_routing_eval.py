import pytest

from app.evaluate_routing import evaluate, load_cases, validate_cases


def test_eval_file_is_valid_and_covers_every_category_and_agent():
    cases = load_cases()
    assert len(cases) >= 20
    assert {c["category"] for c in cases} == {"single", "multi", "out_of_scope"}
    assert {a for c in cases for a in c["expected"]} == {
        "kubernetes",
        "observability",
        "security",
    }


def test_validation_rejects_bad_cases():
    with pytest.raises(ValueError):
        validate_cases(
            [{"id": "x", "incident": "t", "expected": ["terraform"], "category": "single"}]
        )
    with pytest.raises(ValueError):
        validate_cases(
            [{"id": "x", "incident": "t", "expected": ["kubernetes"], "category": "multi"}]
        )
    duplicate = {"id": "x", "incident": "t", "expected": [], "category": "out_of_scope"}
    with pytest.raises(ValueError):
        validate_cases([duplicate, dict(duplicate)])


def test_metric_math_with_a_stub_router():
    cases = [
        {"id": "a", "incident": "a", "expected": ["kubernetes"], "category": "single"},
        {"id": "b", "incident": "b", "expected": ["kubernetes"], "category": "single"},
        {
            "id": "c",
            "incident": "c",
            "expected": ["observability"],
            "category": "single",
        },
    ]
    predictions = {
        "a": ["kubernetes"],
        "b": [],
        "c": ["kubernetes", "observability"],
    }
    report = evaluate(cases, router=predictions.__getitem__)

    assert report.exact_matches == 1
    kubernetes = report.per_agent["kubernetes"]
    assert (kubernetes.true_positives, kubernetes.false_positives) == (1, 1)
    assert kubernetes.false_negatives == 1
    assert kubernetes.precision == 0.5 and kubernetes.recall == 0.5
    assert report.per_agent["observability"].precision == 1.0
    assert report.per_agent["security"].precision is None
    assert [m["id"] for m in report.misses] == ["b", "c"]


def test_out_of_scope_accounting():
    cases = [
        {"id": "a", "incident": "a", "expected": [], "category": "out_of_scope"},
        {"id": "b", "incident": "b", "expected": [], "category": "out_of_scope"},
    ]
    predictions = {"a": [], "b": ["security"]}
    report = evaluate(cases, router=predictions.__getitem__)
    assert report.out_of_scope_total == 2
    assert report.out_of_scope_correct == 1


def test_baseline_router_meets_its_gate_on_core_cases():
    core = [case for case in load_cases() if not case.get("hard")]
    report = evaluate(core)
    assert report.exact_accuracy >= 0.9
    assert report.out_of_scope_correct == report.out_of_scope_total