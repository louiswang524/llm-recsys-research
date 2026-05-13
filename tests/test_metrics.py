import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import math
from llm_recsys.evaluation.metrics import hit_at_k, ndcg_at_k, mrr, compute_metrics


# ── hit_at_k ─────────────────────────────────────────────────────────────────

def test_hit_at_k_target_at_top():
    assert hit_at_k(["a", "b", "c"], "a", k=1) == 1.0

def test_hit_at_k_target_outside_k():
    assert hit_at_k(["a", "b", "c"], "c", k=2) == 0.0

def test_hit_at_k_target_exactly_at_k():
    assert hit_at_k(["a", "b", "c"], "c", k=3) == 1.0

def test_hit_at_k_target_missing():
    assert hit_at_k(["a", "b", "c"], "z", k=10) == 0.0


# ── ndcg_at_k ────────────────────────────────────────────────────────────────

def test_ndcg_at_k_rank_1():
    # rank 1 → log2(1+2) = log2(3) ≈ 1.585, NDCG = 1/log2(2) = 1.0
    assert ndcg_at_k(["a", "b", "c"], "a", k=3) == pytest_approx(1.0 / math.log2(2))

def test_ndcg_at_k_rank_2():
    assert ndcg_at_k(["a", "b", "c"], "b", k=3) == pytest_approx(1.0 / math.log2(3))

def test_ndcg_at_k_outside_k():
    assert ndcg_at_k(["a", "b", "c"], "c", k=2) == 0.0

def test_ndcg_at_k_missing():
    assert ndcg_at_k(["a", "b"], "z", k=5) == 0.0

def test_ndcg_decreases_with_rank():
    a = ndcg_at_k(["x", "a", "b"], "a", k=3)
    b = ndcg_at_k(["a", "x", "b"], "a", k=3)
    assert b > a  # better rank → higher NDCG


# ── mrr ──────────────────────────────────────────────────────────────────────

def test_mrr_rank_1():
    assert mrr(["a", "b", "c"], "a") == pytest_approx(1.0)

def test_mrr_rank_2():
    assert mrr(["a", "b", "c"], "b") == pytest_approx(0.5)

def test_mrr_rank_3():
    assert mrr(["a", "b", "c"], "c") == pytest_approx(1 / 3)

def test_mrr_missing():
    assert mrr(["a", "b", "c"], "z") == 0.0


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_compute_metrics_perfect():
    # All targets at rank 1
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    targets = ["a", "x"]
    m = compute_metrics(ranked, targets, k_values=[1, 3])
    assert m["HR@1"] == 1.0
    assert m["HR@3"] == 1.0
    assert m["MRR"] == 1.0

def test_compute_metrics_no_hits():
    ranked = [["a", "b", "c"]]
    targets = ["z"]
    m = compute_metrics(ranked, targets, k_values=[1, 3])
    assert m["HR@1"] == 0.0
    assert m["HR@3"] == 0.0
    assert m["MRR"] == 0.0

def test_compute_metrics_returns_all_keys():
    m = compute_metrics([["a"]], ["a"], k_values=[1, 5, 10])
    for k in [1, 5, 10]:
        assert f"HR@{k}" in m
        assert f"NDCG@{k}" in m
    assert "MRR" in m

def test_compute_metrics_partial():
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    targets = ["a", "z"]  # first is hit@1, second is hit@3 only
    m = compute_metrics(ranked, targets, k_values=[1, 3])
    assert m["HR@1"] == pytest_approx(0.5)
    assert m["HR@3"] == pytest_approx(1.0)

def test_compute_metrics_ndcg_order():
    # second user has target at rank 3 — NDCG should be < HR
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    targets = ["a", "z"]
    m = compute_metrics(ranked, targets, k_values=[3])
    assert m["NDCG@3"] < m["HR@3"]


# ── tiny helper — no pytest dep required ─────────────────────────────────────

def pytest_approx(val, rel=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - val) < rel * max(abs(val), 1e-12)
        def __repr__(self):
            return f"approx({val})"
    return _Approx()
