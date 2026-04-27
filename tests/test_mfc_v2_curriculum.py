# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Unit tests for the Monotone Frontier Curriculum **v2** (minimal variant).

Loads ``verl/utils/curriculum.py`` in isolation so ``import verl`` (which
pulls ``ray``) is not required.  These tests must pass alongside the v1
suite in ``test_mfc_curriculum.py`` — v2 is additive, v1 must not regress.
"""

import importlib.util
from pathlib import Path
from unittest import mock


def _load_curriculum_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "verl" / "utils" / "curriculum.py"
    spec = importlib.util.spec_from_file_location(
        "cgrpo_curriculum_standalone_mfc_v2", path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_curriculum_module()
MFCV2 = _mod.MonotoneFrontierCurriculumStateV2
MFCV1 = _mod.MonotoneFrontierCurriculumState
AdaBack = _mod.PerSampleCurriculumState


# -- Init & API parity -----------------------------------------------------


def test_v2_initial_rho_max_default():
    st = MFCV2(default_rho_max=1.0, tau=0.5)
    rho = st.get_rho("s0", num_steps=10)
    assert st.states["s0"]["rho_max"] == 1.0
    assert 0.0 <= rho <= 1.0


def test_v2_default_tau_is_0_5():
    st = MFCV2()
    assert st.tau == 0.5


def test_v2_state_dict_includes_variant_v2():
    sd = MFCV2().state_dict()
    assert sd["method"] == "mfc"
    assert sd["variant"] == "v2"
    assert "tau" in sd


def test_v2_load_rejects_v1_state_dict():
    """Cross-variant load must raise to prevent silent state corruption."""
    v2 = MFCV2()
    bad = {"method": "mfc", "variant": "v1", "p_probe": 0.25, "states": {}}
    try:
        v2.load_state_dict(bad)
    except ValueError as e:
        assert "v2" in str(e)
    else:
        raise AssertionError("expected ValueError on v1->v2 load")


def test_v2_load_accepts_unmarked_state_dict():
    """Backward compat: state_dict without `variant` key is accepted (treated as v2)."""
    v2 = MFCV2(tau=0.4)
    v2.load_state_dict({"tau": 0.7, "states": {"x": {"rho_max": 0.3, "rho": 0.0,
                                                      "last_success": False,
                                                      "visits": 1, "num_steps": 8}}})
    assert v2.tau == 0.7
    assert "x" in v2.states


# -- Sampling: discrete Uniform{0, ..., g_curr} ---------------------------


def test_v2_sample_uses_discrete_lattice():
    """Each visit must draw an integer step count and yield exactly k/n."""
    st = MFCV2()
    st.get_rho("a", num_steps=10)  # register
    st.states["a"]["rho_max"] = 0.4  # g_curr = round(0.4 * 10) = 4
    with mock.patch.object(_mod.random, "randint", return_value=2) as mr:
        rho = st.get_rho("a", num_steps=10)
    mr.assert_called_with(0, 4)
    assert rho == 0.2
    assert st.states["a"]["rho"] == 0.2


def test_v2_sampled_rho_is_always_on_lattice():
    """Every sampled rho is exactly k / num_steps for some k."""
    import random as r

    r.seed(42)
    st = MFCV2()
    n = 16
    st.get_rho("a", num_steps=n)
    st.states["a"]["rho_max"] = 0.5  # g_curr = 8
    for _ in range(500):
        rho = st.get_rho("a", num_steps=n)
        k = round(rho * n)
        assert abs(rho - k / n) < 1e-9, f"non-lattice rho sampled: {rho}"
        assert 0 <= k <= 8, f"out-of-range step count: {k}"


def test_v2_sample_includes_both_endpoints():
    """g=0 (target) and g=g_curr (frontier) must BOTH be reachable."""
    st = MFCV2()
    st.get_rho("a", num_steps=10)
    st.states["a"]["rho_max"] = 0.5  # g_curr = 5
    # Force g=0 (the on-target lattice point)
    with mock.patch.object(_mod.random, "randint", return_value=0):
        assert st.get_rho("a", num_steps=10) == 0.0
    # Force g=5 (the frontier itself; included in v2's discrete set)
    with mock.patch.object(_mod.random, "randint", return_value=5):
        assert st.get_rho("a", num_steps=10) == 0.5


def test_v2_discrete_uniform_is_unbiased():
    """Empirical histogram over {0, 1, ..., g_curr} must be near-uniform.

    Locks down the structural fix vs. continuous Uniform(0, rho_max),
    where the endpoints would each get half-width probability mass.
    """
    import random as r

    r.seed(7)
    st = MFCV2()
    n = 10
    st.get_rho("a", num_steps=n)
    st.states["a"]["rho_max"] = 0.5  # g_curr = 5 -> 6 levels
    counts = [0] * 6
    N = 6000
    for _ in range(N):
        rho = st.get_rho("a", num_steps=n)
        counts[round(rho * n)] += 1
    expected = N / 6
    # ±15% tolerance — well within multinomial fluctuation at N=6000
    for k, c in enumerate(counts):
        assert 0.85 * expected < c < 1.15 * expected, (
            f"non-uniform: counts={counts}, level {k} got {c} (expected {expected:.0f})"
        )


def test_v2_sample_at_zero_when_rho_max_is_zero():
    st = MFCV2()
    st.get_rho("a", num_steps=10)
    st.states["a"]["rho_max"] = 0.0
    rho = st.get_rho("a", num_steps=10)
    assert rho == 0.0


def test_v2_g_curr_zero_collapses_to_deterministic_zero():
    """When rho_max corresponds to g_curr=0, the sampler must always yield 0."""
    st = MFCV2()
    st.get_rho("a", num_steps=10)
    # rho_max=0.04 → round(0.4)=0 → g_curr=0; randint(0, 0) = 0 always
    st.states["a"]["rho_max"] = 0.04
    for _ in range(50):
        assert st.get_rho("a", num_steps=10) == 0.0


def test_v2_degenerate_num_steps_returns_zero():
    """num_steps == 0 has no lattice; sampler degenerates to 0 deterministically."""
    st = MFCV2()
    rho = st.get_rho("a", num_steps=0)
    assert rho == 0.0


# -- Update: monotone descent + non-regression ----------------------------


def test_v2_update_descends_on_success():
    st = MFCV2(tau=0.5)
    st.get_rho("a", num_steps=10)  # register
    st.states["a"]["rho_max"] = 0.6
    st.states["a"]["rho"] = 0.4
    st.update("a", avg_reward=0.7, num_steps=10)
    assert st.states["a"]["rho_max"] == 0.4
    assert st.states["a"]["last_success"] is True
    assert st.states["a"]["visits"] == 1


def test_v2_update_no_descent_when_rho_used_above_rho_max():
    """Even if successful, rho_max must not increase."""
    st = MFCV2(tau=0.5)
    st.get_rho("a", num_steps=10)
    st.states["a"]["rho_max"] = 0.3
    st.states["a"]["rho"] = 0.5  # somehow higher than rho_max (shouldn't happen via sampling)
    st.update("a", avg_reward=0.9, num_steps=10)
    assert st.states["a"]["rho_max"] == 0.3  # unchanged


def test_v2_failure_does_not_change_rho_max():
    st = MFCV2(tau=0.5)
    st.get_rho("a", num_steps=10)
    st.states["a"]["rho_max"] = 0.6
    st.states["a"]["rho"] = 0.3
    st.update("a", avg_reward=0.1, num_steps=10)
    assert st.states["a"]["rho_max"] == 0.6
    assert st.states["a"]["last_success"] is False
    # No rho_min creation, no rho_min field at all
    assert "rho_min" not in st.states["a"]


def test_v2_repeated_failures_do_not_lock_out_low_rho():
    """Critical structural difference vs AdaBack: failure at low rho must NOT
    raise a lower bound; future visits can still try low rho."""
    st = MFCV2(tau=0.5)
    st.get_rho("a", num_steps=10)
    st.states["a"]["rho_max"] = 0.5
    # 5 failures at the lowest lattice rho (rho=0).
    for _ in range(5):
        st.states["a"]["rho"] = 0.0
        st.update("a", avg_reward=0.0, num_steps=10)
    # rho_max must still allow sampling at g=0 next visit.
    assert st.states["a"]["rho_max"] == 0.5
    # Force the sampler to pick g=0 (the on-target rho); v2 must still allow
    # this visit to land at rho=0.0 — no rho_min-style lockout.
    with mock.patch.object(_mod.random, "randint", return_value=0):
        rho = st.get_rho("a", num_steps=10)
    assert rho == 0.0


def test_v2_snap_to_zero_when_below_one_step_threshold():
    """Once rho_max would round to 0 teacher steps, snap exactly to 0."""
    st = MFCV2(tau=0.5)
    st.get_rho("a", num_steps=10)  # threshold = 1/(2*10) = 0.05
    st.states["a"]["rho_max"] = 1.0
    st.states["a"]["rho"] = 0.04  # below threshold
    st.update("a", avg_reward=0.7, num_steps=10)
    # 0.04 < 1/20 = 0.05 → snap to 0
    assert st.states["a"]["rho_max"] == 0.0


# -- Monotonicity over a sequence -----------------------------------------


def test_v2_rho_max_is_strictly_monotone_non_increasing():
    st = MFCV2(tau=0.5)
    history = []
    st.get_rho("a", num_steps=10)
    history.append(st.states["a"]["rho_max"])
    # Mix of successes and failures at random rhos.
    rng_seq = [
        (0.7, 0.6),  # success at 0.6 -> rho_max=0.6
        (0.3, 0.2),  # success at 0.2 -> rho_max=0.2
        (0.0, 0.05),  # failure at 0.05 -> no change
        (0.0, 0.1),  # failure at 0.1 -> no change
        (0.6, 0.15),  # success at 0.15 — but 0.15 > rho_max=0.2? actually 0.15 < 0.2, descend
    ]
    for avg, rho_used in rng_seq:
        st.states["a"]["rho"] = rho_used
        st.update("a", avg_reward=avg, num_steps=10)
        history.append(st.states["a"]["rho_max"])
    for prev, curr in zip(history, history[1:]):
        assert curr <= prev, f"non-monotone: {prev} -> {curr}"


# -- Metrics --------------------------------------------------------------


def test_v2_metrics_contains_unified_curriculum_keys():
    st = MFCV2(tau=0.5)
    st.get_rho("a", num_steps=8)
    st.update("a", avg_reward=0.6, num_steps=8)
    st.get_rho("b", num_steps=8)
    st.update("b", avg_reward=0.0, num_steps=8)
    m = st.get_metrics()
    # mfc/* keys (v2-shape)
    for key in [
        "mfc/variant_v2_active",
        "mfc/mean_rho_max",
        "mfc/median_rho_max",
        "mfc/frac_at_zero",
        "mfc/frac_effective_zero",
        "mfc/recent_success_rate",
        "mfc/mean_rho_used",
        "mfc/num_tracked",
        "mfc/tau",
    ]:
        assert key in m, f"missing v2 metric {key}"
    # curriculum/* unified panel keys (cross-method comparable with AdaBack / v1)
    for key in [
        "curriculum/method",
        "curriculum/mean_frontier",
        "curriculum/frac_at_zero_strict",
        "curriculum/frac_at_zero_loose",
        "curriculum/frac_effective_zero",
        "curriculum/mean_rho_used",
        "curriculum/num_tracked",
    ]:
        assert key in m, f"missing unified metric {key}"
    assert m["curriculum/method"] == "mfc_v2"


def test_v2_state_dict_roundtrip():
    st = MFCV2(tau=0.4)
    st.get_rho("a", num_steps=10)
    st.update("a", avg_reward=0.5, num_steps=10)
    st.get_rho("b", num_steps=10)
    st.update("b", avg_reward=0.0, num_steps=10)

    st2 = MFCV2()
    st2.load_state_dict(st.state_dict())
    assert st2.tau == st.tau
    assert st2.states == st.states


# -- v1 must still work (regression guard) --------------------------------


def test_v1_still_emits_unified_metrics_after_change():
    """The v2 work added unified curriculum/* keys to v1 too; v1 algorithm
    behaviour itself must be unchanged."""
    v1 = MFCV1(p_probe=0.25, safety_K=3, delta_safe=0.1, n_rollouts=8)
    v1.get_rho("a", num_steps=10)
    v1.update("a", avg_reward=0.5, num_steps=10)
    m = v1.get_metrics()
    # New unified keys
    assert "curriculum/method" in m
    assert m["curriculum/method"] == "mfc_v1"
    assert "curriculum/mean_frontier" in m
    # Existing v1-specific keys unchanged
    for key in [
        "mfc/mean_rho_star",
        "mfc/safety_trigger_total",
        "mfc/probe_fraction",
        "mfc/epsilon",
    ]:
        assert key in m


def test_adaback_emits_unified_metrics():
    """AdaBack now also exposes curriculum/* keys for cross-method overlays."""
    ad = AdaBack(tau=0.4, p_zero=0.1)
    ad.get_rho("a", num_steps=10)
    ad.update("a", avg_reward=0.5, num_steps=10)
    m = ad.get_metrics()
    assert m["curriculum/method"] == "adaptive"
    assert "curriculum/mean_frontier" in m
    assert "curriculum/frac_at_zero_strict" in m
    # Existing AdaBack keys still present
    assert "adaptive/mean_rho" in m
    assert "adaptive/frac_at_zero" in m


def test_unified_metric_keys_are_identical_across_methods():
    """Cross-method overlays only work if all three methods emit the same
    set of curriculum/* keys.  Lock that down here."""
    ad = AdaBack()
    ad.get_rho("a", 8)
    ad.update("a", 0.5, 8)

    v1 = MFCV1(n_rollouts=8)
    v1.get_rho("a", 8)
    v1.update("a", 0.5, 8)

    v2 = MFCV2()
    v2.get_rho("a", 8)
    v2.update("a", 0.5, 8)

    def ckeys(state):
        return {k for k in state.get_metrics() if k.startswith("curriculum/")}

    k_ad, k_v1, k_v2 = ckeys(ad), ckeys(v1), ckeys(v2)
    assert k_ad == k_v1 == k_v2, (
        f"unified curriculum/* keys diverge: "
        f"adaptive={k_ad - k_v2} vs v2={k_v2 - k_ad} vs v1={k_v1 ^ k_v2}"
    )
