# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Unit tests for the Monotone Frontier Curriculum (MFC) state.

Loads ``verl/utils/curriculum.py`` in isolation so ``import verl`` (which
pulls ``ray``) is not required.
"""

import importlib.util
from pathlib import Path
from unittest import mock


def _load_curriculum_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "verl" / "utils" / "curriculum.py"
    spec = importlib.util.spec_from_file_location(
        "cgrpo_curriculum_standalone_mfc", path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_curriculum_module()
MFC = _mod.MonotoneFrontierCurriculumState
AdaBack = _mod.PerSampleCurriculumState


# -- Initialization & API --------------------------------------------------


def test_mfc_initial_rho_star_is_snapped_to_top_lattice():
    """default_rho_star=1.0 is equivalent to (num_steps - 1) / num_steps."""
    st = MFC(default_rho_star=1.0, n_rollouts=8)
    st.get_rho("s0", num_steps=10)
    assert st.states["s0"]["rho_star"] == 0.9  # = 9 / 10, top lattice for n=10


def test_mfc_initial_rho_star_lattice_respects_num_steps():
    st = MFC(default_rho_star=1.0, n_rollouts=8)
    st.get_rho("a", num_steps=8)
    st.get_rho("b", num_steps=5)
    assert st.states["a"]["rho_star"] == 7 / 8
    assert st.states["b"]["rho_star"] == 4 / 5


def test_mfc_epsilon_derived_from_n_rollouts():
    st = MFC(n_rollouts=8)
    assert abs(st.epsilon - 1.0 / 8) < 1e-12
    st2 = MFC(n_rollouts=4)
    assert abs(st2.epsilon - 1.0 / 4) < 1e-12


def test_g_from_rho_is_fp_safe_on_boundary():
    """3/7 * 7 and 2/5 * 5 should round-trip cleanly."""
    for n in range(2, 30):
        for g in range(n):
            rho = g / n
            assert MFC._g_from_rho(rho, n) == g, f"g={g}, n={n}, rho={rho}"


# -- Mechanism 2: exploit / probe sampling ---------------------------------


def test_mfc_exploit_uses_rho_star():
    st = MFC(p_probe=0.25, n_rollouts=8)
    st.get_rho("a", num_steps=10)
    st.states["a"]["rho_star"] = 0.4  # lattice (4/10)
    with mock.patch.object(_mod.random, "random", return_value=0.9):
        rho = st.get_rho("a", num_steps=10)
    assert rho == 0.4
    assert st.states["a"]["mode"] == "exploit"
    assert st.states["a"]["rho"] == 0.4


def test_mfc_probe_is_strictly_below_frontier_in_step_space():
    """With rho_star=0.6 and num_steps=10, g_curr=6, probe ∈ {0..5}."""
    st = MFC(p_probe=0.25, n_rollouts=8)
    st.get_rho("b", num_steps=10)
    st.states["b"]["rho_star"] = 0.6  # g_curr = 6
    with mock.patch.object(_mod.random, "random", return_value=0.0), \
         mock.patch.object(_mod.random, "randint", return_value=2):  # g_probe = 2
        rho = st.get_rho("b", num_steps=10)
    assert rho == 0.2  # = 2 / 10
    assert st.states["b"]["mode"] == "probe"
    assert rho < 0.6


def test_mfc_probe_never_reuses_current_g_curr_value():
    """Discrete probing must strictly advance by ≥1 teacher step."""
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("c", num_steps=10)
    st.states["c"]["rho_star"] = 0.3  # g_curr = 3
    # Exhaustively sample many probes; every one should give g_probe ∈ {0,1,2}
    g_curr = MFC._g_from_rho(0.3, 10)
    for _ in range(200):
        rho = st.get_rho("c", num_steps=10)
        st.states["c"]["rho_star"] = 0.3  # keep the frontier fixed for the test
        g_probe = MFC._g_from_rho(rho, 10)
        assert g_probe < g_curr


def test_mfc_probe_degenerates_when_g_curr_zero():
    """At the discrete floor, every visit is exploit at rho=0."""
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("d", num_steps=10)
    st.states["d"]["rho_star"] = 0.0
    rho = st.get_rho("d", num_steps=10)
    assert rho == 0.0
    assert st.states["d"]["mode"] == "exploit"


# -- Mechanism 1: ratchet --------------------------------------------------


def test_mfc_ratchet_advances_on_probe_success():
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("p", num_steps=10)
    st.states["p"]["rho_star"] = 0.5  # g_curr = 5
    with mock.patch.object(_mod.random, "random", return_value=0.0), \
         mock.patch.object(_mod.random, "randint", return_value=2):  # g_probe = 2
        st.get_rho("p", num_steps=10)
    # avg_reward >= epsilon (1/8) → success; frontier advances to 2/10 = 0.2
    st.update("p", avg_reward=0.25, num_steps=10)
    assert st.states["p"]["rho_star"] == 0.2
    assert st.states["p"]["safety_counter"] == 0


def test_mfc_no_regression_on_probe_failure():
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("q", num_steps=10)
    st.states["q"]["rho_star"] = 0.5
    with mock.patch.object(_mod.random, "random", return_value=0.0), \
         mock.patch.object(_mod.random, "randint", return_value=1):  # g_probe = 1
        st.get_rho("q", num_steps=10)
    # avg_reward < epsilon -> probe failure; frontier stays (Case C)
    st.update("q", avg_reward=0.0, num_steps=10)
    assert st.states["q"]["rho_star"] == 0.5
    assert st.states["q"]["safety_counter"] == 0


def test_mfc_exploit_success_keeps_rho_star():
    st = MFC(p_probe=0.0, n_rollouts=8)
    st.get_rho("r", num_steps=10)
    st.states["r"]["rho_star"] = 0.3
    st.get_rho("r", num_steps=10)  # exploit at rho_star=0.3
    st.update("r", avg_reward=0.5, num_steps=10)
    # Frontier cannot ADVANCE because rho_used == rho_star
    assert st.states["r"]["rho_star"] == 0.3
    assert st.states["r"]["safety_counter"] == 0


def test_mfc_discrete_probe_can_reach_exact_zero():
    """
    Regression test for the "continuous probe can never hit 0" bug.
    When g_probe=0 is sampled and the probe succeeds, rho_star becomes
    exactly 0 (not a tiny positive float).
    """
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("z", num_steps=10)
    st.states["z"]["rho_star"] = 0.1  # g_curr = 1, so probe ∈ {0}
    # No mock needed: randint(0, 0) deterministically returns 0.
    st.get_rho("z", num_steps=10)
    assert st.states["z"]["rho"] == 0.0
    st.update("z", avg_reward=1.0, num_steps=10)
    assert st.states["z"]["rho_star"] == 0.0


# -- Mechanism 1: safety valve (lattice-aware) -----------------------------


def test_mfc_safety_valve_advances_at_least_one_step():
    """Even if delta_safe < 1/n, the safety valve must advance by ≥ 1 step."""
    st = MFC(p_probe=0.0, safety_K=1, delta_safe=0.01, n_rollouts=8)
    st.get_rho("s1", num_steps=10)
    st.states["s1"]["rho_star"] = 0.3  # g = 3
    st.get_rho("s1", num_steps=10)
    st.update("s1", avg_reward=0.0, num_steps=10)
    # 0.3 + 0.01 = 0.31 rounds to g=3; min-advance guard pushes to g=4
    assert st.states["s1"]["rho_star"] == 0.4


def test_mfc_safety_valve_after_K_exploit_failures():
    st = MFC(p_probe=0.0, safety_K=2, delta_safe=0.2, n_rollouts=8)
    st.get_rho("s2", num_steps=10)
    st.states["s2"]["rho_star"] = 0.3  # g = 3

    st.get_rho("s2", num_steps=10)
    st.update("s2", avg_reward=0.0, num_steps=10)
    assert st.states["s2"]["rho_star"] == 0.3
    assert st.states["s2"]["safety_counter"] == 1

    st.get_rho("s2", num_steps=10)
    st.update("s2", avg_reward=0.0, num_steps=10)
    # 0.3 + 0.2 = 0.5 rounds to g=5 (continuous bump beats +1 guard)
    assert st.states["s2"]["rho_star"] == 0.5
    assert st.states["s2"]["safety_counter"] == 0
    assert st.states["s2"]["safety_triggered_total"] == 1


def test_mfc_safety_counter_resets_on_success():
    st = MFC(p_probe=0.0, safety_K=3, n_rollouts=8)
    st.get_rho("s3", num_steps=10)
    st.states["s3"]["rho_star"] = 0.3

    st.get_rho("s3", num_steps=10)
    st.update("s3", avg_reward=0.0, num_steps=10)  # fail -> counter=1
    assert st.states["s3"]["safety_counter"] == 1

    st.get_rho("s3", num_steps=10)
    st.update("s3", avg_reward=0.5, num_steps=10)  # success -> reset
    assert st.states["s3"]["safety_counter"] == 0


def test_mfc_safety_valve_clamps_at_top_lattice():
    """Even with a huge delta_safe, rho_star is clamped to (n-1)/n."""
    st = MFC(p_probe=0.0, safety_K=1, delta_safe=0.9, n_rollouts=8)
    st.get_rho("s4", num_steps=10)
    st.states["s4"]["rho_star"] = 0.5
    st.get_rho("s4", num_steps=10)
    st.update("s4", avg_reward=0.0, num_steps=10)
    assert st.states["s4"]["rho_star"] == 0.9  # 9/10 = top lattice


def test_mfc_rho_star_stays_on_lattice_after_safety():
    """Every safety trigger should leave rho_star exactly on g/n."""
    st = MFC(p_probe=0.0, safety_K=1, delta_safe=0.15, n_rollouts=8)
    st.get_rho("s5", num_steps=10)
    st.states["s5"]["rho_star"] = 0.2
    for _ in range(3):
        st.get_rho("s5", num_steps=10)
        st.update("s5", avg_reward=0.0, num_steps=10)
        rho = st.states["s5"]["rho_star"]
        g = round(rho * 10)
        assert abs(rho - g / 10) < 1e-9


# -- Checkpoint & metrics --------------------------------------------------


def test_mfc_state_dict_roundtrip():
    st = MFC(p_probe=0.3, safety_K=2, delta_safe=0.2, n_rollouts=4)
    st.get_rho("a", num_steps=5)
    st.update("a", avg_reward=0.3, num_steps=5)
    st.get_rho("b", num_steps=5)
    st.update("b", avg_reward=0.0, num_steps=5)

    st2 = MFC(n_rollouts=4)
    st2.load_state_dict(st.state_dict())
    assert st2.p_probe == st.p_probe
    assert st2.safety_K == st.safety_K
    assert st2.delta_safe == st.delta_safe
    assert st2.n_rollouts == st.n_rollouts
    assert st2.epsilon == st.epsilon
    assert st2.states == st.states


def test_mfc_metrics_keys_present():
    st = MFC(n_rollouts=4)
    st.get_rho("a", num_steps=5)
    st.update("a", avg_reward=0.5, num_steps=5)
    st.get_rho("b", num_steps=5)
    st.update("b", avg_reward=0.0, num_steps=5)
    m = st.get_metrics()
    required = [
        "mfc/mean_rho_star",
        "mfc/median_rho_star",
        "mfc/min_rho_star",
        "mfc/max_rho_star",
        "mfc/frac_effective_zero",
        "mfc/frac_effective_below_2_steps",
        "mfc/frac_at_zero",
        "mfc/frac_rho_star_below_0_05",
        "mfc/frac_rho_star_below_0_1",
        "mfc/frac_rho_star_at_zero_exact",
        "mfc/frac_at_one",
        "mfc/mean_rho_used",
        "mfc/probe_fraction",
        "mfc/mean_g_star",
        "mfc/mean_visits",
        "mfc/num_tracked",
        "mfc/safety_trigger_total",
        "mfc/mean_safety_counter",
        "mfc/epsilon",
    ]
    for key in required:
        assert key in m, f"missing metric key {key}"
    assert m["mfc/num_tracked"] == 2


def test_mfc_frac_effective_zero_counts_discrete_zero():
    """A sample with rho_star = 0.04 and num_steps=8 maps to g=0."""
    st = MFC(n_rollouts=8)
    # Register two samples, then manually plant a near-zero frontier for one.
    st.get_rho("a", num_steps=8)
    st.get_rho("b", num_steps=8)
    st.states["a"]["rho_star"] = 0.04  # round(0.32) = 0 → effective zero
    st.states["b"]["rho_star"] = 0.5  # round(4.0) = 4 → not effective zero
    m = st.get_metrics()
    assert m["mfc/frac_effective_zero"] == 0.5
    # The continuous rho_star_below_0_05 should also flag the 0.04 sample.
    assert m["mfc/frac_rho_star_below_0_05"] == 0.5


def test_mfc_get_metrics_empty_returns_empty_dict():
    st = MFC(n_rollouts=8)
    assert st.get_metrics() == {}


# -- AdaBack regression: make sure MFC did not break the sibling class ----


def test_adaback_still_exposes_original_api():
    st = AdaBack(tau=0.4, p_zero=0.1)
    rho = st.get_rho("x", num_steps=10)
    assert 0.0 <= rho <= 1.0
    st.update("x", avg_reward=0.5, num_steps=10)
    assert "x" in st.states
    m = st.get_metrics()
    assert "adaptive/mean_rho" in m
    assert "adaptive/frac_at_zero" in m


def test_adaback_has_faithful_rho_max_metrics():
    """The newly-added unbiased convergence metrics based on rho_max."""
    st = AdaBack(tau=0.4, p_zero=0.0, default_rho=0.5, min_step_delta=1)
    st.get_rho("x", num_steps=10)
    st.update("x", avg_reward=0.0, num_steps=10)  # fail → rho_min up
    m = st.get_metrics()
    assert "adaptive/frac_rho_max_below_0_05" in m
    assert "adaptive/frac_rho_max_below_0_1" in m


def test_compute_guidance_steps_shared_between_classes():
    steps = ["s1", "s2", "s3", "s4"]
    assert (
        MFC.compute_guidance_steps(steps, 0.5)
        == AdaBack.compute_guidance_steps(steps, 0.5)
    )
    assert MFC.compute_guidance_steps([], 0.5) == []
    assert MFC.compute_guidance_steps(steps, 0.0) == []
