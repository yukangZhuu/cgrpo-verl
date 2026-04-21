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


def test_mfc_initial_rho_star_is_default():
    st = MFC(default_rho_star=1.0, n_rollouts=8)
    rho = st.get_rho("s0", num_steps=10)
    assert st.states["s0"]["rho_star"] == 1.0
    assert 0.0 <= rho <= 1.0


def test_mfc_epsilon_derived_from_n_rollouts():
    st = MFC(n_rollouts=8)
    assert abs(st.epsilon - 1.0 / 8) < 1e-12
    st2 = MFC(n_rollouts=4)
    assert abs(st2.epsilon - 1.0 / 4) < 1e-12


# -- Mechanism 2: exploit / probe sampling ---------------------------------


def test_mfc_exploit_uses_rho_star():
    st = MFC(p_probe=0.25, n_rollouts=8)
    st.get_rho("a", num_steps=10)  # register
    st.states["a"]["rho_star"] = 0.4
    # Force random() >= p_probe -> exploit branch
    with mock.patch.object(_mod.random, "random", return_value=0.9):
        rho = st.get_rho("a", num_steps=10)
    assert rho == 0.4
    assert st.states["a"]["mode"] == "exploit"
    assert st.states["a"]["rho"] == 0.4


def test_mfc_probe_samples_below_rho_star():
    st = MFC(p_probe=0.25, n_rollouts=8)
    st.get_rho("b", num_steps=10)  # register
    st.states["b"]["rho_star"] = 0.6
    # Force random() < p_probe -> probe branch; and control the uniform draw
    with mock.patch.object(_mod.random, "random", return_value=0.0), \
         mock.patch.object(_mod.random, "uniform", return_value=0.2):
        rho = st.get_rho("b", num_steps=10)
    assert rho == 0.2
    assert st.states["b"]["mode"] == "probe"
    assert rho < st.states["b"]["rho_star"]


def test_mfc_degenerates_to_exploit_when_rho_star_is_zero():
    st = MFC(p_probe=1.0, n_rollouts=8)  # always probe
    st.get_rho("c", num_steps=10)
    st.states["c"]["rho_star"] = 0.0
    rho = st.get_rho("c", num_steps=10)
    assert rho == 0.0
    assert st.states["c"]["mode"] == "exploit"


# -- Mechanism 1: ratchet --------------------------------------------------


def test_mfc_ratchet_advances_on_probe_success():
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("p", num_steps=10)
    st.states["p"]["rho_star"] = 0.5
    # Force probe at 0.2
    with mock.patch.object(_mod.random, "random", return_value=0.0), \
         mock.patch.object(_mod.random, "uniform", return_value=0.2):
        st.get_rho("p", num_steps=10)
    # avg_reward >= epsilon (1/8) -> success; frontier advances to 0.2
    st.update("p", avg_reward=0.25, num_steps=10)
    assert st.states["p"]["rho_star"] == 0.2
    assert st.states["p"]["safety_counter"] == 0


def test_mfc_no_regression_on_probe_failure():
    st = MFC(p_probe=1.0, n_rollouts=8)
    st.get_rho("q", num_steps=10)
    st.states["q"]["rho_star"] = 0.5
    with mock.patch.object(_mod.random, "random", return_value=0.0), \
         mock.patch.object(_mod.random, "uniform", return_value=0.1):
        st.get_rho("q", num_steps=10)
    # avg_reward < epsilon -> probe failure; frontier stays (Case C)
    st.update("q", avg_reward=0.0, num_steps=10)
    assert st.states["q"]["rho_star"] == 0.5
    # Safety counter MUST NOT increment on probe failures (mode == "probe")
    assert st.states["q"]["safety_counter"] == 0


def test_mfc_exploit_success_keeps_rho_star():
    st = MFC(p_probe=0.0, n_rollouts=8)  # always exploit
    st.get_rho("r", num_steps=10)
    st.states["r"]["rho_star"] = 0.3
    st.get_rho("r", num_steps=10)  # exploit -> rho=0.3
    st.update("r", avg_reward=0.5, num_steps=10)
    # Frontier cannot ADVANCE because rho_used == rho_star
    assert st.states["r"]["rho_star"] == 0.3
    assert st.states["r"]["safety_counter"] == 0


# -- Mechanism 1: safety valve ---------------------------------------------


def test_mfc_safety_valve_after_K_exploit_failures():
    st = MFC(p_probe=0.0, safety_K=2, delta_safe=0.1, n_rollouts=8)
    st.get_rho("s", num_steps=10)
    st.states["s"]["rho_star"] = 0.3

    # Two exploit failures at the frontier -> safety valve triggers
    st.get_rho("s", num_steps=10)
    st.update("s", avg_reward=0.0, num_steps=10)
    assert st.states["s"]["rho_star"] == 0.3
    assert st.states["s"]["safety_counter"] == 1

    st.get_rho("s", num_steps=10)
    st.update("s", avg_reward=0.0, num_steps=10)
    # After K=2 exploit failures: regression by delta_safe, counter reset
    assert abs(st.states["s"]["rho_star"] - 0.4) < 1e-9
    assert st.states["s"]["safety_counter"] == 0
    assert st.states["s"]["safety_triggered_total"] == 1


def test_mfc_safety_counter_resets_on_success():
    st = MFC(p_probe=0.0, safety_K=3, n_rollouts=8)
    st.get_rho("t", num_steps=10)
    st.states["t"]["rho_star"] = 0.3

    st.get_rho("t", num_steps=10)
    st.update("t", avg_reward=0.0, num_steps=10)  # fail -> counter=1
    assert st.states["t"]["safety_counter"] == 1

    st.get_rho("t", num_steps=10)
    st.update("t", avg_reward=0.5, num_steps=10)  # success -> counter reset
    assert st.states["t"]["safety_counter"] == 0


def test_mfc_safety_valve_clamps_at_1():
    st = MFC(p_probe=0.0, safety_K=1, delta_safe=0.5, n_rollouts=8)
    st.get_rho("u", num_steps=10)
    st.states["u"]["rho_star"] = 0.9
    st.get_rho("u", num_steps=10)
    st.update("u", avg_reward=0.0, num_steps=10)
    # 0.9 + 0.5 clamped to 1.0
    assert st.states["u"]["rho_star"] == 1.0


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
    for key in [
        "mfc/mean_rho_star",
        "mfc/median_rho_star",
        "mfc/frac_at_zero",
        "mfc/frac_below_0_1",
        "mfc/frac_at_one",
        "mfc/mean_rho_used",
        "mfc/probe_fraction",
        "mfc/mean_visits",
        "mfc/num_tracked",
        "mfc/safety_trigger_total",
        "mfc/epsilon",
    ]:
        assert key in m, f"missing metric key {key}"
    assert m["mfc/num_tracked"] == 2


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


def test_compute_guidance_steps_shared_between_classes():
    steps = ["s1", "s2", "s3", "s4"]
    assert (
        MFC.compute_guidance_steps(steps, 0.5)
        == AdaBack.compute_guidance_steps(steps, 0.5)
    )
    assert MFC.compute_guidance_steps([], 0.5) == []
    assert MFC.compute_guidance_steps(steps, 0.0) == []
