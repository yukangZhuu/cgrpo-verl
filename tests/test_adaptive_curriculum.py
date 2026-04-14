# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Unit tests for per-sample adaptive curriculum (AdaBack-style).

Loads ``verl/utils/curriculum.py`` in isolation so ``import verl`` (ray) is not required.
"""

import importlib.util
from pathlib import Path
from unittest import mock


def _load_curriculum_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "verl" / "utils" / "curriculum.py"
    spec = importlib.util.spec_from_file_location("cgrpo_curriculum_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_curriculum_module()
PerSampleCurriculumState = _mod.PerSampleCurriculumState


def test_compute_guidance_steps_empty_and_bounds():
    steps = ["a", "b", "c", "d"]
    assert PerSampleCurriculumState.compute_guidance_steps([], 0.5) == []
    assert PerSampleCurriculumState.compute_guidance_steps(steps, 0.0) == []
    assert PerSampleCurriculumState.compute_guidance_steps(steps, -0.1) == []
    assert len(PerSampleCurriculumState.compute_guidance_steps(steps, 1.0)) == 3


def test_p_zero_persists_rho_for_update():
    """When p_zero fires, update() must use rho=0, not a stale s['rho']."""
    st = PerSampleCurriculumState(tau=0.4, p_zero=1.0, default_rho=0.9, min_step_delta=1)
    with mock.patch.object(_mod.random, "random", return_value=0.0):
        rho = st.get_rho("s0", num_steps=10)
    assert rho == 0.0
    assert st.states["s0"]["rho"] == 0.0
    assert st.states["s0"].get("last_forced_zero") is True
    st.update("s0", avg_reward=1.0, num_steps=10)
    assert st.states["s0"]["rho"] == 0.0
    assert st.states["s0"]["visits"] == 1
    assert st.states["s0"]["rho_min"] == 0.0
    # rho_max is set from rollout rho=0, then min_step_delta widening (clamped at 0).
    assert st.states["s0"]["rho_max"] >= 0.0


def test_update_below_tau_raises_rho_min():
    st = PerSampleCurriculumState(tau=0.5, p_zero=0.0, default_rho=0.5, min_step_delta=1)
    with mock.patch.object(_mod.random, "uniform", return_value=0.3):
        st.get_rho("s1", num_steps=10)
    st.update("s1", avg_reward=0.0, num_steps=10)
    assert st.states["s1"]["rho_min"] == 0.3
    assert st.states["s1"]["rho_max"] == 1.0


def test_min_step_delta_widens_narrow_interval():
    st = PerSampleCurriculumState(tau=0.5, p_zero=0.0, default_rho=0.5, min_step_delta=1)
    st.states["s2"] = {
        "rho_min": 0.0,
        "rho_max": 1.0,
        "rho": 0.001,
        "visits": 0,
    }
    st.update("s2", avg_reward=1.0, num_steps=100)
    # Widening is symmetric around mid and clamped to [0, 1]; may still be < 1/n near 0.
    assert st.states["s2"]["rho_max"] - st.states["s2"]["rho_min"] > 0.001


def test_state_dict_roundtrip():
    st = PerSampleCurriculumState()
    st.get_rho("a", 5)
    st.update("a", 0.2, 5)
    st2 = PerSampleCurriculumState()
    st2.load_state_dict(st.state_dict())
    assert st2.states == st.states
    assert st2.tau == st.tau
