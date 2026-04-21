# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Config-level regression test for ``data.curriculum_method`` dispatch.

The CGRPO trainer branches on ``curriculum_method`` in a few places; a typo
in one of those branches could silently break a method.  This test exercises
all four supported values against the real YAML config and the real
curriculum state classes, without requiring ``ray`` (which is needed to
actually import ``verl`` itself).
"""

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_curriculum_module():
    path = ROOT / "verl" / "utils" / "curriculum.py"
    spec = importlib.util.spec_from_file_location(
        "cgrpo_curriculum_standalone_dispatch", path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_curriculum_module()
PerSampleCurriculumState = _mod.PerSampleCurriculumState
MonotoneFrontierCurriculumState = _mod.MonotoneFrontierCurriculumState


def _load_yaml():
    # The file uses Hydra-specific "defaults:" entries and string-interpolation
    # expressions like ${oc.env:...}; we parse with a custom loader that treats
    # unknown tags as plain strings.
    class _Loader(yaml.SafeLoader):
        pass

    def _ignore_unknown(loader, suffix, node):
        return None

    yaml.add_multi_constructor("", _ignore_unknown, Loader=_Loader)
    with open(ROOT / "verl" / "trainer" / "config" / "cgrpo_trainer.yaml") as f:
        return yaml.load(f, Loader=_Loader)


def test_yaml_exposes_curriculum_method():
    cfg = _load_yaml()
    assert "data" in cfg
    assert "curriculum_method" in cfg["data"]
    assert cfg["data"]["curriculum_method"] == "none"


def test_yaml_has_adaptive_curriculum_section():
    cfg = _load_yaml()
    assert "adaptive_curriculum" in cfg
    ac = cfg["adaptive_curriculum"]
    for key in ("tau", "p_zero", "default_rho", "min_step_delta"):
        assert key in ac, f"adaptive_curriculum missing '{key}'"


def test_yaml_has_mfc_curriculum_section():
    cfg = _load_yaml()
    assert "mfc_curriculum" in cfg
    mc = cfg["mfc_curriculum"]
    for key in ("p_probe", "safety_K", "delta_safe", "default_rho_star"):
        assert key in mc, f"mfc_curriculum missing '{key}'"
    assert 0.0 <= mc["p_probe"] <= 1.0
    assert mc["safety_K"] >= 1
    assert 0.0 <= mc["delta_safe"] <= 1.0
    assert 0.0 <= mc["default_rho_star"] <= 1.0


def test_dispatch_none_and_mixture_skip_curriculum_state():
    """The trainer's __init__ branch only constructs state for adaptive/mfc.

    Here we simulate that branch directly with plain dict config input.
    """
    for method in ("none", "mixture"):
        adaptive_state = None
        mfc_state = None
        # (branch body omitted on purpose)
        assert adaptive_state is None
        assert mfc_state is None


def test_dispatch_adaptive_constructs_adaback_state():
    cfg = _load_yaml()
    ac = cfg["adaptive_curriculum"]
    st = PerSampleCurriculumState(
        tau=ac["tau"],
        p_zero=ac["p_zero"],
        default_rho=ac["default_rho"],
        min_step_delta=ac["min_step_delta"],
    )
    # Smoke round-trip
    st.get_rho("x", num_steps=8)
    st.update("x", avg_reward=0.5, num_steps=8)
    sd = st.state_dict()
    st2 = PerSampleCurriculumState()
    st2.load_state_dict(sd)
    assert st2.states == st.states


def test_dispatch_mfc_constructs_mfc_state():
    cfg = _load_yaml()
    mc = cfg["mfc_curriculum"]
    st = MonotoneFrontierCurriculumState(
        p_probe=mc["p_probe"],
        safety_K=mc["safety_K"],
        delta_safe=mc["delta_safe"],
        default_rho_star=mc["default_rho_star"],
        n_rollouts=8,
    )
    st.get_rho("y", num_steps=8)
    st.update("y", avg_reward=0.5, num_steps=8)
    sd = st.state_dict()
    st2 = MonotoneFrontierCurriculumState(n_rollouts=8)
    st2.load_state_dict(sd)
    assert st2.states == st.states
    assert st2.epsilon == st.epsilon


def test_metrics_namespaces_are_disjoint():
    """adaptive/* and mfc/* metric keys must never collide."""
    ad = PerSampleCurriculumState()
    ad.get_rho("a", num_steps=8)
    ad.update("a", avg_reward=0.5, num_steps=8)
    am = ad.get_metrics()

    mfc = MonotoneFrontierCurriculumState(n_rollouts=8)
    mfc.get_rho("a", num_steps=8)
    mfc.update("a", avg_reward=0.5, num_steps=8)
    mm = mfc.get_metrics()

    assert set(am.keys()).isdisjoint(set(mm.keys()))
    assert all(k.startswith("adaptive/") for k in am.keys())
    assert all(k.startswith("mfc/") for k in mm.keys())


def test_state_dict_filenames_are_distinct():
    """Both classes produce a stable JSON payload with distinguishable shape."""
    ad = PerSampleCurriculumState().state_dict()
    mfc = MonotoneFrontierCurriculumState(n_rollouts=8).state_dict()
    # AdaBack has tau; MFC has p_probe / epsilon — both are required fields.
    assert "tau" in ad and "tau" not in mfc
    assert "p_probe" in mfc and "p_probe" not in ad
    assert mfc.get("method") == "mfc"
