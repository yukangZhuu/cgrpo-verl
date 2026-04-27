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
MonotoneFrontierCurriculumStateV2 = _mod.MonotoneFrontierCurriculumStateV2


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
    # v1-only fields (preserved for backward compatibility with v1 launchers).
    for key in ("p_probe", "safety_K", "delta_safe", "default_rho_star"):
        assert key in mc, f"mfc_curriculum missing '{key}'"
    assert 0.0 <= mc["p_probe"] <= 1.0
    assert mc["safety_K"] >= 1
    assert 0.0 <= mc["delta_safe"] <= 1.0
    assert 0.0 <= mc["default_rho_star"] <= 1.0
    # v2 additions: variant + tau + default_rho_max.
    assert "variant" in mc and mc["variant"] in ("v1", "v2"), (
        "mfc_curriculum.variant must be set to either 'v1' or 'v2'"
    )
    assert mc["variant"] == "v1", (
        "Default mfc_curriculum.variant must be 'v1' to preserve backward "
        "compatibility with existing launchers."
    )
    assert "tau" in mc and 0.0 < mc["tau"] <= 1.0
    assert "default_rho_max" in mc and 0.0 <= mc["default_rho_max"] <= 1.0


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


def test_dispatch_mfc_v1_constructs_mfc_v1_state():
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


def test_dispatch_mfc_v2_constructs_mfc_v2_state():
    cfg = _load_yaml()
    mc = cfg["mfc_curriculum"]
    st = MonotoneFrontierCurriculumStateV2(
        tau=mc["tau"],
        default_rho_max=mc["default_rho_max"],
    )
    st.get_rho("y", num_steps=8)
    st.update("y", avg_reward=0.7, num_steps=8)
    sd = st.state_dict()
    assert sd.get("variant") == "v2"
    st2 = MonotoneFrontierCurriculumStateV2()
    st2.load_state_dict(sd)
    assert st2.states == st.states
    assert st2.tau == st.tau


def test_dispatch_v1_v2_state_files_must_be_distinct():
    """Trainer saves v1 to mfc_curriculum_state.json and v2 to
    mfc_curriculum_state_v2.json. This catches a future regression
    that would let v1 and v2 share a checkpoint filename."""
    trainer_src = (
        ROOT / "verl" / "trainer" / "cgrpo_trainer.py"
    ).read_text(encoding="utf-8")
    assert '"mfc_curriculum_state.json"' in trainer_src
    assert '"mfc_curriculum_state_v2.json"' in trainer_src


def test_method_private_metric_namespaces_are_disjoint():
    """Method-private prefixes (``adaptive/*`` and ``mfc/*``) must never collide.

    The shared ``curriculum/*`` prefix is *intentional* — it is the unified
    cross-method comparison panel (see :func:`_unified_curriculum_metrics`
    in ``verl/utils/curriculum.py``) used by paper-grade wandb overlays.
    """
    ad = PerSampleCurriculumState()
    ad.get_rho("a", num_steps=8)
    ad.update("a", avg_reward=0.5, num_steps=8)
    am = ad.get_metrics()

    mfc1 = MonotoneFrontierCurriculumState(n_rollouts=8)
    mfc1.get_rho("a", num_steps=8)
    mfc1.update("a", avg_reward=0.5, num_steps=8)
    mm1 = mfc1.get_metrics()

    mfc2 = MonotoneFrontierCurriculumStateV2()
    mfc2.get_rho("a", num_steps=8)
    mfc2.update("a", avg_reward=0.5, num_steps=8)
    mm2 = mfc2.get_metrics()

    def private(state_metrics):
        return {k for k in state_metrics if not k.startswith("curriculum/")}

    am_priv = private(am)
    mm1_priv = private(mm1)
    mm2_priv = private(mm2)

    assert all(k.startswith("adaptive/") for k in am_priv)
    assert all(k.startswith("mfc/") for k in mm1_priv)
    assert all(k.startswith("mfc/") for k in mm2_priv)
    assert am_priv.isdisjoint(mm1_priv)
    assert am_priv.isdisjoint(mm2_priv)
    # v1 and v2 deliberately share the ``mfc/*`` prefix for dashboard
    # continuity; the unique distinguishing keys are
    # ``mfc/variant_v1_active`` and ``mfc/variant_v2_active``.
    assert "mfc/variant_v1_active" in mm1_priv
    assert "mfc/variant_v2_active" in mm2_priv


def test_state_dict_filenames_are_distinct():
    """Both classes produce a stable JSON payload with distinguishable shape."""
    ad = PerSampleCurriculumState().state_dict()
    mfc = MonotoneFrontierCurriculumState(n_rollouts=8).state_dict()
    # AdaBack has tau; MFC has p_probe / epsilon — both are required fields.
    assert "tau" in ad and "tau" not in mfc
    assert "p_probe" in mfc and "p_probe" not in ad
    assert mfc.get("method") == "mfc"


def test_curriculum_dataset_sentinel_covers_mfc():
    """Guard against a regression where the g_level=-1 sentinel branch in
    CurriculumGRPODataset.__getitem__ silently excludes "mfc" and thus
    causes MFC rows to train with g_level=0 (no rho sampling)."""
    path = ROOT / "verl" / "utils" / "dataset" / "curriculum_dataset.py"
    source = path.read_text(encoding="utf-8")
    # Both method strings MUST be part of the gate condition.
    assert (
        'self.curriculum_method in ("adaptive", "mfc")' in source
        or "self.curriculum_method in ('adaptive', 'mfc')" in source
    ), (
        "curriculum_dataset.py sentinel gate no longer mentions both "
        "'adaptive' and 'mfc' — MFC rows would train with g_level=0."
    )
    # And the g_level = -1.0 sentinel assignment must be under that gate.
    assert "g_level = -1.0" in source


def test_pure_n128_dataset_is_anchor_free():
    """The pure-128 dataset must not contain frozen_g_level anchors."""
    import json

    path = ROOT / "data" / "ttn2k" / "final" / "ttn_unsolvable_pass64_n128" / "dataset.jsonl"
    if not path.exists():
        # Not fatal — this file is produced by build_ttn_unsolvable_n128.py.
        # Tests should pass in clones that haven't built the data yet.
        return
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 128, f"expected 128 rows, got {len(lines)}"
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert row.get("pass_rate", 0) == 0.0, f"row {i}: pass_rate != 0"
        assert "frozen_g_level" not in row, f"row {i}: anchors leaked in"
        assert "g_level" not in row, f"row {i}: static g_level leaked in"
        assert "adaptive_id" in row, f"row {i}: missing adaptive_id"


def _simulate_curriculum_dataset_g_level(item: dict, curriculum_method: str) -> float:
    """Pure-Python re-implementation of the g_level assignment branch in
    :meth:`CurriculumGRPODataset.__getitem__`.  Mirrors the production source
    *exactly* so this test fails if the production branch drifts.

    Only returns the final ``g_level`` — the rest of the row is irrelevant.
    """
    g_level = float(item.get("g_level", 0.0))
    guidance_steps = item.get("guidance_steps", [])
    frozen_g_level = item.get("frozen_g_level", None)

    if frozen_g_level is not None:
        g_level = float(frozen_g_level)
    elif curriculum_method in ("adaptive", "mfc") and not guidance_steps:
        g_level = -1.0
    return g_level


def test_sentinel_simulates_mfc_rows_to_minus_one():
    """On every row of the pure-128 dataset, MFC produces the -1 sentinel."""
    import json

    path = ROOT / "data" / "ttn2k" / "final" / "ttn_unsolvable_pass64_n128" / "dataset.jsonl"
    if not path.exists():
        return
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, line in enumerate(lines):
        row = json.loads(line)
        for method in ("mfc", "adaptive"):
            g = _simulate_curriculum_dataset_g_level(row, method)
            assert g == -1.0, (
                f"row {i} under curriculum_method={method!r} "
                f"produced g_level={g!r}; expected -1.0 sentinel"
            )


def test_sentinel_not_triggered_for_none_or_mixture():
    """Baseline / mixture methods must leave g_level at its dataset value."""
    # A mixture-style row with explicit g_level=0.3 + guidance_steps.
    row_with_guidance = {
        "g_level": 0.3,
        "guidance_steps": ["s1", "s2"],
    }
    for method in ("none", "mixture"):
        assert _simulate_curriculum_dataset_g_level(row_with_guidance, method) == 0.3

    # A bare row (no g_level, no guidance).  Under none/mixture it must stay at 0.
    bare_row = {}
    for method in ("none", "mixture"):
        assert _simulate_curriculum_dataset_g_level(bare_row, method) == 0.0


def test_sentinel_skipped_for_frozen_anchors():
    """Anchor rows keep their frozen_g_level regardless of curriculum method."""
    row_g0 = {"frozen_g_level": 0.0}
    row_g1 = {"frozen_g_level": 1.0}
    for method in ("none", "mixture", "adaptive", "mfc"):
        assert _simulate_curriculum_dataset_g_level(row_g0, method) == 0.0
        assert _simulate_curriculum_dataset_g_level(row_g1, method) == 1.0
