# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training metrics tracker and per-sample adaptive curriculum for CGRPO.
"""

import logging
import math
import random
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Lightweight metrics tracker (used by all modes)
# ======================================================================

class TrainingMetricsTracker:
    """
    Tracks training statistics for logging and monitoring.
    No curriculum scheduling — purely observational.
    """

    def __init__(self, ema_alpha: float = 0.1, window_size: int = 20):
        self.ema_alpha = ema_alpha
        self.sr_ema = 0.0
        self.total_steps = 0
        self.recent_rewards = deque(maxlen=window_size)

    def update(self, batch_success_rate: float, batch_size: int = 1) -> dict:
        self.total_steps += 1
        self.sr_ema = self.ema_alpha * batch_success_rate + (1 - self.ema_alpha) * self.sr_ema
        self.recent_rewards.append(batch_success_rate)

        window_mean = sum(self.recent_rewards) / len(self.recent_rewards)

        return {
            "training/sr_batch": batch_success_rate,
            "training/total_steps": self.total_steps,
            "training/sr_ema": self.sr_ema,
            "training/sr_window_mean": window_mean,
        }


# ======================================================================
# Per-sample adaptive curriculum (AdaBack-style)
# ======================================================================

class PerSampleCurriculumState:
    """
    Tracks per-sample supervision ratio intervals for AdaBack-style curriculum.

    Each sample *i* maintains ``[rho_min_i, rho_max_i]``.  At each encounter
    ``rho_i`` is sampled uniformly from this interval, converted to a discrete
    number of guidance steps, and after rollout the interval is updated based
    on the average reward compared to threshold ``tau``.

    Samples with ``frozen_g_level`` are never updated — they serve as anchors.
    """

    def __init__(
        self,
        tau: float = 0.4,
        p_zero: float = 0.1,
        default_rho: float = 0.5,
        min_step_delta: int = 1,
    ):
        self.tau = tau
        self.p_zero = p_zero
        self.default_rho = default_rho
        self.min_step_delta = max(min_step_delta, 1)

        # sample_id -> {"rho_min": float, "rho_max": float, "rho": float, "visits": int}
        self.states: dict[str, dict] = {}

        # Global EMA of rho bounds — used to initialise unseen samples
        self._global_rho_min_ema = 0.0
        self._global_rho_max_ema = 1.0
        self._ema_alpha = 0.05

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_rho(self, sample_id: str, num_steps: int) -> float:
        """
        Sample a supervision ratio for *sample_id*.

        Returns a float in [0, 1] representing the fraction of teacher steps
        to reveal.  With probability ``p_zero`` the ratio is forced to 0
        (train-test gap closure).
        """
        if sample_id not in self.states:
            self.states[sample_id] = {
                "rho_min": 0.0,
                "rho_max": 1.0,
                "rho": self.default_rho,
                "visits": 0,
            }

        s = self.states[sample_id]

        if random.random() < self.p_zero:
            # Must persist rho=0 so :meth:`update` uses the same value as this rollout.
            s["rho"] = 0.0
            s["last_forced_zero"] = True
            return 0.0

        s["last_forced_zero"] = False
        rho_min, rho_max = s["rho_min"], s["rho_max"]

        if rho_max <= rho_min:
            rho = rho_min
        else:
            rho = random.uniform(rho_min, rho_max)

        rho = max(0.0, min(1.0, rho))
        s["rho"] = rho
        return rho

    def update(self, sample_id: str, avg_reward: float, num_steps: int) -> None:
        """
        Update the rho interval for *sample_id* based on rollout feedback.

        AdaBack rule (arXiv:2506.18110 §2.1):
          * reward < tau  → rho_min = rho   (need more supervision)
          * reward >= tau → rho_max = rho, rho_min = 0  (can try harder)

        A ``min_step_delta`` guard ensures the interval is wide enough that
        the next sampled rho maps to a different discrete step count.
        """
        if sample_id not in self.states:
            return

        s = self.states[sample_id]
        rho = s["rho"]
        s["visits"] += 1

        if avg_reward < self.tau:
            s["rho_min"] = rho
        else:
            s["rho_max"] = rho
            s["rho_min"] = 0.0

        # Enforce a minimum gap so the next sample can differ by >= 1 step
        if num_steps > 0:
            min_gap = self.min_step_delta / num_steps
            if s["rho_max"] - s["rho_min"] < min_gap:
                mid = (s["rho_max"] + s["rho_min"]) / 2
                s["rho_min"] = max(0.0, mid - min_gap / 2)
                s["rho_max"] = min(1.0, mid + min_gap / 2)

        # Update global EMA
        self._global_rho_min_ema = (
            self._ema_alpha * s["rho_min"]
            + (1 - self._ema_alpha) * self._global_rho_min_ema
        )
        self._global_rho_max_ema = (
            self._ema_alpha * s["rho_max"]
            + (1 - self._ema_alpha) * self._global_rho_max_ema
        )

    # ------------------------------------------------------------------
    # Step discretisation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_guidance_steps(
        steps: list[str], rho: float
    ) -> list[str]:
        """
        Convert a continuous rho ∈ [0, 1] to a discrete slice of *steps*.

        ``rho = 0`` → no guidance (empty list).
        ``rho = 1`` → reveal all steps except the last one.
        """
        n = len(steps)
        if n == 0 or rho <= 0:
            return []
        num_reveal = round(rho * n)
        num_reveal = max(0, min(num_reveal, n - 1))
        return steps[:num_reveal]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "p_zero": self.p_zero,
            "default_rho": self.default_rho,
            "min_step_delta": self.min_step_delta,
            "states": dict(self.states),
            "_global_rho_min_ema": self._global_rho_min_ema,
            "_global_rho_max_ema": self._global_rho_max_ema,
        }

    def load_state_dict(self, d: dict) -> None:
        self.tau = d.get("tau", self.tau)
        self.p_zero = d.get("p_zero", self.p_zero)
        self.default_rho = d.get("default_rho", self.default_rho)
        self.min_step_delta = d.get("min_step_delta", self.min_step_delta)
        self.states = d.get("states", {})
        self._global_rho_min_ema = d.get("_global_rho_min_ema", 0.0)
        self._global_rho_max_ema = d.get("_global_rho_max_ema", 1.0)
        logger.info(
            f"PerSampleCurriculumState restored: {len(self.states)} samples tracked"
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Aggregate metrics for wandb / logging."""
        if not self.states:
            return {}

        rhos = [s["rho"] for s in self.states.values()]
        rho_mins = [s["rho_min"] for s in self.states.values()]
        rho_maxs = [s["rho_max"] for s in self.states.values()]
        visits = [s["visits"] for s in self.states.values()]

        n = len(rhos)
        # Note: ``frac_at_zero`` below is defined on ``s["rho"]`` — the
        # *last sampled* rho per visit.  That signal is polluted by
        # (a) p_zero-forced zeros and (b) Uniform-from-interval draws that
        # happen to land below 0.05 while ``rho_max`` is still far above 0.
        # The unbiased convergence indicator is ``frac_rho_max_below_0_05``
        # (below), which reflects the state of the per-sample interval
        # rather than any single realisation.
        at_zero = sum(1 for r in rhos if r < 0.05) / n
        at_full = sum(1 for r in rhos if r > 0.95) / n
        frac_rho_max_below_0_05 = sum(1 for rm in rho_maxs if rm < 0.05) / n
        frac_rho_max_below_0_1 = sum(1 for rm in rho_maxs if rm < 0.1) / n

        return {
            "adaptive/mean_rho": sum(rhos) / n,
            "adaptive/mean_rho_min": sum(rho_mins) / n,
            "adaptive/mean_rho_max": sum(rho_maxs) / n,
            "adaptive/frac_at_zero": at_zero,
            "adaptive/frac_at_full": at_full,
            "adaptive/frac_rho_max_below_0_05": frac_rho_max_below_0_05,
            "adaptive/frac_rho_max_below_0_1": frac_rho_max_below_0_1,
            "adaptive/mean_visits": sum(visits) / n,
            "adaptive/num_tracked": n,
        }


# ======================================================================
# Monotone Frontier Curriculum (MFC) — designed for unsolvable-only data
# ======================================================================

class MonotoneFrontierCurriculumState:
    """
    Per-sample curriculum state for the Monotone Frontier Curriculum (MFC).

    Each sample ``i`` maintains a single frontier point ``rho_star_i`` —
    the lowest supervision ratio at which the sample has been empirically
    verified to produce a usable rollout (at least one correct rollout
    among ``n``).  The frontier is monotonically non-increasing over
    training, except via a rate-limited "safety valve" that allows small
    regressions when the current frontier has been repeatedly failed
    in exploit mode.

    Mechanisms (see 第三轮实验/MFC_算法设计_spec.md):

    * **Monotone frontier (ratchet)** — success at ``rho <= rho_star``
      advances the frontier; probe failures (``rho < rho_star``) do not
      regress it.  Exploit-mode failures at ``rho == rho_star`` are
      counted; after ``safety_K`` consecutive such failures, the
      frontier is bumped up (continuously by ``delta_safe``, then snapped
      to the discrete lattice with a guaranteed ``+1`` step advance) and
      the counter resets.

    * **Frontier-biased visit sampling** — each visit uses a single
      ``rho_used`` for all ``n`` rollouts (GRPO intra-group identity
      preserved).  With probability ``1 - p_probe`` the visit is in
      "exploit" mode (``rho_used = rho_star``); otherwise "probe" mode,
      which samples in the *discrete step space*:
      ``g_curr = round(rho_star * num_steps)`` (clipped to
      ``[0, num_steps - 1]``); ``g_probe ~ Uniform{0, ..., g_curr - 1}``;
      ``rho_used = g_probe / num_steps``.  This guarantees every probe
      advances by at least one teacher-step relative to the current
      exploit hint, and makes ``rho_star = 0`` reachable in finite time
      (rather than only asymptotically as under continuous probing).

    Success threshold is the derived quantity ``epsilon = 1 / n_rollouts``
    — a visit is "successful" iff at least one of its ``n`` rollouts is
    correct.

    Invariants maintained by this class:

    * ``rho_star_i`` is always a discrete lattice point ``g / num_steps_i``
      after any probe success or safety-valve event.  (The initial value
      ``default_rho_star = 1.0`` is equivalent to the top lattice point
      ``(num_steps_i - 1) / num_steps_i`` modulo
      :meth:`compute_guidance_steps` clipping; it is snapped on first
      :meth:`get_rho` as a defensive no-op.)
    * After a safety-valve event, ``g_star`` strictly increases by at
      least ``1`` (so the valve always makes progress — avoiding the
      silent "bump by 0.01 but round() still returns the same ``g``"
      failure mode).

    Samples with ``frozen_g_level`` (anchors) are never registered in
    ``self.states`` and therefore never updated — the caller is
    responsible for skipping them before invoking :meth:`get_rho` /
    :meth:`update`.
    """

    def __init__(
        self,
        p_probe: float = 0.25,
        safety_K: int = 3,
        delta_safe: float = 0.1,
        default_rho_star: float = 1.0,
        n_rollouts: int = 8,
    ):
        self.p_probe = float(p_probe)
        self.safety_K = int(max(1, safety_K))
        self.delta_safe = float(max(0.0, delta_safe))
        self.default_rho_star = float(min(1.0, max(0.0, default_rho_star)))
        self.n_rollouts = int(max(1, n_rollouts))
        self.epsilon = 1.0 / self.n_rollouts

        # sample_id -> dict (see _init_entry for schema)
        self.states: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Small additive bias used whenever we convert ``rho * n`` to an integer
    # step count.  It pushes values on or just below the banker's-rounding
    # half-step (e.g. 0.4999999... produced by 1/3 * 3 type FP chatter) up to
    # the "expected" integer, avoiding off-by-one oscillation in the discrete
    # lattice invariants.
    _FP_ROUND_EPS = 1e-9

    @classmethod
    def _g_from_rho(cls, rho: float, num_steps: int) -> int:
        """
        Map a continuous ``rho`` to the discrete step count actually used by
        :meth:`compute_guidance_steps`.  Returns an integer in
        ``[0, max(0, num_steps - 1)]``.  ``num_steps <= 0`` degenerates to 0.
        """
        if num_steps <= 0:
            return 0
        g = int(math.floor(rho * num_steps + 0.5 + cls._FP_ROUND_EPS))
        if g < 0:
            g = 0
        if g > num_steps - 1:
            g = num_steps - 1
        return g

    @classmethod
    def _snap_to_lattice(cls, rho: float, num_steps: int) -> float:
        """
        Project ``rho`` onto the per-sample discrete lattice
        ``{g / num_steps : g ∈ [0, num_steps - 1]}``.  Idempotent.
        ``num_steps <= 0`` degenerates to ``0.0``.
        """
        if num_steps <= 0:
            return 0.0
        return cls._g_from_rho(rho, num_steps) / num_steps

    def _init_entry(self, num_steps: int) -> dict:
        # Snap the default frontier to the top lattice point for this sample
        # so that ``rho_star`` is on-lattice from the first visit onward.
        rho_star0 = self._snap_to_lattice(self.default_rho_star, num_steps)
        return {
            "rho_star": rho_star0,
            "rho": rho_star0,
            "mode": "exploit",
            "safety_counter": 0,
            "safety_triggered_total": 0,
            "visits": 0,
            "num_steps": int(max(0, num_steps)),
            # Most-recent update bookkeeping (set by :meth:`update`).
            "rho_star_before": rho_star0,
            "rho_star_after": rho_star0,
            "last_success": False,
        }

    # ------------------------------------------------------------------
    # Core API (mirrors PerSampleCurriculumState for polymorphic use)
    # ------------------------------------------------------------------

    def get_rho(self, sample_id: str, num_steps: int) -> float:
        """
        Sample a supervision ratio for *sample_id* following the MFC
        exploit / probe policy.

        The probe branch works in the *discrete step space* so each probe
        strictly advances the hint by at least one teacher-step relative to
        the current exploit level.  ``rho_star`` is kept on the per-sample
        lattice as a side-effect.
        """
        if sample_id not in self.states:
            self.states[sample_id] = self._init_entry(num_steps)

        s = self.states[sample_id]
        s["num_steps"] = int(max(0, num_steps))

        # Defensive: snap rho_star to the lattice every visit (idempotent
        # after the first hit, cheap, and guarantees the invariant even if
        # state was loaded from an old checkpoint that predates this class).
        s["rho_star"] = self._snap_to_lattice(s["rho_star"], num_steps)
        rho_star = s["rho_star"]
        g_curr = self._g_from_rho(rho_star, num_steps)

        # Already at (or effectively at) the discrete floor — no room to
        # probe lower.  All visits collapse to the exploit branch at the
        # current on-lattice value (which may be exactly 0).
        if g_curr <= 0 or num_steps <= 0:
            s["mode"] = "exploit"
            s["rho"] = rho_star
            return rho_star

        if random.random() < self.p_probe:
            # Discrete probe: strictly below g_curr, uniform over step space.
            g_probe = random.randint(0, g_curr - 1)
            s["mode"] = "probe"
            s["rho"] = g_probe / num_steps
        else:
            s["mode"] = "exploit"
            s["rho"] = rho_star

        return s["rho"]

    def update(self, sample_id: str, avg_reward: float, num_steps: int) -> None:
        """
        Update ``rho_star_i`` based on the rollout group's ``avg_reward``.

        The three cases (A/B/C) correspond directly to the spec in
        §3.4 of 第三轮实验/MFC_算法设计_spec.md.
        """
        if sample_id not in self.states:
            return

        s = self.states[sample_id]
        s["num_steps"] = int(max(0, num_steps))
        rho_used = s["rho"]
        mode = s["mode"]
        rho_star_before = s["rho_star"]
        s["rho_star_before"] = rho_star_before
        s["visits"] += 1

        success = avg_reward >= self.epsilon
        s["last_success"] = success

        if success:
            # Case A — frontier advances (monotone non-increasing).
            # Probe ``rho_used`` is already on the per-sample lattice; exploit
            # ``rho_used == rho_star`` which is also on-lattice by invariant.
            if rho_used <= s["rho_star"]:
                s["rho_star"] = rho_used
            s["safety_counter"] = 0
        else:
            if mode == "exploit":
                # Case B — failure at the current frontier in exploit mode.
                s["safety_counter"] += 1
                if s["safety_counter"] >= self.safety_K:
                    s["rho_star"] = self._bump_safety_on_lattice(
                        s["rho_star"], num_steps
                    )
                    s["safety_counter"] = 0
                    s["safety_triggered_total"] += 1
            else:
                # Case C — probe failure: ratchet, no regression.
                pass

        s["rho_star_after"] = s["rho_star"]

    def _bump_safety_on_lattice(self, rho_star: float, num_steps: int) -> float:
        """
        Apply the safety-valve regression: continuous bump by
        ``delta_safe``, snapped to the discrete lattice, with a guaranteed
        ``+1`` step advance so a small ``delta_safe`` never silently no-ops.
        The result is clamped to the top lattice point ``(num_steps - 1) / num_steps``.
        """
        if num_steps <= 0:
            return 0.0
        g_curr = self._g_from_rho(rho_star, num_steps)
        g_cont_bumped = self._g_from_rho(
            min(1.0, rho_star + self.delta_safe), num_steps
        )
        g_new = max(g_curr + 1, g_cont_bumped)
        g_new = min(num_steps - 1, g_new)
        return g_new / num_steps

    # ------------------------------------------------------------------
    # Step discretisation (shared semantics with AdaBack)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_guidance_steps(
        steps: list, rho: float
    ) -> list:
        """Reuse the same discretisation rule as :class:`PerSampleCurriculumState`."""
        return PerSampleCurriculumState.compute_guidance_steps(steps, rho)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "method": "mfc",
            "p_probe": self.p_probe,
            "safety_K": self.safety_K,
            "delta_safe": self.delta_safe,
            "default_rho_star": self.default_rho_star,
            "n_rollouts": self.n_rollouts,
            "epsilon": self.epsilon,
            "states": dict(self.states),
        }

    def load_state_dict(self, d: dict) -> None:
        self.p_probe = float(d.get("p_probe", self.p_probe))
        self.safety_K = int(d.get("safety_K", self.safety_K))
        self.delta_safe = float(d.get("delta_safe", self.delta_safe))
        self.default_rho_star = float(
            d.get("default_rho_star", self.default_rho_star)
        )
        self.n_rollouts = int(d.get("n_rollouts", self.n_rollouts))
        self.epsilon = 1.0 / self.n_rollouts
        self.states = d.get("states", {})
        logger.info(
            f"MonotoneFrontierCurriculumState restored: "
            f"{len(self.states)} samples tracked"
        )

    # ------------------------------------------------------------------
    # Metrics (mfc/* prefix for wandb)
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """
        Aggregate metrics for wandb / logging.

        Two complementary "frac at zero"-style metrics are exported:

        * ``mfc/frac_at_zero`` — fraction with the continuous frontier
          ``rho_star < 0.05``.  Directly comparable to AdaBack's
          ``adaptive/frac_rho_max_below_0_05``.
        * ``mfc/frac_effective_zero`` — fraction whose discrete hint
          ``round(rho_star * num_steps)`` is 0 (i.e. the model trains on a
          *functionally* zero-guidance prompt).  This is the most faithful
          operationalisation of the paper's "entered no-shift training"
          surrogate objective.
        """
        if not self.states:
            return {}

        rho_stars = [s["rho_star"] for s in self.states.values()]
        rhos_used = [s["rho"] for s in self.states.values()]
        visits = [s["visits"] for s in self.states.values()]
        safety_triggers = [s["safety_triggered_total"] for s in self.states.values()]
        safety_counters = [s["safety_counter"] for s in self.states.values()]
        modes = [s["mode"] for s in self.states.values()]
        num_steps_list = [s.get("num_steps", 0) for s in self.states.values()]

        n = len(rho_stars)

        rho_stars_sorted = sorted(rho_stars)
        median = rho_stars_sorted[n // 2] if n > 0 else 0.0

        # Discrete-aware effective zero.
        g_stars = [
            self._g_from_rho(r, ns)
            for r, ns in zip(rho_stars, num_steps_list)
        ]
        frac_effective_zero = sum(1 for g in g_stars if g == 0) / n
        frac_effective_below_2 = sum(1 for g in g_stars if g < 2) / n

        # Continuous-threshold convergence (apples-to-apples with AdaBack).
        frac_rho_star_below_0_05 = sum(1 for v in rho_stars if v < 0.05) / n
        frac_rho_star_below_0_1 = sum(1 for v in rho_stars if v < 0.1) / n
        frac_rho_star_at_zero_exact = sum(1 for v in rho_stars if v <= 0.0) / n
        frac_at_one = sum(1 for v in rho_stars if v > 0.999) / n

        return {
            "mfc/mean_rho_star": sum(rho_stars) / n,
            "mfc/median_rho_star": median,
            "mfc/min_rho_star": min(rho_stars) if rho_stars else 0.0,
            "mfc/max_rho_star": max(rho_stars) if rho_stars else 0.0,
            # Faithful convergence indicators
            "mfc/frac_effective_zero": frac_effective_zero,
            "mfc/frac_effective_below_2_steps": frac_effective_below_2,
            "mfc/frac_at_zero": frac_rho_star_below_0_05,
            "mfc/frac_rho_star_below_0_05": frac_rho_star_below_0_05,
            "mfc/frac_rho_star_below_0_1": frac_rho_star_below_0_1,
            "mfc/frac_rho_star_at_zero_exact": frac_rho_star_at_zero_exact,
            "mfc/frac_at_one": frac_at_one,
            # Visit statistics
            "mfc/mean_rho_used": sum(rhos_used) / n,
            "mfc/probe_fraction": sum(1 for m in modes if m == "probe") / n,
            "mfc/mean_g_star": sum(g_stars) / n if g_stars else 0.0,
            "mfc/mean_visits": sum(visits) / n,
            "mfc/num_tracked": n,
            # Safety-valve statistics
            "mfc/safety_trigger_total": sum(safety_triggers),
            "mfc/mean_safety_counter": sum(safety_counters) / n,
            "mfc/epsilon": self.epsilon,
        }
