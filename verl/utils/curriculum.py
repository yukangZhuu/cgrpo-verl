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
        at_zero = sum(1 for r in rhos if r < 0.05) / n
        at_full = sum(1 for r in rhos if r > 0.95) / n

        return {
            "adaptive/mean_rho": sum(rhos) / n,
            "adaptive/mean_rho_min": sum(rho_mins) / n,
            "adaptive/mean_rho_max": sum(rho_maxs) / n,
            "adaptive/frac_at_zero": at_zero,
            "adaptive/frac_at_full": at_full,
            "adaptive/mean_visits": sum(visits) / n,
            "adaptive/num_tracked": n,
        }
