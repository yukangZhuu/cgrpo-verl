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
Lightweight training metrics tracker for CGRPO.
Tracks per-step success rate and batch statistics for logging only — no scheduling.
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)


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
