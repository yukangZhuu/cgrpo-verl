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
Curriculum Manager for Curriculum-GRPO (C-GRPO).
Implements backward chaining learning strategy.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for Curriculum Manager."""
    initial_k: int = 1
    max_k: int = 10
    ema_alpha: float = 0.2
    base_threshold: float = 0.92
    threshold_decay: float = 0.95
    patience: int = 1000
    min_steps_per_k: int = 100
    warmup_steps: int = 0
    early_stop_enabled: bool = True
    early_stop_threshold: float = 0.85
    early_stop_min_steps: int = 200


class CurriculumManager:
    """
    Manages curriculum learning for C-GRPO.
    
    The curriculum level k determines how many steps the student model
    needs to generate (starting from the end of teacher's solution).
    
    Curriculum progression:
    - k=1: Student generates only the last step
    - k=2: Student generates the last 2 steps
    - ...
    - k=max_k: Student generates all steps
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """
        Initialize Curriculum Manager.
        
        Args:
            config: Curriculum configuration. Uses defaults if None.
        """
        self.config = config or CurriculumConfig()
        self.k = self.config.initial_k
        self.sr_ema = 0.0
        self.patience_counter = 0
        self.total_steps = 0
        self.steps_at_current_k = 0
        self.history = []
        self._should_stop = False
        self._steps_at_max_k = 0
        
        logger.info(f"CurriculumManager initialized with k={self.k}, max_k={self.config.max_k}")
    
    def get_threshold(self) -> float:
        """
        Calculate dynamic threshold based on current k.
        
        Threshold decreases as k increases to make progression easier.
        
        Returns:
            Current success rate threshold for advancement.
        """
        threshold = self.config.base_threshold * (self.config.threshold_decay ** (self.k - 1))
        return max(threshold, 0.5)
    
    def update(self, batch_success_rate: float, batch_size: int = 1) -> dict:
        """
        Update curriculum based on batch performance.
        
        Args:
            batch_success_rate: Success rate of current batch (0.0 to 1.0).
            batch_size: Number of samples in the batch.
        
        Returns:
            Dictionary containing curriculum metrics.
        """
        self.total_steps += 1
        self.steps_at_current_k += 1
        
        self.sr_ema = (
            self.config.ema_alpha * batch_success_rate + 
            (1 - self.config.ema_alpha) * self.sr_ema
        )
        
        advanced = False
        threshold = self.get_threshold()
        
        if self._should_advance():
            self._advance_curriculum()
            advanced = True
        
        if self.is_completed():
            self._steps_at_max_k += 1
            if self._check_early_stop():
                self._should_stop = True
        
        metrics = {
            "curriculum/k": self.k,
            "curriculum/sr_ema": self.sr_ema,
            "curriculum/threshold": threshold,
            "curriculum/patience_counter": self.patience_counter,
            "curriculum/advanced": float(advanced),
            "curriculum/total_steps": self.total_steps,
            "curriculum/steps_at_current_k": self.steps_at_current_k,
            "curriculum/steps_at_max_k": self._steps_at_max_k,
            "curriculum/should_stop": float(self._should_stop),
        }
        
        self.history.append({
            "step": self.total_steps,
            "k": self.k,
            "batch_sr": batch_success_rate,
            "sr_ema": self.sr_ema,
            "threshold": threshold,
        })
        
        return metrics
    
    def _should_advance(self) -> bool:
        """
        Check if curriculum should advance to next level.
        
        Returns:
            True if should advance, False otherwise.
        """
        if self.k >= self.config.max_k:
            return False
        
        if self.total_steps < self.config.warmup_steps:
            return False
        
        if self.steps_at_current_k < self.config.min_steps_per_k:
            return False
        
        threshold = self.get_threshold()
        
        if self.sr_ema > threshold:
            logger.info(
                f"Curriculum advancing: sr_ema={self.sr_ema:.4f} > threshold={threshold:.4f}"
            )
            return True
        
        self.patience_counter += 1
        if self.patience_counter >= self.config.patience:
            logger.info(
                f"Curriculum advancing (patience exhausted): "
                f"patience_counter={self.patience_counter}"
            )
            return True
        
        return False
    
    def _advance_curriculum(self):
        """Advance curriculum to next level."""
        self.k += 1
        self.patience_counter = 0
        self.steps_at_current_k = 0
        logger.info(f"Curriculum advanced to k={self.k}")
    
    def _check_early_stop(self) -> bool:
        """
        Check if training should stop early.
        
        Early stopping conditions:
        1. k has reached max_k
        2. Success rate EMA exceeds early_stop_threshold
        3. Minimum steps at max_k have been completed
        
        Returns:
            True if should stop, False otherwise.
        """
        if not self.config.early_stop_enabled:
            return False
        
        if not self.is_completed():
            return False
        
        if self._steps_at_max_k < self.config.early_stop_min_steps:
            return False
        
        if self.sr_ema >= self.config.early_stop_threshold:
            logger.info(
                f"Early stopping triggered: k={self.k} (max_k), "
                f"sr_ema={self.sr_ema:.4f} >= threshold={self.config.early_stop_threshold:.4f}, "
                f"steps_at_max_k={self._steps_at_max_k}"
            )
            return True
        
        return False
    
    def should_stop(self) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop.
        """
        return self._should_stop
    
    def get_current_k(self) -> int:
        """Get current curriculum level."""
        return self.k
    
    def set_k(self, k: int):
        """
        Manually set curriculum level.
        
        Args:
            k: New curriculum level.
        """
        if 1 <= k <= self.config.max_k:
            self.k = k
            self.steps_at_current_k = 0
            self.patience_counter = 0
            logger.info(f"Curriculum manually set to k={self.k}")
        else:
            raise ValueError(f"k must be between 1 and {self.config.max_k}, got {k}")
    
    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing."""
        return {
            "k": self.k,
            "sr_ema": self.sr_ema,
            "patience_counter": self.patience_counter,
            "total_steps": self.total_steps,
            "steps_at_current_k": self.steps_at_current_k,
            "history": self.history,
            "_should_stop": self._should_stop,
            "_steps_at_max_k": self._steps_at_max_k,
            "config": {
                "initial_k": self.config.initial_k,
                "max_k": self.config.max_k,
                "ema_alpha": self.config.ema_alpha,
                "base_threshold": self.config.base_threshold,
                "threshold_decay": self.config.threshold_decay,
                "patience": self.config.patience,
                "min_steps_per_k": self.config.min_steps_per_k,
                "warmup_steps": self.config.warmup_steps,
                "early_stop_enabled": self.config.early_stop_enabled,
                "early_stop_threshold": self.config.early_stop_threshold,
                "early_stop_min_steps": self.config.early_stop_min_steps,
            }
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from dictionary."""
        self.k = state_dict["k"]
        self.sr_ema = state_dict["sr_ema"]
        self.patience_counter = state_dict["patience_counter"]
        self.total_steps = state_dict["total_steps"]
        self.steps_at_current_k = state_dict["steps_at_current_k"]
        self.history = state_dict.get("history", [])
        self._should_stop = state_dict.get("_should_stop", False)
        self._steps_at_max_k = state_dict.get("_steps_at_max_k", 0)
        
        if "config" in state_dict:
            config_dict = state_dict["config"]
            self.config = CurriculumConfig(**config_dict)
        
        logger.info(f"CurriculumManager loaded: k={self.k}, sr_ema={self.sr_ema:.4f}")
    
    def get_progress_percentage(self) -> float:
        """
        Get curriculum progress as percentage.
        
        Returns:
            Progress percentage (0.0 to 100.0).
        """
        return (self.k / self.config.max_k) * 100.0
    
    def is_completed(self) -> bool:
        """Check if curriculum is completed (reached max_k)."""
        return self.k >= self.config.max_k
    
    def get_summary(self) -> str:
        """Get summary string for logging."""
        return (
            f"Curriculum(k={self.k}/{self.config.max_k}, "
            f"sr_ema={self.sr_ema:.4f}, "
            f"threshold={self.get_threshold():.4f}, "
            f"progress={self.get_progress_percentage():.1f}%)"
        )
