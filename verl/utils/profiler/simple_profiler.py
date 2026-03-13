import time
from contextlib import contextmanager
from collections import defaultdict

class Profiler:
    """
    A simple performance profiler to track execution time of code blocks.
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
    
    @contextmanager
    def context_manager(self, name: str):
        """
        Context manager to measure execution time of a block.
        
        Args:
            name: Name of the code block/operation to profile.
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings[name].append(duration)
    
    def get_metrics(self) -> dict:
        """
        Get aggregated metrics for all profiled operations.
        
        Returns:
            Dictionary with average timing metrics.
        """
        metrics = {}
        for name, durations in self.timings.items():
            if durations:
                # Calculate average duration
                avg_duration = sum(durations) / len(durations)
                metrics[f"timing/{name}"] = avg_duration
                # Clear history after reporting to keep metrics current (per-step)
                # or keep it if you want cumulative average. 
                # For training loops, usually we want per-step or moving average.
                # Here we clear it to report "last step" or "since last report" metrics.
                self.timings[name] = []
        return metrics
