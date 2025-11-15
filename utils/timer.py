"""
utils/timer.py

Timing and profiling utilities for:
  - Measuring inference, retrieval, and generation steps
  - Logging latency and throughput
"""

import time
from typing import Optional, Dict, Any
from contextlib import ContextDecorator

try:
    from utils.logging_utils import get_logger
    _logger = get_logger("idr_timer")
except Exception:
    _logger = None


# ------------------------------------------------------------
# 🕒 Basic Timer Context
# ------------------------------------------------------------

class Timer(ContextDecorator):
    """
    Context manager for measuring execution time.

    Example:
        with Timer("diffusion step"):
            run_diffusion()
    """

    def __init__(self, name: str = "block", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            msg = f"⏱️  [{self.name}] took {self.elapsed:.3f}s"
            print(msg) if _logger is None else _logger.info(msg)
        return False  # don't suppress exceptions

    def get_elapsed(self) -> float:
        """Return elapsed time after context exits."""
        return self.elapsed or 0.0


# ------------------------------------------------------------
# ⏳ Timer Manager (for multi-component tracking)
# ------------------------------------------------------------

class TimerManager:
    """
    Manage multiple timers (e.g., retrieval, diffusion, constraint, evaluation).
    """

    def __init__(self):
        self.records: Dict[str, float] = {}

    def start(self, name: str):
        """Start a timer for a specific step."""
        self.records[name] = -time.time()  # negative start time placeholder

    def stop(self, name: str):
        """Stop a timer and record elapsed time."""
        if name not in self.records or self.records[name] >= 0:
            raise ValueError(f"Timer '{name}' not active.")
        self.records[name] = time.time() + self.records[name]  # record duration

    def get(self, name: str) -> float:
        """Get elapsed time for a specific timer."""
        return max(0.0, self.records.get(name, 0.0))

    def summary(self) -> Dict[str, float]:
        """Return all recorded times."""
        return {k: round(v, 3) for k, v in self.records.items() if v >= 0}

    def log_summary(self):
        """Log or print summary nicely."""
        summary = self.summary()
        if not summary:
            print("No timers recorded.")
            return
        msg_lines = ["⏲️  Timer Summary:"]
        for name, t in summary.items():
            msg_lines.append(f"  - {name:<20s}: {t:.3f}s")
        msg = "\n".join(msg_lines)
        print(msg) if _logger is None else _logger.info(msg)


# ------------------------------------------------------------
# 🧮 Utility Function
# ------------------------------------------------------------

def timed_function(name: str):
    """
    Decorator for timing a function call.
    Example:
        @timed_function("retrieval")
        def retrieve_data(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            msg = f"⏱️  Function '{name}' executed in {elapsed:.3f}s"
            print(msg) if _logger is None else _logger.info(msg)
            return result
        return wrapper
    return decorator


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Timer and TimerManager...\n")

    with Timer("mock diffusion generation"):
        time.sleep(1.2)

    manager = TimerManager()
    manager.start("retrieval")
    time.sleep(0.5)
    manager.stop("retrieval")

    manager.start("constraint_check")
    time.sleep(0.3)
    manager.stop("constraint_check")

    manager.log_summary()

    @timed_function("fid_computation")
    def dummy_fid():
        time.sleep(0.4)

    dummy_fid()