import time


def elapsed_ms(started_at: float) -> float:
    """Return the number of milliseconds elapsed since ``started_at``."""
    return round((time.perf_counter() - started_at) * 1000, 1)
