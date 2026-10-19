"""Plain-text run-based ETA for the benchmark CLI.

This is independent from the opt-in JSONL event emitter: progress advances
only after completed measured runs, never after warmups or token events.
"""

from __future__ import annotations

import time
from typing import Callable


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class TerminalRunProgress:
    """Estimate time remaining from completed measured runs.

    Elapsed time starts before the first per-shape warmup; it therefore
    includes warmups and other overhead without counting them as completed
    measured runs. The ETA remains unknown until the first measured run ends.
    """

    BAR_WIDTH = 20

    def __init__(self, total_runs: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.total_runs = max(0, total_runs)
        self.completed_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self._successful_seconds = 0.0
        self._last_run_at: float | None = None
        self._clock = clock
        self._started_at: float | None = None

    def start(self) -> str:
        if self._started_at is None:
            self._started_at = self._clock()
            self._last_run_at = self._started_at
        return self._render()

    def complete_run(self, *, success: bool = True) -> str:
        """Advance an attempted measured run, excluding failed runs from ETA samples."""
        if self._started_at is None:
            self.start()
        if self.completed_runs < self.total_runs:
            now = self._clock()
            if success:
                self.successful_runs += 1
                self._successful_seconds += max(0.0, now - self._last_run_at)
            else:
                self.failed_runs += 1
            self.completed_runs += 1
            self._last_run_at = now
        return self._render()

    def _render(self) -> str:
        elapsed = max(0.0, self._clock() - self._started_at) if self._started_at is not None else 0.0
        fraction = self.completed_runs / self.total_runs if self.total_runs else 0.0
        filled = int(self.BAR_WIDTH * fraction)
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        if self.total_runs > 0 and self.completed_runs == self.total_runs:
            eta = "00:00"
        elif self.successful_runs:
            avg_seconds = self._successful_seconds / self.successful_runs
            eta = _format_duration(avg_seconds * (self.total_runs - self.completed_runs))
        else:
            eta = "--:--"
        failures = (
            f" | OK {self.successful_runs} FAIL {self.failed_runs}"
            if self.failed_runs else ""
        )
        return (
            f"[{bar}] {int(100 * fraction):3d}% | "
            f"{self.completed_runs}/{self.total_runs} | "
            f"Elapsed {_format_duration(elapsed)} | ETA {eta}{failures}"
        )
