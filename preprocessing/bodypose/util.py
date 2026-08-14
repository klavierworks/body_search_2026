"""Progress reporting and small shared helpers."""

from __future__ import annotations

import shutil
import sys
import time


def human_count(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


def human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


class Progress:
    """A single-line progress meter that degrades to plain lines when piped."""

    def __init__(self, total: int | None, label: str = "", interval: float = 0.2):
        self.total = total
        self.label = label
        self.interval = interval
        self.count = 0
        self.start = time.monotonic()
        self._last_draw = 0.0
        self._tty = sys.stderr.isatty()
        self.suffix = ""

    def advance(self, n: int = 1, suffix: str = "") -> None:
        self.count += n
        if suffix:
            self.suffix = suffix
        now = time.monotonic()
        if now - self._last_draw < self.interval:
            return
        self._last_draw = now
        self._draw()

    def _line(self) -> str:
        elapsed = time.monotonic() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        parts = [self.label] if self.label else []
        if self.total:
            pct = 100.0 * self.count / self.total
            parts.append(f"{self.count:,}/{self.total:,} ({pct:5.1f}%)")
            if rate > 0:
                remaining = (self.total - self.count) / rate
                parts.append(f"eta {human_duration(remaining)}")
        else:
            parts.append(f"{self.count:,}")
        parts.append(f"{rate:.0f}/s")
        parts.append(human_duration(elapsed))
        if self.suffix:
            parts.append(self.suffix)
        return "  ".join(parts)

    def _draw(self) -> None:
        line = self._line()
        if self._tty:
            width = shutil.get_terminal_size((100, 24)).columns
            sys.stderr.write("\r\033[2K" + line[: width - 1])
            sys.stderr.flush()
        elif self.count % 5000 == 0:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()

    def done(self) -> None:
        if self._tty:
            sys.stderr.write("\r\033[2K")
        sys.stderr.write(self._line() + "\n")
        sys.stderr.flush()


def note(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
