"""
Lightweight training progress display. Pure stdlib - the dependency
contract (4 core deps, always) rules out tqdm.

In a terminal (TTY) it renders a single self-updating line:

    Screening models [############........]  9/14 gradient_boosting 12.3s

In non-TTY contexts (piped output, CI logs, some notebooks) it prints one
milestone line per step instead, so logs stay readable.
"""
from __future__ import annotations

import sys
import time

__all__ = ["ProgressBar"]

_BAR_WIDTH = 24


class ProgressBar:
    """Minimal progress bar: ``update()`` per step, ``close()`` when done."""

    def __init__(self, total: int, desc: str = "", enabled: bool = True, stream=None):
        self.total = max(int(total), 1)
        self.desc = desc
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = bool(enabled)
        self.count = 0
        self.start = time.monotonic()
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_len = 0

    def update(self, label: str = ""):
        """Advance one step and re-render, tagging the step with ``label``."""
        self.count = min(self.count + 1, self.total)
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.start
        if self._tty:
            filled = int(_BAR_WIDTH * self.count / self.total)
            bar = "#" * filled + "." * (_BAR_WIDTH - filled)
            line = f"{self.desc} [{bar}] {self.count:>2}/{self.total} {label} {elapsed:.1f}s"
            pad = " " * max(self._last_len - len(line), 0)
            self.stream.write("\r" + line + pad)
            self.stream.flush()
            self._last_len = len(line)
        else:
            self.stream.write(f"{self.desc}: {self.count}/{self.total} {label} ({elapsed:.1f}s)\n")
            self.stream.flush()

    def close(self, final: str = ""):
        """Finish the bar; optionally replace it with a final message."""
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.start
        message = final or f"{self.desc} done: {self.count}/{self.total} in {elapsed:.1f}s"
        if self._tty:
            pad = " " * max(self._last_len - len(message), 0)
            self.stream.write("\r" + message + pad + "\n")
        else:
            self.stream.write(message + "\n")
        self.stream.flush()
