from __future__ import annotations

import time
import typing as t


class Timer:
    """A timer context to log time and hours"""

    def __init__(self):
        self.start = float
        self.end = float
        self.exec_time = float

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_: t.Any) -> None:
        self.end = time.perf_counter()
        self.exec_time = self.end - self.start
