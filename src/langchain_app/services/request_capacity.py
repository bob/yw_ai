"""Bound simultaneous paid work across sessions in one application process."""

from functools import lru_cache
import os
from threading import BoundedSemaphore


class RequestCapacityError(RuntimeError):
    """A safe, retryable failure when all application request slots are busy."""

    def __init__(self) -> None:
        super().__init__("The assistant is busy. Please try again shortly.")


class RequestCapacity:
    """Share only capacity counters; never share user or conversation data."""

    def __init__(self, maximum: int = 4) -> None:
        if type(maximum) is not int or not 1 <= maximum <= 32:
            raise ValueError("ASSISTANT_MAX_CONCURRENT_REQUESTS must be an integer from 1 to 32")
        self._slots = BoundedSemaphore(maximum)

    def acquire(self) -> bool:
        return self._slots.acquire(blocking=False)

    def release(self) -> None:
        self._slots.release()


@lru_cache(maxsize=1)
def get_request_capacity() -> RequestCapacity:
    """Read operator configuration once, without adding an unbounded queue."""
    return RequestCapacity(int(os.getenv("ASSISTANT_MAX_CONCURRENT_REQUESTS", "4")))
