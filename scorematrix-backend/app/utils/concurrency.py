"""Concurrency helpers that are safe across event loops (important for tests)."""

from __future__ import annotations

import asyncio
import weakref


class BoundedConcurrency:
    """An asyncio semaphore that is created lazily *per running event loop*.

    A plain ``asyncio.Semaphore`` can end up bound to the wrong loop when a
    service object outlives a loop (e.g. between test-client portals). This
    wrapper hands out one semaphore per loop and is used as an async context
    manager.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
            weakref.WeakKeyDictionary()
        )

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._by_loop.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self.limit)
            self._by_loop[loop] = sem
        return sem

    async def __aenter__(self) -> None:
        await self._semaphore().acquire()

    async def __aexit__(self, *exc_info) -> None:
        self._semaphore().release()
