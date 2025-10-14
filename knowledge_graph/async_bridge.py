import asyncio
import atexit
import functools
import threading
from typing import Callable, Coroutine, TypeVar, cast

T = TypeVar("T")

_thread_state = threading.local()


def get_persistent_loop() -> asyncio.AbstractEventLoop:
    """
    Return a persistent event loop bound to the current thread.

    This ensures that synchronous wrappers always reuse the same live event loop
    instead of creating and closing a new loop on every call. Reusing a loop
    prevents issues where async resources (like HTTP clients) become tied to a
    loop that has been closed, which commonly manifests as 'Event loop is
    closed' errors for newcomers to async.
    """
    loop = getattr(_thread_state, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_state.loop = loop
        atexit.register(shutdown_loop, loop)
    return loop


def shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Cleanly shut down the persistent thread-local event loop at process exit.

    This cancels any pending tasks before closing the loop.
    """
    if loop.is_closed():
        return
    try:
        pending = asyncio.all_tasks(loop=loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def async_to_sync(
    async_func: Callable[..., Coroutine[None, None, T]],
) -> Callable[..., T]:
    """
    Decorator that converts async methods to synchronous interface

    This decorator wraps async methods to provide a synchronous interface by
    automatically managing the event loop. It uses a per-thread loop, so async resources
    remain bound to a live loop across calls. If there is no running loop, it creates a
    new persistent event loop bound to the current thread.

    The decorator preserves the original function's return type, so type checkers should
    understand that sync wrappers return the actual objects, not coroutines.

    Args:
        async_func: An async function that returns a Coroutine[Any, Any, T]

    Returns:
        A synchronous function that returns T (the unwrapped result type)

    Raises:
        RuntimeError: if called from an already-running event loop, it asks the caller
        to use the async version directly

    Example:
        @async_to_sync
        async def get_data(self) -> MyData:
            return await self.get_data_async()
        # Type checker knows this returns MyData, not Awaitable[MyData]
        data = session.get_data()
    """

    @functools.wraps(async_func)
    def wrapper(self, *args, **kwargs) -> T:
        try:
            running = asyncio.get_running_loop()
            if running.is_running():
                raise RuntimeError(
                    f"Cannot call sync version of {async_func.__name__} from async context. "
                    f"Use {async_func.__name__}_async directly."
                )
        except RuntimeError:
            # Not in a running loop in this thread → OK to use our persistent loop
            pass

        loop = get_persistent_loop()
        return loop.run_until_complete(async_func(self, *args, **kwargs))

    return cast(Callable[..., T], wrapper)
