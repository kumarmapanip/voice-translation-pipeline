import asyncio

RETRIABLE = (429, 500, 503, 504)


def _status(e: Exception):
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code if isinstance(code, int) else None


async def with_retries(fn, *args, attempts: int = 3, base_delay: float = 0.5):
    """Gives transient errors (rate limits, server hiccups, dropped
    connections) a couple of retries with backoff. Anything else is a real
    bug and propagates immediately."""
    for i in range(attempts):
        try:
            return await fn(*args)
        except Exception as e:
            transient = _status(e) in RETRIABLE or isinstance(e, (OSError, TimeoutError))
            if not transient or i == attempts - 1:
                raise
            await asyncio.sleep(base_delay * 2**i)
