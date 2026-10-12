"""
One kept-open HTTP connection per server, instead of a new one per request.

WHY
    A voice turn is several requests in a row to the same place: the words to
    the recogniser, the model, then one request per spoken sentence. Opening a
    fresh connection for each — which is what `async with httpx.AsyncClient()`
    per call does — pays TCP and TLS setup every time. Through the subscription
    relay that is two to three round trips to Frankfurt per request, roughly
    0.1-0.2 s each, in front of every sentence the assistant says.

    Kept open, only the first request of a session pays it.

HOW LONG A CONNECTION IS KEPT
    Shorter than the server keeps it: a client that reuses a connection the
    server has just closed gets "Server disconnected without sending a
    response" on the next request — measured live, the first sentence after
    a pause failed that way (uvicorn closed idle connections after 5 s, this
    kept them 120 s). Our servers keep them SERVER_KEEPALIVE seconds
    (backend/relay.py, backend/gpu); this lets go well before.
    Something in between (a home router, a mobile network) may still drop an
    idle connection earlier, so a request the server closed WITHOUT sending
    any response is sent once more on a fresh connection (_RetryStale).

WHY KEYED BY EVENT LOOP
    An httpx.AsyncClient belongs to the loop that first used it. The engine runs
    on main.py's loop, but the settings panel and one-shot calls run on others;
    sharing one client across loops raises at the worst possible moment. So
    there is one per (loop, server, TLS trust), and a client whose loop has
    closed is simply replaced.
"""
from __future__ import annotations

import asyncio
import threading
import weakref

_lock = threading.Lock()
KEEPALIVE = 50.0            # < the servers' SERVER_KEEPALIVE (75 s)
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict]" = weakref.WeakKeyDictionary()


def _retry_stale_transport(verify, trust_env: bool, limits):
    import httpx

    class _RetryStale(httpx.AsyncBaseTransport):
        """One more try on a fresh connection when the server closed the
        connection without answering at all: the request was never seen,
        so sending it again does not do anything twice."""

        def __init__(self):
            self._inner = httpx.AsyncHTTPTransport(verify=verify, trust_env=trust_env, limits=limits)

        async def handle_async_request(self, request):
            try:
                return await self._inner.handle_async_request(request)
            except httpx.RemoteProtocolError as e:
                if "without sending a response" not in str(e):
                    raise
                return await self._inner.handle_async_request(request)

        async def aclose(self):
            await self._inner.aclose()

    return _RetryStale()


def async_client(base: str, *, verify=True, trust_env: bool = True, timeout: float = 60.0):
    """A kept-open AsyncClient for `base` on the running loop."""
    import httpx

    loop = asyncio.get_running_loop()
    key = (base, id(verify) if not isinstance(verify, bool) else verify, trust_env, float(timeout))
    with _lock:
        per_loop = _clients.get(loop)
        if per_loop is None:
            per_loop = _clients[loop] = {}
        cl = per_loop.get(key)
        if cl is None or cl.is_closed:
            limits = httpx.Limits(max_keepalive_connections=8, keepalive_expiry=KEEPALIVE)
            cl = per_loop[key] = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=15.0), trust_env=trust_env,
                transport=_retry_stale_transport(verify, trust_env, limits))
    return cl
