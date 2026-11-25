"""HTTP plumbing shared by the REST and proxy servers."""
from __future__ import annotations

import asyncio
import logging
import secrets
import sys
from collections import OrderedDict
from typing import Optional
from urllib.parse import parse_qs

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.datastructures import Headers

from smrti.servers import config as cfg

logger = logging.getLogger("smrti.servers")

# How long uvicorn waits for open connections at shutdown, and how long the
# lifespan then waits for in-flight stores. Together they stay inside the 10
# seconds `smrti stop` gives a server before it kills it.
GRACEFUL_SHUTDOWN_SECONDS = 3
DRAIN_SECONDS = 5

# The cookie the visualizer authenticates with (see _viz_login). It holds a
# random session token, never the key, and is honoured only for reads.
KEY_COOKIE = "smrti_session"
# Live visualizer sessions: token -> the key that opened it. Bounded, oldest
# dropped first; a server restart ends them all.
_viz_sessions: OrderedDict[str, str] = OrderedDict()
_VIZ_SESSIONS_MAX = 256

# What a presented key may act on: None is every tenant (SMRTI_API_KEY), a
# tuple is the tenants a SMRTI_API_KEYS entry lists, and _NO_KEY is a key
# this server does not know.
_NO_KEY = object()

# The process-wide logs hold every tenant's records. A key bound to some
# tenants reads them filtered to its own, but clearing them would erase the
# other tenants' records too, so only the all-tenants key may.
_ALL_TENANT_ROUTES = {("DELETE", "/llm-calls"), ("DELETE", "/decisions")}


def _same(supplied: str, key: str) -> bool:
    # Compared as bytes: compare_digest raises TypeError on a non-ASCII str,
    # which turned a stray character in a client's header into a 500.
    return secrets.compare_digest(supplied.encode(), key.encode())


def _tenants_of(supplied: str):
    """The tenants *supplied* may act on, or _NO_KEY when it is no key of ours."""
    if not supplied:
        return _NO_KEY
    if cfg.API_KEY and _same(supplied, cfg.API_KEY):
        return None
    for key, tenants in cfg.API_KEYS.items():
        if _same(supplied, key):
            return tenants
    return _NO_KEY


def _bearer(request: Request) -> str:
    # The scheme is case-insensitive (RFC 9110), so "bearer <key>" is the
    # same credential and must be recognised — and stripped — the same way.
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth[:7].lower() == "bearer " else ""


def _error(status: int, message: str, kind: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": code}},
    )


def _presented_tenants(request: Request):
    """The tenants of the first known key the request presents, or _NO_KEY.

    The visualizer's session cookie counts only on a read. A browser sends a
    cookie to every port of a host and treats them as one site, so a page on
    any other local port could otherwise write through it.
    """
    session_key = ""
    if request.method in ("GET", "HEAD"):
        session_key = _viz_sessions.get(request.cookies.get(KEY_COOKIE, ""), "")
    presented = (
        request.headers.get("x-api-key") or "",
        _bearer(request),
        session_key,
    )
    for supplied in presented:
        tenants = _tenants_of(supplied)
        if tenants is not _NO_KEY:
            return tenants
    return _NO_KEY


def _named_tenants(request: Request) -> list[str]:
    """Every tenant the request names: the proxy's header and the viz query."""
    named = (request.headers.get("x-smrti-tenant-id"), request.query_params.get("tenant_id"))
    return [t for t in named if t]


def _refusal(request: Request, tenants: tuple[str, ...]) -> Optional[str]:
    """Why a key bound to *tenants* may not make this request, if it may not."""
    for tenant in _named_tenants(request):
        if tenant not in tenants:
            return f"This API key may not act on tenant {tenant!r}"
    # Another database's tenants are not the ones this key was bound to in
    # the server's own: the alternate databases belong to the operator.
    if request.query_params.get("db"):
        return "This API key may not open another database"
    if (request.method, request.url.path) in _ALL_TENANT_ROUTES:
        return "This API key may not clear a log other tenants share"
    return None


async def api_key_middleware(request: Request, call_next):
    """Authenticate every request and bind it to the tenants its key may use.

    With neither SMRTI_API_KEY nor SMRTI_API_KEYS set, every request passes
    unchanged (local use). Otherwise the key comes from ``X-Api-Key``,
    ``Authorization: Bearer`` — so proxy clients can keep their upstream
    credentials in Authorization and supply the smrti key via X-Api-Key —
    or the visualizer's cookie (see _viz_login), and one this server does
    not know is a 401.

    A key from SMRTI_API_KEYS may act only on its own tenants. The check is
    made here, once, for every route: a request naming another tenant, an
    alternate ``?db=``, or a clear of a shared log is a 403 before any
    handler runs. The tenant it acts on is resolved here too, onto
    ``request.state.tenant`` — the one it named, else its key's first — so
    no handler applies a default of its own to a bound key.
    """
    if not (cfg.API_KEY or cfg.API_KEYS):
        return await call_next(request)
    if request.url.path == "/viz/login" and request.method == "POST":
        return await _viz_login(request)
    tenants = _presented_tenants(request)
    if tenants is _NO_KEY:
        if request.url.path == "/viz" and request.method == "GET":
            return HTMLResponse(_VIZ_LOGIN_FORM, status_code=401)
        return _error(401, "Invalid or missing API key", "authentication_error", "invalid_api_key")
    if tenants is not None:
        refusal = _refusal(request, tenants)
        if refusal:
            return _error(403, refusal, "permission_error", "tenant_not_allowed")
        named = _named_tenants(request)
        request.state.tenant = named[0] if named else tenants[0]
    request.state.tenants = tenants
    return await call_next(request)


# What a browser gets for /viz without a key: a form that posts the key in
# the request body, which neither the access log nor the history records.
_VIZ_LOGIN_FORM = """<!doctype html><meta charset="utf-8"><title>smrti visualizer</title>
<form method="post" action="viz/login"><label>API key
<input type="password" name="key" autocomplete="current-password" autofocus required></label>
<button type="submit">Open</button></form>"""


async def _viz_login(request: Request):
    """Trade a key posted from the login form for a cookie, then send the
    browser on to /viz.

    A browser navigating to the visualizer, and the EventSource it opens for
    the live log, send no header, so with a key configured the page could
    not load at all. The key arrives once in a POST body — a key in the URL
    landed in the access log and the history — and the browser keeps only a
    random session token, in an HttpOnly, SameSite=Strict cookie the auth
    check accepts on reads.
    """
    body = (await request.body()).decode("utf-8", "replace")
    supplied = (parse_qs(body).get("key") or [""])[0]
    if _tenants_of(supplied) is _NO_KEY:
        return HTMLResponse(_VIZ_LOGIN_FORM, status_code=401)
    token = secrets.token_urlsafe(32)
    _viz_sessions[token] = supplied
    while len(_viz_sessions) > _VIZ_SESSIONS_MAX:
        _viz_sessions.popitem(last=False)
    response = RedirectResponse("../viz", status_code=303)
    response.set_cookie(
        KEY_COOKIE, token, httponly=True, samesite="strict",
        secure=request.url.scheme == "https",
    )
    return response


def upstream_authorization(request: Request) -> str:
    """The client's Authorization header, unless it carries this server's own key.

    A client that authenticates to Smrti with ``Authorization: Bearer <key>``
    has no upstream credential in that header, and forwarding it handed the
    Smrti key to the model provider and to the extraction endpoint. A client
    that authenticated with X-Api-Key keeps its Authorization: that is the
    documented way to pass the upstream key through.
    """
    bearer = _bearer(request)
    if bearer and _tenants_of(bearer) is not _NO_KEY:
        return ""
    return request.headers.get("authorization", "")


def api_root(url: str) -> str:
    """*url* without a trailing ``/v1``: every caller appends ``/v1/<route>``.

    OpenAI-compatible servers document their base URL with the ``/v1`` on —
    the README's Docker example names ``…:11434/v1`` — and joining another
    one onto it requested ``/v1/v1/chat/completions``.
    """
    root = url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def tenant_for(request: Request, named: Optional[str], default: str) -> str:
    """The tenant a request acts on.

    For a key bound to tenants it is what the middleware resolved, already
    checked against the key. Otherwise — the all-tenants key, or no key
    configured — it is the tenant the request named, else *default*.
    """
    return getattr(request.state, "tenant", None) or named or default


def visible(request: Request, tenant_id: Optional[str]) -> bool:
    """Whether this request's key may see a record filed under *tenant_id*."""
    tenants = getattr(request.state, "tenants", None)
    return tenants is None or tenant_id in tenants


class BodyLimit:
    """Refuse a request body over SMRTI_MAX_BODY_BYTES with a 413.

    A declared Content-Length over the limit is refused before anything is
    read; a chunked body is counted as it arrives and refused once it passes
    the limit. Without it a single request could hold any amount of memory.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        limit = cfg.MAX_BODY_BYTES
        if scope["type"] != "http" or limit <= 0:
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await _too_large(limit)(scope, receive, send)
            return
        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, f"Request body is larger than {limit} bytes")
            return message

        await self.app(scope, counting_receive, send)


def _too_large(limit: int) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": f"Request body is larger than {limit} bytes"})


async def drain(tasks: set) -> None:
    """Wait, for at most DRAIN_SECONDS, for in-flight background work.

    Each task is a memory a client was already answered as though it were
    stored, so shutdown waits for them, and the wait is bounded because one
    hung model call must not hold the process up.
    """
    if tasks:
        await asyncio.wait(list(tasks), timeout=DRAIN_SECONDS)


def cached_instance(cache: OrderedDict, key, mint, lock, limit: int, tasks: set):
    """The instance *cache* holds under *key*, minting one with *mint* when absent.

    *lock* guards the dictionary and nothing else: minting writes the space's
    personality row, which can wait seconds on an epoch's write lock, so it
    runs outside it, and a request for a cached instance — the event loop's
    included — never waits behind another's mint. Two requests minting one
    key at once keep the first stored. Instances evicted past *limit* get
    the epoch they are owed (see :func:`consolidate_evicted`), also outside
    the lock.
    """
    with lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    fresh = mint()
    evicted = []
    with lock:
        if key not in cache:
            while len(cache) >= limit:
                evicted.append(cache.popitem(last=False)[1])
            cache[key] = fresh
        cache.move_to_end(key)
        mem = cache[key]
    for old in evicted:
        consolidate_evicted(old, tasks)
    return mem


def consolidate_evicted(mem, tasks: set) -> None:
    """Run the epoch an instance evicted from a cache is owed.

    The reflect loop consolidates the instances it can see, and an evicted
    instance takes its activity count with it: with more active spaces than
    a cache holds, spaces left before their epoch came round, and none of
    them ever aged. One with activity since its last epoch is consolidated
    as it leaves, in the executor, and tracked in *tasks* so shutdown waits
    for it. Nothing runs when the reflect loop is off.
    """
    from smrti.servers import reflect_loop

    ops = getattr(mem, "ops_since_reflect", None)
    if not isinstance(ops, int) or ops <= 0 or reflect_loop.REFLECT_INTERVAL <= 0:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        mem.reflect()  # no event loop to keep free
        return
    future = loop.run_in_executor(None, mem.reflect)
    tasks.add(future)
    future.add_done_callback(tasks.discard)
    future.add_done_callback(_log_failure)


def _log_failure(future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.error("reflect of an evicted instance failed", exc_info=future.exception())


def serve(app, host: str, port: int) -> None:
    """Run *app* under uvicorn, ending the live log streams as shutdown begins.

    uvicorn waits for open connections to close before it runs the lifespan
    shutdown, and an SSE subscriber to /llm-calls/stream never closes its
    own: one open visualizer held the server, and the drain of in-flight
    stores behind it, until the process was killed. The streams are ended
    first, and the wait for the rest is bounded. Ctrl-C and a failed startup
    end the process the way ``uvicorn.run`` ends it.
    """
    import uvicorn
    from uvicorn.main import STARTUP_FAILURE

    from smrti import call_log

    class _Server(uvicorn.Server):
        async def shutdown(self, sockets=None):
            call_log.close_streams()
            await super().shutdown(sockets=sockets)

    server = _Server(uvicorn.Config(
        app, host=host, port=port, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS
    ))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    if not server.started:
        sys.exit(STARTUP_FAILURE)
