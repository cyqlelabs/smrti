"""HTTP plumbing shared by the REST and proxy servers."""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from smrti.servers import config as cfg

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
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


def _error(status: int, message: str, kind: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": code}},
    )


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
    unchanged (local use). Otherwise the key comes from ``X-Api-Key`` or
    ``Authorization: Bearer`` — so proxy clients can keep their upstream
    credentials in Authorization and supply the smrti key via X-Api-Key —
    and one this server does not know is a 401.

    A key from SMRTI_API_KEYS may act only on its own tenants. The check is
    made here, once, for every route: a request naming another tenant, an
    alternate ``?db=``, or a clear of a shared log is a 403 before any
    handler runs. The tenant it acts on is resolved here too, onto
    ``request.state.tenant`` — the one it named, else its key's first — so
    no handler applies a default of its own to a bound key.
    """
    if not (cfg.API_KEY or cfg.API_KEYS):
        return await call_next(request)
    for supplied in (request.headers.get("x-api-key") or "", _bearer(request)):
        tenants = _tenants_of(supplied)
        if tenants is not _NO_KEY:
            break
    else:
        return _error(401, "Invalid or missing API key", "authentication_error", "invalid_api_key")
    if tenants is not None:
        refusal = _refusal(request, tenants)
        if refusal:
            return _error(403, refusal, "permission_error", "tenant_not_allowed")
        named = _named_tenants(request)
        request.state.tenant = named[0] if named else tenants[0]
    request.state.tenants = tenants
    return await call_next(request)


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
