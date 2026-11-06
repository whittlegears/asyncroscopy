"""Bearer-token verification for the MCP endpoints (the instrument side of ATHENA WG1's endpoint proxy).

WG1's design puts a token service (minting and verification) in front of every instrument: a
user reaches an instrument through the token service, an endpoint API proxy, and an endpoint
proxy that terminates TLS and checks the token, then the PyTango / Tiled / MCP servers. This
module is the check at the MCP server itself. It is optional: without an ``mcp.auth`` section
nothing here runs and the endpoint behaves as before (caller from ``X-Asyncroscopy-Caller``).

With ``mcp.auth`` set:

* every request needs a valid bearer JWT: signature (a public key file or the token service's
  JWKS), expiry, issuer and audience are checked, and ``exp`` and ``sub`` must be present;
* the token's ``sub`` is the caller recorded in the trace and in approval requests, and the
  ``who`` of an approve, deny or set_mode; ``X-Asyncroscopy-Caller`` is ignored;
* the token's scopes give it roles, and each tool needs one, taken from the tool's descriptor
  (``asyncroscopy.safety.decide.required_role``), not from a second list:

  ========= ==================================================================
  role      tools
  ========= ==================================================================
  read      low-risk tools (get_*, list_*, job_status, view_data, ...)
  operate   high-risk tools (move_stage, acquire_*, <job>_start, run_job, ...);
            the approval gate still applies to them
  approve   operator-only commands (Approvals approve, deny, set_mode, ...),
            which only the operator endpoint offers
  ========= ==================================================================

  A token sees only the tools its roles allow (tools/list is filtered too).

The config is validated here without importing FastMCP, so the launcher can check it cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from asyncroscopy.core.net import is_loopback_host

if TYPE_CHECKING:
    from fastmcp.server.auth import AccessToken

    from asyncroscopy.mcp.mcp_server import MCPServer

ROLES = ("read", "operate", "approve")
# Scopes a WG1 token carries, and the roles each grants. A deployment can map its own scope names
# (mcp.auth.roles); a role is granted when the token has any scope listed for it.
DEFAULT_ROLE_SCOPES: dict[str, list[str]] = {
    "read": ["instrument:read", "instrument:operate", "instrument:approve"],
    "operate": ["instrument:operate"],
    "approve": ["instrument:approve"],
}
# Approvals commands whose "who" names the person deciding; with auth it is the token's subject.
DECIDING_COMMANDS = frozenset({"approve", "deny", "set_mode", "set_auto_allowed"})


@dataclass(frozen=True)
class AuthSettings:
    """The ``mcp.auth`` section, checked."""

    issuer: str
    audience: str
    jwks_url: str = ""
    public_key_file: str = ""
    algorithm: str = "RS256"
    # Claims whose values are matched against the role scopes ("scope" is space-separated, "scp" or a
    # roles claim may be a list).
    role_claims: tuple[str, ...] = ("scope", "scp")
    roles: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ROLE_SCOPES.items()})

    @classmethod
    def from_config(cls, raw: dict | None, base_dir: Path | None = None) -> AuthSettings | None:
        """``None`` for no section (or ``{}``); a ValueError naming the problem for a bad one."""
        if not raw:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"mcp.auth must be a mapping, not {raw!r}")
        known = {"jwks_url", "public_key_file", "issuer", "audience", "algorithm", "role_claims", "roles"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"mcp.auth has unknown keys {sorted(unknown)}; allowed: {sorted(known)}")
        jwks_url = str(raw.get("jwks_url") or "")
        key_file = str(raw.get("public_key_file") or "")
        if bool(jwks_url) == bool(key_file):
            raise ValueError("mcp.auth needs exactly one of jwks_url (the token service's JWKS) or public_key_file (a PEM public key)")
        if key_file:
            path = Path(key_file)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            if not path.is_file():
                raise ValueError(f"mcp.auth.public_key_file {path} does not exist")
            key_file = str(path)
        for name in ("issuer", "audience"):
            if not str(raw.get(name) or "").strip():
                raise ValueError(f"mcp.auth.{name} is required: a token minted for another service or instrument must not open this one")
        algorithm = str(raw.get("algorithm") or "RS256")
        if algorithm.upper().startswith("HS"):
            raise ValueError("mcp.auth.algorithm must be asymmetric (RS*, ES*, PS*, EdDSA): the instrument holds only a public key")
        claims = raw.get("role_claims") or ["scope", "scp"]
        claims = [claims] if isinstance(claims, str) else list(claims)
        roles = {k: list(v) for k, v in DEFAULT_ROLE_SCOPES.items()}
        for role, scopes in (raw.get("roles") or {}).items():
            if role not in ROLES:
                raise ValueError(f"mcp.auth.roles has unknown role {role!r}; roles are {list(ROLES)}")
            scopes = [scopes] if isinstance(scopes, str) else list(scopes or [])
            if not all(isinstance(s, str) and s for s in scopes):
                raise ValueError(f"mcp.auth.roles.{role} must be a list of scope names")
            roles[role] = scopes
        return cls(
            issuer=str(raw["issuer"]).strip(),
            audience=str(raw["audience"]).strip(),
            jwks_url=jwks_url,
            public_key_file=key_file,
            algorithm=algorithm,
            role_claims=tuple(str(c) for c in claims),
            roles=roles,
        )

    def to_json(self) -> str:
        """For the server's --auth-json (the launcher has already resolved the key path)."""
        return json.dumps({
            "issuer": self.issuer, "audience": self.audience, "jwks_url": self.jwks_url, "public_key_file": self.public_key_file,
            "algorithm": self.algorithm, "role_claims": list(self.role_claims), "roles": self.roles,
        })

    def roles_of(self, claims: dict[str, Any]) -> set[str]:
        """Roles a token's claims grant."""
        values: set[str] = set()
        for claim in self.role_claims:
            value = claims.get(claim)
            if isinstance(value, str):
                values.update(value.split())
            elif isinstance(value, (list, tuple)):
                values.update(str(v) for v in value)
        return {role for role, scopes in self.roles.items() if values & set(scopes)}


def stamp_who(dev_class: str, command_name: str, call_args: tuple, subject: str | None) -> tuple:
    """An approve, deny, set_mode or set_auto_allowed of an Approvals device records the token's subject as ``who``."""
    if dev_class.lower() != "approvals" or command_name not in DECIDING_COMMANDS or not call_args or not subject:
        return call_args
    try:
        body = json.loads(call_args[0])
    except (TypeError, ValueError):
        return call_args
    if not isinstance(body, dict):
        return call_args
    body["who"] = subject
    return (json.dumps(body, sort_keys=True, separators=(",", ":")), *call_args[1:])


def is_loopback(host: str) -> bool:
    return is_loopback_host(host)  # one rule for every bind check (asyncroscopy.core.net)

def remote_without_auth(bound_host: str, simulated: bool, auth: AuthSettings | dict | None) -> str | None:
    """Why a server must not start: real hardware (simulated: false, or not known) reachable from other
    machines with no token check. None when it may start."""
    if auth or simulated or is_loopback(bound_host):
        return None
    return (
        f"this stack is not simulated and the MCP endpoint would bind {bound_host!r}, reachable from other machines, "
        "with no token check. Set mcp.auth (the token service's jwks_url or a public_key_file, issuer, audience; "
        "docs/Operation/remote_access.md), or keep http_host at 127.0.0.1 and reach it through the endpoint proxy."
    )


# ---------------------------------------------------------------------- FastMCP wiring
def _verifier(settings: AuthSettings):
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    class _StrictJWTVerifier(JWTVerifier):
        """FastMCP's JWT check, plus: a token must expire and must name its subject."""

        async def load_access_token(self, token: str):
            access = await super().load_access_token(token)
            if access is None or access.claims.get("exp") is None or not access.claims.get("sub"):
                return None
            return access

    key = Path(settings.public_key_file).read_text(encoding="utf-8") if settings.public_key_file else None
    return _StrictJWTVerifier(public_key=key, jwks_uri=settings.jwks_url or None, issuer=settings.issuer,
                              audience=settings.audience, algorithm=settings.algorithm)


def current_token() -> AccessToken | None:
    """The verified access token of the current request, or None (no token, or not over HTTP)."""
    try:
        from fastmcp.server.dependencies import get_access_token

        return get_access_token()
    except Exception:
        return None


def token_subject() -> str | None:
    token = current_token()
    return (token.claims.get("sub") or token.subject) if token is not None else None


def attach(server: MCPServer, raw: AuthSettings | dict | None) -> AuthSettings | None:
    """Put token verification and the role gate on ``server.mcp``; returns the settings, or None (no auth)."""
    settings = raw if isinstance(raw, AuthSettings) else AuthSettings.from_config(raw)
    if settings is None:
        return None
    from fastmcp.exceptions import ToolError
    from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

    from asyncroscopy.safety.decide import required_role

    class RoleGate(Middleware):
        """Refuses a tool call whose token lacks the tool's role, and lists only the tools a token may call.

        HTTP requests without a valid token never get here (FastMCP answers 401); a call with no token
        at all (an in-memory client) is refused here, so the gate fails closed on every transport.
        """

        def _roles(self) -> tuple[Any, set[str]]:
            token = current_token()
            return token, (settings.roles_of(token.claims) if token is not None else set())

        def _needs(self, name: str) -> str:
            return required_role(server.descriptors.get(name))

        async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
            token, roles = self._roles()
            name = context.message.name
            if token is None:
                raise ToolError(f"{name} refused: this endpoint needs a valid bearer token (mcp.auth)")
            names = [name]
            if name == "call_tool":  # call_tool runs another tool: the token needs that tool's role too
                names.append(str((context.message.arguments or {}).get("name", "")))
            for checked in names:
                need = self._needs(checked)
                if need not in roles:
                    scopes = ", ".join(settings.roles[need]) or "(none configured)"
                    raise ToolError(f"{checked} refused: it needs the {need} role (scope {scopes}); the token of {token_subject()!r} grants {sorted(roles) or 'no role'}")
            return await call_next(context)

        async def on_list_tools(self, context: MiddlewareContext, call_next: CallNext) -> list:
            tools = await call_next(context)
            _token, roles = self._roles()
            return [t for t in tools if self._needs(t.name) in roles]

    def may_call(name: str) -> bool:
        token = current_token()
        return token is not None and required_role(server.descriptors.get(name)) in settings.roles_of(token.claims)

    server.mcp.auth = _verifier(settings)
    server.mcp.add_middleware(RoleGate())
    server.may_call = may_call
    return settings
