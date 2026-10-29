#!/usr/bin/env python
"""DEVELOPMENT ONLY: a stand-in for ATHENA WG1's token service, to test mcp.auth on this machine.

NOT FOR PRODUCTION. In a deployment, tokens come from WG1's token minting / verification
service, and the MCP endpoint checks them against that service's JWKS (mcp.auth.jwks_url). This
script makes a local RSA key pair under outputs/dev_tokens/ (gitignored) and mints short-lived
RS256 tokens with the scopes you choose, so the token path can be exercised against a twin.

    # once: a key pair, and the mcp.auth section that trusts it
    uv run startup_scripts/dev_tokens.py init

    # a token for an agent that may read and operate (default 15 minutes)
    uv run startup_scripts/dev_tokens.py mint --sub alice --scope instrument:read --scope instrument:operate

    # give it to the agents
    set ASYNCROSCOPY_MCP_TOKEN=<the token>        (PowerShell: $env:ASYNCROSCOPY_MCP_TOKEN = "<the token>")

Scopes (mcp.auth.roles maps them to roles): instrument:read (low-risk tools),
instrument:operate (high-risk tools, still through the approval gate), instrument:approve
(approve, deny and set_mode on the operator endpoint).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_DIR / "outputs" / "dev_tokens"
DEV_ISSUER = "asyncroscopy-dev-tokens (NOT WG1)"
DEV_AUDIENCE = "asyncroscopy-mcp"
MAX_TTL_S = 24 * 3600  # dev tokens are short-lived; nothing here should outlive a day
BANNER = "[dev_tokens] DEVELOPMENT ONLY: a stand-in for WG1's token service. Never use these keys or tokens in production."


def init(key_dir: Path, force: bool = False) -> dict:
    """Write private.pem and public.pem under key_dir; returns the mcp.auth section that trusts them."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_path, public_path = key_dir / "private.pem", key_dir / "public.pem"
    if private_path.exists() and not force:
        raise FileExistsError(f"{private_path} exists; pass --force to make a new pair (tokens minted with the old one stop working)")
    key_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    public_path.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    try:
        shown = public_path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        shown = str(public_path)
    return {"public_key_file": shown, "issuer": DEV_ISSUER, "audience": DEV_AUDIENCE}


def mint(key_dir: Path, subject: str, scopes: list[str], ttl_s: int = 900, issuer: str = DEV_ISSUER,
         audience: str = DEV_AUDIENCE, extra: dict | None = None) -> str:
    """An RS256 JWT signed with key_dir/private.pem. ``ttl_s`` may be negative (an expired token, for tests)."""
    from joserfc import jwk, jwt  # FastMCP's own JWT library, so nothing new to install

    if ttl_s > MAX_TTL_S:
        raise ValueError(f"dev tokens live at most {MAX_TTL_S} s")
    now = int(time.time())
    claims = {"iss": issuer, "aud": audience, "sub": subject, "iat": now, "nbf": min(now, now + ttl_s),
              "exp": now + ttl_s, "jti": uuid.uuid4().hex, "scope": " ".join(scopes), "dev_only": True, **(extra or {})}
    private = jwk.import_key((key_dir / "private.pem").read_bytes(), "RSA")
    return jwt.encode({"alg": "RS256", "typ": "JWT"}, claims, private)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Where the key pair lives (default outputs/dev_tokens, gitignored).")
    sub = parser.add_subparsers(dest="action", required=True)
    p_init = sub.add_parser("init", help="Make a key pair and print the mcp.auth section that trusts it.")
    p_init.add_argument("--force", action="store_true")
    p_mint = sub.add_parser("mint", help="Print a short-lived token.")
    p_mint.add_argument("--sub", required=True, help="The token's subject: the caller the trace and approvals record.")
    p_mint.add_argument("--scope", action="append", default=[], help="A scope, e.g. instrument:read (repeat for more).")
    p_mint.add_argument("--ttl", type=int, default=900, help="Seconds until it expires (default 900, at most a day).")
    p_mint.add_argument("--audience", default=DEV_AUDIENCE)
    p_mint.add_argument("--issuer", default=DEV_ISSUER)
    args = parser.parse_args(argv)

    print(BANNER, file=sys.stderr)
    if args.action == "init":
        section = init(args.dir, force=args.force)
        print("Add this to the MCP config (the key file path is read from the project directory):\n")
        print("  auth:")
        for key, value in section.items():
            print(f"    {key}: {json.dumps(value)}")
        return 0
    if not args.scope:
        print("[dev_tokens] no --scope given: the token would grant no role", file=sys.stderr)
    print(mint(args.dir, args.sub, args.scope, args.ttl, issuer=args.issuer, audience=args.audience))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
