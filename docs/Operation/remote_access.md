# Remote access: tokens, TLS and the endpoint proxy

A facility can put every instrument behind the same chain:

```
user / agent ──> Token Minting / Verification Service ──> Endpoint API Proxy
             ──> Endpoint Proxy (TLS encryption and token auth.) ──> PyTango / Tiled / MCP servers
```

This page covers the instrument end of that chain: what this repository provides, what the facility provides, and what an instrument owner configures. Nothing here is needed on a digital twin used on one machine (the quickstart and the hackathon setup are unchanged).

## Who owns what

| Part | Owner | Here |
| :-- | :-- | :-- |
| Token minting and verification service (identity, scopes, JWKS) | The facility | Not here. `startup_scripts/dev_tokens.py` is a **development stand-in** for testing only. |
| Endpoint API proxy, VPN, the proxy blueprint | The facility | Not here. |
| Token check at the MCP server (signature, expiry, issuer, audience; roles from scopes) | this repo | `mcp.auth` in the MCP config; `asyncroscopy/mcp/endpoint_auth.py`. |
| Risk policy, Instrument Trace | this repo | Unchanged; the token's subject is now the caller they record. |

The MCP server checks the token itself even behind the proxy, so a request that reaches port 8000 some other way (another process on the host, a misconfigured proxy) is still refused.

## The MCP server's `auth` section

```yaml
mcp:
  # ...
  auth:
    jwks_url: https://tokens.example.org/.well-known/jwks.json   # The facility's token service
    # public_key_file: outputs/dev_tokens/public.pem                 # or one PEM key (relative: the project directory)
    issuer: https://tokens.example.org                             # must match the token's iss
    audience: asyncroscopy:spectra300                                 # this instrument; must match aud
    # algorithm: RS256                                                # asymmetric only (RS*, ES*, PS*, EdDSA)
    # role_claims: [scope, scp]                                       # claims whose values are matched below
    # roles:                                                          # role -> scopes that grant it (these are the defaults)
    #   read:    [instrument:read, instrument:operate, instrument:approve]
    #   operate: [instrument:operate]
    #   approve: [instrument:approve]
    #   preapprove: [instrument:preapprove]                           # a workflow server's ASYNCROSCOPY_OPERATOR_TOKEN, and nothing else
```

Exactly one of `jwks_url` and `public_key_file`; `issuer` and `audience` are required, so a token minted for another instrument or service does not open this one. `run_mcp.py` refuses a config with a missing or unknown key.

With `auth` set:

- **Every request needs `Authorization: Bearer <token>`.** A missing, expired, wrongly signed, wrong-issuer or wrong-audience token is answered with HTTP 401. A token must carry `exp` and `sub`.
- **The token's `sub` is the caller.** It is recorded in the Instrument Trace. `X-Asyncroscopy-Caller` is ignored (agents still send it; it does no harm).
- **Roles gate tools, from the existing risk policy** (`asyncroscopy/safety/policy.py`), not from a second list:

  | Role | Tools |
  | :-- | :-- |
  | `read` | low-risk tools: `get_*`, `list_*`, `job_status`, `view_data`, `endpoint_info`, safe stops ... |
  | `operate` | high-risk tools: `move_stage`, `acquire_*`, `<job>_start`, `run_job` ... |
  | `approve` | operator-only commands, offered only on the operator endpoint |

  A risk override in `mcp.safety.risk` moves a tool between `read` and `operate` with it. A token sees only the tools its roles allow in `tools/list`.

Without `auth`, nothing changes: no token is asked for, and the caller is the self-reported `X-Asyncroscopy-Caller`.

**Refusal to start.** A stack that is not simulated (its devices' `simulated` property is false, or cannot be read) whose MCP endpoint would bind beyond 127.0.0.1 (`http_host: 0.0.0.0` with `allow_remote_bind: true`, or a network address) must have `auth`: `run_mcp.py` refuses to start it, and so does the server when started directly. Twins may still bind every interface without tokens. The recommended hardware setup keeps `http_host: 127.0.0.1` and lets the proxy be the only thing on the network.

## Host and Origin checks

Every MCP HTTP endpoint (`run_mcp.py`'s agent, operator and monitor endpoints; `run_llm.py`'s agent server and
workflow operator endpoint) answers only requests whose `Host` header, and `Origin` header when there is one, names
the bind host, `localhost`, `127.0.0.1` or `::1`; with `http_host: 0.0.0.0`, also this computer's own name and
addresses. Anything else gets HTTP 403 before the MCP handler sees it, so a web page whose domain is made to resolve to
127.0.0.1 (DNS rebinding) cannot drive the instrument from a browser on this machine (`asyncroscopy/mcp/host_guard.py`).

A proxy, or a machine that reaches the endpoint by another name, needs that name listed:

```yaml
mcp:            # run_mcp.py; for run_llm.py the same key goes under agent:
  allowed_hosts: [instrument.lab.example]
```

The operator endpoint that `mcp.operator_port` starts takes none of these: it answers this computer only.

## What an instrument owner configures

1. Ask the facility for the token service's JWKS URL and issuer, and an audience name for this instrument; agree the scopes (`instrument:read`, `instrument:operate`, `instrument:approve` unless the facility names them otherwise; map them with `mcp.auth.roles`).
2. Add the `auth` section to the agent endpoint's MCP config (and to the operator endpoint's, if it is ever reached from another machine).
3. Keep the MCP servers and Tiled on 127.0.0.1. Put a proxy in front that terminates TLS, forwards `/mcp` to 8000 and `/analysis/mcp` to 8003, and passes the bearer token through.
4. Leave the operator endpoint (8001) unforwarded unless the facility decides operators may use it remotely; if so, give it `auth` too (operator-only commands then need `instrument:approve`).

The analysis server (`pytemlib_mcp.analysis_server`, port 8003) does not verify tokens yet; keep it on 127.0.0.1 behind the proxy.

## Clients

- **asyncroscopy agents** read `ASYNCROSCOPY_MCP_TOKEN` and send it as `Authorization: Bearer` to the instrument and analysis endpoints (`AgentConfig.token`, `asyncroscopy/agents/config.py`; any `McpServer` without its own Authorization header picks it up too). A token expires: get a new one from the token service before a long run.
- **SciAgentGUI** (a separate repository) already stores an MCP bearer token per server in the OS keychain; enter the bearer token there for a server that has `mcp.auth`.
- **Scripts** with the FastMCP client: `StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"})`.

## Testing with development tokens

`startup_scripts/dev_tokens.py` is a stand-in for the facility's token service, **for development and tests only, never for production**. It writes an RSA key pair under `outputs/dev_tokens/` (`outputs/` is gitignored) and mints short-lived RS256 tokens (15 minutes by default, at most a day) marked `"dev_only": true`.

```bash
uv run startup_scripts/dev_tokens.py init        # prints the auth section to paste into the MCP config
uv run startup_scripts/dev_tokens.py mint --sub alice --scope instrument:read --scope instrument:operate
```

Then, in the shell that runs the agent:

```powershell
$env:ASYNCROSCOPY_MCP_TOKEN = "<the token>"
```

A token with only `instrument:read` lists `get_stage` but not `move_stage`; `instrument:operate` can call `move_stage`. `tests/endpoints/test_endpoint_auth.py` runs this whole flow in-process.

## The Tiled API key is separate

Tiled's API key (`ASYNCROSCOPY_TILED_API_KEY`, `docs/Tiled_server/data_integration.md`) is unchanged. It is a **service-to-service** secret: the DATA device, the MCP server and SciAgentGUI use it to reach Tiled on the instrument host. It is not a user identity and does not replace a facility token. Users and agents reach data through the MCP endpoint's `list_data` and `view_data`, which the token gates (`read`). Forwarding Tiled through the proxy (port 9443) gives anyone holding the Tiled key direct access, so forward it only to networks that should hold that key, or not at all. Having Tiled accept bearer tokens itself (Tiled supports external authenticators) is the facility's decision and is not done here.

## Not done

- No token has come from the facility's real service yet; the JWKS path is exercised only by FastMCP's verifier, and the tests use a local public key file.
- The analysis server and Tiled do not verify bearer tokens.
- Token refresh is the client's job; agents do not refresh tokens.
