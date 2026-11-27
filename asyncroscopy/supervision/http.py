"""Operator HTTP API, served next to ``/mcp`` by the MCP server.

This is the operator's channel: SciAgentGUI (or the built-in page at
``/supervision/ui``) uses it to watch the queue, pause, step, skip, take over,
approve, stop, browse the atlas and run recipes. The agent reaches the
instrument only through MCP tools, which have no way to approve, resume or
change policy. When ``supervision.operator_token`` is set, every request here
needs ``Authorization: Bearer <token>`` (or ``?token=`` for the page and images).

Endpoints (JSON unless noted)::

    GET  /supervision/state                       queue state + recent actions
    GET  /supervision/events?since=N              events after sequence number N
    POST /supervision/pause | resume | step | abort
    POST /supervision/take-over {note} | release
    POST /supervision/stop                        instrument stop, bypasses the queue
    POST /supervision/actions {tool, args}        queue a device tool as the operator
    POST /supervision/actions/<id>/skip | approve | deny
    GET  /supervision/atlas                       specimen, mount, mapping, sites
    POST /supervision/atlas/sites {name, notes, landmark, fov_m, imsize}
    GET  /supervision/atlas/sites/<id>            site with visits
    PATCH /supervision/atlas/sites/<id> {name, notes, landmark}
    POST /supervision/atlas/sites/<id>/return {params}   queue "return and acquire"
    GET  /supervision/atlas/sites/<id>/reference.png
    GET  /supervision/atlas/sites/<id>/visits/<vid>       visit detail
    GET  /supervision/atlas/sites/<id>/visits/<vid>.png
    POST /supervision/atlas/calibrate {fov_m, imsize}
    POST /supervision/atlas/remount {mount_id}
    POST /supervision/atlas/landmarks/reacquire {params}
    GET  /supervision/recipes                     list
    POST /supervision/recipes/from-actions {name, action_ids, description, params}
    GET | PUT | DELETE /supervision/recipes/<name>
    POST /supervision/recipes/<name>/check
    POST /supervision/recipes/<name>/run {params}
    GET  /supervision/capabilities
    GET  /supervision/ui                          HTML operator page
"""

from __future__ import annotations

import hmac
import json
import re
from pathlib import Path
from typing import Any

import anyio
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from asyncroscopy.atlas.store import to_png
from asyncroscopy.supervision.service import Supervisor

_UI = Path(__file__).with_name("operator_ui.html")
_CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
         "Access-Control-Allow-Headers": "Authorization, Content-Type"}


def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(json.loads(json.dumps(data, default=str)), status_code=status, headers=_CORS)


def _error(message: str, status: int) -> JSONResponse:
    return _json({"error": message}, status)


def _png(data: bytes) -> Response:
    return Response(data, media_type="image/png", headers={**_CORS, "Cache-Control": "no-store"})


class OperatorAPI:
    def __init__(self, supervisor: Supervisor, who: str = "operator"):
        self.sup = supervisor
        self.who = who
        q = supervisor.queue
        s = supervisor
        # (method, regex) -> handler(match, body) ; handlers are sync and run in a worker thread
        self.routes: list[tuple[str, re.Pattern, Any]] = [
            ("GET", r"state", lambda m, b, r: q.snapshot(int(r.query_params.get("limit", 200)))),
            ("GET", r"events", lambda m, b, r: {"events": q.events_since(int(r.query_params.get("since", 0))),
                                                "state": q.state()}),
            ("POST", r"pause", lambda m, b, r: (q.pause(b.get("note", "")), q.state())[1]),
            ("POST", r"resume", lambda m, b, r: (q.resume(), q.state())[1]),
            ("POST", r"step", lambda m, b, r: (q.step(), q.state())[1]),
            ("POST", r"abort", lambda m, b, r: {"aborting": q.abort(), "state": q.state()}),
            ("POST", r"take-over", lambda m, b, r: (q.take_over(self.who, b.get("note", "")), q.state())[1]),
            ("POST", r"release", lambda m, b, r: (q.release(self.who), q.state())[1]),
            ("POST", r"stop", lambda m, b, r: q.stop(self.who)),
            ("POST", r"actions", lambda m, b, r: s.submit_tool(b["tool"], b.get("args") or {}, "operator")),
            ("POST", r"actions/(?P<id>[^/]+)/skip", lambda m, b, r: {"ok": q.skip(m["id"], self.who)}),
            ("POST", r"actions/(?P<id>[^/]+)/approve", lambda m, b, r: {"ok": q.approve(m["id"], self.who)}),
            ("POST", r"actions/(?P<id>[^/]+)/deny", lambda m, b, r: {"ok": q.deny(m["id"], self.who)}),
            ("GET", r"actions/(?P<id>[^/]+)", lambda m, b, r: q.get(m["id"]).to_json()),
            ("GET", r"atlas", lambda m, b, r: s.atlas.summary()),
            ("POST", r"atlas/sites", lambda m, b, r: s.save_site(
                b.get("name") or "site", origin="operator", notes=b.get("notes", ""),
                landmark=bool(b.get("landmark")), fov_m=b.get("fov_m"), imsize=b.get("imsize"), wait=False)),
            ("GET", r"atlas/sites/(?P<id>[^/]+)/reference\.png", lambda m, b, r: _png(to_png(s.atlas.reference(m["id"])))),
            ("GET", r"atlas/sites/(?P<id>[^/]+)/visits/(?P<vid>[^/]+)\.png",
             lambda m, b, r: _png(to_png(s.atlas.visit_image(m["id"], m["vid"])))),
            ("GET", r"atlas/sites/(?P<id>[^/]+)/visits/(?P<vid>[^/]+)", lambda m, b, r: s.atlas.visit(m["id"], m["vid"])),
            ("POST", r"atlas/sites/(?P<id>[^/]+)/return", lambda m, b, r: s.return_to_site(
                m["id"], origin="operator", params=b.get("params"), wait=False)),
            ("GET", r"atlas/sites/(?P<id>[^/]+)", lambda m, b, r: s.atlas.get(m["id"]).to_json()),
            ("PATCH", r"atlas/sites/(?P<id>[^/]+)", lambda m, b, r: s.atlas.update_site(m["id"], **b).to_json()),
            ("POST", r"atlas/calibrate", lambda m, b, r: s.calibrate_mapping(
                origin="operator", fov_m=b.get("fov_m"), imsize=b.get("imsize"), wait=False)),
            ("POST", r"atlas/remount", lambda m, b, r: {"mount_id": s.new_mount(b.get("mount_id"))}),
            ("POST", r"atlas/landmarks/reacquire", lambda m, b, r: s.reacquire_landmarks(
                origin="operator", params=b.get("params"), wait=False)),
            ("GET", r"recipes", lambda m, b, r: {"recipes": s.recipes.list()}),
            ("POST", r"recipes/from-actions", lambda m, b, r: s.recipe_from_actions(
                b["name"], list(b["action_ids"]), b.get("description", ""), b.get("params"))),
            ("POST", r"recipes/(?P<name>[^/]+)/check", lambda m, b, r: s.check_recipe(m["name"])),
            ("POST", r"recipes/(?P<name>[^/]+)/run", lambda m, b, r: s.run_recipe(
                m["name"], b.get("params"), origin="operator", wait=False)),
            ("GET", r"recipes/(?P<name>[^/]+)", lambda m, b, r: s.recipes.load(m["name"]).to_json()),
            ("PUT", r"recipes/(?P<name>[^/]+)", lambda m, b, r: s.recipes.update(m["name"], b).to_json()),
            ("DELETE", r"recipes/(?P<name>[^/]+)", lambda m, b, r: {"deleted": s.recipes.delete(m["name"])}),
            ("GET", r"capabilities", lambda m, b, r: {**s.capabilities().to_json(), "policy": s.policy.to_json()}),
        ]
        self.routes = [(meth, re.compile(pat + r"/?$"), fn) for meth, pat, fn in self.routes]

    def _authorized(self, request: Request) -> bool:
        token = self.sup.config.operator_token
        if not token:
            return True
        header = request.headers.get("authorization", "")
        given = header[7:] if header.lower().startswith("bearer ") else request.query_params.get("token", "")
        return hmac.compare_digest(given, token)

    async def handle(self, request: Request) -> Response:
        if request.method == "OPTIONS":
            return Response(status_code=204, headers=_CORS)
        path = request.path_params.get("path", "").strip("/")
        if path in ("", "ui") and request.method == "GET":
            return HTMLResponse(_UI.read_text(encoding="utf-8"), headers=_CORS)
        if not self._authorized(request):
            return _error("operator token required", 401)
        body: dict = {}
        if request.method in ("POST", "PUT", "PATCH"):
            raw = await request.body()
            if raw:
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    return _error("request body is not JSON", 400)
                if not isinstance(body, dict):
                    return _error("request body must be a JSON object", 400)
        for method, pattern, fn in self.routes:
            if method != request.method:
                continue
            match = pattern.match(path)
            if match is None:
                continue
            try:
                result = await anyio.to_thread.run_sync(fn, match.groupdict(), body, request)
            except KeyError as exc:
                return _error(str(exc.args[0]) if exc.args else "not found", 404)
            except (ValueError, TypeError, FileExistsError) as exc:
                return _error(str(exc), 400)
            except Exception as exc:
                return _error(f"{type(exc).__name__}: {exc}", 409)
            return result if isinstance(result, Response) else _json(result)
        return _error(f"no route {request.method} /supervision/{path}", 404)


def register_operator_api(mcp, supervisor: Supervisor) -> OperatorAPI:
    api = OperatorAPI(supervisor)
    mcp.custom_route("/supervision/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])(
        api.handle)
    mcp.custom_route("/supervision", methods=["GET"])(api.handle)
    return api
