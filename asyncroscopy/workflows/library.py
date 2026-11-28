"""The workflow library holds proposed workflows and the approvals that let them run on an instrument.

A workflow has one of three statuses.

* **shipped** is a module in ``agent/workflows/``. It is reviewed in the repository like any other code.
* **approved** was proposed and then approved by a person. The approval names the workflow by its hash, which
  covers its code and the code of every block it uses. If either changes it is a draft again.
* **draft** was proposed and is not approved, or changed after it was approved. It is kept with the words of the
  person who asked for it in ``requests.json``, and with what the conversation established when the proposing app
  sent it (``context``).

A shipped or approved workflow may run on a real instrument when the operator allows hardware. A draft runs on a
digital twin only. Only a person can approve or reject, on the command line with
``python -m asyncroscopy.workflows approve NAME``, or through the workflow operator endpoint
(``asyncroscopy.mcp.agent_mcp_server --endpoint-role operator``; ``docs/MCP/workflow_approval.md``), which an app
such as the GUI calls for a person and which no agent is given. No tool a model can call approves anything.

An approval can carry these.

* **Rehearsals.** Every run of a draft on a digital twin is saved in ``rehearsals.json`` with the ``stack_id`` and
  ``stack_fingerprint`` of the stack it ran on, every distinct call it made (``calls_made``) and what the twin
  reported each instrument tool to be (``actions``), and its ``results``:
  the data it acquired and derived, its time and dose, and its twin score (``asyncroscopy.sim.truth_score``). The person approving sees the runs of this exact code. The command
  line requires none; the operator endpoint requires one that passed on the stack it is connected to
  (``require_rehearsal``): it ended in ``done`` and every cleanup call succeeded, on the same fingerprint.
* **An instrument.** ``instrument`` limits the approval to one stack, named by the endpoint's ``stack_id``. Left
  out, it holds on every instrument.
* **An end date.** ``days`` makes the approval expire.
* **Approvers.** If the library has ``approvers.json``, a list of login names, only those logins may approve.
  Without it anyone who can run the command on this host may. The record keeps the login and the name given.

A person can also **reject** a draft with a reason. The rejection is kept in ``rejections.json`` with the hash it
was given for, the draft stays a draft (it still runs on a twin), and an approval later replaces it. Rejecting an
approved workflow revokes its approval first.

The source of every approval is also kept in ``last_approved.json``, which revoking and rejecting leave in place,
so a person reviewing a changed workflow sees what changed since it was last approved.

The files under ``root`` are ``drafts/NAME.py``, ``approved/NAME.py``, ``approvals.json``, ``rehearsals.json``,
``requests.json``, ``rejections.json``, ``last_approved.json`` and optionally ``approvers.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .authoring import NotAWorkflow, load_source
from .graph import Workflow, all_workflows

DEFAULT_LIBRARY = Path("outputs/workflows/library")
SHIPPED, APPROVED, DRAFT = "shipped", "approved", "draft"
ANY_INSTRUMENT = "any"
KEPT_REHEARSALS = 10  # per workflow, newest last


def rehearsal_passed(entry: dict) -> bool:
    return entry.get("state") == "done" and not entry.get("cleanup_failed")


def fingerprint_of(entry: dict) -> str | None:
    return (entry.get("stack_fingerprint") or {}).get("digest")


def simulated(fingerprint: dict | None) -> bool:
    return bool(((fingerprint or {}).get("simulated") or {}).get("value"))


def action_of(tool: str, info: dict | None) -> dict | None:
    """What an endpoint reports a tool to be, the same on a twin and on an instrument whatever the tool is called: its
    role and command, and the kinds of its device (``endpoint_info``'s ``tools`` and ``device_kinds``). A native tool
    is its name. None when the endpoint does not describe it."""
    entry = ((info or {}).get("tools") or {}).get(tool)
    if not entry:
        return None
    device = entry.get("device")
    return {"role": entry.get("role"), "command": entry.get("command") or tool,
            "kinds": sorted(((info or {}).get("device_kinds") or {}).get(device) or []) if device else None}


def matched_calls(rehearsal: dict, info: dict | None) -> list[dict]:
    """A rehearsal's ``calls_made``, each with the one instrument tool on ``info`` whose ``action_of`` is the twin
    tool's (``instrument_tool``), or why there is none (``held_because``)."""
    actions = rehearsal.get("actions") or {}
    out = []
    for call in rehearsal.get("calls_made") or []:
        if call.get("server") != "instrument":
            out.append(dict(call))
            continue
        wanted = actions.get(call["tool"])
        found = [name for name in sorted((info or {}).get("tools") or {}) if wanted is not None and action_of(name, info) == wanted]
        if info is None:
            why = "no match: will be held (the instrument endpoint could not be reached)"
        elif wanted is None:
            why = "no match: will be held (the rehearsal kept no description of this tool from the twin's endpoint_info)"
        elif not found:
            why = (f"no match: will be held (the instrument offers no tool with role {wanted['role']} and command {wanted['command']} "
                   f"on a device of kinds {wanted['kinds']})")
        elif len(found) > 1:
            why = f"no match: will be held ({len(found)} instrument tools match: {', '.join(found)})"
        else:
            why = None
        out.append({**call, "instrument_tool": found[0] if why is None else None, "held_because": why})
    return out


def needs_approval(name: str) -> str:
    return (f"{name} is a draft: nobody has approved it, so it runs on a digital twin only. A person approves it once with "
            f"`python -m asyncroscopy.workflows approve {name}`, after reading it (`python -m asyncroscopy.workflows view` "
            "shows it with every other workflow).")


class Library:
    def __init__(self, root: str | Path = DEFAULT_LIBRARY) -> None:
        self.root = Path(root)

    def _file(self, kind: str, name: str) -> Path:
        return self.root / ("approved" if kind == APPROVED else "drafts") / f"{name}.py"

    def _approvals(self) -> dict:
        path = self.root / "approvals.json"
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return found if isinstance(found, dict) else {}

    def _write_approvals(self, approvals: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "approvals.json").write_text(json.dumps(approvals, indent=1, sort_keys=True), encoding="utf-8")

    def _read(self, file: str, kind: type):
        try:
            found = json.loads((self.root / file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return found if isinstance(found, kind) else None

    def record_rehearsal(self, workflow: Workflow, record: dict, info: dict, results: dict | None = None) -> None:
        """Save a finished twin run of a proposed workflow as a rehearsal, under the stack ``info`` describes."""
        if workflow.name in all_workflows() or record.get("state") not in ("done", "failed", "stopped"):
            return
        entry = {"hash": record.get("hash"), "run_id": record.get("run_id"), "state": record.get("state"), "at": record.get("ended") or time.time(),
                 "calls": record.get("calls"), "inputs": record.get("inputs") or {}, "error": record.get("error"),
                 "problems": list(record.get("problems") or []),
                 "cleanup_failed": sorted(str(c.get("tool")) for c in record.get("cleanup") or [] if isinstance(c, dict) and not c.get("ok")),
                 "stack_id": info.get("stack_id"), "stack_fingerprint": info.get("stack_fingerprint"),
                 "calls_made": list(record.get("calls_made") or []), "results": dict(results if results is not None else record.get("results") or {}),
                 "actions": {c["tool"]: action_of(c["tool"], info) for c in record.get("calls_made") or [] if c.get("server") == "instrument"}}
        every = self._read("rehearsals.json", dict) or {}
        every[workflow.name] = [*(every.get(workflow.name) or []), entry][-KEPT_REHEARSALS:]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rehearsals.json").write_text(json.dumps(every, indent=1, sort_keys=True, default=str), encoding="utf-8")

    def rehearsals(self, name: str, hash_: str | None = None) -> list[dict]:
        """The saved twin runs of ``name``, oldest first. With ``hash_`` only the runs of that exact code."""
        runs = (self._read("rehearsals.json", dict) or {}).get(str(name)) or []
        return [r for r in runs if hash_ is None or r.get("hash") == hash_]

    def rehearsal_problem(self, name: str, hash_: str, stack_fingerprint: dict | None) -> tuple[list[dict], str | None]:
        """The rehearsals of this exact code that passed on the stack with ``stack_fingerprint``, and why there is
        none, or None. A rehearsal passed when it reached ``done`` (no block failed) and every cleanup call of its
        blocks succeeded."""
        runs = self.rehearsals(name, hash_)
        passed = [r for r in runs if rehearsal_passed(r)]
        wanted = (stack_fingerprint or {}).get("digest")
        if not passed:
            return [], (f"this exact code has never run to the end on a digital twin with every cleanup call succeeding "
                        f"({len(runs)} rehearsal(s) of it, none in 'done' with its cleanup done)")
        if not wanted:
            return [], "the instrument endpoint reports no stack_fingerprint, so no rehearsal can be matched to the stack it is approved on"
        if simulated(stack_fingerprint):
            here = [r for r in passed if fingerprint_of(r) == wanted]
            mismatch = f"this stack's fingerprint is {wanted}. Rehearse it on this stack"
        else:
            software = stack_fingerprint.get("software")
            here = [r for r in passed if simulated(r.get("stack_fingerprint")) and (r.get("stack_fingerprint") or {}).get("software") == software]
            mismatch = (f"this instrument is not simulated, so a rehearsal counts when it ran on a digital twin with the same "
                        f"software ({software}). Rehearse it on a twin running this software")
        if here:
            return here, None
        elsewhere = sorted({f"{r.get('stack_id') or 'an unnamed stack'} ({fingerprint_of(r) or 'no fingerprint'})" for r in passed})
        return [], f"this exact code passed only on another stack or another version of it: {', '.join(elsewhere)}; {mismatch}"

    def approvers(self) -> list[str] | None:
        """The logins in ``approvers.json`` that may approve, or None when there is no such file."""
        found = self._read("approvers.json", list)
        return None if found is None else [str(login) for login in found]

    def approval(self, name: str) -> dict | None:
        return self._approvals().get(str(name))

    def rejection(self, name: str) -> dict | None:
        """The person's rejection of ``name`` as {"hash", "reason", "by", "login", "at", "via"}, or None."""
        return (self._read("rejections.json", dict) or {}).get(str(name))

    def _write_rejections(self, rejections: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rejections.json").write_text(json.dumps(rejections, indent=1, sort_keys=True), encoding="utf-8")

    def request(self, name: str) -> dict | None:
        """What the person asked for when the draft ``name`` was proposed, as {"request", "at", "hash"} and "context"
        when one was sent, or None."""
        return (self._read("requests.json", dict) or {}).get(str(name))

    def last_approved(self, name: str) -> dict | None:
        """The code of ``name`` as a person last approved it, {"hash", "by", "at", "source"}, or None if never."""
        return (self._read("last_approved.json", dict) or {}).get(str(name))

    def hardware_problem(self, name: str, status: str, stack_id: str | None = None, now: float | None = None) -> str | None:
        """Why the workflow ``name`` may not run on the instrument ``stack_id``, or None. ``status`` comes from ``find``."""
        if status == SHIPPED:
            return None
        if status != APPROVED:
            return needs_approval(name)
        record = self.approval(name) or {}
        wanted = record.get("instrument") or ANY_INSTRUMENT
        if wanted != ANY_INSTRUMENT and wanted != stack_id:
            return (f"{name} is approved for the instrument {wanted}, not this one ({stack_id or 'which does not say what it is'}). "
                    f"A person approves it for this one with `python -m asyncroscopy.workflows approve {name}`.")
        expires = record.get("expires")
        if expires is not None and (now if now is not None else time.time()) > float(expires):
            return (f"the approval of {name} ended on {time.strftime('%Y-%m-%d', time.localtime(float(expires)))}. "
                    f"A person approves it again with `python -m asyncroscopy.workflows approve {name}`.")
        return None

    def _load(self, path: Path) -> Workflow | None:
        try:
            workflow = load_source(path.read_text(encoding="utf-8"))
        except (OSError, NotAWorkflow):
            return None
        return workflow if workflow.name == path.stem else None  # ignore a file whose name does not match

    def find(self, name: str) -> tuple[Workflow, str]:
        """Return the workflow and its status, which is "shipped", "approved" or "draft".

        Raises KeyError when there is none by that name. An approved workflow whose hash no longer matches its
        approval is a draft.
        """
        name = str(name)
        shipped = all_workflows()
        if name in shipped:
            return shipped[name], SHIPPED
        approved = self._load(self._file(APPROVED, name))
        if approved is not None:
            record = self._approvals().get(name) or {}
            return approved, APPROVED if record.get("hash") == approved.hash() else DRAFT
        draft = self._load(self._file(DRAFT, name))
        if draft is not None:
            return draft, DRAFT
        raise KeyError(f"there is no workflow named {name!r}")

    def workflow(self, name: str) -> Workflow:
        return self.find(name)[0]

    def listed(self) -> list[dict]:
        """Every workflow with its status, shipped first. Each entry has "name", "summary", "inputs" and "status"."""
        out = [{**workflow.describe(), "status": SHIPPED} for _, workflow in sorted(all_workflows().items())]
        names = {entry["name"] for entry in out}
        approvals = self._approvals()
        for kind in (APPROVED, DRAFT):
            folder = self._file(kind, "x").parent
            for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
                if path.stem in names:
                    continue
                try:
                    workflow, status = self.find(path.stem)
                except KeyError:
                    continue
                names.add(path.stem)
                entry = {**workflow.describe(), "status": status}
                if status == APPROVED:
                    record = approvals.get(path.stem) or {}
                    entry["approved_by"] = record.get("by")
                    entry["approved_for"] = record.get("instrument") or ANY_INSTRUMENT
                    if record.get("expires") is not None:
                        entry["approval_ends"] = time.strftime("%Y-%m-%d", time.localtime(float(record["expires"])))
                else:
                    if kind == APPROVED:
                        entry["note"] = "it, or a block it uses, changed after it was approved: it needs approving again"
                    asked = self.request(path.stem)
                    if asked:
                        entry["requested"] = asked.get("request")
                    rejected = self.rejection(path.stem)
                    if rejected and rejected.get("hash") == workflow.hash():
                        entry["rejected"] = rejected.get("reason")
                out.append(entry)
        return out

    def propose(self, source: str, tool_names: list[str] | None = None, analysis_tool_names: list[str] | None = None,
                request: str | None = None, context: str | None = None) -> dict:
        """Check a workflow's text and keep it as a draft, with ``request`` and ``context`` shown to the approver, when
        it has no problems; returns the lint result with "draft" and "status"."""
        refused = {"ok": False, "name": None, "hash": None, "acts": False, "tools": [], "blocks": [], "draft": None, "status": None}
        try:
            workflow = load_source(source)
        except NotAWorkflow as exc:
            return {**refused, "problems": list(exc.problems)}
        if workflow.name in all_workflows():
            return {**refused, "name": workflow.name, "problems": [f"{workflow.name} is a shipped workflow's name; give yours another name"]}
        if self._file(APPROVED, workflow.name).exists():
            return {**refused, "name": workflow.name,
                    "problems": [f"an approved workflow is already named {workflow.name}; give yours another name (a person may revoke that one)"]}
        result = workflow.lint(tool_names, analysis_tool_names)
        if not result["ok"]:
            return {**result, "draft": None, "status": None}
        path = self._file(DRAFT, workflow.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        requests = self._read("requests.json", dict) or {}
        if str(request or "").strip():
            requests[workflow.name] = {"request": str(request).strip(), "at": time.time(), "hash": result["hash"],
                                       **({"context": str(context).strip()} if str(context or "").strip() else {})}
        else:
            requests.pop(workflow.name, None)
        (self.root / "requests.json").write_text(json.dumps(requests, indent=1, sort_keys=True), encoding="utf-8")
        return {**result, "draft": str(path), "status": DRAFT}

    def approve(self, name: str, by: str, *, login: str | None = None, instrument: str | None = None, days: float | None = None,
                require_rehearsal: bool = False, via: str = "cli", expected_hash: str | None = None,
                instrument_info: dict | None = None, rehearsal_run_id: str | None = None) -> dict:
        """Record a person's approval of the draft ``name`` as it is now (``docs/MCP/workflow_approval.md``).

        Raises KeyError when there is no such draft, PermissionError when this login may not approve, and ValueError
        when a check, the rehearsal or ``expected_hash`` refuses it.
        """
        if not str(by or "").strip():
            raise ValueError("an approval names the person who gives it")
        allowed = self.approvers()
        if allowed is not None and str(login or "") not in allowed:
            raise PermissionError(f"{login or 'an unnamed login'} is not in this library's approvers.json; those who may approve: {allowed}")
        if days is not None and not float(days) > 0:
            raise ValueError("days must be above 0 (leave it out for an approval that holds until revoked)")
        path = self._file(DRAFT, str(name))
        if not path.exists() and self._file(APPROVED, str(name)).exists():
            path = self._file(APPROVED, str(name))  # changed after approval, approved again
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise KeyError(f"there is no draft named {name!r} to approve") from exc
        try:
            workflow = load_source(source)
        except NotAWorkflow as exc:
            raise ValueError(f"{name} is not a workflow any more: {exc}") from exc
        result = workflow.lint()
        if workflow.name != str(name) or not result["ok"]:
            raise ValueError(f"{name} does not pass its checks: {result['problems'] or 'its file and its name differ'}")
        if expected_hash is not None and str(expected_hash) != result["hash"]:
            raise ValueError(f"{workflow.name} has not been approved: it changed since it was reviewed (reviewed {expected_hash}, "
                             f"now {result['hash']}). Review it again.")
        rehearsed, unrehearsed = self.rehearsal_problem(workflow.name, result["hash"], (instrument_info or {}).get("stack_fingerprint"))
        if require_rehearsal and unrehearsed:
            raise ValueError(f"{workflow.name} has not been approved: {unrehearsed}. Run it on the twin first (run_workflow on a simulated stack).")
        if require_rehearsal and not rehearsal_run_id:
            raise ValueError(f"{workflow.name} has not been approved: name the rehearsal the person reviewed (rehearsal_run_id, "
                             "review_workflow's rehearsal.run_id)")
        chosen = None
        if rehearsal_run_id:
            chosen = rehearsed[-1] if rehearsed and not unrehearsed else None
            if chosen is None or chosen.get("run_id") != str(rehearsal_run_id):
                raise ValueError(f"{workflow.name} has not been approved: the rehearsal it would be approved on is "
                                 f"{(chosen or {}).get('run_id') or 'none'}, not {rehearsal_run_id}. Review it again.")
        now = time.time()
        approved = self._file(APPROVED, workflow.name)
        approved.parent.mkdir(parents=True, exist_ok=True)
        approved.write_text(source, encoding="utf-8")
        if path != approved:
            path.unlink(missing_ok=True)
        record = {"hash": result["hash"], "by": str(by).strip(), "login": login, "at": now, "acts": result["acts"], "tools": result["tools"],
                  "instrument": str(instrument).strip() if instrument and str(instrument).strip() else ANY_INSTRUMENT,
                  "expires": now + float(days) * 86400.0 if days is not None else None,
                  "rehearsed": {**{k: chosen.get(k) for k in ("run_id", "at", "calls", "inputs", "stack_id", "actions")},
                                "stack_fingerprint": fingerprint_of(chosen), "calls_made": matched_calls(chosen, instrument_info)} if chosen else None,
                  "request": (self.request(workflow.name) or {}).get("request"), "via": str(via)}
        self._write_approvals({**self._approvals(), workflow.name: record})
        signed = self._read("last_approved.json", dict) or {}
        signed[workflow.name] = {"hash": result["hash"], "by": record["by"], "at": now, "source": source}
        (self.root / "last_approved.json").write_text(json.dumps(signed, indent=1, sort_keys=True), encoding="utf-8")
        rejections = self._read("rejections.json", dict) or {}
        if rejections.pop(workflow.name, None) is not None:
            self._write_rejections(rejections)
        return {"name": workflow.name, **record}

    def reject(self, name: str, reason: str, *, by: str | None = None, login: str | None = None, via: str = "cli",
               expected_hash: str | None = None) -> dict:
        """Record a person's rejection of ``name`` as it is now; an approval is revoked first and the text stays a draft."""
        if not str(reason or "").strip():
            raise ValueError("a rejection gives the reason, so whoever wrote the workflow can fix it")
        allowed = self.approvers()
        if allowed is not None and str(login or "") not in allowed:
            raise PermissionError(f"{login or 'an unnamed login'} is not in this library's approvers.json; those who may reject: {allowed}")
        workflow, status = self.find(str(name))
        if status == SHIPPED:
            raise ValueError(f"{name} is shipped with the software; it is reviewed in the repository, not here")
        if expected_hash is not None and str(expected_hash) != workflow.hash():
            raise ValueError(f"{workflow.name} has not been rejected: it changed since it was reviewed (reviewed {expected_hash}, "
                             f"now {workflow.hash()}). Review it again.")
        if status == APPROVED or self._file(APPROVED, workflow.name).exists():
            self.revoke(workflow.name)
        record = {"hash": workflow.hash(), "reason": str(reason).strip(), "by": str(by).strip() if by and str(by).strip() else None,
                  "login": login, "at": time.time(), "via": str(via)}
        self._write_rejections({**(self._read("rejections.json", dict) or {}), workflow.name: record})
        return {"name": workflow.name, "status": DRAFT, **record}

    def revoke(self, name: str) -> dict:
        """Withdraw an approval; the workflow becomes a draft again. Raises KeyError when it is not approved."""
        approvals = self._approvals()
        approved = self._file(APPROVED, str(name))
        if str(name) not in approvals and not approved.exists():
            raise KeyError(f"{name!r} is not an approved workflow")
        approvals.pop(str(name), None)
        self._write_approvals(approvals)
        if approved.exists():
            draft = self._file(DRAFT, str(name))
            draft.parent.mkdir(parents=True, exist_ok=True)
            draft.write_text(approved.read_text(encoding="utf-8"), encoding="utf-8")
            approved.unlink()
        return {"name": str(name), "status": DRAFT}
