"""The workflow library holds proposed workflows and the approvals that let them run on an instrument.

A workflow has one of three statuses.

* **shipped** is a module under ``graphs/``. It is reviewed in the repository like any other code.
* **approved** was proposed and then approved by a person. The approval names the workflow by its hash, which
  covers its code and the code of every block it uses. If either changes it is a draft again.
* **draft** was proposed and is not approved, or changed after it was approved. It is kept with the words of the
  person who asked for it in ``requests.json``.

A shipped or approved workflow may run on a real instrument when the operator allows hardware. A draft runs on a
digital twin only. Only a person can approve, on the command line with
``python -m asyncroscopy.agents.workflows approve NAME``. No tool a model can call approves anything.

An approval can carry these.

* **Rehearsals.** Every run of a draft on a digital twin is saved in ``rehearsals.json``. The person approving sees
  the runs of this exact code. None is required.
* **An instrument.** ``instrument`` limits the approval to one stack, named by the endpoint's ``stack_id``. Left
  out, it holds on every instrument.
* **An end date.** ``days`` makes the approval expire.
* **Approvers.** If the library has ``approvers.json``, a list of login names, only those logins may approve.
  Without it anyone who can run the command on this host may. The record keeps the login and the name given.

The files under ``root`` are ``drafts/NAME.py``, ``approved/NAME.py``, ``approvals.json``, ``rehearsals.json``,
``requests.json`` and optionally ``approvers.json``.
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


def needs_approval(name: str) -> str:
    return (f"{name} is a draft: nobody has approved it, so it runs on a digital twin only. A person approves it once with "
            f"`python -m asyncroscopy.agents.workflows approve {name}`, after reading it (`python -m asyncroscopy.agents.workflows view` "
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

    def record_rehearsal(self, workflow: Workflow, record: dict) -> None:
        """Save a finished run of ``workflow`` on a digital twin. ``record`` is the runner's record.

        Shipped workflows are skipped because they need no approval. Runs that did not reach an end are skipped too.
        """
        if workflow.name in all_workflows() or record.get("state") not in ("done", "failed", "stopped"):
            return
        entry = {"hash": record.get("hash"), "run_id": record.get("run_id"), "state": record.get("state"), "at": record.get("ended") or time.time(),
                 "calls": record.get("calls"), "inputs": record.get("inputs") or {}, "error": record.get("error")}
        every = self._read("rehearsals.json", dict) or {}
        every[workflow.name] = [*(every.get(workflow.name) or []), entry][-KEPT_REHEARSALS:]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rehearsals.json").write_text(json.dumps(every, indent=1, sort_keys=True, default=str), encoding="utf-8")

    def rehearsals(self, name: str, hash_: str | None = None) -> list[dict]:
        """The saved twin runs of ``name``, oldest first. With ``hash_`` only the runs of that exact code."""
        runs = (self._read("rehearsals.json", dict) or {}).get(str(name)) or []
        return [r for r in runs if hash_ is None or r.get("hash") == hash_]

    def approvers(self) -> list[str] | None:
        """The logins in ``approvers.json`` that may approve, or None when there is no such file."""
        found = self._read("approvers.json", list)
        return None if found is None else [str(login) for login in found]

    def approval(self, name: str) -> dict | None:
        return self._approvals().get(str(name))

    def request(self, name: str) -> dict | None:
        """What the person asked for when the draft ``name`` was proposed, as {"request", "at"}, or None."""
        return (self._read("requests.json", dict) or {}).get(str(name))

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
                    f"A person approves it for this one with `python -m asyncroscopy.agents.workflows approve {name}`.")
        expires = record.get("expires")
        if expires is not None and (now if now is not None else time.time()) > float(expires):
            return (f"the approval of {name} ended on {time.strftime('%Y-%m-%d', time.localtime(float(expires)))}. "
                    f"A person approves it again with `python -m asyncroscopy.agents.workflows approve {name}`.")
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
                out.append(entry)
        return out

    def propose(self, source: str, tool_names: list[str] | None = None, analysis_tool_names: list[str] | None = None,
                request: str | None = None) -> dict:
        """Check a workflow's text and keep it as a draft when it has no problems.

        ``request`` is the words of the person who asked for this workflow. It is kept with the draft and shown to
        whoever approves it.

        Returns the lint result with "draft", which is its path or None, and "status". ``problems`` are sentences
        for the author to fix. A workflow with problems is not kept and none of it is run.
        """
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
            requests[workflow.name] = {"request": str(request).strip(), "at": time.time(), "hash": result["hash"]}
        else:
            requests.pop(workflow.name, None)
        (self.root / "requests.json").write_text(json.dumps(requests, indent=1, sort_keys=True), encoding="utf-8")
        return {**result, "draft": str(path), "status": DRAFT}

    def approve(self, name: str, by: str, *, login: str | None = None, instrument: str | None = None, days: float | None = None) -> dict:
        """Record a person's approval of the draft ``name`` as it is now. Returns the approval record.

        ``by`` is the person's name.
        ``login`` is the account giving it, checked against ``approvers.json`` when that file exists.
        ``instrument`` is the ``stack_id`` it is approved for. None or "any" means every instrument.
        ``days`` is how long it holds. None means until revoked.

        The draft becomes the approved workflow of that name and its hash is what is approved. Raises KeyError when
        there is no such draft, PermissionError when this login may not approve, and ValueError when the workflow
        no longer passes its checks or ``by`` is empty.
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
            path = self._file(APPROVED, str(name))  # it changed after approval and is being approved again
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
        rehearsed = [r for r in self.rehearsals(workflow.name, result["hash"]) if r.get("state") == "done"]
        now = time.time()
        approved = self._file(APPROVED, workflow.name)
        approved.parent.mkdir(parents=True, exist_ok=True)
        approved.write_text(source, encoding="utf-8")
        if path != approved:
            path.unlink(missing_ok=True)
        record = {"hash": result["hash"], "by": str(by).strip(), "login": login, "at": now, "acts": result["acts"], "tools": result["tools"],
                  "instrument": str(instrument).strip() if instrument and str(instrument).strip() else ANY_INSTRUMENT,
                  "expires": now + float(days) * 86400.0 if days is not None else None,
                  "rehearsed": {k: rehearsed[-1].get(k) for k in ("run_id", "at", "calls", "inputs")} if rehearsed else None,
                  "request": (self.request(workflow.name) or {}).get("request")}
        self._write_approvals({**self._approvals(), workflow.name: record})
        return {"name": workflow.name, **record}

    def revoke(self, name: str) -> dict:
        """Withdraw an approval. The workflow becomes a draft again and its text is kept.

        Raises KeyError when it is not approved.
        """
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
