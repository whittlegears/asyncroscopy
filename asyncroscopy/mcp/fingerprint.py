"""The stack fingerprint ``endpoint_info`` reports: what software, device code and device configuration serve this
endpoint, read from the stack itself. ``docs/MCP/permissions.md`` describes the fields."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import asyncroscopy
from asyncroscopy.mcp.descriptors import DeviceInfo

module_property = "module_name"
simulated_by = "declared: the simulated Tango property of every device with a command that acts (run_servers' top-level simulated:)"
checkout = Path(asyncroscopy.__file__).resolve().parents[1]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=checkout, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def software() -> dict[str, Any]:
    try:
        package = version("asyncroscopy")
    except PackageNotFoundError:
        package = None
    commit = _git("rev-parse", "HEAD")
    changes = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {"package": "asyncroscopy", "version": package, "commit": commit, "uncommitted_changes": None if changes is None else bool(changes)}


def module_sha256(module_name: str | None) -> str | None:
    if not module_name:
        return None
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        return None
    return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()


def device_properties(proxy: Any) -> dict[str, list[str]] | None:
    if proxy is None:
        return None
    try:
        names = sorted(proxy.get_property_list("*"))
        found = proxy.get_property(names) if names else {}
    except Exception:
        return None
    return {name: [str(v) for v in found.get(name) or []] for name in names}


def device_code(dev_class: str, properties: Mapping[str, list[str]] | None) -> dict[str, Any]:
    module = ((properties or {}).get(module_property) or [None])[0]
    return {"class": dev_class, "module": module, "module_sha256": module_sha256(module)}


def stack_fingerprint(devices: Mapping[str, DeviceInfo], simulated: bool) -> dict[str, Any]:
    properties = {prefix: device_properties(device.proxy) for prefix, device in sorted(devices.items())}
    found = {
        "software": software(),
        "devices": {prefix: device_code(device.dev_class, properties[prefix]) for prefix, device in sorted(devices.items())},
        "config_digest": digest(properties),
        "unreadable_properties": sorted(prefix for prefix, values in properties.items() if values is None),
        "simulated": {"value": bool(simulated), "established_by": simulated_by},
    }
    return {**found, "digest": digest(found)}
