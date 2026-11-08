"""Every ASYNCROSCOPY_* variable and output path, resolved in one place when asked.

``settings.get()`` reads the environment at call time (never at import), so a test or a launcher can
set a variable first. It keeps nothing between calls: each call is a fresh, frozen ``Settings``.
Library code takes explicit values where it can, and otherwise calls ``settings.get()`` inside the
function that needs a value. No other code reads an ASYNCROSCOPY_* variable. The table of keys is in
``docs/Operation/settings.md``.

A variable that is unset gives the default below; ``None`` means the code using it has its own default
or the feature is off. Paths left unset fall under ``outputs_root``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

OUTPUTS_ENV = "ASYNCROSCOPY_OUTPUTS"

DEFAULT_LOG_ROTATE_BYTES = 64 * 2**20
DEFAULT_LOG_ROTATE_KEEP = 20


def _flag(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    return text.strip().lower() not in ("", "0", "false", "no", "off")


def _number(env: Mapping[str, str], name: str, default, kind):
    """``env[name]`` as ``kind``; a missing or blank value is ``default``. A value that is not a number names the variable."""
    text = env.get(name, "")
    if not text.strip():
        return default
    try:
        return kind(text)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {text!r}") from None


@dataclass(frozen=True)
class Settings:
    # Where servers write
    outputs_root: Path
    log_dir: Path
    trace_dir: Path
    log_rotate_bytes: int
    log_rotate_keep: int
    agent_memory: Path
    task_trace: str
    instrument_notes: str | None
    acquisition_dir: str | None
    integrity_log: Path
    integrity_key_file: Path
    integrity_key: str | None = field(repr=False)
    dose_ledger: Path

    # Addresses and credentials
    mcp_url: str | None
    operator_url: str | None
    analysis_url: str | None
    mcp_token: str | None = field(repr=False)
    tiled_uri: str | None
    tiled_public_uri: str
    tiled_api_key: str | None = field(repr=False)
    plain_tiled_api_key: str | None = field(repr=False)  # TILED_API_KEY: the name Tiled's own tools use

    # Agents
    model: str | None
    model_provider: str | None
    num_ctx: int | None
    agent_profile: str
    agent_tools: tuple[str, ...]
    agent_prompt: str | None
    caller: str | None
    mapper: str | None
    microscope: str | None
    poll_s: float
    approval_wait_s: float
    analysis_wait_s: float
    claim_check: bool
    auto_promote_after: int

    # Extensions
    plugins: str
    skills_path: tuple[Path, ...]

    # GUIs and test switches
    qt_api: str
    write_golden: bool
    write_benchmarks: bool
    parity_autoscript: str | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        outputs = Path(env.get(OUTPUTS_ENV) or "outputs").resolve()
        return cls(
            outputs_root=outputs,
            log_dir=Path(env.get("ASYNCROSCOPY_LOG_DIR") or outputs / "logs"),
            trace_dir=outputs / "trace",
            log_rotate_bytes=_number(env, "ASYNCROSCOPY_LOG_ROTATE_BYTES", DEFAULT_LOG_ROTATE_BYTES, int),
            log_rotate_keep=_number(env, "ASYNCROSCOPY_LOG_ROTATE_KEEP", DEFAULT_LOG_ROTATE_KEEP, int),
            agent_memory=Path(env.get("ASYNCROSCOPY_AGENT_MEMORY") or outputs / "agent_memory"),
            task_trace=env.get("ASYNCROSCOPY_TASK_TRACE", str(outputs / "trace" / "agent_tasks.jsonl")),  # "" turns it off
            instrument_notes=env.get("ASYNCROSCOPY_INSTRUMENT_NOTES") or None,
            acquisition_dir=env.get("ASYNCROSCOPY_ACQUISITION_DIR") or None,
            integrity_log=Path(env.get("ASYNCROSCOPY_INTEGRITY_LOG") or outputs / "integrity" / "manifest.jsonl"),
            integrity_key_file=Path(env.get("ASYNCROSCOPY_INTEGRITY_KEY_FILE") or outputs / "integrity" / "key"),
            integrity_key=env.get("ASYNCROSCOPY_INTEGRITY_KEY") or None,
            dose_ledger=Path(env.get("ASYNCROSCOPY_DOSE_LEDGER") or outputs / "dose" / "ledger.jsonl"),
            mcp_url=env.get("ASYNCROSCOPY_MCP_URL") or None,
            operator_url=env.get("ASYNCROSCOPY_OPERATOR_URL") or None,
            analysis_url=env.get("ASYNCROSCOPY_ANALYSIS_URL") or None,
            mcp_token=env.get("ASYNCROSCOPY_MCP_TOKEN") or None,
            tiled_uri=env.get("ASYNCROSCOPY_TILED_URI") or None,
            tiled_public_uri=env.get("ASYNCROSCOPY_TILED_PUBLIC_URI", "").strip().rstrip("/"),
            tiled_api_key=env.get("ASYNCROSCOPY_TILED_API_KEY"),  # "" is a key an operator set on purpose
            plain_tiled_api_key=env.get("TILED_API_KEY") or None,
            model=env.get("ASYNCROSCOPY_MODEL") or None,
            model_provider=env.get("ASYNCROSCOPY_MODEL_PROVIDER") or None,
            num_ctx=_number(env, "ASYNCROSCOPY_NUM_CTX", None, int),
            agent_profile=env.get("ASYNCROSCOPY_AGENT_PROFILE", "default"),
            agent_tools=tuple(p.strip() for p in env.get("ASYNCROSCOPY_AGENT_TOOLS", "").split(",") if p.strip()),
            agent_prompt=env.get("ASYNCROSCOPY_AGENT_PROMPT") or None,
            caller=env.get("ASYNCROSCOPY_CALLER") or None,
            mapper=env.get("ASYNCROSCOPY_MAPPER") or None,
            microscope=env.get("ASYNCROSCOPY_MICROSCOPE") or None,
            poll_s=_number(env, "ASYNCROSCOPY_POLL_S", 1.0, float),
            approval_wait_s=_number(env, "ASYNCROSCOPY_APPROVAL_WAIT_S", 1800.0, float),
            analysis_wait_s=_number(env, "ASYNCROSCOPY_ANALYSIS_WAIT_S", 3600.0, float),
            claim_check=_flag(env.get("ASYNCROSCOPY_CLAIM_CHECK"), True),
            auto_promote_after=_number(env, "ASYNCROSCOPY_AUTO_PROMOTE_AFTER", 0, int),
            plugins=env.get("ASYNCROSCOPY_PLUGINS", ""),
            skills_path=tuple(Path(p) for p in env.get("ASYNCROSCOPY_SKILLS_PATH", "").split(os.pathsep) if p),
            qt_api=env.get("ASYNCROSCOPY_QT_API", "").lower(),
            write_golden=env.get("ASYNCROSCOPY_WRITE_GOLDEN") == "1",
            write_benchmarks=env.get("ASYNCROSCOPY_WRITE_BENCHMARKS") == "1",
            parity_autoscript=env.get("ASYNCROSCOPY_PARITY_AUTOSCRIPT") or None,
        )

    def output(self, *parts: str) -> Path:
        """A path under ``outputs_root``, e.g. ``output("trace", "analysis.jsonl")``."""
        return self.outputs_root.joinpath(*parts)


def get() -> Settings:
    """The settings as the environment gives them now."""
    return Settings.from_env()
