
# Contributing Guide

Thank you for contributing to this project.

Set up with [quickstart](Operation/quickstart.md); add hardware with [new_instrument](Adding_New_Hardware/new_instrument.md).

This repository is designed as infrastructure for scientific instrument control. Stability, clarity, and long-term maintainability take priority over rapid feature additions.

Before submitting a pull request, please review the engineering principles below.

---

# Engineering Philosophy(Subject to Revision)

This project prioritizes:

* Clarity over abstraction
* Explicit contracts over implicit assumptions
* Modularity over coupling
* Stability over rapid feature growth
* Infrastructure-grade reliability over experimentation

The migration to PyTango represents a transition toward a stable, extensible control architecture suitable for laboratory deployment.

---


# Core Principles(Elaborated)

## 1. Strong Typing Across Public Interfaces

**Keywords:** Reliability, Maintainability, Explicitness

* All public functions and classes must use explicit type annotations.
* Return contracts must be typed and deterministic.
* Avoid untyped dynamic structures in public APIs.

**Rationale:**
Typing improves reliability and refactoring safety. Distributed systems require strict clarity at communication boundaries.

---

## 2. Modular Device Architecture

**Keywords:** Extensibility, Isolation, Separation of Concerns

* Each detector must be implemented as an independent module or device.
* The microscope device orchestrates but does not own detector logic.
* Hardware abstraction, orchestration, and data transport must remain separated.

**Rationale:**
Modularity enables adding new detectors without modifying core logic and prevents cascading failures.

---

## 3. Simplicity Over Cleverness

**Keywords:** Explainability, Debuggability, Predictability

* Prefer explicit, readable logic.
* Avoid hidden control flow or meta-programming unless necessary.
* Code should be understandable in one reading session.

**Rationale:**
Scientific control systems must be explainable and auditable. Maintainability outweighs compactness.

---

## 4. Explicit Communication Contracts

**Keywords:** Determinism, Interoperability, Stability

* Acquisition commands return a Tiled key; a command that returns image bytes uses a deterministic DevEncoded format.
* Metadata must include required fields (e.g., shape, dtype).
* No implicit reconstruction assumptions.

**Rationale:**
Clear contracts prevent integration drift and simplify external ecosystem integration.

---

## 5. Testing Requirements

**Keywords:** Regression Prevention, Confidence, Automation

* All new features must include tests.
* Unit tests for device logic.
* Integration tests using `tango.test_context`.
* Failure mode tests (e.g., connection loss, fallback behavior).

**Rationale:**
Hardware-facing systems require high confidence. Tests prevent regression and enable safe refactoring.

---

## 6. Minimal Core Dependencies

**Keywords:** Portability, Stability, Deployment Simplicity

* The core library should depend only on essential packages (e.g., PyTango, NumPy).
* Heavy scientific or ML dependencies belong in adapter layers.
* Avoid unnecessary runtime dependencies.

**Rationale:**
Minimal dependencies reduce installation complexity and improve portability across laboratory environments.

---

## 7. Clear Error Semantics

**Keywords:** Transparency, Diagnostics, Observability

* Raise explicit Tango exceptions for hardware failures.
* Avoid silent failures.
* Log actionable diagnostic information.

**Rationale:**
Instrument downtime is costly. Errors must be clear and traceable.

---

## 8. Deterministic Logging and State Reporting

**Keywords:** Traceability, Reproducibility

* Use structured logging.
* Avoid hidden mutable global state.
* Device state transitions must be explicit.

**Rationale:**
Scientific reproducibility requires traceable execution history.

---

## 9. Documentation Standards

**Keywords:** Usability, Adoption

* Provide minimal working client notebooks.
* Include practical examples for common workflows.
* Document reconstruction logic and failure handling.

**Rationale:**
Most users interact through notebooks. Examples serve as executable documentation.

---

## 10. Backward Compatibility

**Keywords:** Stability, Upgrade Path

* Do not break public APIs without versioning.
* Use clear deprecation notices.
* Maintain compatibility where feasible.

**Rationale:**
Device interfaces become infrastructure. Stability protects downstream users.

---

# Layering

Packages depend on the ones below them, never the other way:

```
core  <-  contracts / trace / data / safety  <-  instruments / jobs  <-  mcp  <-  agents
```

* `core/` holds plumbing with no asyncroscopy imports: file locks and sequenced appends
  (`fileio.py`), bind-address rules (`net.py`) and the shared names (`names.py`: headers,
  envelope verbs, job states, data-key patterns, action roles, the agent graph names).
* `trace/writer.py` writes the tool log and the Instrument Trace.
* `safety/` decides risk and approval (`policy.py` for the name rules, `decide.py` for the one
  approval decision, `estop.py` for the emergency stop).
* `instruments/` and `jobs/` are the devices. Each base class declares what its commands are
  (`asyncroscopy.utils.commands.command(risk=, role=, stop=, routine=, operator_only=)`) and serves
  the declarations as its `action_hints` attribute; vendor backends and twins inherit them.
* The scanning-beam instruments share `instruments/scanning_beam.py` (`ScanningBeamInstrument`, with its twin mixin and map
  profile beside it); `ElectronMicroscope` and `ScanningElectronMicroscope` add what is their own. `instruments/common/sample.py`
  is the SAMPLE device (which sample is loaded, and its dose). The Mapper's analysis goes through
  `jobs/mapping/stitch_service.py`, so `jobs/` needs no MCP client (`stitch_mcp.py` is loaded on first use).
* `mcp/` serves the devices: `mcp_server.py` (discovery, wrapping, interceptors), `descriptors.py`,
  and `components/` for the native tools. It learns what an instrument is only from the devices
  it discovers (their declarations), never by importing `instruments/`.
* `agents/` holds everything tuned to a model: graphs (`mapping`, `general`, `team`, `goal`,
  `swarm`), middleware, `guidance.py`, `catalog.py` (what each tool is, read from the endpoint's
  `endpoint_info` once per run) and `profiles/` (model-specific data, the team graph's tool split
  among it). The agent MCP server (`mcp/agent_mcp_server.py`) imports the agent layer only when it
  starts.
* `analysis_server/` is the analysis MCP server (`python -m asyncroscopy.analysis_server`): it stores,
  schedules and serves pyTEMlib's `image_tools` (stitching, scoring, placement) with only `core`,
  `contracts` and `trace` below it, never `mcp`, `agents`, `instruments` or `jobs`.
* `settings.py` is the one place an `ASYNCROSCOPY_*` variable or output path is read (table in
  `docs/Operation/settings.md`); `settings.get()` reads the environment at call time, so nothing is read at import.

`tests/test_layering.py` imports each package in a fresh interpreter and fails if a lower package
loads anything from `asyncroscopy.mcp` or `asyncroscopy.agents`, if `mcp` loads anything from
`instruments` or `agents`, if `jobs` loads anything from `instruments`, or if the analysis server
loads anything from `mcp`, `agents`, `instruments` or `jobs`.

---

# Pull Request Checklist

Before submitting a PR, ensure:

* [ ] Public interfaces are typed
* [ ] Tests are added or updated
* [ ] No unnecessary dependencies introduced
* [ ] Communication contracts preserved
* [ ] Documentation updated if behavior changed
* [ ] Error handling follows project standards

---

# Code Style Expectations

* Prefer explicit over implicit logic
* Keep functions small and focused
* Avoid deep inheritance hierarchies
* Avoid premature abstraction
* Document assumptions clearly

---

# Questions or Design Changes

For architectural changes:

* Open an issue before implementation.
* Clearly explain trade-offs.
* Provide rationale aligned with project principles.

Major design shifts should be documented as Architecture Decision Records (ADR).

