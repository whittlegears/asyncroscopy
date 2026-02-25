
# Contributing Guide

Thank you for contributing to this project.
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

* Image commands must return a deterministic DevEncoded format.
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

