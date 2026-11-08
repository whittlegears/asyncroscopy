"""The error a job device's verbs answer with."""

from __future__ import annotations

from typing import Any

class JobError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details  # extra fields of the error object, e.g. violations
