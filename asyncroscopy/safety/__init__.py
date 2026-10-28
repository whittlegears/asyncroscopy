"""Approval mode for agent-driven instrument control. See docs/Operation/approval_mode.md."""

from .policy import OPERATOR_ONLY, risk_of

__all__ = ["OPERATOR_ONLY", "risk_of"]
