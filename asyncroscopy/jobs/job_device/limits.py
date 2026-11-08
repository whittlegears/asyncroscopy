"""Check what a plan's steps will set against the instrument's limits, at reserve."""

from __future__ import annotations


from .errors import JobError

VIOLATIONS_SHOWN = 20  # violations listed in a refusal; the total is always given


def _limit(value) -> float | None:
    """A Tango attribute config bound as a number; None for "Not specified"."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LimitChecks:
    """At reserve: refuse a plan that would set an attribute outside its limits."""

    def _check_limits(self, job: dict) -> dict:
        """Check what step_targets says the steps will set; raise plan_rejected listing every violation."""
        try:
            targets = list(self.step_targets(job) or [])
        except Exception as exc:
            raise JobError("device_error", f"{type(self).__name__}.step_targets failed: {type(exc).__name__}: {exc}") from exc
        found: dict[tuple[str, str], tuple | str] = {}
        violations, unchecked = [], {}
        for target in targets:
            key = (target["device"], target["attribute"])
            if "min" in target or "max" in target:
                low, high = _limit(target.get("min")), _limit(target.get("max"))
            else:
                if key not in found:
                    found[key] = self._attribute_limits(*key)
                if isinstance(found[key], str):
                    unchecked[f"{key[0]}/{key[1]}"] = found[key]
                    continue
                low, high = found[key]
            value = float(target["value"])
            if (low is not None and value < low) or (high is not None and value > high):
                violations.append({"step": target["step"], "device": key[0], "attribute": key[1], "value": value, "min": low, "max": high})
        if unchecked and getattr(self, "require_limits", True):
            shown = "; ".join(f"{k} ({r})" for k, r in list(unchecked.items())[:3])
            raise JobError(
                "plan_rejected", f"the limits of {len(unchecked)} attribute(s) the plan sets could not be read, so the plan cannot be checked: {shown}. Nothing moved.",
                unchecked=[{"target": k, "reason": r} for k, r in unchecked.items()][:VIOLATIONS_SHOWN],
            )
        if violations:
            shown = "; ".join(
                f"step {v['step']} sets {v['device']}/{v['attribute']} to {v['value']:g}, outside [{v['min']}, {v['max']}]" for v in violations[:3]
            )
            more = f" (and {len(violations) - 3} more)" if len(violations) > 3 else ""
            raise JobError(
                "plan_rejected", f"{len(violations)} step(s) outside the instrument's limits: {shown}{more}. Nothing moved.",
                violations=violations[:VIOLATIONS_SHOWN], violations_total=len(violations),
            )
        return {"targets": len(targets), "unchecked": [{"target": k, "reason": r} for k, r in unchecked.items()]}

    def _attribute_limits(self, device: str, attribute: str) -> tuple[float | None, float | None] | str:
        """(min, max) from the attribute's Tango config, or why they could not be read."""
        try:
            config = self.proxy(device, timeout_s=3.0).get_attribute_config(attribute)
        except Exception as exc:  # DevFailed: device down or no such attribute
            return f"could not read its limits: {exc}".splitlines()[0][:200]
        return _limit(config.min_value), _limit(config.max_value)
