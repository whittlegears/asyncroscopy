"""Claim check: a final answer may only claim what this run's tool calls did.

Benchmark runs with a small model (qwen3:8b) ended with answers such as "the
defocus was set back to 0 nm" when the last call set +20 nm, lists of image
keys no tool returned, and "you can view the spectrum" when the user asked what
the sample contains. On an instrument a claim with nothing behind it is a
safety problem, so before a final answer (no tool calls) reaches the user,
``ClaimCheckMiddleware`` checks it against the run's tool calls, without
another model call. Only claims that can be checked exactly are flagged:

==================== =================================================================
claim                unsupported when
==================== =================================================================
key                  a data key (``*.h5`` ...) or map key that no tool reply, request
                     or system prompt in this conversation contains
setting value        "<setting> set back to / reset to / is now <value>": the last value
                     set or read for it differs (or none was); "set <setting> to
                     <value>": no successful set call used that value. Lengths are
                     metres in tool calls; a unitless value is checked only when 0
failed acquisition   a data key given as a result that only a failed acquisition named (one
                     taken with the beam blanked saves noise and fails); unless negated
acquired at          "I took images at <setting> <value>, <value> and <value>": no acquisition
                     ran while that setting had one of the values (the defocus series that
                     skipped its 0 nm image); unchecked if an acquisition ran with it unknown
action               "I took / acquired / captured ...", "moved to ...": no successful
                     call of that kind (an acquisition, a stage move, or a job) at all;
                     "a higher-magnification image", "zoomed in": no set_fov call succeeded
placement            "<key> was placed", "placed it on ...": place_image said "unplaced" for
                     that key (its latest call), or never said "placed" for any image
deferred read        the request asks what the data contain, the answer leaves viewing
                     them to the user, and no tool read any data (view_data, get_region)
missing positions    the request asks for coordinates / positions / where things are, a
                     reply gave points' x_m, y_m, and the answer states none of them
field name           the answer names a reply's field (``result_keys.map``) where its value belongs
stitched map         "the stitched map is available", "the map key is ...": a job's stitch failed
                     (``stitch_error``) and no later reply gave a map; unless negated
map quality          "good quality", "seamless", "well stitched": most of the map's tiles matched no
                     neighbour and sit where the stage reported them; unless negated. When the
                     request asks how good the map is, an answer that never says so is flagged too
unit                 a number copied from a reply field whose name gives its length unit
                     (``total_nm``, ``fov_m``) but stated in another length unit
==================== =================================================================

Negated and hedged sentences ("could not", "was not", "previously acquired")
are not claims. Facts come from ``goals.trace_from_messages`` and ``goals.Facts``,
the same reading of tool calls the goal checker uses.

On an unsupported claim the model gets one correction turn (a human message
named ``claim_check``) listing exactly what is unsupported; it may make the
missing call or correct the answer. If its next final answer still has
unsupported claims, a short note naming them is appended to that answer:
claims are never silently removed. ``ASYNCROSCOPY_CLAIM_CHECK=0`` turns the
check off.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from asyncroscopy import settings
from asyncroscopy.core.names import DATA_KEY_IN_TEXT
from asyncroscopy.agents.catalog import ToolCatalog
from asyncroscopy.agents.goals import MAP_KEY, Facts, trace_from_messages
from asyncroscopy.agents.guards import CLAIM_CHECK, is_request
from asyncroscopy.agents.guidance import stage_placed

READERS = ("view_data", "get_region")

NEGATION = re.compile(r"\b(?:not|never|no|unable|failed|cannot|couldn't|didn't|wasn't|weren't|haven't|hasn't|without|could|would|will|can|should|may|might|previously|earlier|already|existing|stored|old)\b|n't\b", re.I)
NUMBER = r"(?P<num>[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
UNIT = r"(?:\s*(?P<unit>nanomet(?:er|re)s?|micromet(?:er|re)s?|millimet(?:er|re)s?|picomet(?:er|re)s?|met(?:er|re)s?|nm|pm|µm|μm|um|mm|m)\b)?"
SCALE = {"nm": 1e-9, "pm": 1e-12, "µm": 1e-6, "μm": 1e-6, "um": 1e-6, "mm": 1e-3, "m": 1.0,
         "nanomet": 1e-9, "micromet": 1e-6, "millimet": 1e-3, "picomet": 1e-12, "met": 1.0}
_VALUE = NUMBER.replace("?P<num>", "?:") + UNIT.replace("?P<unit>", "?:")
VALUES = rf"{_VALUE}(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+){_VALUE})*"  # "-20 nm, 0 nm, and +20 nm"
FINAL_STATE =r"(?:(?:set|put|returned|reset|restored|brought|changed|moved)\s+back\s+to|(?:reset|restored|returned)\s+to|is\s+now(?:\s+at)?|now\s+at|left\s+at|ends?\s+at)"
SETTING_ALIASES = {"fov": ("fov", "field of view"), "defocus": ("defocus", "focus")}

ACQUIRED = re.compile(
    # Only the agent saying it did it: "the latest acquired image" describes data, it claims nothing.
    r"(?:\b(?:I|we)\s+(?:then\s+|also\s+|first\s+|have\s+|successfully\s+|just\s+)*|\b(?:and|then)\s+)(?:took|acquired|captured|recorded|collected)\s+(?:(?:an?|the|one|two|three|four|five|some|several|new|\d+)\s+)?(?:[\w-]+\s+){0,3}?"
    r"(?:images?|pictures?|micrographs?|spectr(?:um|a)|scans?|diffractograms?|data|frames?)\b"
    r"|\b(?:images?|pictures?|micrographs?|spectr(?:um|a)|scans?|diffractograms?)\s+(?:was|were|has\s+been|have\s+been)\s+(?:taken|acquired|captured|recorded|collected)\b"
    r"|\bI\s+(?:then\s+|also\s+|first\s+|have\s+|successfully\s+)*imaged\b",
    re.I,
)
MOVED = re.compile(
    r"\b(?:I|we)\s+(?:then\s+|also\s+|first\s+|have\s+|successfully\s+)*(?:moved|went|navigated|drove)\b"
    r"|\bstage\s+(?:was|has\s+been)\s+moved\b"
    r"|\b(?:moved|went|navigated)\s+(?:the\s+stage\s+)?(?:back\s+)?to\s+(?:the\s+)?(?:best|region|position|point|area|feature|particle|centre|center|origin|coordinates?)\b",
    re.I,
)
MAGNIFIED = re.compile(r"\b(?:higher|high|increased|greater|more)[- ]magnification\b|\bzoom(?:ed)?\s+in(?:to)?\b|\b(?:smaller|narrower)\s+field\s+of\s+view\b", re.I)
CONTENT_QUESTION = re.compile(
    r"\bwhat\b[^.?!]{0,40}\b(?:contains?|made\s+of|composed\s+of|composition|is\s+in\s+(?:it|the\s+(?:sample|spectrum|image|data)))\b"
    r"|\b(?:which|what)\s+elements?\b|\bcomposition\b|\belements?\s+(?:are|is)\s+(?:present|there)\b",
    re.I,
)
DEFERS = re.compile(
    r"\byou\s+(?:can|could|may|might)\s+(?:now\s+)?(?:view|inspect|analy[sz]e|look\s+at|open|examine|check)\b"
    r"|\bavailable\s+for\s+(?:further\s+)?(?:analysis|viewing|inspection|review)\b"
    r"|\bcan\s+be\s+(?:viewed|analy[sz]ed|inspected|examined)\b|\bfor\s+further\s+analysis\b",
    re.I,
)

# The unit a table's column header or a sentence gives its bare numbers: "x (nm)", "in nm", "in metres".
HEADER_UNIT = re.compile(r"\((?P<header>nm|pm|µm|μm|um|mm|m)\)|\bin\s+(?P<phrase>nanomet(?:er|re)s|micromet(?:er|re)s|met(?:er|re)s|nm|µm|μm|um|m)\b", re.I)

# "at the current position" asks for nothing: only a plural, "position of", or "where ... is" asks where.
POSITION_QUESTION = re.compile(r"\b(?:coordinates?|positions|locations|position\s+of|where\s+(?:it\s+|they\s+)?(?:is|are|was|were|lies|lie))\b", re.I)

FIELD_UNIT = re.compile(r"_(nm|um|pm|mm|m)(?:_per_s)?$")
STATED_LENGTH = re.compile(NUMBER + r"`?" + UNIT)  # "`0.00017` meters": a small model puts copied numbers in backticks
QUALITY_QUESTION = re.compile(r"\bhow\s+(?:good|well)\b|\bquality\b|\bseams?\b", re.I)
STAGE_PLACED = re.compile(r"\bstage(?:'s)?\s+(?:reported|position|placed|put)|\brough\b|\bmatched\s+no\b|\bno\s+(?:image\s+)?match|\bunmatched\b"
                          r"|\bmisalign|\bunplaced\b|\bfallback\b|\bseams?\b", re.I)
FIELD_NAME = re.compile(r"\bresult_keys\.\w+")
STITCHED = re.compile(
    r"\bstitched\s+(?:map|image|mosaic)\s+(?:is|was)\s+(?:now\s+)?(?:available|ready|done|complete|saved)\b|\bmap\s+key\s+is\b"
    r"|\bstitch(?:ing)?\s+(?:succeeded|completed|worked|finished)\b|\bwas\s+stitched\b",
    re.I,
)

GOOD_MAP = re.compile(
    r"\b(?:good|high|excellent|great)\s+(?:quality|stitch(?:ing)?)\b|\bseamless(?:ly)?\b|\b(?:well|accurately|cleanly|correctly)[- ]stitched\b"
    r"|\baligned\s+(?:well|correctly|accurately)\b|\bquality\s+(?:is|was)\s+(?:very\s+|quite\s+)?(?:good|high|excellent|great)\b",
    re.I,
)

# "placed where the stage reported them" is a map's tiles laid out by stage position, not a place_image result.
PLACED = re.compile(r"\b(?:was|were|been|is|are|got|successfully|I|we)\s+(?:\w+\s+)?placed\b(?!\s+(?:where|by|at|from)\s+(?:the\s+)?stage)|\bplaced\s+(?:it|them|the\s+\w+(?:\s+\w+)?)\s+(?:on|in)\b|\bplacement\s+(?:succeeded|worked)\b", re.I)

# An answer that ends by announcing work instead of reporting it ("I'll now take an EDS spectrum on it.").
ANNOUNCES = re.compile(
    r"\b(?:I(?:'ll| will| am going to|'m going to)|let's|let\s+(?:us|me)|next,?\s+I(?:'ll| will))\s+(?:now\s+|then\s+|first\s+|go\s+ahead\s+and\s+)?"
    r"(?:do|take|acquire|record|collect|measure|find|locate|move|place|image|map|run|start|proceed|go|zoom|focus|scan|check|identify|select|prepare"
    r"|adjust|retry|try|continue|change|use|repeat|reduce|increase|set|call|retrieve|get|read)\b",
    re.I,
)

CORRECTION = ("Check before you answer: your answer says things no tool call in this run did or returned:\n{items}\n"
              "Make the missing tool call now if the request needs it, or answer again without these claims. "
              "Say only what the tool replies show.")
NOTE = "\n\n[Claim check: these statements could not be verified from this run's tool calls:\n{items}]"


@dataclass
class Claim:
    kind: str  # key, failed_acquisition, setting, acquired_at, action, placement, unfinished, deferred_read, missing_positions, field_name, stitch, map_quality, unit
    text: str  # what the answer says
    why: str  # why no tool call supports it

    def line(self) -> str:
        return f'- "{self.text}": {self.why}'


def enabled_from_env() -> bool:
    return settings.get().claim_check


def is_correction(message: BaseMessage) -> bool:
    return getattr(message, "type", "") == "human" and not is_request(message)


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _request(messages) -> str:
    return next((_text(m) for m in reversed(messages) if is_request(m)), "")


def _sentences(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+|;", answer) if s.strip()]


def _negated(sentence: str, match) -> bool:
    """A claim is hedged or negated when one of those words comes shortly before it in its sentence, or within it."""
    return bool(NEGATION.search(sentence[max(0, match.start() - 40):match.end()]))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            parsed = _json(value)
            return float(parsed) if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) else None
    return None


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _setting_of(tool: str, verb: str) -> str | None:
    match = re.match(rf"^[A-Z][A-Za-z0-9]*_{verb}_(\w+)$", tool)
    return match.group(1) if match and match.group(1) not in ("stage", "parameters") else None


class _Settings:
    """Values set and read for each setting in this run, from the calls in order."""

    def __init__(self, calls, tool_names, catalog: ToolCatalog):
        self.names = {n for t in tool_names for n in [_setting_of(t, "set")] if n} | {n for c in calls for n in [_setting_of(c.name, "set")] if n}
        self.last: dict[str, tuple[float, str]] = {}
        self.set: dict[str, list[float]] = {}
        self.setter: dict[str, str] = {}
        self.acquired: list[dict[str, float]] = []  # the known settings in effect at each successful acquisition
        for call in calls:
            if call.error:
                continue
            if catalog.category(call.name) == "acquire":
                self.acquired.append({name: value for name, (value, _) in self.last.items()})
            name = _setting_of(call.name, "set")
            if name:
                value = _number(call.args.get(name))
                numbers = [v for v in (_number(a) for a in call.args.values()) if v is not None]
                if value is None and len(numbers) == 1:
                    value = numbers[0]
                if value is not None:
                    self.set.setdefault(name, []).append(value)
                    self.last[name] = (value, call.name)
                    self.setter[name] = call.name
                continue
            name = _setting_of(call.name, "get")
            if name and _number(call.result) is not None:
                self.last[name] = (_number(call.result), call.name)
                continue
            for node in _walk(_json(call.result)):  # a reply that reports settings (get_parameters: defocus_m, fov_m)
                for key, value in node.items():
                    setting = key[:-2] if key.endswith("_m") else key
                    if setting in self.names and _number(value) is not None:
                        self.last[setting] = (_number(value), call.name)

    def phrases(self) -> list[tuple[str, str]]:
        """(setting, how an answer may name it), longest phrase first."""
        out = []
        for name in self.names:
            for phrase in {name.replace("_", " "), *SETTING_ALIASES.get(name, ())}:
                out.append((name, phrase))
        return sorted(out, key=lambda p: -len(p[1]))


def _scale(unit: str | None) -> float | None:
    if not unit:
        return None
    unit = unit.lower()
    for prefix, scale in SCALE.items():
        if unit == prefix or (len(prefix) > 2 and unit.startswith(prefix)):
            return scale
    return None


def _matches(stated: float, scale: float | None, actual: float) -> bool | None:
    """Whether a tool value (SI) agrees with a stated value; None when it cannot be checked exactly."""
    if scale is None:
        return abs(actual) <= 1e-9 if stated == 0 else None
    target = stated * scale
    return abs(actual - target) <= max(0.05 * abs(target), 0.5 * scale)


def _show(value: float, scale: float | None, unit: str | None) -> str:
    return f"{value / scale:g} {unit}" if scale else f"{value:g}"


def _setting_claims(answer: str, settings: _Settings) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[tuple[str, int]] = set()
    for sentence in _sentences(answer):
        for name, phrase in settings.phrases():
            p = re.escape(phrase)
            patterns = (
                re.compile(rf"\b{p}\b[^.;\n]{{0,40}}?\b{FINAL_STATE}\s+{NUMBER}{UNIT}", re.I),
                re.compile(rf"\bset\s+(?:the\s+)?{p}\s+(?:back\s+)?(?:to|at)\s+{NUMBER}{UNIT}", re.I),
                re.compile(rf"\b{p}\s+(?:was|were|has\s+been)\s+set\s+(?:back\s+)?(?:to|at)\s+{NUMBER}{UNIT}", re.I),
            )
            for pattern in patterns:
                for m in pattern.finditer(sentence):
                    key = (sentence, m.start())
                    if key in seen or _negated(sentence, m):
                        continue
                    seen.add(key)
                    stated = float(m.group("num").replace("−", "-"))
                    unit = m.group("unit")
                    scale = _scale(unit)
                    said = m.group(0).strip()
                    if pattern is patterns[0] or re.search(r"\bback\b", said, re.I):
                        if name not in settings.last:
                            claims.append(Claim("setting", said, f"no tool call in this run set or read the {phrase}"))
                            continue
                        actual, source = settings.last[name]
                        if _matches(stated, scale, actual) is False:
                            # Name the exact call that would make the claim true: a small model told only that a claim is
                            # unsupported repeated its answer unchanged.
                            setter = f"{source.split('_')[0]}_set_{name}"
                            target = json.dumps({name: stated * scale if scale else stated})
                            claims.append(Claim("setting", said, f"the last {phrase} set or read in this run was {_show(actual, scale, unit)} ({source}); "
                                                                 f"to make it true, call {setter} with {target} now"))
                    else:
                        values = settings.set.get(name, [])
                        if not values:
                            claims.append(Claim("setting", said, f"no successful set_{name} call in this run"))
                        elif all(_matches(stated, scale, v) is False for v in values):
                            shown = ", ".join(_show(v, scale, unit) for v in values)
                            claims.append(Claim("setting", said, f"the {phrase} values set in this run were {shown}"))
    return claims


def _at_values(sentence: str, phrase: str):
    """"at defocus -20, 0 and +20 nm" in ``sentence``: the match and each (stated number, scale, unit), or None."""
    m = re.search(rf"\bat\s+(?:an?\s+|the\s+)?{re.escape(phrase)}s?(?:\s+values?)?(?:\s+of)?\s+(?P<list>{VALUES})", sentence, re.I)
    if not m:
        return None
    items = list(re.finditer(rf"{NUMBER}{UNIT}", m.group("list")))
    unit = next((i.group("unit") for i in reversed(items) if i.group("unit")), None)  # "-20, 0 and +20 nm"
    return m, [(float(i.group("num").replace("−", "-")), _scale(i.group("unit") or unit), i.group("unit") or unit) for i in items]


def _acquired_at_claims(answer: str, settings: _Settings, request: str = "") -> list[Claim]:
    """"I took images at defocus -20 nm, 0 nm and +20 nm": each value needs an acquisition while the setting had it.
    Unchecked when an acquisition ran with the setting unknown (it may have had the value). A value the request
    asked for is to be acquired, not dropped from the answer."""
    claims: list[Claim] = []
    for sentence in _sentences(answer):
        acquired = ACQUIRED.search(sentence)
        if not acquired or _negated(sentence, acquired):
            continue
        checked: set[str] = set()
        for name, phrase in settings.phrases():
            found = _at_values(sentence, phrase)
            if name in checked or not found or not settings.acquired:
                continue
            checked.add(name)
            during = [a.get(name) for a in settings.acquired]
            if None in during:
                continue
            m, items = found
            asked = [stated * scale if scale else stated for s in _sentences(request) for stated, scale, _ in (_at_values(s, phrase) or (None, []))[1] if scale or stated == 0]
            setter = settings.setter.get(name)
            for stated, scale, unit in items:
                if any(_matches(stated, scale, v) is not False for v in during):
                    continue
                value = stated * scale if scale else stated
                shown = _show(value, scale, unit)
                if setter and any(_matches(stated, scale, v) for v in asked):
                    why = (f"the request asks for an acquisition at {phrase} {shown}, and none ran while the {phrase} was {shown}; "
                           f"do it now: call {setter} with {json.dumps({name: value})}, acquire again, then restore the {phrase}. Do not answer without it")
                else:
                    how = f"; to make it true, call {setter} with {json.dumps({name: value})}, acquire again, then restore the {phrase}" if setter else ""
                    why = f"no acquisition in this run ran while the {phrase} was {shown}{how}"
                claims.append(Claim("acquired_at", m.group(0).strip(), why))
    return claims


def _failed_acquisition_claims(answer: str, calls, catalog: ToolCatalog) -> list[Claim]:
    """A key given as a result that only a failed acquisition named (saved with the beam blanked: noise, not the sample)."""
    made = {k for c in calls if c.ok and catalog.category(c.name) in ("acquire", "job") for k in DATA_KEY_IN_TEXT.findall(c.result)}
    failed = {k: c.result for c in calls if c.error and catalog.category(c.name) == "acquire" for k in DATA_KEY_IN_TEXT.findall(c.result) if k not in made}
    claims = []
    for sentence in _sentences(answer):
        for key in dict.fromkeys(DATA_KEY_IN_TEXT.findall(sentence)):
            if key in failed and not NEGATION.search(sentence):
                # The reason's first sentence only, and no blind "acquire again": a model that had already retaken the
                # image took one more when told to, and still listed this key.
                why = failed[key].split("ToolError:", 1)[-1].strip().split(". ")[0][:120]
                claims.append(Claim("failed_acquisition", key, f"that acquisition failed ({why}), so this file is not a result: leave this key "
                                                                "out and report the key of the successful acquisition that replaced it; "
                                                                "only if none did, make that acquisition again"))
    return claims


def _action_claims(answer: str, calls, catalog: ToolCatalog) -> list[Claim]:
    """Claims of acquiring, moving or zooming that no successful call backs. Not checked without the endpoint's
    catalog: which calls acquire or move is what the endpoint says, not what their names suggest."""
    if not catalog.known:
        return []
    ok = [c for c in calls if c.ok]
    jobs = any(catalog.category(c.name) == "job" for c in ok)
    acquired = jobs or any(catalog.category(c.name) == "acquire" for c in ok)
    moved = jobs or any(catalog.category(c.name) == "move" or c.name.endswith("_aim_image_shift") for c in ok)
    # A job zoomed only if its plan imaged finer than its tiles (a revisit or a zoom); a plain map changed nothing.
    magnified = any(catalog.role(c.name) == "optics.fov" or (catalog.category(c.name) == "job" and re.search(r"revisit|zoom|images", json.dumps(c.args, default=str) + str(c.result))) for c in ok)
    claims = []
    for sentence in _sentences(answer):
        for pattern, done, why in ((ACQUIRED, acquired, "no acquisition or job call succeeded in this run"),
                                   (MOVED, moved, "no stage move or job call succeeded in this run"),
                                   (MAGNIFIED, magnified, "no set_fov call succeeded in this run, so every image has the field of view it started with")):
            if done:
                continue
            m = pattern.search(sentence)
            if m and not _negated(sentence, m):
                claims.append(Claim("action", m.group(0).strip(), why))
    return claims


def _placements(calls) -> dict[str, str]:
    """The latest placement status for each image key: from place_image, or from a job that placed images (a Mapper
    zoom reports each tier's {"image", "placement": {"status"}})."""
    out: dict[str, str] = {}
    for call in calls:
        if call.error:
            continue
        reply = _json(call.result)
        if call.name.endswith("place_image"):
            if isinstance(reply, dict) and reply.get("status") in ("placed", "unplaced"):
                out[str(reply.get("child") or call.args.get("image"))] = reply["status"]
            continue
        for node in _walk(reply):
            placement = node.get("placement")
            if node.get("image") and isinstance(placement, dict) and placement.get("status") in ("placed", "unplaced"):
                out[str(node["image"])] = placement["status"]
    return out


def _placement_claims(answer: str, calls) -> list[Claim]:
    placements = _placements(calls)
    claims = []
    for sentence in _sentences(answer):
        m = PLACED.search(sentence)
        if not m or _negated(sentence, m):
            continue
        keys = [k for k in DATA_KEY_IN_TEXT.findall(sentence) if k in placements]
        refused = [k for k in keys if placements[k] == "unplaced"]
        if refused:
            claims.append(Claim("placement", sentence[:160], f"place_image said unplaced for {', '.join(refused)}; report its reason instead"))
        elif not keys and "placed" not in placements.values():
            claims.append(Claim("placement", m.group(0).strip(), "no place_image call in this run said placed"))
    return claims


def _field_name_claims(answer: str) -> list[Claim]:
    """The answer names a reply's field instead of its value (qwen3:8b: "The map key is `result_keys.map`")."""
    return [Claim("field_name", name, "this is the name of a field in a reply, not a key: give the value the reply holds there, or say there is none")
            for name in dict.fromkeys(FIELD_NAME.findall(answer))]


def _stitch_claims(answer: str, calls) -> list[Claim]:
    """A stitched map claimed when the job's stitch failed and no later call made one (a 2704-tile map whose stitch
    timed out was reported as "the stitched map is available")."""
    error = None
    for call in calls:
        if call.error:
            continue
        for node in _walk(_json(call.result)):
            if node.get("stitch_error"):
                error = str(node["stitch_error"])
            if isinstance(node.get("result_keys"), dict) and node["result_keys"].get("map"):
                error = None
    if error is None:
        return []
    claims = []
    for sentence in _sentences(answer):
        m = STITCHED.search(sentence)
        if m and not _negated(sentence, m):
            claims.append(Claim("stitch", sentence[:160], f"the stitch failed ({error[:200]}): say so, and give the run key the reply names"))
    return claims


def _unit_claims(answer: str, calls) -> list[Claim]:
    """A reply's number stated in a length unit other than its field's (qwen3:8b gave total_nm 0.00017 as metres)."""
    fields: dict[float, tuple[str, float]] = {}
    for call in calls:
        if call.error:
            continue
        for node in _walk(_json(call.result)):
            for key, value in node.items():
                m = FIELD_UNIT.search(str(key))
                number = _number(value)
                if m and number:
                    fields.setdefault(number, (str(key), SCALE[m.group(1)]))
    claims = []
    for m in STATED_LENGTH.finditer(answer):
        stated, scale = _number(m.group("num").replace("−", "-")), _scale(m.group("unit"))
        if stated is None or scale is None:
            continue
        for value, (key, field_scale) in fields.items():
            if abs(stated - value) <= 1e-9 * abs(value) and field_scale != scale:
                claims.append(Claim("unit", m.group(0).strip("` "), f"this is {key} from a reply: {value} in the unit its name gives, not {m.group('unit')}"))
                break
    return claims


def _map_quality_claims(answer: str, request: str, calls) -> list[Claim]:
    """A map called good or seamless when most of its tiles matched no neighbour (qwen3:8b called a map whose 100 tiles
    all sat where the stage reported them "of good quality", from its pixel statistics); and, when the request asks
    how good the map is, an answer that leaves that out (corrected, it listed the key and tile size instead)."""
    placed = None
    for call in calls:
        if not call.error:
            for node in _walk(_json(call.result)):
                placed = stage_placed(node) or placed
    if placed is None:
        return []
    why = (f"{placed[0]} of {placed[1]} tiles matched no neighbour and sit where the stage reported them: "
           "say the map is a rough layout whose seams can be off by the stage's error")
    claims = []
    for sentence in _sentences(answer):
        m = GOOD_MAP.search(sentence)
        if m and not _negated(sentence, m):
            claims.append(Claim("map_quality", sentence[:160], why))
    if not claims and QUALITY_QUESTION.search(request) and not STAGE_PLACED.search(answer):
        claims.append(Claim("map_quality", request[:160], why))
    return claims


def _unfinished_claims(answer: str) -> list[Claim]:
    """The answer's last sentences announce a next step the model never took (a small model ending its turn early)."""
    tail = _sentences(answer)[-2:]
    if any("?" in sentence for sentence in tail):
        return []  # a question to the user is a fair place to stop
    for sentence in tail:
        m = ANNOUNCES.search(sentence)
        if m:
            return [Claim("unfinished", sentence.strip()[:160], "the answer announces this step but ends without doing it; make the call now, or say why it cannot be done")]
    return []


def _missing_positions_claims(answer: str, request: str, calls) -> list[Claim]:
    """The request asks where things are, a reply gave their x_m, y_m, and the answer states none of those positions
    (qwen3:8b answered "Here are their stage coordinates:" with no coordinates after a list_pois of 19 points)."""
    if not POSITION_QUESTION.search(request):
        return []
    found: dict[str, set] = {}  # each tool's distinct positions (a point's links may repeat it)
    for call in calls:
        if call.error:
            continue
        for node in _walk(_json(call.result)):
            x, y = _number(node.get("x_m")), _number(node.get("y_m"))
            if x is not None and y is not None:
                found.setdefault(call.name, set()).add((x, y))
    values = [abs(v) for points in found.values() for point in points for v in point if v]
    if not values:
        return []
    text = MAP_KEY.sub(" ", DATA_KEY_IN_TEXT.sub(" ", answer))
    # A table headed "x (nm)" leaves its numbers without a unit: they are read in the units its headers name. Only
    # headers: "x_m" and "spectrum" end in unit letters, and read as units they made tile numbers into positions.
    headed = {_scale(m["header"] or m["phrase"]) for m in HEADER_UNIT.finditer(text)} - {None}
    for m in re.finditer(NUMBER + UNIT, text):
        stated = abs(float(m["num"].replace("−", "-")))
        scales = [_scale(m["unit"])] if m["unit"] else [1.0, *headed]
        # Within 2% and 1 nm of rounding: _matches allows half a unit, and "tile 2" read in µm matched every point.
        if stated and any(abs(stated * scale - value) <= 0.02 * value + 1e-9 for scale in scales if scale for value in values):
            return []
    tool, points = list(found.items())[-1]
    # Not "make the missing call": qwen3:8b called list_pois again, and its second copy filled the 8k context.
    return [Claim("missing_positions", "the positions asked for",
                  f"the request asks where, and the {tool} reply above gave x_m, y_m for {len(points)} point(s), but the "
                  f"answer states none of them; do not call {tool} again: answer now, listing each point's id with its "
                  f"x_m and y_m copied from that reply")]


def check_answer(messages: list, answer: str, tool_names: Iterable[str] = (), context: str = "",
                 catalog: ToolCatalog | None = None) -> list[Claim]:
    """The claims in ``answer`` that the tool calls in ``messages`` do not support (see the module docstring).

    ``tool_names``: the tools the agent has (they name the settings it can set); ``context``: other text
    the model was given (its system prompt), whose keys it may name; ``catalog``: what the endpoint's tools
    are (endpoint_info), without which claims about acquiring and moving are not checked.
    """
    catalog = catalog if catalog is not None else ToolCatalog.unknown()
    calls = trace_from_messages(messages)
    facts = Facts.from_calls(calls, catalog)
    claims: list[Claim] = []
    # Keys: any key a reply, a request or the prompt gave the model may be named (an error reply too).
    given = " ".join(_text(m) for m in messages if getattr(m, "type", "") in ("tool", "human")) + " " + context
    known = facts.seen_keys | set(DATA_KEY_IN_TEXT.findall(given)) | set(MAP_KEY.findall(given))
    named = list(dict.fromkeys(DATA_KEY_IN_TEXT.findall(answer) + MAP_KEY.findall(answer)))
    for key in named:
        if key not in known:
            claims.append(Claim("key", key, "no tool returned this key in this run"))
    claims += _failed_acquisition_claims(answer, calls, catalog)
    settings = _Settings(calls, tool_names, catalog)
    claims += _setting_claims(answer, settings)
    claims += _acquired_at_claims(answer, settings, _request(messages))
    claims += _action_claims(answer, calls, catalog)
    claims += _placement_claims(answer, calls)
    claims += _field_name_claims(answer)
    claims += _stitch_claims(answer, calls)
    claims += _map_quality_claims(answer, _request(messages), calls)
    claims += _unit_claims(answer, calls)
    claims += _unfinished_claims(answer)
    claims += _missing_positions_claims(answer, _request(messages), calls)
    if CONTENT_QUESTION.search(_request(messages)) and not any(c.ok and c.name in READERS for c in calls):
        m = DEFERS.search(answer)
        if m:
            claims.append(Claim("deferred_read", m.group(0).strip(),
                                "the request asks what the data contain, but no tool read them; read them (view_data) and say what they show"))
    return claims


def corrections_used(messages: list) -> int:
    """Correction turns since the user's latest request."""
    count = 0
    for m in reversed(messages):
        if is_correction(m):
            count += 1
        elif getattr(m, "type", "") == "human":
            break
    return count


def with_note(message: AIMessage, claims: list[Claim]) -> AIMessage:
    """The answer with a visible note naming the claims that could not be verified (nothing removed)."""
    text = _text(message) + NOTE.format(items="\n".join(c.line() for c in claims))
    return AIMessage(text, id=message.id, tool_calls=[], response_metadata=getattr(message, "response_metadata", {}) or {})


def _final(result: list) -> AIMessage | None:
    ai = next((m for m in reversed(result) if isinstance(m, AIMessage)), None)
    return ai if ai is not None and not ai.tool_calls else None


class ClaimCheckMiddleware(AgentMiddleware):
    """Checks a final answer's claims against the run's tool calls; one correction turn, then a visible note.

    Put it last in the middleware list so it wraps the model call itself: the correction is one more
    model call inside the same step, and the first answer, the correction and the new answer all go
    into the state (and so into traces).
    """

    def __init__(self, tool_names: Iterable[str] = (), enabled: bool | None = None, max_corrections: int = 1,
                 catalog: ToolCatalog | None = None) -> None:
        super().__init__()
        self.tool_names = set(tool_names)
        self.catalog = catalog
        self.enabled = enabled_from_env() if enabled is None else enabled
        self.max_corrections = max_corrections

    def _claims(self, request, history: list, answer: AIMessage) -> list[Claim]:
        system = getattr(request, "system_message", None)
        return check_answer(history, _text(answer), self.tool_names | {getattr(t, "name", "") for t in request.tools or []},
                            context=_text(system) if system is not None else "", catalog=self.catalog)

    def _plan(self, request, response):
        """(answer, claims, history) when the response is a final answer with unsupported claims, else None."""
        if not self.enabled:
            return None
        result = response.result if isinstance(response, ModelResponse) else [response]
        answer = _final(result)
        if answer is None:
            return None
        history = list(request.state.get("messages", [])) if getattr(request, "state", None) else list(request.messages)
        claims = self._claims(request, history, answer)
        return (answer, claims, history) if claims else None

    def _finish(self, request, history, answer, correction, retry):
        result = retry.result if isinstance(retry, ModelResponse) else [retry]
        second = _final(result)
        if second is not None:
            left = self._claims(request, [*history, answer, correction], second)
            if left:
                result = [with_note(second, left) if m is second else m for m in result]
        return ModelResponse(result=[answer, correction, *result], structured_response=getattr(retry, "structured_response", None))

    def _correct(self, request, planned):
        answer, claims, history = planned
        if corrections_used(history) >= self.max_corrections:
            return None, ModelResponse(result=[with_note(answer, claims)])
        correction = HumanMessage(CORRECTION.format(items="\n".join(c.line() for c in claims)), name=CLAIM_CHECK)
        return correction, request.override(messages=[*request.messages, answer, correction])

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        response = handler(request)
        planned = self._plan(request, response)
        if planned is None:
            return response
        correction, next_request = self._correct(request, planned)
        if correction is None:
            return next_request
        return self._finish(request, planned[2], planned[0], correction, handler(next_request))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelCallResult:
        response = await handler(request)
        planned = self._plan(request, response)
        if planned is None:
            return response
        correction, next_request = self._correct(request, planned)
        if correction is None:
            return next_request
        return self._finish(request, planned[2], planned[0], correction, await handler(next_request))
