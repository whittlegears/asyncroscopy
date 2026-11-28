"""Skills as passages an agent retrieves while it works.

A skill's manual is split into passages: its ``##`` sections, and a long section into
groups of its list items or paragraphs. Each passage keeps the skill's title and its
section heading, so it reads on its own. ``search`` ranks passages for a query with
BM25 (word frequency weighed by how rare a word is across all passages) plus how well
the passage's skill matches (``registry.match_score``), so the right passage of the
right skill comes first.

The agent graphs (``asyncroscopy.agents.skill_middleware``) search before every model
call with the request and what the tools last said, so a failed call brings the advice
for that error and a stitched map brings what to do with a map, step by step. This is
retrieval in process over a few dozen short texts: no model call and no embedding,
well under a millisecond, and the same conversation always gets the same passages.

``rank_texts`` ranks any short texts (memory notes, lessons) with the same BM25.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from asyncroscopy.skills.registry import NEGATED_WORD, STOP, Skill, _stamp, _stem, match_score, read_front_matter

MAX_PASSAGE_CHARS = 600  # a section longer than this is split by its list items or paragraphs
PER_SKILL = 2  # passages one skill gives a step, so the best of several skills come rather than one manual
MIN_SCORE = 3.0  # below this a passage shares too little with the query to help
SKILL_WEIGHT = 0.5  # how much the skill's own match (keywords, name, description) adds to each of its passages
HEADING_WEIGHT = 2  # a heading word counts as this many words of text
K1, B = 1.2, 0.75  # BM25's usual constants

_passages: dict[str, tuple[tuple[int, int] | None, tuple[Passage, ...]]] = {}


@dataclass(frozen=True)
class Passage:
    skill: str  # the skill's name
    title: str  # the manual's first heading
    heading: str  # the section's heading, "" for text before the first section
    text: str
    order: int  # its place in the manual

    @property
    def label(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title


def tokens(text: str) -> list[str]:
    """Stemmed content words of ``text`` in order, repeats kept (``registry.content_words`` keeps a set)."""
    return [_stem(w) for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if w not in STOP]


ITEM = re.compile(r"(?m)^(?=(?:\d+\.|[-*])\s)")


def _pieces(text: str) -> list[str]:
    """A long section as pieces of at most MAX_PASSAGE_CHARS where it can: its top-level list items, or its paragraphs."""
    parts = [p.strip() for p in ITEM.split(text) if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for part in parts:
        if out and len(out[-1]) + len(part) + 1 <= MAX_PASSAGE_CHARS:
            out[-1] += "\n" + part
        else:
            out.append(part)
    return out


def split_manual(skill: str, body: str) -> tuple[Passage, ...]:
    """A skill's manual (``body``, front matter already removed) as passages."""
    title_match = re.search(r"(?m)^#\s+(.+)$", body)
    title = title_match.group(1).strip() if title_match else skill
    if title_match:
        body = body[: title_match.start()] + body[title_match.end():]
    out: list[Passage] = []
    for section in re.split(r"(?m)^(?=##\s)", body):
        section = section.strip()
        if not section:
            continue
        heading = ""
        if section.startswith("##"):
            first, _, section = section.partition("\n")
            heading = first.lstrip("#").strip()
            section = section.strip()
            if not section:
                continue
        for piece in [section] if len(section) <= MAX_PASSAGE_CHARS else _pieces(section):
            out.append(Passage(skill, title, heading, piece, len(out)))
    return tuple(out)


def passages_of(skill: Skill) -> tuple[Passage, ...]:
    """A skill's passages, split once per version of its file."""
    stamp = _stamp(skill.path)
    key = f"{skill.name}\0{skill.path}"
    cached = _passages.get(key)
    if cached is None or cached[0] != stamp or stamp is None:
        cached = (stamp, split_manual(skill.name, read_front_matter(skill.path)[1]))
        _passages[key] = cached
    return cached[1]


class Bm25:
    """BM25 over a few documents, each a list of tokens."""

    def __init__(self, docs: Sequence[list[str]]) -> None:
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.mean = (sum(self.lengths) / len(docs)) if docs else 0.0
        frequency = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def score(self, i: int, query: Iterable[str]) -> float:
        doc, length, total = self.docs[i], self.lengths[i], 0.0
        for term in set(query):
            tf = doc.get(term, 0)
            if tf:
                total += self.idf[term] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / (self.mean or 1)))
        return total


def query_tokens(parts: Sequence[tuple[str, float]]) -> list[tuple[list[str], float]]:
    """Each weighted query part as tokens, with words it negates ("without EDS") left out."""
    return [(tokens(NEGATED_WORD.sub(" ", text)), weight) for text, weight in parts if text and weight > 0]


def search(parts: Sequence[tuple[str, float]], skills: Sequence[Skill], *, k: int = 3, min_score: float = MIN_SCORE,
           boost: dict[str, float] | None = None, skip: set[str] | None = None, per_skill: int = PER_SKILL) -> list[tuple[Passage, float]]:
    """The ``k`` passages of ``skills`` that best answer the query, best first, each with its score.

    ``parts``: (text, weight) pairs, such as the request at 1.0 and the latest tool reply at 0.6. At most per_skill passages come from one skill.
    ``boost``: points added to every passage of a skill, for one an event calls for (a failed call
    and ``recover_from_errors``). ``skip``: skills whose passages are not wanted (already in the prompt whole).
    """
    skip, boost = skip or set(), boost or {}
    pool = [p for s in skills if s.name not in skip and s.available for p in passages_of(s)]
    if not pool:
        return []
    index = Bm25([tokens(p.text) + tokens(p.heading) * HEADING_WEIGHT + tokens(p.title) for p in pool])
    queries = query_tokens(parts)
    by_name = {s.name: s for s in skills}
    skill_match = {s.name: sum(w * match_score(s, text) for text, w in parts if text) for s in skills}
    scored = []
    for i, passage in enumerate(pool):
        text_score = sum(w * index.score(i, q) for q, w in queries)
        if text_score <= 0:
            continue  # a passage sharing no word with the query is never chosen for its skill's name alone
        score = text_score + SKILL_WEIGHT * skill_match.get(passage.skill, 0.0) + boost.get(passage.skill, 0.0)
        if score >= min_score and (by_name[passage.skill].matched or passage.skill in boost):
            scored.append((score, -passage.order, passage))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    out: list[tuple[Passage, float]] = []
    taken: Counter[str] = Counter()
    for score, _order, passage in scored:
        if taken[passage.skill] < per_skill:
            taken[passage.skill] += 1
            out.append((passage, round(score, 3)))
        if len(out) == k:
            break
    return out


def passages_text(passages: Sequence[Passage], budget: int) -> tuple[str, list[Passage]]:
    """The passages written for a prompt, in their manual's order within each skill, while they fit in ``budget``.
    Returns (text, the passages that fit)."""
    fitted, used = [], 0
    for passage in passages:
        size = len(passage.text) + len(passage.label) + 8
        if used + size > budget:
            continue
        fitted.append(passage)
        used += size
    first: dict[str, int] = {}
    for i, passage in enumerate(fitted):
        first.setdefault(passage.skill, i)  # skills in the order of their best passage
    fitted.sort(key=lambda p: (first[p.skill], p.order))
    # The label leaves out the skill's name, because a model may call the name as a tool.
    return "\n\n".join(f"--- {p.label} ---\n{p.text}" for p in fitted), fitted


def rank_texts(texts: Sequence[str], parts: Sequence[tuple[str, float]], min_score: float = 0.0) -> list[int]:
    """Indexes of ``texts`` that share words with the query, best first (ties keep the given order)."""
    if not texts:
        return []
    index = Bm25([tokens(t) for t in texts])
    queries = query_tokens(parts)
    scored = [(sum(w * index.score(i, q) for q, w in queries), -i) for i in range(len(texts))]
    return [-neg for score, neg in sorted(scored, reverse=True) if score > min_score]
