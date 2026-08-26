"""Tests for the usage log and the capped rank prior (Phase E)."""

import re

import numpy as np
import pytest

from asyncroscopy.skills import SkillsService
from asyncroscopy.skills.index import SkillIndex
from asyncroscopy.skills.store import SkillStore
from asyncroscopy.skills.usage import (
    UsageLog,
    apply_usage_prior,
    prior_cap,
    prior_damping,
    task_hash,
)

dimensions = 64


class StubEmbedder:
    def __init__(self, model="stub-embed"):
        self.model = model

    def embed(self, texts):
        vectors = np.zeros((len(texts), dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                vectors[row][hash(word) % dimensions] += 1.0
        return vectors


def _service(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    index = SkillIndex(root / "index.db", StubEmbedder())
    return SkillsService(store, index, UsageLog(root / "usage.db"))


def _payload(skill_id, text, description="", enabled=True):
    return {
        "id": skill_id,
        "name": skill_id,
        "description": description,
        "text": text,
        "enabled": enabled,
    }


class TestUsageLog:
    def test_record_and_stats_round_trip(self, tmp_path):
        log = UsageLog(tmp_path / "usage.db")
        assert log.record(["focus", "focus", "report"], task_hash("align it"), True) == 3
        assert log.record(["focus"], task_hash("again"), False) == 1
        stats = log.stats()
        assert stats["focus"]["loads"] == 3
        assert stats["focus"]["successes"] == 2
        assert stats["focus"]["failures"] == 1
        assert stats["focus"]["last_used_at"] > 0
        assert stats["report"] == {
            "loads": 1,
            "successes": 1,
            "failures": 0,
            "last_used_at": stats["report"]["last_used_at"],
        }

    def test_blank_ids_are_dropped_and_an_empty_record_is_a_no_op(self, tmp_path):
        log = UsageLog(tmp_path / "usage.db")
        assert log.record(["", "  "], task_hash("x"), True) == 0
        assert log.stats() == {}

    def test_priors_are_smoothed_and_signed(self, tmp_path):
        log = UsageLog(tmp_path / "usage.db")
        log.record(["winner"], "t1", True)
        log.record(["winner"], "t2", True)
        log.record(["loser"], "t3", False)
        priors = log.priors()
        assert 0 < priors["winner"] == 2 / (2 + prior_damping)
        assert priors["loser"] == -1 / (1 + prior_damping)


class TestCappedPrior:
    def test_the_prior_can_flip_a_near_tie(self):
        results = [
            {"id": "a", "score": 0.0300, "matched_section": "metadata"},
            {"id": "b", "score": 0.0295, "matched_section": "metadata"},
        ]
        reranked = apply_usage_prior(results, {"b": 1.0})
        assert [entry["id"] for entry in reranked] == ["b", "a"]

    def test_the_cap_holds_a_clear_semantic_winner_stays_on_top(self):
        results = [
            {"id": "right", "score": 0.0300, "matched_section": "metadata"},
            {"id": "popular", "score": 0.0200, "matched_section": "metadata"},
        ]
        reranked = apply_usage_prior(results, {"popular": 1.0, "right": -1.0})
        assert [entry["id"] for entry in reranked] == ["right", "popular"]
        assert reranked[0]["score"] == pytest.approx(0.03 * (1 - prior_cap))
        assert reranked[1]["score"] == pytest.approx(0.02 * (1 + prior_cap))

    def test_priors_beyond_unit_range_are_clamped(self):
        results = [{"id": "a", "score": 0.02, "matched_section": "metadata"}]
        [entry] = apply_usage_prior(results, {"a": 50.0})
        assert entry["score"] == pytest.approx(0.02 * (1 + prior_cap))
        assert entry["usage_prior"] == 1.0


class TestServiceRanking:
    plasmon = _payload(
        "plasmon-fit",
        "# Plasmon\nFit the plasmon peak with a Drude model on the spectrum.",
        description="Fit plasmon peaks in low loss spectra.",
    )
    drude = _payload(
        "drude-notes",
        "# Drude\nNotes on the Drude model for the plasmon peak of a spectrum.",
        description="Drude model notes for plasmon peaks in spectra.",
    )

    def test_logged_outcomes_shift_a_near_tie(self, tmp_path):
        service = _service(tmp_path)
        service.sync_from_payload([self.plasmon, self.drude])
        query = "fit the plasmon peak of this spectrum"
        baseline = [entry["id"] for entry in service.find_skills(query, k=2)]
        trailing = baseline[1]
        for run in range(20):
            service.record_usage([trailing], f"t{run}", True)
        reranked = [entry["id"] for entry in service.find_skills(query, k=2)]
        assert reranked[0] == trailing
        assert set(reranked) == set(baseline)

    def test_usage_never_resurrects_a_disabled_skill(self, tmp_path):
        service = _service(tmp_path)
        service.sync_from_payload([self.plasmon, self.drude])
        for run in range(20):
            service.record_usage(["drude-notes"], f"t{run}", True)
        disabled = dict(self.drude, enabled=False)
        service.sync_from_payload([self.plasmon, disabled])
        results = service.find_skills("fit the plasmon peak of this spectrum", k=5)
        assert all(entry["id"] != "drude-notes" for entry in results)

    def test_usage_report_includes_never_loaded_skills(self, tmp_path):
        service = _service(tmp_path)
        service.sync_from_payload([self.plasmon, self.drude])
        service.record_usage(["plasmon-fit"], "t1", True)
        service.record_usage(["plasmon-fit"], "t2", False)
        report = service.usage_report()
        by_id = {entry["id"]: entry for entry in report}
        assert by_id["plasmon-fit"]["loads"] == 2
        assert by_id["plasmon-fit"]["successes"] == 1
        assert by_id["plasmon-fit"]["failures"] == 1
        assert by_id["drude-notes"]["loads"] == 0
        assert by_id["drude-notes"]["last_used_at"] is None
        assert report[0]["id"] == "plasmon-fit"

    def test_record_usage_without_a_log_refuses(self, tmp_path):
        root = tmp_path / "skills"
        service = SkillsService(
            SkillStore(root), SkillIndex(root / "index.db", StubEmbedder()), None
        )
        with pytest.raises(RuntimeError, match="no usage log"):
            service.record_usage(["x"], "t", True)


class TestTaskHash:
    def test_stable_and_short(self):
        assert task_hash("focus the probe") == task_hash("focus the probe")
        assert len(task_hash("focus the probe")) == 16
        assert task_hash("a") != task_hash("b")
