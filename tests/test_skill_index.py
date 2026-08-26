"""Tests for the hybrid skill index, with a deterministic stub embedder."""

import re

import numpy as np
import pytest

from asyncroscopy.skills import SkillsService
from asyncroscopy.skills.index import SkillIndex, SkillIndexUnavailable, chunk_skill
from asyncroscopy.skills.store import SkillRecord, SkillStore

dimensions = 64


class StubEmbedder:
    """Bag-of-words hash embedding: shared words give aligned vectors."""

    def __init__(self, model="stub-embed"):
        self.model = model
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                vectors[row][hash(word) % dimensions] += 1.0
        return vectors


class DownEmbedder(StubEmbedder):
    def embed(self, texts):
        raise SkillIndexUnavailable("stub embedder is down")


def _record(skill_id, text, name=None, description="", enabled=True):
    return SkillRecord(
        id=skill_id,
        name=name or skill_id,
        description=description,
        text=text,
        enabled=enabled,
    )


def _service(tmp_path, embedder):
    store = SkillStore(tmp_path / "skills")
    index = SkillIndex(tmp_path / "skills" / "index.db", embedder)
    return SkillsService(store, index)


alignment = _record(
    "probe-alignment",
    "# Focus procedure\nFocus the probe and correct the stigmator before every acquisition.",
    name="Probe alignment",
    description="How to focus and align the probe before acquiring an image.",
)
eds_skill = _record(
    "eds-quantification",
    "# Quantify\nIntegrate peaks and report weight percent from spectra.",
    name="EDS quantification",
    description="Turn spectra into composition numbers.",
)
report_skill = _record(
    "report-checklist",
    "# Reports\nEvery analysis report needs axes, units, and provenance.",
    name="Report checklist",
    description="What a finished analysis report must contain.",
)


class TestChunking:
    def test_metadata_is_always_the_first_chunk(self):
        chunks = chunk_skill(alignment)
        assert chunks[0][0] == "metadata"
        assert "Probe alignment" in chunks[0][1]

    def test_headings_split_the_body_into_sections(self):
        record = _record("multi", "intro\n# One\nfirst\n# Two\nsecond")
        sections = [section for section, _ in chunk_skill(record)]
        assert sections == ["metadata", "body", "One", "Two"]


class TestSync:
    def test_sync_embeds_metadata_and_heading_chunks(self, tmp_path):
        embedder = StubEmbedder()
        index = SkillIndex(tmp_path / "index.db", embedder)
        report = index.sync([alignment])
        assert report["skills"] == 1
        assert report["chunks_embedded"] == len(chunk_skill(alignment))

    def test_unchanged_skills_are_not_reembedded(self, tmp_path):
        embedder = StubEmbedder()
        index = SkillIndex(tmp_path / "index.db", embedder)
        index.sync([alignment])
        calls_before = len(embedder.calls)
        report = index.sync([alignment])
        assert report["chunks_embedded"] == 0
        assert len(embedder.calls) == calls_before

    def test_a_changed_skill_is_reembedded(self, tmp_path):
        embedder = StubEmbedder()
        index = SkillIndex(tmp_path / "index.db", embedder)
        index.sync([alignment])
        changed = _record(alignment.id, alignment.text + "\nNew line.", name=alignment.name)
        report = index.sync([changed])
        assert report["chunks_embedded"] > 0

    def test_a_new_embedding_model_forces_a_full_reembed(self, tmp_path):
        index = SkillIndex(tmp_path / "index.db", StubEmbedder(model="one"))
        index.sync([alignment])
        replacement = SkillIndex(tmp_path / "index.db", StubEmbedder(model="two"))
        report = replacement.sync([alignment])
        assert report["chunks_embedded"] == len(chunk_skill(alignment))


class TestSearch:
    def test_search_ranks_the_matching_skill_first(self, tmp_path):
        index = SkillIndex(tmp_path / "index.db", StubEmbedder())
        index.sync([alignment, eds_skill, report_skill])
        results = index.search("how do I focus the probe before acquisition")
        assert results[0]["id"] == "probe-alignment"

    def test_exact_jargon_is_found_by_keyword_fusion(self, tmp_path):
        index = SkillIndex(tmp_path / "index.db", StubEmbedder())
        haadf = _record("haadf-imaging", "# HAADF\nUse the HAADF detector at 200 kV.")
        index.sync([haadf, eds_skill, report_skill])
        results = index.search("HAADF")
        assert results[0]["id"] == "haadf-imaging"

    def test_disabled_skills_never_surface(self, tmp_path):
        index = SkillIndex(tmp_path / "index.db", StubEmbedder())
        disabled = _record(alignment.id, alignment.text, name=alignment.name, enabled=False)
        index.sync([disabled, eds_skill])
        results = index.search("focus the probe")
        assert all(result["id"] != "probe-alignment" for result in results)

    def test_a_deleted_index_rebuilds_to_the_same_results(self, tmp_path):
        db = tmp_path / "index.db"
        index = SkillIndex(db, StubEmbedder())
        index.sync([alignment, eds_skill])
        first = index.search("focus the probe")
        db_bytes = db.read_bytes()
        assert db_bytes
        rebuilt = SkillIndex(tmp_path / "fresh.db", StubEmbedder())
        rebuilt.sync([alignment, eds_skill])
        assert rebuilt.search("focus the probe") == first

    def test_search_refuses_with_a_reason_when_the_embedder_is_down(self, tmp_path):
        index = SkillIndex(tmp_path / "index.db", StubEmbedder())
        index.sync([alignment])
        index.embedder = DownEmbedder()
        with pytest.raises(SkillIndexUnavailable, match="down"):
            index.search("focus")

    def test_an_empty_index_refuses_rather_than_returning_nothing(self, tmp_path):
        index = SkillIndex(tmp_path / "index.db", StubEmbedder())
        with pytest.raises(SkillIndexUnavailable, match="empty"):
            index.search("anything")


class TestService:
    def test_service_round_trip_sync_search_load(self, tmp_path):
        service = _service(tmp_path, StubEmbedder())
        report = service.sync_from_payload(
            [
                {"id": r.id, "name": r.name, "description": r.description, "text": r.text, "enabled": True}
                for r in (alignment, eds_skill)
            ]
        )
        assert report["written"] == 2
        results = service.find_skills("focus the probe before acquisition")
        assert results[0]["id"] == "probe-alignment"
        assert results[0]["name"]
        assert "Focus the probe" in service.load_skill("probe-alignment")

    def test_loading_an_unknown_skill_names_the_known_ones(self, tmp_path):
        service = _service(tmp_path, StubEmbedder())
        service.sync_from_payload(
            [{"id": "only-one", "name": "Only", "description": "", "text": "# T", "enabled": True}]
        )
        with pytest.raises(KeyError, match="only-one"):
            service.load_skill("missing")
