"""Tests for the filesystem skill store."""

import json

import pytest

from asyncroscopy.skills.store import SkillStore, parse_frontmatter


def _payload(skill_id, text="# Body", **overrides):
    skill = {"id": skill_id, "name": skill_id, "description": "", "text": text, "enabled": True}
    skill.update(overrides)
    return skill


class TestReplaceAll:
    def test_replace_all_writes_agentskills_layout(self, tmp_path):
        store = SkillStore(tmp_path)
        report = store.replace_all([_payload("probe-alignment", "# Align the probe")])
        assert report == {"written": 1, "removed": 0}
        assert (tmp_path / "probe-alignment" / "SKILL.md").read_text(encoding="utf-8") == "# Align the probe"
        assert (tmp_path / "probe-alignment" / "provenance.json").is_file()

    def test_replace_all_removes_skills_missing_from_payload(self, tmp_path):
        store = SkillStore(tmp_path)
        store.replace_all([_payload("keep"), _payload("drop")])
        report = store.replace_all([_payload("keep")])
        assert report == {"written": 1, "removed": 1}
        assert not (tmp_path / "drop").exists()

    def test_replace_all_never_deletes_root_files(self, tmp_path):
        store = SkillStore(tmp_path)
        (tmp_path / "index.db").write_text("precious", encoding="utf-8")
        store.replace_all([])
        assert (tmp_path / "index.db").read_text(encoding="utf-8") == "precious"

    def test_a_traversing_id_is_refused(self, tmp_path):
        store = SkillStore(tmp_path)
        with pytest.raises(ValueError):
            store.replace_all([_payload("../escape")])

    def test_provenance_round_trips(self, tmp_path):
        store = SkillStore(tmp_path)
        store.replace_all([_payload("tracked", version=3, agent_authored=True, source="marketplace")])
        record = store.read("tracked")
        assert record.version == 3
        assert record.agent_authored is True
        assert record.source == "marketplace"
        provenance = json.loads((tmp_path / "tracked" / "provenance.json").read_text(encoding="utf-8"))
        assert "synced_at" in provenance


class TestFrontmatter:
    def test_the_synced_name_and_description_win_over_frontmatter(self, tmp_path):
        store = SkillStore(tmp_path)
        text = "---\nname: Frontmatter Name\ndescription: Frontmatter blurb.\n---\n# Steps"
        store.replace_all([_payload("probe-alignment", text, name="GUI Name", description="GUI blurb.")])
        record = store.read("probe-alignment")
        assert record.name == "GUI Name"
        assert record.description == "GUI blurb."

    def test_frontmatter_supplies_name_and_description_when_the_sync_carries_none(self, tmp_path):
        store = SkillStore(tmp_path)
        text = "---\nname: Probe Alignment\ndescription: Align before acquiring.\n---\n# Steps"
        store.replace_all([_payload("probe-alignment", text, name="", description="")])
        record = store.read("probe-alignment")
        assert record.name == "Probe Alignment"
        assert record.description == "Align before acquiring."

    def test_missing_frontmatter_falls_back_to_first_line(self, tmp_path):
        store = SkillStore(tmp_path)
        store.replace_all([_payload("bare", "# The heading line\nbody")])
        record = store.read("bare")
        assert record.name == "bare"
        assert record.description == "The heading line"

    def test_parse_frontmatter_ignores_non_scalar_lines(self):
        name, description = parse_frontmatter("---\nname: X\nlist:\n- item\n---\nbody")
        assert name == "X"
        assert description == ""
