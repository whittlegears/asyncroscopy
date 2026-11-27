"""The skill registry reads a SKILL.md once and again only when it changes."""

import os

from asyncroscopy.skills import registry


def write_skill(folder, name, description="does a thing"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\nThe steps.\n", encoding="utf-8")
    return path


def parse_counter(monkeypatch):
    parses = []
    real = registry._parse
    monkeypatch.setattr(registry, "_parse", lambda path: (parses.append(path), real(path))[1])
    return parses


def test_repeated_listings_parse_nothing_new(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path / "lab"))
    write_skill(tmp_path / "lab" / "one", "lab_one")
    parses = parse_counter(monkeypatch)
    first = registry.list_skills()
    count = len(parses)
    assert any(s.name == "lab_one" for s in first)
    second = registry.list_skills()
    assert len(parses) == count and [s.name for s in first] == [s.name for s in second]


def test_a_changed_file_and_a_new_folder_are_seen(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path / "lab"))
    path = write_skill(tmp_path / "lab" / "one", "lab_one", "first words")
    assert next(s for s in registry.list_skills() if s.name == "lab_one").description == "first words"
    write_skill(tmp_path / "lab" / "one", "lab_one", "other words, longer")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert next(s for s in registry.list_skills() if s.name == "lab_one").description == "other words, longer"
    write_skill(tmp_path / "lab" / "two", "lab_two")
    assert {"lab_one", "lab_two"} <= {s.name for s in registry.list_skills()}


def test_a_skill_md_added_to_an_existing_folder_is_seen(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path / "lab"))
    (tmp_path / "lab" / "later").mkdir(parents=True)
    assert "lab_later" not in {s.name for s in registry.list_skills()}
    write_skill(tmp_path / "lab" / "later", "lab_later")
    assert "lab_later" in {s.name for s in registry.list_skills()}


def test_what_one_caller_marks_on_a_skill_does_not_reach_the_next(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path / "lab"))
    (tmp_path / "lab" / "one").mkdir(parents=True)
    (tmp_path / "lab" / "one" / "SKILL.md").write_text("---\nname: lab_one\ndescription: d\nrequires: [Nope_*]\n---\nx\n", encoding="utf-8")
    assert next(s for s in registry.list_skills({"Other_tool"}) if s.name == "lab_one").problem
    assert next(s for s in registry.list_skills() if s.name == "lab_one").problem is None


def test_a_broken_file_stays_listed_as_unreadable_and_mends_when_fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path / "lab"))
    folder = tmp_path / "lab" / "bad"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("no front matter here", encoding="utf-8")
    for _ in range(2):
        assert next(s for s in registry.list_skills() if s.name == "bad").problem
    write_skill(folder, "bad", "now fine")
    assert next(s for s in registry.list_skills() if s.name == "bad").problem is None
