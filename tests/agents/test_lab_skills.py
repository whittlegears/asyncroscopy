"""The lab's own skills, written from a client (asyncroscopy.skills.lab), no model."""

import pytest

from asyncroscopy.skills import lab, list_skills, load_skill, select_skills
from asyncroscopy.skills.index import search

MANUAL = "# Tilt series\n\nTake a tilt series of the region.\n\n## Steps\n\n1. Tilt by 2 degrees and take an image each time.\n"


@pytest.fixture(autouse=True)
def lab_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_LAB_SKILLS", str(tmp_path / "lab"))
    return tmp_path / "lab"


def named(name):
    return next((s for s in list_skills() if s.name == name), None)


def test_a_plain_manual_is_saved_with_front_matter_and_is_used_like_any_skill(lab_folder):
    path = lab.save("Tilt Series!", MANUAL)
    assert path == lab_folder / "tilt_series" / "SKILL.md"
    skill = named("tilt_series")
    assert skill.source == "lab" and skill.available and skill.description == "Take a tilt series of the region."
    assert load_skill("tilt_series").startswith("# Tilt series")
    assert skill.acts  # a skill that does not say it only reads counts as acting
    assert any(p.skill == "tilt_series" for p, _ in search([("take a tilt series", 1.0)], list_skills()))


def test_a_skill_with_front_matter_keeps_it_and_takes_the_given_name():
    lab.save("survey", "---\nname: something_else\ndescription: Survey the grid.\nkeywords: [grid survey]\nacts: false\n---\n# Survey\n\nLook.")
    skill = named("survey")
    assert skill.keywords == ["grid survey"] and skill.acts is False and skill.description == "Survey the grid."


def test_a_skill_the_linter_refuses_is_not_written(lab_folder):
    with pytest.raises(ValueError):
        lab.save("unsafe", "# Unsafe\n\n" + "x " * 6000)  # longer than any prompt holds
    assert named("unsafe") is None and not list(lab_folder.glob("*/.*.tmp"))


def test_a_lab_skill_replaces_a_shipped_one_of_the_same_name_and_deleting_it_brings_it_back():
    shipped = named("focus").description
    lab.save("focus", "# Our focus\n\nOur lab's way to focus.")
    assert named("focus").source == "lab"
    lab.delete("focus")
    assert named("focus").source == "shipped" and named("focus").description == shipped


def test_a_shipped_skill_is_switched_off_not_deleted():
    with pytest.raises(KeyError):
        lab.delete("mapping")
    lab.switch("mapping", on=False)
    skill = named("mapping")
    assert skill.off and not skill.available and skill.problem == "switched off on this endpoint"
    assert "mapping" not in [s.name for s in select_skills("Map a 600 nm square around the current stage position.", {"run_job", "wait_for_job", "Mapper_reserve", "map_info"})]
    lab.switch("mapping", on=True)
    assert named("mapping").available


def test_deleting_a_lab_skill_forgets_that_it_was_switched_off():
    lab.save("tilt", MANUAL)
    lab.switch("tilt", on=False)
    lab.delete("tilt")
    assert "tilt" not in lab.switched_off()
