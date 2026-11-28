"""Agent skills: short procedures matched to a request and put in an agent's prompt. The skills themselves are in agent/skills (see agent/README.md); this package finds and loads them."""

from .registry import Skill, list_skills, load_skill, select_skills, skills_text

__all__ = ["Skill", "list_skills", "load_skill", "select_skills", "skills_text"]
