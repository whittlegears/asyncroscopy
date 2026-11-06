"""asyncroscopy.registry.Registry: one lookup for every kind of named plug-in class."""

import sys
import types

import pytest

from asyncroscopy.plugins import Plugins
from asyncroscopy.registry import Registry


class Thing:
    name = ""


def _registry(builtins=None):
    return Registry("asyncroscopy.test_no_such_group", builtins, kind="thing", plugins=Plugins("ASYNCROSCOPY_TEST_NO_PLUGINS"))


def test_register_get_and_unregister():
    things = _registry()

    @things.register
    class A(Thing):
        name = "a"

    assert things.get("a") is A
    assert "a" in things and things.all() == {"a": A}
    things.unregister("a")
    with pytest.raises(ValueError, match="unknown thing 'a'; registered: \\[\\]"):
        things.get("a")


def test_a_class_without_a_name_is_refused():
    with pytest.raises(ValueError, match="needs a name"):
        _registry().register(Thing)


def test_a_builtin_module_is_imported_only_when_its_name_is_asked_for(monkeypatch):
    things = _registry({"b": "asyncroscopy_test_builtin_b"})
    module = types.ModuleType("asyncroscopy_test_builtin_b")
    imported = []

    def fake_import(name):
        imported.append(name)

        class B(Thing):
            name = "b"

        things.register(B)
        return module

    monkeypatch.setattr("asyncroscopy.registry.importlib.import_module", fake_import)
    assert "b" not in things
    assert things.get("b").name == "b"
    assert imported == ["asyncroscopy_test_builtin_b"]


def test_a_broken_builtin_is_named_in_the_unknown_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "asyncroscopy_test_missing_module", raising=False)
    things = _registry({"c": "asyncroscopy_test_missing_module"})
    with pytest.raises(ValueError, match="asyncroscopy_test_missing_module.*failed to load"):
        things.get("c")
    assert list(things.errors()) == ["built-in module 'asyncroscopy_test_missing_module'"]


def test_each_registry_keeps_its_own_state():
    first, second = _registry(), _registry()

    @first.register
    class D(Thing):
        name = "d"

    assert "d" in first and "d" not in second
