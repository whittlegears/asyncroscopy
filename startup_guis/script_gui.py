#!/usr/bin/env python
"""One startup window for every startup script, built from the YAML it opens.

The config list is every YAML in configs/ that a startup script runs (startup_scripts/registry.py).
Opening one picks its script, and the form is that file's own keys: each top-level mapping is a
section, and each mapping of named entries (such as devices) is a list of switches. The script's
SETTINGS add the type, choices and defaults of the keys it declares, and its load_config decides
what is valid. Display text comes from ui_hints.py and is optional, so a new key, device, config or
script needs no change here.

    python startup_guis/script_gui.py [config.yaml]
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from startup_guis import ui_hints  # noqa: E402
from startup_guis.qt_compat import (  # noqa: E402
    HORIZONTAL,
    NO_WRAP,
    POINTING_HAND_CURSOR,
    VERTICAL,
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTimer,
    QVBoxLayout,
    QWidget,
    app_exec,
)
from startup_guis.shared import (  # noqa: E402
    BODY_FONT,
    COLORS,
    CONFIG_DIR,
    GENERATED_CONFIG_DIR,
    TEXT_FONT,
    TITLE_FONT,
    CheckBox,
    CollapsibleSection,
    ManagedCommand,
    action_button,
    append_terminal_text,
    apply_theme,
    configure_splitter,
    configure_terminal,
    load_yaml,
    scrollable,
    section_label,
    set_status_badge,
    status_badge,
    write_yaml,
    yaml_text,
)
from startup_scripts import registry  # noqa: E402
from startup_scripts.config_schema import ENABLED_KEY, MISSING, Setting, coerce, get_path, set_path  # noqa: E402

TEXT_KINDS = ("str", "int", "path", "dir", "list", "value")


@dataclass(frozen=True)
class Field:
    """One key the window shows: declared by the script, or found in the YAML ("value": any YAML scalar)."""

    path: str
    kind: str
    default: Any = None
    required: bool = False
    choices: tuple[str, ...] = ()

    @property
    def section(self) -> str:
        return self.path.split(".", 1)[0] if "." in self.path or self.kind == "entries" else ""


def infer_kind(value: Any) -> str:
    """The widget kind for a key the script does not declare, from its YAML value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list" if all(not isinstance(item, (dict, list)) for item in value) else "yaml"
    if isinstance(value, dict):
        return "entries" if value and all(isinstance(item, dict) or item is None for item in value.values()) else "yaml"
    return "value"


def form_fields(settings: list[Setting], raw: Any) -> list[Field]:
    """Every key to show, in the YAML's own order, then the declared keys the file leaves out."""
    declared = {setting.path: setting for setting in settings}
    paths: list[str] = []
    for key, value in (raw.items() if isinstance(raw, dict) else []):
        key = str(key)
        if key in declared or not isinstance(value, dict) or infer_kind(value) == "entries":
            paths.append(key)
        else:
            paths.extend(f"{key}.{sub}" for sub in value)
    paths.extend(path for path in declared if path not in paths)
    fields = []
    for path in paths:
        setting = declared.get(path)
        if setting is not None:
            fields.append(Field(path, setting.kind, setting.default, setting.required, setting.choices))
        else:
            fields.append(Field(path, infer_kind(get_path(raw, path, None))))
    return fields


def project_path_text(path: Path | str) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def as_bool(value, default: bool) -> bool:
    """A YAML boolean as the scripts read it ("False", "0", "off" are false); unreadable values show the default."""
    try:
        result = coerce(Setting("value", "bool"), value)
    except ValueError:
        return default
    return default if result is None else result


def display_text(field: Field, value) -> str | bool:
    """What a widget shows for a YAML value."""
    if field.kind == "bool":
        return as_bool(value, bool(field.default))
    if value is None or value is MISSING:
        return ""
    if field.kind == "list":
        return ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    if field.kind == "yaml":
        return yaml.safe_dump(value, sort_keys=False).strip() if value not in ({}, []) else ""
    if field.kind in ("path", "dir"):
        return project_path_text(value)
    return str(value)


def yaml_value(field: Field, shown: str | bool):
    """The YAML value for what a widget shows; text that does not parse is kept so validation can report it."""
    if field.kind == "bool":
        return bool(shown)
    text = str(shown).strip()
    if text == "":
        return None
    if field.kind == "int":
        try:
            return int(text)
        except ValueError:
            return text
    if field.kind == "list":
        return [item.strip() for item in text.split(",") if item.strip()]
    if field.kind in ("yaml", "value"):
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text
    return text


def delete_path(raw: dict, path: str) -> None:
    *parents, last = path.split(".")
    node = raw
    for part in parents:
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        node.pop(last, None)


class ScriptGui(QMainWindow):
    def __init__(self, config_path: Path | str | None = None):
        super().__init__()
        self.script: ModuleType | None = None
        self.resize(1280, 960)
        self.setMinimumSize(880, 560)
        self.command = ManagedCommand(self.enqueue_output, self.process_done)
        self.raw: dict = {}
        self.loaded_path: Path | None = None
        self.fields: dict[str, tuple[Field, object]] = {}  # path -> (field, widget)
        self.shown: dict[str, str | bool] = {}  # path -> what the widget showed right after loading
        self.entry_checks: dict[str, dict[str, CheckBox]] = {}  # entries path -> name -> switch
        self.entry_shown: dict[str, dict[str, bool]] = {}
        self.sections: list[QWidget] = []
        self.badge_error = False
        self.validate_timer = QTimer(self)
        self.validate_timer.setSingleShot(True)
        self.validate_timer.setInterval(300)
        self.validate_timer.timeout.connect(self.refresh_preview)
        self.build()
        self.load_config_from_path(Path(config_path) if config_path else CONFIG_DIR / ui_hints.DEFAULT_CONFIG)

    @property
    def script_name(self) -> str:
        return registry.script_name(self.script) if self.script else ""

    @property
    def generated_path(self) -> Path:
        return GENERATED_CONFIG_DIR / f"{self.script_name or 'startup'}_gui.yaml"

    @property
    def status_patterns(self) -> list:
        return ui_hints.STATUS_PATTERNS.get(self.script_name, [])

    # ---------------------------------------------------------------- layout

    def build(self) -> None:
        apply_theme(self)
        root = QSplitter(VERTICAL)
        top = QSplitter(HORIZONTAL)
        for splitter in (root, top):
            configure_splitter(splitter)
        controls, preview, terminal = QWidget(), QWidget(), QWidget()
        top.addWidget(scrollable(controls))
        top.addWidget(preview)
        top.setSizes([760, 480])
        root.addWidget(top)
        root.addWidget(terminal)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([560, 400])
        container = QWidget()
        container.setObjectName("appRoot")
        wrapper = QVBoxLayout(container)
        wrapper.setContentsMargins(14, 14, 14, 14)
        wrapper.addWidget(root)
        self.setCentralWidget(container)
        self.build_controls(controls)
        self.build_preview(preview)
        self.build_terminal(terminal)

    def build_controls(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(10)
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        self.title = QLabel()
        self.title.setFont(TITLE_FONT)
        title_row.addWidget(self.title)
        self.badge = status_badge()
        self.badge.setVisible(False)
        title_row.addWidget(self.badge)
        title_row.addStretch()
        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(260)
        for path, script in registry.config_files(CONFIG_DIR):
            self.config_combo.addItem(f"{path.name}  ·  {ui_hints.title(registry.script_name(script))}", str(path))
        self.config_combo.currentIndexChanged.connect(self.config_chosen)
        title_row.addWidget(self.config_combo)
        layout.addLayout(title_row)

        self.form = QVBoxLayout()
        self.form.setSpacing(10)
        layout.addLayout(self.form)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        start = action_button("Start", "#1f7a35", "#2ea043")
        stop = action_button("Stop", "#b42318", "#dc2626")
        load = QPushButton("Load config file")
        save = QPushButton("Save current config")
        start.clicked.connect(self.start)
        stop.clicked.connect(lambda: self.command.stop())
        load.clicked.connect(self.read_config)
        save.clicked.connect(self.save_config)
        for button in (start, stop, load, save):
            button.setFont(BODY_FONT)
            button.setMinimumHeight(40)
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addStretch()

    def build_form(self) -> None:
        """Replace the form with one for the loaded YAML and its script."""
        for widget in self.sections:
            self.form.removeWidget(widget)
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        self.sections, self.fields, self.entry_checks, self.entry_shown = [], {}, {}, {}
        sections: dict[str, CollapsibleSection] = {}
        for field in form_fields(self.script.SETTINGS, self.raw):
            if field.kind == "entries":
                self.add_entries(field)
                continue
            if field.section not in sections:
                sections[field.section] = CollapsibleSection(ui_hints.section_title(field.section), expanded=False)
                self.form.addWidget(sections[field.section])
                self.sections.append(sections[field.section])
            self.add_field(sections[field.section], field)

    def add_field(self, section: CollapsibleSection, field: Field) -> None:
        title = ui_hints.label(field.path)
        if field.kind == "bool":
            widget = CheckBox(title)
            widget.stateChanged.connect(self.changed)
            section.form.addRow("", widget)
        elif field.kind == "yaml":
            widget = QTextEdit()
            widget.setFont(TEXT_FONT)
            widget.setLineWrapMode(NO_WRAP)
            widget.setMinimumHeight(110)
            widget.textChanged.connect(self.changed)
            section.form.addRow(section_label(title))
            section.form.addRow(widget)
        else:
            if field.choices or field.kind in ("path", "dir"):
                widget = QComboBox()
                widget.setEditable(True)
                widget.addItems(list(field.choices))
                widget.currentTextChanged.connect(self.changed)
            else:
                widget = QLineEdit()
                widget.textChanged.connect(self.changed)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(widget, 1)
            if field.kind in ("path", "dir"):
                browse = QPushButton("Browse")
                browse.setCursor(POINTING_HAND_CURSOR)
                browse.clicked.connect(lambda _=False, f=field, w=widget: self.browse(f, w))
                row_layout.addWidget(browse)
            section.form.addRow(title, row)
        help_text = ui_hints.HELP.get(field.path)
        if help_text:
            widget.setToolTip(help_text)
            if hasattr(widget, "setPlaceholderText"):
                widget.setPlaceholderText(help_text)
            elif hasattr(widget, "lineEdit") and widget.lineEdit() is not None:
                widget.lineEdit().setPlaceholderText(help_text)
        self.fields[field.path] = (field, widget)

    def add_entries(self, field: Field) -> None:
        """A switch per named entry, on unless the entry says enabled: false."""
        section = CollapsibleSection(ui_hints.section_title(field.path), layout_cls=QGridLayout, expanded=True)
        self.form.addWidget(section)
        self.sections.append(section)
        checks, shown = {}, {}
        entries = get_path(self.raw, field.path, {})
        for index, (key, entry) in enumerate(entries.items() if isinstance(entries, dict) else []):
            summary = ui_hints.entry_summary(entry)
            checkbox = CheckBox(f"{key}  ·  {summary}" if summary and summary != key else str(key))
            enabled = as_bool(entry.get(ENABLED_KEY, True), True) if isinstance(entry, dict) else True
            checkbox.setChecked(enabled)
            checkbox.stateChanged.connect(self.changed)
            section.form.addWidget(checkbox, index // 2, index % 2)
            checks[key], shown[key] = checkbox, enabled
        self.entry_checks[field.path], self.entry_shown[field.path] = checks, shown

    def build_preview(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(section_label("Config that Start will use"))
        self.yaml_preview = QTextEdit()
        self.yaml_preview.setFont(TEXT_FONT)
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setLineWrapMode(NO_WRAP)
        layout.addWidget(self.yaml_preview)
        self.validity = QLabel()
        self.validity.setWordWrap(True)
        layout.addWidget(self.validity)

    def build_terminal(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(section_label("Terminal output"))
        self.output = QTextEdit()
        configure_terminal(self.output)
        layout.addWidget(self.output)

    # ---------------------------------------------------------------- field values

    def get_shown(self, path: str) -> str | bool:
        field, widget = self.fields[path]
        if field.kind == "bool":
            return widget.isChecked()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return widget.text()

    def set_shown(self, path: str, value) -> None:
        field, widget = self.fields[path]
        shown = display_text(field, value)
        if field.kind == "bool":
            widget.setChecked(bool(shown))
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(shown)
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(shown)
        else:
            widget.setText(shown)

    def browse(self, field: Field, widget: QComboBox) -> None:
        start = str(PROJECT_DIR / widget.currentText()) if widget.currentText() else str(PROJECT_DIR)
        title = f"Select {ui_hints.label(field.path)}"
        if field.kind == "dir":
            path = QFileDialog.getExistingDirectory(self, title, start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title, start)
        if path:
            widget.setCurrentText(project_path_text(path))

    def changed(self, *_args) -> None:
        self.validate_timer.start()

    # ---------------------------------------------------------------- config

    def config_chosen(self, index: int) -> None:
        path = self.config_combo.itemData(index)
        if path:
            self.load_config_from_path(path)

    def load_config_from_path(self, path: Path | str) -> None:
        path = Path(path)
        try:
            raw = load_yaml(path)
        except (OSError, yaml.YAMLError) as exc:
            self.enqueue_output(f"Could not read {path}: {exc}\n")
            return
        script = registry.script_for(raw)
        if script is None:
            self.enqueue_output(f"{path.name} is not a config for any startup script; not loaded.\n")
            return
        if self.command.running and script is not self.script:
            self.enqueue_output(f"{self.script_name} is running; press Stop before opening a {registry.script_name(script)} config.\n")
            return
        self.script, self.raw, self.loaded_path = script, raw, path
        self.setWindowTitle(ui_hints.title(self.script_name))
        self.title.setText(ui_hints.title(self.script_name))
        self.badge.setVisible(bool(self.status_patterns))
        self.build_form()
        for path_key, (field, _widget) in self.fields.items():
            self.set_shown(path_key, get_path(raw, path_key, field.default))
        self.shown = {path_key: self.get_shown(path_key) for path_key in self.fields}
        self.refresh_preview()
        self.enqueue_output(f"Loaded {path}\n")
        index = self.config_combo.findData(str(path))
        if index != -1:
            self.config_combo.blockSignals(True)
            self.config_combo.setCurrentIndex(index)
            self.config_combo.blockSignals(False)

    def current_config(self) -> dict:
        """The loaded YAML with only what was changed in the window written over it."""
        config = copy.deepcopy(self.raw)
        for path, (field, _widget) in self.fields.items():
            shown = self.get_shown(path)
            if shown == self.shown.get(path):
                continue
            value = yaml_value(field, shown)
            if value is None and not field.required:
                delete_path(config, path)
            else:
                set_path(config, path, value)
        for path, checks in self.entry_checks.items():
            entries = get_path(config, path, {})
            for key, checkbox in checks.items():
                raw_enabled = entries[key].get(ENABLED_KEY, True) if isinstance(entries.get(key), dict) else True
                if checkbox.isChecked() == self.entry_shown[path].get(key) and raw_enabled in (True, False):
                    continue
                if not isinstance(entries.get(key), dict):
                    entries[key] = {}
                entries[key][ENABLED_KEY] = checkbox.isChecked()
        return config

    def check_config(self, path: Path | None = None) -> tuple[Path, str | None]:
        """Write the config and run the script's own loader on it; returns (path, error).

        Live checks use a side file so they never rewrite the file a running script was started from.
        """
        path = path or self.generated_path.with_suffix(".check.yaml")
        try:
            write_yaml(path, self.current_config())
            self.script.load_config(path)
        except Exception as exc:  # the script decides what is invalid
            return path, str(exc)
        return path, None

    def refresh_preview(self) -> None:
        if self.script is None:
            return
        try:
            config = self.current_config()
        except Exception as exc:
            self.yaml_preview.setPlainText(f"Cannot build config: {exc}")
            return
        self.yaml_preview.setPlainText(yaml_text(config))
        _, error = self.check_config()
        if error:
            self.validity.setText(f"Config error: {error}")
            self.validity.setStyleSheet(f"color: {COLORS['danger']};")
        else:
            self.validity.setText("Config is valid.")
            self.validity.setStyleSheet(f"color: {COLORS['accent']};")

    def save_config(self) -> None:
        default = str(self.loaded_path or CONFIG_DIR / "config.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Save config", default, "YAML (*.yaml *.yml);;All files (*)")
        if path:
            write_yaml(Path(path), self.current_config())
            self.enqueue_output(f"Saved {path}\n")

    def read_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config", str(CONFIG_DIR), "YAML (*.yaml *.yml);;All files (*)")
        if path:
            self.load_config_from_path(path)

    # ---------------------------------------------------------------- process

    def launch_command(self, config_path: Path, *extra: str) -> list[str]:
        launcher = project_path_text(registry.script_path(self.script))
        return ["uv", "run", "python", "-u", launcher, "--yaml", str(config_path), *extra]

    def start(self) -> None:
        if self.script is None:
            self.enqueue_output("No config loaded.\n")
            return
        if self.command.running:
            self.enqueue_output("Already running; press Stop first.\n")
            return
        config_path, error = self.check_config(self.generated_path)
        if error:
            self.enqueue_output(f"Config error, not started: {error}\n")
            return
        self.badge_error = False
        set_status_badge(self.badge, None)
        self.command.start(
            self.launch_command(config_path, "--parent-pid", str(os.getpid())),
            cleanup=self.launch_command(config_path, "--cleanup"),
        )

    def enqueue_output(self, text: str) -> None:
        append_terminal_text(self.output, text)
        for pattern, template, is_error in self.status_patterns:
            match = pattern.search(text)
            if match:
                self.badge_error = is_error
                set_status_badge(self.badge, template.format(*match.groups()), error=is_error)
                break

    def process_done(self, returncode: int | None) -> None:
        self.enqueue_output(f"\nProcess exited with return code {returncode}.\n")
        if not self.badge_error:  # keep a failure reason visible after exit
            set_status_badge(self.badge, None)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.command.running:
            self.command.stop(wait=True)
        super().closeEvent(event)


def main(config_path: Path | str | None = None, argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    window = ScriptGui(config_path or (argv[1] if len(argv) > 1 else None))
    window.show()
    return app_exec(app)


if __name__ == "__main__":
    sys.exit(main())
