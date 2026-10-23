#!/usr/bin/env python
"""A startup window built entirely from a startup script's declared settings.

Nothing here knows about Tango, MCP or devices: each field, choice, preset, device row
and validation rule comes from the script module (see startup_scripts/config_schema.py).
Adding an option or a device to a script or YAML needs no change in this file.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from startup_guis.qt_compat import (  # noqa: E402
    HORIZONTAL,
    NO_WRAP,
    POINTING_HAND_CURSOR,
    VERTICAL,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
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
    PresetToggle,
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
from startup_scripts.config_schema import MISSING, Setting, get_path, set_path  # noqa: E402


def project_path_text(path: Path | str) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def display_text(setting: Setting, value) -> str | bool:
    """What a widget shows for a YAML value."""
    if setting.kind == "bool":
        return bool(value)
    if value is None or value is MISSING:
        return ""
    if setting.kind == "list":
        return ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    if setting.kind == "yaml":
        return yaml.safe_dump(value, sort_keys=False).strip() if value not in ({}, []) else ""
    if setting.kind in ("path", "dir"):
        return project_path_text(value)
    return str(value)


def yaml_value(setting: Setting, shown: str | bool):
    """The YAML value for what a widget shows; text that does not parse is kept so validation can report it."""
    if setting.kind == "bool":
        return bool(shown)
    text = str(shown).strip()
    if text == "":
        return None
    if setting.kind == "int":
        try:
            return int(text)
        except ValueError:
            return text
    if setting.kind == "list":
        return [item.strip() for item in text.split(",") if item.strip()]
    if setting.kind == "yaml":
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
    def __init__(self, script, config_path: Path | str | None = None):
        super().__init__()
        self.script = script
        self.devices = getattr(script, "DEVICES", None)
        self.status_patterns = getattr(script, "STATUS_PATTERNS", [])
        self.generated_path = GENERATED_CONFIG_DIR / f"{Path(script.LAUNCHER).stem}_gui.yaml"
        self.setWindowTitle(script.TITLE)
        self.resize(1280, 960)
        self.setMinimumSize(880, 560)
        self.command = ManagedCommand(self.enqueue_output, self.process_done)
        self.raw: dict = {}
        self.loaded_path: Path | None = None
        self.fields: dict[str, tuple[Setting, object]] = {}  # path -> (setting, widget)
        self.shown: dict[str, str | bool] = {}  # path -> what the widget showed right after loading
        self.presets: dict[str, PresetToggle] = {}
        self.device_checks: dict[str, CheckBox] = {}
        self.device_shown: dict[str, bool] = {}
        self.badge_error = False
        self.validate_timer = QTimer(self)
        self.validate_timer.setSingleShot(True)
        self.validate_timer.setInterval(300)
        self.validate_timer.timeout.connect(self.refresh_preview)
        self.build()
        default_path = getattr(script, "GUI_DEFAULT_CONFIG_PATH", script.DEFAULT_CONFIG_PATH)
        self.load_config_from_path(Path(config_path) if config_path else default_path)
        # Open on each setting's first preset, e.g. the digital twin rather than the real instrument.
        for path, toggle in self.presets.items():
            setting, _ = self.fields[path]
            self.set_shown(path, next(iter(setting.presets.values())))

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
        title = QLabel(self.script.TITLE)
        title.setFont(TITLE_FONT)
        title_row.addWidget(title)
        self.badge = status_badge()
        self.badge.setVisible(bool(self.status_patterns))
        title_row.addWidget(self.badge)
        title_row.addStretch()
        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(220)
        self.config_combo.addItems(self.config_files())
        self.config_combo.currentTextChanged.connect(lambda name: name and self.load_config_from_path(CONFIG_DIR / name))
        title_row.addWidget(self.config_combo)
        layout.addLayout(title_row)

        sections: dict[str, CollapsibleSection] = {}
        for setting in self.script.SETTINGS:
            if setting.section not in sections:
                sections[setting.section] = CollapsibleSection(setting.section, expanded=False)
                layout.addWidget(sections[setting.section])
            self.add_field(sections[setting.section], setting)

        if self.devices is not None:
            self.devices_section = CollapsibleSection(self.devices.label, layout_cls=QGridLayout, expanded=True)
            layout.addWidget(self.devices_section)

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

    def add_field(self, section: CollapsibleSection, setting: Setting) -> None:
        if setting.kind == "bool":
            widget = CheckBox(setting.title)
            widget.stateChanged.connect(self.changed)
            section.form.addRow("", widget)
        elif setting.kind == "yaml":
            widget = QTextEdit()
            widget.setFont(TEXT_FONT)
            widget.setLineWrapMode(NO_WRAP)
            widget.setMinimumHeight(110)
            widget.textChanged.connect(self.changed)
            section.form.addRow(section_label(setting.title))
            section.form.addRow(widget)
        else:
            if setting.choices or setting.kind in ("path", "dir"):
                widget = QComboBox()
                widget.setEditable(True)
                widget.addItems(list(setting.choices))
                widget.currentTextChanged.connect(self.changed)
            else:
                widget = QLineEdit()
                widget.textChanged.connect(self.changed)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(widget, 1)
            if setting.kind in ("path", "dir"):
                browse = QPushButton("Browse")
                browse.setCursor(POINTING_HAND_CURSOR)
                browse.clicked.connect(lambda _=False, s=setting, w=widget: self.browse(s, w))
                row_layout.addWidget(browse)
            if setting.presets:
                toggle = PresetToggle(setting.presets, lambda value, p=setting.path: self.set_shown(p, value))
                self.presets[setting.path] = toggle
                row_layout.addWidget(toggle)
            section.form.addRow(setting.title, row)
        if setting.help:
            widget.setToolTip(setting.help)
            if hasattr(widget, "setPlaceholderText"):
                widget.setPlaceholderText(setting.help)
            elif hasattr(widget, "lineEdit") and widget.lineEdit() is not None:
                widget.lineEdit().setPlaceholderText(setting.help)
        self.fields[setting.path] = (setting, widget)

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
        setting, widget = self.fields[path]
        if setting.kind == "bool":
            return widget.isChecked()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return widget.text()

    def set_shown(self, path: str, value) -> None:
        setting, widget = self.fields[path]
        shown = display_text(setting, value)
        if setting.kind == "bool":
            widget.setChecked(bool(shown))
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(shown)
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(shown)
        else:
            widget.setText(shown)

    def browse(self, setting: Setting, widget: QComboBox) -> None:
        start = str(PROJECT_DIR / widget.currentText()) if widget.currentText() else str(PROJECT_DIR)
        if setting.kind == "dir":
            path = QFileDialog.getExistingDirectory(self, f"Select {setting.title}", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {setting.title}", start)
        if path:
            widget.setCurrentText(project_path_text(path))

    def changed(self, *_args) -> None:
        for path, toggle in self.presets.items():
            toggle.set_value(str(self.get_shown(path)).strip())
        self.validate_timer.start()

    # ---------------------------------------------------------------- devices

    def rebuild_devices(self) -> None:
        if self.devices is None:
            return
        grid = self.devices_section.form
        for checkbox in self.device_checks.values():
            grid.removeWidget(checkbox)
            checkbox.deleteLater()
        self.device_checks, self.device_shown = {}, {}
        entries = get_path(self.raw, self.devices.path, {})
        for index, (key, spec) in enumerate((entries or {}).items() if isinstance(entries, dict) else []):
            spec = spec if isinstance(spec, dict) else {}
            summary = self.devices.summary(key, spec) if self.devices.summary else ""
            checkbox = CheckBox(f"{key}  ·  {summary}" if summary else str(key))
            enabled = spec.get(self.devices.enabled_key, True) not in (False, "false", "no", "off", 0)
            checkbox.setChecked(enabled)
            checkbox.stateChanged.connect(self.changed)
            grid.addWidget(checkbox, index // 2, index % 2)
            self.device_checks[key] = checkbox
            self.device_shown[key] = enabled

    # ---------------------------------------------------------------- config

    def config_files(self) -> list[str]:
        names = []
        for path in sorted([*CONFIG_DIR.glob("*.yaml"), *CONFIG_DIR.glob("*.yml")]):
            try:
                if self.script.is_config(load_yaml(path)):
                    names.append(path.name)
            except (OSError, yaml.YAMLError):
                continue
        return names

    def load_config_from_path(self, path: Path | str) -> None:
        path = Path(path)
        try:
            raw = load_yaml(path)
        except (OSError, yaml.YAMLError) as exc:
            self.enqueue_output(f"Could not read {path}: {exc}\n")
            return
        if not self.script.is_config(raw):
            self.enqueue_output(f"{path.name} is not a config for {Path(self.script.LAUNCHER).name}; not loaded.\n")
            return
        self.raw, self.loaded_path = raw, path
        for path_key, (setting, _widget) in self.fields.items():
            self.set_shown(path_key, get_path(raw, path_key, setting.default))
        self.shown = {path_key: self.get_shown(path_key) for path_key in self.fields}
        self.rebuild_devices()
        self.changed()
        self.refresh_preview()
        self.enqueue_output(f"Loaded {path}\n")
        if self.config_combo.findText(path.name) != -1:
            self.config_combo.blockSignals(True)
            self.config_combo.setCurrentText(path.name)
            self.config_combo.blockSignals(False)

    def current_config(self) -> dict:
        """The loaded YAML with only what was changed in the window written over it."""
        config = copy.deepcopy(self.raw)
        for path, (setting, _widget) in self.fields.items():
            shown = self.get_shown(path)
            if shown == self.shown.get(path):
                continue
            value = yaml_value(setting, shown)
            if value is None and not setting.required:
                delete_path(config, path)
            else:
                set_path(config, path, value)
        if self.devices is not None:
            entries = get_path(config, self.devices.path, {})
            for key, checkbox in self.device_checks.items():
                if checkbox.isChecked() == self.device_shown.get(key):
                    continue
                if not isinstance(entries.get(key), dict):
                    entries[key] = {}
                entries[key][self.devices.enabled_key] = checkbox.isChecked()
        return config

    def check_config(self, path: Path | None = None) -> tuple[Path, str | None]:
        """Write the config and run the script's own loader on it; returns (path, error).

        Live checks use a side file so they never rewrite the file a running script was started from.
        """
        path = write_yaml(path or self.generated_path.with_suffix(".check.yaml"), self.current_config())
        try:
            self.script.load_config(path)
        except Exception as exc:  # the script decides what is invalid
            return path, str(exc)
        return path, None

    def refresh_preview(self) -> None:
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
        return ["uv", "run", "python", "-u", self.script.LAUNCHER, "--yaml", str(config_path), *extra]

    def start(self) -> None:
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


def main(script, argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    window = ScriptGui(script)
    window.show()
    return app_exec(app)
