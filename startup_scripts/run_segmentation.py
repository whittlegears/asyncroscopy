#!/usr/bin/env python
"""Register and start only the SAM2 segmentation Tango device."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tango
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager  # noqa: E402
from startup_scripts.config_schema import ConfigError, Setting, read_settings  # noqa: E402
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers  # noqa: E402

DEVICE_NAME = "asyncroscopy/segment/default"
INSTANCE_NAME = "segment_instance"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "Segmentation.yaml"
STATE_NAME = "run_segmentation"

SETTINGS = [
    Setting("tango.host", required=True),
    Setting("tango.port", "int", required=True),
    Setting("data_device_address", default="asyncroscopy/data/default"),
    Setting("model_size", default="facebook/sam2-hiera-large"),
    Setting("compute_device", default="auto", choices=("auto", "cuda", "cpu")),
    Setting("device_timeout_seconds", "int", 120),
]


def is_config(raw) -> bool:
    return isinstance(raw, dict) and "model_size" in raw and "instrument" not in raw


@dataclass(frozen=True)
class TangoConfig:
    host: str
    port: int


@dataclass(frozen=True)
class SegmentationConfig:
    tango: TangoConfig
    data_device_address: str = "asyncroscopy/data/default"
    model_size: str = "facebook/sam2-hiera-large"
    compute_device: str = "auto"
    device_timeout_seconds: int = 120


def load_config(path: Path) -> SegmentationConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f"{path.name} is not valid YAML: {exc}"]) from exc
    values = read_settings(raw, SETTINGS)
    return SegmentationConfig(
        tango=TangoConfig(host=values["tango.host"], port=values["tango.port"]),
        data_device_address=values["data_device_address"],
        model_size=values["model_size"],
        compute_device=values["compute_device"],
        device_timeout_seconds=values["device_timeout_seconds"],
    )


def register_device(config: SegmentationConfig) -> None:
    database = tango.Database(config.tango.host, config.tango.port)
    device_info = tango.DbDevInfo()
    device_info.server = f"SEGMENTATION/{INSTANCE_NAME}"
    device_info._class = "SEGMENTATION"
    device_info.name = DEVICE_NAME

    try:
        database.add_device(device_info)
        print(f"Registered device: {DEVICE_NAME}")
    except tango.DevFailed as exc:
        print(f"Device already registered or error: {exc}")

    database.put_device_property(
        DEVICE_NAME,
        {
            "data_device_address": [config.data_device_address],
            "model_size": [config.model_size],
            "compute_device": [config.compute_device],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_CONFIG_PATH)
    add_lifecycle_args(parser)
    args = parser.parse_args(argv)
    install_shutdown_handlers(args.parent_pid)

    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    if args.cleanup:
        with ProcessManager(name=STATE_NAME):
            pass
        print("Stopped processes left by an earlier run.")
        return 0

    tango_host = f"{config.tango.host}:{config.tango.port}"
    os.environ["TANGO_HOST"] = tango_host
    try:
        register_device(config)
    except Exception as exc:
        print(f"Could not register segmentation at {tango_host}: {exc}", file=sys.stderr)
        return 1

    try:
        with ProcessManager(name=STATE_NAME) as manager:
            managed: ManagedProcess = manager.start_process(
                key="segmentation",
                label="Segmentation Server",
                command=[
                    sys.executable,
                    "-m",
                    "asyncroscopy.mcp.segment",
                    INSTANCE_NAME,
                ],
                env={
                    **os.environ,
                    "TANGO_HOST": tango_host,
                    "PYTHONUNBUFFERED": "1",
                },
                stdout=None,
                stderr=None,
            )

            print("Waiting for segmentation device to initialize...")
            deadline = time.monotonic() + config.device_timeout_seconds
            while time.monotonic() < deadline:
                try:
                    proxy = tango.DeviceProxy(DEVICE_NAME)
                    proxy.ping()
                    state = proxy.state()
                    if state == tango.DevState.ON:
                        print("Segmentation device is ready. Press Ctrl+C to terminate.")
                        break
                    if state == tango.DevState.FAULT:
                        print(f"Segmentation initialization failed: {proxy.status()}")
                        return 1
                except Exception:
                    pass
                time.sleep(1)
            else:
                print("Timed out waiting for segmentation device.", file=sys.stderr)
                return 1

            while managed.running:
                try:
                    managed.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        print("\nShutting down segmentation server...")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
