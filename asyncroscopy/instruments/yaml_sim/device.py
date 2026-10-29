"""A Tango device built from an instrument YAML file (see ``spec.py`` for the format).

Every setting becomes an attribute plus ``get_<name>`` (and ``set_<name>`` when
settable) commands; every command and acquisition in the file becomes a Tango
command, so the MCP server offers each one as an agent tool and the safety
policy classes it like any other command (``get_*`` read-only, ``set_*`` and
``acquire*`` held in approve mode, ``xray_off``-style stops always allowed).
Acquisitions are saved through the DATA device and return its key, so agents
look at them with ``view_data``.

In a launcher config, give the class name the tools should carry and the file::

    instrument:
      class_name: XRDTwin
      module_name: asyncroscopy.instruments.yaml_sim.device
      properties:
        yaml_path: asyncroscopy/instruments/yaml_sim/examples/xrd.yaml
        data_device_address: asyncroscopy/data/default

The server takes its class name from the Tango database entry the launcher
writes (``<class_name>/<instance>``), so the tools are ``XRDTwin_get_two_theta_deg``
and so on, with no Python written for the instrument.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tango
from tango import AttrWriteType, CmdArgType, DevState
from tango.server import Device, device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.yaml_sim.spec import (
    Refused,
    SimInstrument,
    SpecError,
    load_spec,
)
from asyncroscopy.utils.commands import command

PROJECT_DIR = Path(__file__).resolve().parents[3]
TANGO_TYPES = {float: CmdArgType.DevDouble, int: CmdArgType.DevLong64, bool: CmdArgType.DevBoolean, str: CmdArgType.DevString}


class YamlSimDevice(Device):
    yaml_path = device_property(dtype=str, default_value="", doc="The instrument YAML file (absolute, or relative to the working directory or the repository).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device that saves and registers acquisitions; empty saves files under output_directory.")
    output_directory = device_property(dtype=str, default_value="outputs/yaml_sim", doc="Where acquisitions go when there is no DATA device.")

    def init_device(self) -> None:
        Device.init_device(self)
        self.acquisition_save_directory = self.output_directory
        self.sim = None
        try:
            spec = load_spec(self._resolve(self.yaml_path))
        except SpecError as exc:
            message = f"{type(self).__name__}: the instrument file is not valid: {exc}"
            print(message, file=sys.stderr, flush=True)
            self.set_state(DevState.FAULT)
            self.set_status(message)
            return
        self.sim = SimInstrument(spec)
        for name, quantity in spec.settings.items():
            self._build_setting(name, quantity)
        for name, cmd in spec.commands.items():
            self._build_command(name, cmd)
        for name, acquisition in spec.acquisitions.items():
            self._build_acquisition(name, acquisition)
        self.set_state(DevState.ON)
        self.set_status(spec.description or f"Simulated instrument from {spec.path.name}")

    @staticmethod
    def _resolve(path: str) -> Path:
        if not path:
            raise SpecError("the yaml_path device property is empty; set it to the instrument YAML file")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and not candidate.exists() and (PROJECT_DIR / candidate).exists():
            return PROJECT_DIR / candidate
        return candidate

    def _build_setting(self, name, quantity) -> None:
        sim = self.sim

        def read(attr):
            attr.set_value(sim.get(name))

        def write(attr):
            sim.set(name, attr.get_write_value())

        access = AttrWriteType.READ_WRITE if quantity.settable else AttrWriteType.READ
        attr = tango.Attr(name, TANGO_TYPES[quantity.type], access)
        props = tango.UserDefaultAttrProp()
        props.set_description(quantity.describe())
        if quantity.units:
            props.set_unit(quantity.units)
        attr.set_default_properties(props)
        # Tango calls attribute methods with (device, attribute), unlike dynamic commands.
        self.add_attribute(attr, r_meth=lambda _device, a: read(a), w_meth=(lambda _device, a: _refusing(write, a)) if quantity.settable else None)

        def get():
            return sim.get(name)

        get.__name__ = f"get_{name}"
        get.__doc__ = f"Current {quantity.describe()[0].lower()}{quantity.describe()[1:]}"
        self.add_command(command(f=get, dtype_out=quantity.type), device_level=True)
        if quantity.settable:

            def set_(value):
                _refusing(sim.set, name, value)

            set_.__name__ = f"set_{name}"
            set_.__doc__ = f"Set the {quantity.describe()[0].lower()}{quantity.describe()[1:]}"
            self.add_command(command(f=set_, dtype_in=quantity.type, doc_in=f":param {name}: {quantity.describe()}"), device_level=True)

    def _build_command(self, name, cmd) -> None:
        sim = self.sim
        if cmd.input is None:

            def run():
                return _refusing(sim.run_command, name)

            run.__name__ = name
            run.__doc__ = cmd.doc
            self.add_command(command(f=run, dtype_out=str), device_level=True)
        else:

            def run_with(value):
                return _refusing(sim.run_command, name, value)

            run_with.__name__ = name
            run_with.__doc__ = cmd.doc
            self.add_command(command(f=run_with, dtype_in=cmd.input.type, dtype_out=str, doc_in=f":param {cmd.input.name}: {cmd.input.describe()}"), device_level=True)

    def _build_acquisition(self, name, acquisition) -> None:
        def acquire():
            acquisition_type, detector, data, attrs = _refusing(self.sim.acquire, name)
            if data is None:
                return json.dumps(attrs)
            proxy = tango.DeviceProxy(self.data_device_address) if self.data_device_address else None
            return save_acquisition(
                self, proxy, acquisition_type, detector, data,
                dataset_name="spectrum" if data.ndim == 1 else "image",
                dataset_attrs=attrs,
                file_attrs={"instrument": type(self).__name__, "description": self.sim.spec.description},
            )

        acquire.__name__ = name
        acquire.__doc__ = acquisition.doc + (" Returns the data key; look at it with view_data." if acquisition.kind != "value" else " Returns JSON.")
        self.add_command(command(f=acquire, dtype_out=str), device_level=True)


def _refusing(func, *args):
    """Run ``func``; a Refused becomes a ValueError the client sees with its message."""
    try:
        return func(*args)
    except Refused as exc:
        raise ValueError(str(exc)) from None


_classes: dict[str, type] = {}


def device_class(name: str) -> type:
    """A YamlSimDevice subclass called ``name``: the MCP server names tools after the Tango class."""
    if name == YamlSimDevice.__name__:
        return YamlSimDevice
    if name not in _classes:
        _classes[name] = type(name, (YamlSimDevice,), {"__module__": __name__})
    return _classes[name]


def _class_name_for(instance: str) -> str:
    """The class the launcher registered for this server instance (``<class_name>/<instance>``)."""
    try:
        servers = tango.Database().get_server_list(f"*/{instance}").value_string
    except tango.DevFailed:
        return YamlSimDevice.__name__
    return servers[0].split("/", 1)[0] if servers else YamlSimDevice.__name__


if __name__ == "__main__":
    device_class(_class_name_for(sys.argv[1]) if len(sys.argv) > 1 else YamlSimDevice.__name__).run_server()
