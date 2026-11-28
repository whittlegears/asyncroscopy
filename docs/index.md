# Asyncroscopy documentation

:::::{grid} 2
:gutter: 0

::::{grid-item}
:::{image} ./images/architecturev1.png
:height: 200px
:::
::::

::::{grid-item}
:::{image} ./images/architecturev2.png
:height: 200px
:::
::::

:::::

Asyncroscopy controls lab instruments through [PyTango](https://pytango.readthedocs.io). Each instrument is a Tango device. Each part of the instrument (scan unit, camera, stage, detector) is its own Tango device. Notebooks, scripts and LLM agents all talk to the same devices through the Tango database.

The code ships with electron microscopes (Thermo Fisher AutoScript, JEOL PyJEM, several digital twins), an Asylum Research Jupiter AFM and an Opentrons pipetting robot. You can add your own instrument without changes to the startup scripts, the GUI or the MCP server.

## Where to start

1. [Run the digital twin](Operation/run-servers.md) to see a full stack on one computer with no hardware.
2. Read the [architecture](architecture.md) page. It explains the parts and the names they use.
3. [Add your instrument](new_instrument.md): one instrument class, its support devices and one YAML config.
4. [Scale out](scaling.md) to more instruments, more computers, or more capability per instrument.

## Reference

- [Electron microscope family](Microscopy/modify_base_electron_microscope.md) and its [AutoScript implementation](Microscopy/modify_auto_script_microscope.md)
- [Adding a detector](Adding_New_Hardware/add_detector.md) to an electron microscope
- [Digital twins](digital_twin.md)
- [Data and Tiled](Tiled_server/data_integration.md)
- [MCP server](MCP/mcp_server.md) and [agent MCP server](MCP/agent_mcp_server.md)
- [Troubleshooting](Operation/troubleshooting.md)
- [Contributing guide](dev_guide.md)
