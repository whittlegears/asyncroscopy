# Asyncroscopy

Asyncroscopy controls lab instruments through `PyTango`. Each instrument, and each part of it, is a Tango device. Notebooks, scripts and LLM agents use the same devices.

<div class="d-flex justify-content-between align-items-center w-100">
  <img src="docs/images/architecturev1.png" style="width: 36%;" alt="Architecture V1" />
  <img src="docs/images/architecturev2.png" style="width: 60%;" alt="Architecture V2" />
</div>

## Quick start

```bash
uv sync

# Simulated microscope, no hardware needed
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml

# In a second terminal: expose the devices to agents over MCP
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

For a real instrument, use its config, for example `configs/Spectra300.yaml` with `configs/mcp.yaml`.

The startup GUI does the same from a window:

```bash
uv run startup_guis/server_gui.py
```

## Documentation

The [documentation](https://pycroscopy.github.io/asyncroscopy/) explains how to:

- [add your own instrument](docs/new_instrument.md),
- [scale to more instruments and computers](docs/scaling.md),
- [run the servers](docs/Operation/run-servers.md) and the [MCP server](docs/MCP/mcp_server.md).

## Project structure

```text
asyncroscopy/
├── instruments/       # Instrument and support devices, one folder per kind
├── data/              # DATA device and the file writer for Tiled
├── mcp/               # MCP server and the agent swarm
└── utils/             # ProcessManager and helpers
configs/               # One YAML per stack, MCP server or agent
startup_scripts/       # run_servers.py, run_mcp.py, run_llm.py, run_segmentation.py
startup_guis/          # The startup window
notebooks/             # Example workflows
```

## Optional dependencies

```bash
uv sync --extra diffraction                 # abTEM simulation for the diffraction and tilt twins
uv sync --extra agent                       # LLM agent (LangChain, OpenAI)
uv sync --extra agent --extra ollama        # local models through Ollama
uv sync --extra segment                     # SAM2 segmentation device (PyTorch)
uv sync --extra jupiter                     # Asylum Research Jupiter AFM
```

On Linux and Windows, the `segment` extra installs PyTorch from its CUDA 13.0 package index. macOS uses the normal PyPI build. Check CUDA on the GPU host before you start the segmentation device:

```bash
uv sync --extra segment --reinstall-package torch --reinstall-package torchvision
uv run --extra segment python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Notes

The earlier Twisted-based code is in the `twisted-legacy` branch.
