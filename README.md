# Asyncroscopy

**Asynchronous control framework for smart microscopy.** Built on `PyTango` to be hardware-agnostic.

<div class="d-flex justify-content-between align-items-center w-100">
  <img src="docs/images/architecturev1.png" style="width: 36%;" alt="Architecture V1" />
  <img src="docs/images/architecturev2.png" style="width: 60%;" alt="Architecture V2" />
</div>

## Quick Start

```bash
# Install all dependencies
uv sync

# Start the server stack with a configuration
uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml

# Or use the DigitalTwin (no hardware required)
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
```

Start the MCP server in a second terminal for agent/AI integration:
```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

For interactive GUI-based startup:
```bash
uv run startup_guis/server_gui.py
uv run startup_guis/mcp_gui.py
```

## Project Structure

```text
asyncroscopy/
├── agent/             # LangGraph graphs, skills registry, model providers
├── data/              # Data management device
├── instruments/       # Hardware device implementations
└── mcp/               # FastMCP server and the LLM Tango device
skills/                # Hermes-style SKILL.md files used by the agent
langgraph.json         # LangGraph Studio entry points
```

## Configuration Files (`configs/`)

| Config | Purpose |
|--------|---------|
| `Spectra300.yaml` | Real Thermo Fisher Spectra 300 setup |
| `DigitalTwin.yaml` | Simulated microscope for development/testing |
| `diffraction.yaml` | DigitalTwin with diffraction simulation |
| `digital_twin_tilt.yaml` | ASE/abTEM multislice silicon lamella tilt twin |
| `mcp.yaml` | MCP server configuration (not hardware) |

These are some examples of the available configs, which define the instrument class, supporting devices, Tango connection, and Tiled settings.

## Optional Dependencies

```bash
# Diffraction and tilt-twin multislice simulation (abTEM-based)
uv sync --extra diffraction

# AI agent support: LangGraph + LangChain (OpenAI/Anthropic clients included)
uv sync --extra agent

# Local models through Ollama (requires --extra agent)
uv sync --extra agent --extra ollama

# LangGraph Studio (`uv run langgraph dev`, requires --extra agent)
uv sync --extra agent --extra studio
```

## AI Agent

`asyncroscopy/agent/` holds the LangGraph layer shared by the notebook, LangGraph Studio, and the LLM Tango device:

```bash
cp .env.example .env                      # provider, model, MCP URL, API keys (never in YAML)
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
uv run jupyter lab notebooks/11_Test_AI_Agent.ipynb   # Ollama, API key, or LLM device
uv run langgraph dev                                  # visualise/run the graphs in Studio
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml   # optional LLM Tango device
```

- Deterministic workflows (fixed graphs, e.g. `image_eds_survey`) live in `asyncroscopy/agent/graphs/workflows/`.
- Skills (`skills/<name>/SKILL.md`) give the ReAct agent searchable, versioned procedures; see `skills/README.md`.
- Docs: `docs/Agent/`.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Architecture

- **Abstraction Layer:** Instruments are defined as abstract base classes, with specific implementations for different hardware. As such, models can be defined for various microscopes, digital twins, etc.
- **Device Orchestration:** Supporting devices (camera, EDS, stage, etc.) are initialized first and linked to the instrument via Tango properties.
- **Execution Order:** The instrument serves as the final integration point, instantiated only after all prerequisite supporting devices are running.
- **Data:** Data from devices is saved to Tiled.

## Documentation

Refer to the [asyncroscopy documentation](https://pycroscopy.github.io/asyncroscopy/).

## Notebooks

Various example workflows in `notebooks/`, including:

- `00_Testing.ipynb` - Connection tests
- `01_Aberrations.ipynb` - Probe aberration controls
- `02_Image_Acquisition.ipynb` - HAADF image acquisition
- `03_Stage_Movement_Sample_Map.ipynb` - Stage navigation
- `04_Image_EDS_Point_Spectra.ipynb` - EDS spectrum acquisition
- `05_Digital_Twin_EDS.ipynb` - DigitalTwin EDS simulation
- `06_Digital_Twin_Tilt.ipynb` - DigitalTwin tilt control
- `11_Test_AI_Agent.ipynb` - LangGraph agent, skills, and workflow demo (Ollama / API key / LLM device)
- `15_MAPED.ipynb` - Multi-angle precession electron diffraction
- `16_Alpha_Tilt_Diffraction_Map.ipynb` - Tracked alpha-tilt diffraction mapping

## Notes
* The previous Twisted-based implementation is preserved in the `twisted-legacy` branch for reference.
