# Asyncroscopy

**Asynchronous control framework for smart microscopy.** Built on `PyTango` to be hardware-agnostic.

<div class="d-flex justify-content-between align-items-center w-100">
  <img src="docs/images/architecturev1.png" style="width: 36%;" alt="Architecture V1" />
  <img src="docs/images/architecturev2.png" style="width: 60%;" alt="Architecture V2" />
</div>

## Quick Start

**New here? Follow [docs/Operation/quickstart.md](docs/Operation/quickstart.md).** It is the tested path (installing uv, the exact `uv sync` and test commands, Windows fixes) from clone to a running digital twin, MCP endpoints, analysis server and a local-model agent mapping a sample. In short:

```bash
git clone --depth 1 https://github.com/pycroscopy/asyncroscopy.git   # on Windows, avoid deeply nested folders
cd asyncroscopy
uv sync --extra agent --extra ollama   # Windows: uv not found, or "Missing expected target directory"? see quickstart sections 1-2
uv run pytest -q -m "not slow"
# PowerShell without uv on PATH: write & $uv for uv (quickstart section 1)

# Digital twin (no hardware required); one terminal each
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml        # agent endpoint, 127.0.0.1:8000
uv run startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml  # operator endpoint, 127.0.0.1:8001
uv run python -m pyTEMlib.mcp_server --tiled http://localhost:9091 --api-key secret --port 8003

# Then check the whole stack
uv run startup_scripts/check_stack.py --pytemlib http://127.0.0.1:8003/mcp
```

On a real instrument use its config instead, for example `configs/Spectra300.yaml` with `configs/mcp.yaml`.

**Bringing your own instrument? Follow [docs/Adding_New_Hardware/new_instrument.md](docs/Adding_New_Hardware/new_instrument.md)**, a worked example with a simulated nanoindenter.

`check_stack.py` confirms every piece talks to the others; without `--pytemlib` it skips stitching and scoring. It checks the STEM workflow; for another instrument, see [Adding a new instrument](docs/Adding_New_Hardware/new_instrument.md).

### Agentic control at a glance

| What | Where |
| :-- | :-- |
| A realistic, seeded digital twin (stage error, drift, scan distortion, focus, noise, EDS spectra, dose) with hidden ground truth for scoring workflows | `configs/digital_twin_mapping.yaml`, [docs/digital_twin.md](docs/digital_twin.md) |
| Long jobs (map acquisition) as reserve / start / job_status / abort / results / release | `Mapper` device, [asyncroscopy/contracts](asyncroscopy/contracts/README.md) |
| An approval mode (`auto` / `approve`) that holds risky agent calls for an operator, and an Instrument Trace of every call | [docs/Operation/approval_mode.md](docs/Operation/approval_mode.md) |
| Stitching, zoomable maps in Tiled, region scoring and points of interest | pyTEMlib MCP server, [docs/Operation/agentic_mapping.md](docs/Operation/agentic_mapping.md) |
| LangGraph agents (`mapping`, `general`) that run in LangGraph Studio, and skills agents load on demand | `asyncroscopy/agents`, `asyncroscopy/skills`, `langgraph.json` |
| Calibrating the twin to your instrument | `asyncroscopy/calibration` |

Start only the segmentation Tango device against an existing Tango/DATA/Tiled stack:
```bash
uv run --extra segment startup_scripts/run_segmentation.py --yaml configs/Segmentation.yaml
```

Start the oriented-particle digital twin:

```bash
uv run --extra diffraction python startup_scripts/run_servers.py --yaml configs/digital_twin_particles.yaml
```

`configs/Segmentation.yaml` sets `compute_device: cuda`. On Linux and Windows,
the `segment` extra installs PyTorch from its CUDA 13.0 package index; macOS
continues to use the normal PyPI build. The launcher fails at startup instead
of silently using the CPU when CUDA is unavailable. Install and verify on the
GPU host before launching:

```bash
uv sync --extra segment --reinstall-package torch --reinstall-package torchvision
uv run --extra segment python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
```

For interactive GUI-based startup:
```bash
uv run startup_guis/server_gui.py
uv run startup_guis/mcp_gui.py
```

## Project Structure

```text
asyncroscopy/
├── APIservers/        # Vendor API bridges (Gatan EELS server)
├── agents/            # LangGraph agents (mapping graph, general agent) and the CLI
├── calibration/       # Dose, drift and stage-repeatability measurement series
├── contracts/         # JSON schemas for plans and job envelopes
├── data/              # Data management device (Tiled)
├── instruments/       # Hardware device implementations, the digital twin, the Mapper
├── jobs/              # JobDevice: reserve/start/abort for long-running work
├── mcp/               # FastMCP server for AI agents
├── safety/            # Risk policy and the Approvals device
├── skills/            # Skill loader and bundled skills (mapping/SKILL.md)
├── utils/             # The command decorator and shared helpers
└── testing.py         # instrument_context and TestMCPServer for your own tests
```

## Configuration Files (`configs/`)

| Config | Purpose |
|--------|---------|
| `Spectra300.yaml` | Real Thermo Fisher Spectra 300 setup |
| `DigitalTwin.yaml` | Simulated microscope for development/testing |
| `diffraction.yaml` | DigitalTwin with diffraction simulation |
| `digital_twin_tilt.yaml` | ASE/abTEM multislice silicon lamella tilt twin |
| `digital_twin_mapping.yaml` | DigitalTwin with realistic instrument effects, a 2 um sample, Mapper and Approvals |
| `mcp_dt.yaml` / `mcp_operator.yaml` | Agent MCP endpoint (port 8000) and operator endpoint for the GUI (port 8001) |
| `qwen-llm.yaml` | Agent server on a local Qwen3 model (laptop GPU); `gemma-llm.yaml` is the 31B workstation version |
| `mcp.yaml` | MCP server configuration (not hardware) |

These are some examples of the available configs, which define the instrument class, supporting devices, Tango connection, and Tiled settings.

## Optional Dependencies

`uv sync` installs exactly the extras you name and removes the others, so list every extra you want in one command (on Windows, add `--python <path>` if the quickstart's Python-link fix was needed) (for example `uv sync --extra agent --extra ollama --extra diffraction`). `uv run --extra X ...` adds an extra without removing anything.

```bash
# Diffraction and tilt-twin multislice simulation (abTEM-based)
uv sync --extra diffraction

# AI agents (LangChain/LangGraph) with local Ollama models
uv sync --extra agent --extra ollama

# LangGraph Studio (uv run langgraph dev) and developer tools
uv sync --extra agent --extra ollama --extra dev
```

## Running Tests

```bash
uv run pytest -q -m "not slow"   # about 1,200 tests: 5-7 min on a laptop; device shutdown logs can follow for a minute more
uv run pytest -q                 # everything, including timing tests: about a minute and a half
```

See the [quickstart](docs/Operation/quickstart.md) for what to expect.

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
- `03_EELS_Acquisition.ipynb` - EELS acquisition
- `04_Image_EDS_Point_Spectra.ipynb` - EDS spectrum acquisition
- `05_Digital_Twin_EDS.ipynb` - DigitalTwin EDS simulation
- `06_Segmentation.ipynb` - Image acquisition for segmentation
- `09_Hackathon_Digital_Twin.ipynb` - Hackathon walkthrough on the diffraction twin (`configs/diffraction.yaml`; needs `--extra diffraction`)
- `11_MCP_server_usecase_on_twin.ipynb` - MCP server on the digital twin
- `11_Test_AI_Agent.ipynb` - MCP agent testing
- `13_Stage_Movement_Sample_Map.ipynb` - Stage navigation and sample maps
- `15_MAPED.ipynb` - Multi-angle precession electron diffraction (`--extra diffraction`)
- `16_Alpha_Tilt_Diffraction_Map.ipynb` - Tracked alpha-tilt diffraction mapping (`--extra diffraction`)
- `18_Polycrystalline_Gold_Digital_Twin.ipynb` - Polycrystalline gold twin (`--extra diffraction`)

Run them with the project environment, for example `uv run --extra diffraction jupyter lab` (asyncroscopy is installed into `.venv`, so imports work from `notebooks/`).

## Notes
* The previous Twisted-based implementation is preserved in the `twisted-legacy` branch for reference.
