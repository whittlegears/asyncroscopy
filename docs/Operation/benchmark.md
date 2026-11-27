# Agent benchmark on the digital twin

A scored task suite for agents that control a microscope. It's built for
hackathons and for regression-testing our own routines. The twin acts as
referee: it knows the true focus, the on-zone tilt and the corrector's real
coefficients, and the agent doesn't. The referee and simulation commands
(`get_ground_truth`, `get_aberrations_coeff_sim`, `set_aberrations_coeff_sim`,
`inject_link_drop`) are blocked in `configs/mcp_benchmark.yaml`, so an agent
working through MCP can't read the answers.

## Tasks

| Task | Setup (seeded) | Pass | Score |
|---|---|---|---|
| `focus` | Defocus 30–60 nm off true focus. | Within 10 nm | 1 / (1 + (error / 10 nm)²) |
| `aberrations` | C1 15–40 nm, A1 4–10 nm. | Both below 2 nm | Mean of the per-aberration scores |
| `zone_axis` | Alpha 0.2–0.45° off zone. | Within 0.05° | 1 / (1 + (error / 0.05°)²) |
| `recovery` | Corrector link dropped. | Corrector ON and measuring | 1 or 0 |

## Run it

```bash
uv sync --extra diffraction
uv run startup_scripts/run_servers.py --yaml configs/benchmark_twin.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp_benchmark.yaml

# Our baseline (the asyncroscopy.autonomy routines) as a leaderboard row
uv run python -m asyncroscopy.bench baseline --seeds 3 --board leaderboard.jsonl

# Score a team's agent
uv run python -m asyncroscopy.bench setup --task zone_axis --seed 7 --state run.json   # prints the instructions
#   ... the agent works through MCP ...
uv run python -m asyncroscopy.bench score --state run.json --team "team name" --board leaderboard.jsonl
```

`score` adds a row to the leaderboard. Supervision is on in the benchmark MCP
config, so the row also gets a summary of the queue history for the run: how
many actions, how many instrument steps, and how many refusals.

## Baseline on the live benchmark stack

Seeds 0–2 for focus, aberrations and recovery; seeds 0–7 for zone axis.

| Task | Result |
|---|---|
| Focus | 3/3 pass, 2.7–3.5 nm error, 7–11 s |
| Aberrations | 3/3 pass, 1.1–1.8 nm residual |
| Recovery | 3/3 pass |
| Zone axis | 5–7/8 pass across runs. Every miss is 0.06–0.07°. Passes are 0.002–0.05°. About 1 min each. |

The zone-axis misses are the instrument's limit, not the routine's. The twin
gives every tilt move about ±0.045° of error, and a 4-pattern Laue
measurement resolves about ±0.033°. Even a perfect landing on the true zone
falls inside 0.05° only about 3 times in 4.

## What the benchmark found in our own code

Running the benchmark against our own routines turned up these problems, all
now fixed:

- **Laue metric and large cameras:** the metric was fooled by a 1024 px
  camera's noise floor. It now block-averages and keeps only significant
  signal.
- **Chasing noise:** zone-axis refinement kept stepping on noise. Each extra
  step re-rolled the stage's tilt error.
- **Noisy response estimates:** a ±step finite difference and a Broyden update
  from *commanded* moves corrupted the response estimate when every move
  carries tilt error. The routine now fits the response across the coarse
  scan.
- **Step-like response:** with a wide probe the response is step-like near
  zone. The routine bisects on the sign, then lands from one side and lands
  again if the measurement shows a miss.
- **Launcher watchdog:** a stray launcher's watchdog resurrected servers that
  another launcher owned. The watchdog now detects the conflict and stops.
