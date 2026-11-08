# Settings

Every `ASYNCROSCOPY_*` variable is read in one place, `asyncroscopy/settings.py`, at call time (`settings.get()`), never when a module is imported. Set a variable before the process that needs it starts, or before the call in a script or test. Paths left unset fall under `ASYNCROSCOPY_OUTPUTS`.

| Variable | What it sets | Default |
|---|---|---|
| `ASYNCROSCOPY_OUTPUTS` | The folder servers write under | `outputs` in the working directory |
| `ASYNCROSCOPY_LOG_DIR` | Server logs (Tiled's go in `tiled/` inside it) | `<outputs>/logs` |
| `ASYNCROSCOPY_LOG_ROTATE_BYTES` | Size at which a trace, tool-log or manifest file is rotated (renamed `<name>.<n>`); `0` never | `67108864` (64 MiB) |
| `ASYNCROSCOPY_LOG_ROTATE_KEEP` | Rotated trace and tool-log files kept per log, oldest deleted first; `0` keeps all. Integrity manifest segments are never deleted | `20` |
| `ASYNCROSCOPY_ACQUISITION_DIR` | Where the DATA device saves acquisitions | the config's `tiled.acquisition_dir` |
| `ASYNCROSCOPY_AGENT_MEMORY` | Agent notes and learned skills | `<outputs>/agent_memory` |
| `ASYNCROSCOPY_TASK_TRACE` | Agent task events; empty turns them off | `<outputs>/trace/agent_tasks.jsonl` |
| `ASYNCROSCOPY_INSTRUMENT_NOTES` | A team's own `INSTRUMENT.md`, read as well as the memory folder's | none |
| `ASYNCROSCOPY_INTEGRITY_LOG` | The seal manifest | `<outputs>/integrity/manifest.jsonl` |
| `ASYNCROSCOPY_INTEGRITY_KEY` | The seal key, as hex (wins over the key file) | none |
| `ASYNCROSCOPY_INTEGRITY_KEY_FILE` | File holding the seal key; made on first use | `<outputs>/integrity/key` |
| `ASYNCROSCOPY_DOSE_LEDGER` | The per-sample dose ledger | `<outputs>/dose/ledger.jsonl` |
| `ASYNCROSCOPY_MCP_URL` | Agent MCP endpoint | `http://127.0.0.1:8000/mcp` |
| `ASYNCROSCOPY_OPERATOR_URL` | Operator MCP endpoint (benchmarks) | the stack's `operator:` |
| `ASYNCROSCOPY_ANALYSIS_URL` | Analysis MCP server | what the endpoint reports |
| `ASYNCROSCOPY_MCP_TOKEN` | Bearer token for endpoints with `mcp.auth` | none |
| `ASYNCROSCOPY_TILED_URI` | Tiled address (DATA device default, fallback when DATA does not answer) | `http://127.0.0.1:9091` |
| `ASYNCROSCOPY_TILED_PUBLIC_URI` | The Tiled address DATA gives clients, when it differs from the one it binds | the Tiled address |
| `ASYNCROSCOPY_TILED_API_KEY` | Tiled API key | `secret` |
| `TILED_API_KEY` | Tiled's own name for the key; the analysis server uses it when the one above is unset | none |
| `ASYNCROSCOPY_MODEL` | Model tag or provider model name | the agent profile's |
| `ASYNCROSCOPY_MODEL_PROVIDER` | `ollama`, or any `init_chat_model` provider | the agent profile's |
| `ASYNCROSCOPY_NUM_CTX` | Ollama context window | the agent profile's |
| `ASYNCROSCOPY_AGENT_PROFILE` | Agent profile: a name in `agents/profiles`, or a YAML path | `default` |
| `ASYNCROSCOPY_AGENT_TOOLS` | General agent's tool globs, comma-separated | the STEM set, else every tool |
| `ASYNCROSCOPY_AGENT_PROMPT` | General agent's system prompt template | built in |
| `ASYNCROSCOPY_CALLER` | Name recorded in the trace | set per graph |
| `ASYNCROSCOPY_MAPPER` | Which Mapper the mapping graph uses, by tool prefix | the only one |
| `ASYNCROSCOPY_MICROSCOPE` | Which microscope the mapping graph uses, by tool prefix | the only one |
| `ASYNCROSCOPY_POLL_S` | Mapping graph's poll interval, seconds | `1.0` |
| `ASYNCROSCOPY_APPROVAL_WAIT_S` | Mapping graph's wait for an approval, seconds | `1800` |
| `ASYNCROSCOPY_ANALYSIS_WAIT_S` | Mapping graph's wait for stitching and scoring, seconds | `3600` |
| `ASYNCROSCOPY_CLAIM_CHECK` | `0`, `false`, `no` or `off` turns off checking final answers against tool calls | on |
| `ASYNCROSCOPY_AUTO_PROMOTE_AFTER` | Successful uses after which a learned skill promotes itself; `0` never | `0` |
| `ASYNCROSCOPY_PLUGINS` | Plug-in modules or files, comma- or `os.pathsep`-separated | none |
| `ASYNCROSCOPY_SKILLS_PATH` | Extra skill folders, `os.pathsep`-separated | none |
| `ASYNCROSCOPY_QT_API` | `pyqt5` selects PyQt5 for the desktop GUIs | PyQt6 |
| `ASYNCROSCOPY_WRITE_GOLDEN` | Tests only: `1` rewrites golden files | off |
| `ASYNCROSCOPY_WRITE_BENCHMARKS` | Tests only: `1` writes benchmark results | off |
| `ASYNCROSCOPY_PARITY_AUTOSCRIPT` | Tests only: a file holding the AutoScript class to compare | the packaged one |

The Tango device properties that name files (`trace_path`, `state_directory`, `mode_file` and others) are set in the stack's YAML, not here. `TILED_RETRY_ATTEMPTS` and `TILED_RETRY_TIMEOUT` are read by Tiled itself; asyncroscopy only gives them defaults.
