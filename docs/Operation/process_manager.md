# Process Manager

`ProcessManager` (`asyncroscopy/utils/process_manager.py`) provides the infrastructure for spawning and managing configuration-defined subprocesses in Asyncroscopy. It is used by `startup_scripts/run_servers.py`, `startup_scripts/run_mcp.py`, and `startup_scripts/run_llm.py`.

## Responsibilities

- **Lifecycle Management:** Orchestrates starting, tracking, and stopping child processes.
- **Cleanup:** Ensures all processes are terminated cleanly on exit (even if the parent crashes).
- **State Tracking:** Records active PIDs in a state file within `.processes/` to detect and kill stale processes from previous crashed sessions during startup.
- **Cross-Platform Support:** Handles differences in process management between Windows and POSIX systems.

## Architecture

### `ManagedProcess`

A dataclass representing a single child subprocess. It contains identifiers for the process, the underlying `subprocess.Popen` handle, and buffers to store the most recent output lines. 

### `ProcessManager`

The controller orchestrating multiple `ManagedProcess` instances. It should be initialized with standard context manager syntax (e.g. `with ProcessManager() as manager`). Includes:

- **`start_process(...)`**: Launches a process and tracks the process handle
- **`stop_process(managed)`**: Attempts a graceful termination (`SIGTERM`) followed by a forced kill (`SIGKILL`) if the process exceeds the configured `timeout`.
- **`shutdown_all()`**: Concurrently initiates shutdown of all tracked processes, following the `timeout` period before escalating to force kills.

## Example Usage in Launcher Scripts

`run_servers.py`:

1. **Initialization**: `ProcessManager` is instantiated, clearing any stale processes found in `.processes/`.
2. **Spawning**: It iterates through the configuration (devices, tiled, instrument) and calls `start_process` for each service.
3. **Execution**: The manager maintains the `active_processes` list.
4. **Shutdown**: Upon exit (or `Ctrl+C`), the manager's context manager (`__exit__`) invokes `shutdown_all()` to ensure a clean state for the next run.

## Whole-tree shutdown helpers

`process_manager.py` also exports module-level helpers used by the startup GUIs
and launcher scripts to guarantee that nothing outlives a stop request:

- **`kill_process_tree(root, grace, log)`**: snapshots every descendant of
  `root` (device servers live in their own sessions, so signalling the root's
  group alone is not enough), asks the root's group to shut down (`SIGTERM`, or
  Ctrl+Break on Windows), waits up to `grace` seconds so a launcher can unwind
  its `ProcessManager`, then force-kills every survivor, each survivor's process
  group and the root. Returns the PIDs that had to be force-killed.
- **`install_shutdown_signal_handler()`**: routes `SIGTERM`, `SIGHUP` and
  `SIGBREAK` into a single `KeyboardInterrupt`, so an external stop unwinds the
  `with ProcessManager()` block exactly like Ctrl+C. Used by `run_servers.py`
  and `run_mcp.py`.
- **`watch_parent_from_environment()`** / **`watch_parent_process(pid)`**: a
  daemon thread that raises `KeyboardInterrupt` in the main thread as soon as
  the process named by `ASYNCROSCOPY_PARENT_PID` disappears. The startup GUIs
  set that variable when launching, so a GUI that is force-killed still takes
  its servers down with it.
- **`assign_to_kill_on_close_job(pid)`**: on Windows, places `pid` and
  everything it spawns in a job object with `KILL_ON_JOB_CLOSE`; the tree dies
  when the owning process does. Returns `None` elsewhere.
- **`list_descendant_pids(pid)`** and **`pid_alive(pid)`**: the portable
  process-table queries the helpers above are built on.
