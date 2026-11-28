# Troubleshooting

Most startup problems have one of three causes: an old server that still runs, a stale `.db` file, or a slow network. See [running the servers](run-servers.md) for a normal start.

## New data is on disk but not in Tiled

An old Tiled server still serves an old save directory. Only one Tiled server must run, on the current save path.

Find and stop the old process on the Tiled port (default `9091`). On Windows:

```powershell
netstat -ano | findstr :9091
taskkill /PID <PID> /F
```

To prevent this, let `run_servers.py` manage Tiled. It starts Tiled on the save path from the config and stops it on Ctrl+C. A new `data.save_path` also restarts the managed Tiled server on the new path.

## The stack fails to start after you pull new changes

Delete the stale `.db` files, both the Tiled catalog and the Tango database. Then start again. A clean run builds both databases again.

## A strange error or a timeout at startup

Old servers probably still run. On Windows the PIDs change often, so a kill by PID is not reliable.

- Run `run_servers.py` again. It clears old processes first. Without `--yaml`, answer yes to "Clear old processes first".
- Run `run_servers.py --yaml <config> --cleanup` to stop only what an earlier run of that config left behind.
- To stop one device server by name:

  ```python
  import tango
  db = tango.Database()
  print(db.get_server_list("CORRECTOR/*"))
  tango.DeviceProxy("dserver/CORRECTOR/corrector_instance").command_inout("Kill")
  ```

- Add `--debug` to write each server's output to a log file.

## The database does not start

`tango.Database()` only connects to a database that runs. It does not start one. `run_servers.py` starts it, or you can start it by hand:

```bash
TANGO_HOST=localhost:9094 uv run python -m tango.databaseds.database 2
```

If it is still not ready after about 2 minutes, set `tango.reset_database_file: true` in the config, or tick "Delete tango_database.db before start" in the GUI. Then try again.

## Servers stop responding after many calls

This was seen after repeated `place_beam` calls. Stop the notebook kernel, turn on the database file reset, and start the servers again.

## Timeouts from a remote connection

A slow or unstable network is the usual cause, not the code. Run the command again. If it keeps happening, work on the instrument PC itself through remote desktop, so the connection to the servers stays local.

## A package version mismatch

Delete `.venv`, then run `uv sync` to build the environment again.
