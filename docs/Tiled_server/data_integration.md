# DATA acquisition workflow

See more at https://github.com/bluesky/tiled.

`AutoScriptMicroscope` saves real AutoScript acquisitions on the microscope side
and returns the registered Tiled key through Tango. `asyncroscopy/data/data.py`
is the Tango data device for registering those files with the Tiled HTTP server.

The default format is one HDF5 file per acquisition event: each correlated
output is a dataset, with parsed AutoScript XML leaf metadata as HDF5 attributes.
Image acquisitions can instead write `.tiff` (Velox-compatible, one file per
detector) via `scan.output_format = ".tiff"` for scanned images or
`camera.output_format = ".tiff"` for camera images; spectra and STEM data are
always HDF5.

Vendors return two shapes and `save_acquisition` absorbs both: AutoScript adorned
images bundle pixels and metadata, while PyJEM (JEOL) returns raw pixels plus a
separate `get_detectorsetting()` dict passed as `dataset_attrs`. Metadata lands as
HDF5 attributes for `.h5`; for `.tiff`, adorned images save it natively while
raw-array vendors get the dict json-encoded into the TIFF `ImageDescription` tag.

Acquisition commands that feed this pipeline include `acquire_scanned_image`,
`acquire_spectrum`, `acquire_camera_image`, and `acquire_scanned_data_advanced`.

What a command returns — and how you read it back — depends on the format:

- **`.h5`** (default): returns one Tiled key, read nested.
  `client[key]["image"]["HAADF"]` (one sub-dataset per detector), `["spectrum"]`,
  or `["stem_data"]`; single camera frames use `image`.
- **`.tiff`**: returns a shared *stem*. Rebuild one key per detector as
  `client[f"{stem}_{DET}.tiff"]`; `.read()` returns the array directly (no nesting).

## Notebook setup

Connect to the DATA Tango device once at the beginning of a workflow. The
`save_path` directory should be visible to the Tiled HTTP server, and the
microscope device should have `data_device_address` set to this Tango device.

```python
import json
import tango

data = tango.DeviceProxy("asyncroscopy/data/default")
data.set_timeout_millis(120_000)
data.host = "127.0.0.1"   # the default; on the lab PC, the address Tiled serves on
data.port = 9091
data.save_path = "/path/served/by/tiled"
```

Changing `data.save_path` creates the directory and restarts a DATA-managed
Tiled HTTP server in the background (`get_config()` says "restarting for the new
settings" until it serves the new path). Each acquisition is registered explicitly
after it is written; DATA does not run a filesystem watcher. Write a file fully,
then register it: Tiled opens the file to read its structure, and on Windows an
HDF5 file still open for writing in another process cannot be read. (The Mapper
writes its run file as `maps/map_<run>.h5.partial` and renames it to `.h5` when the
job finishes or is released, so a map in progress is never registered.) `startup_scripts/run_servers.py` sets
the extended Tango timeout automatically.

Acquire as usual. With the default `.h5` the return value is the Tiled key; with
`.tiff` it is the shared stem (see the format contract above):

```python
key = mic.acquire_scanned_image(["HAADF", "BF-S"])   # .h5  → client[key]["image"]["HAADF"]
# scan.output_format = ".tiff"                        # .tiff → client[f"{key}_HAADF.tiff"].read()

camera_key = mic.acquire_camera_image()               # .h5  → client[camera_key]["image"]
# camera.output_format = ".tiff"                      # .tiff → client[f"{stem}_BM-Ceta.tiff"].read()

# Select Flucam through the same CAMERA device and acquisition command:
# camera.camera_detector = "Flucam"
# flucam_key = mic.acquire_camera_image()
```

## Server Roles

There are two data-related servers:

- `asyncroscopy/data/default` is the DATA Tango device server. It belongs to asyncroscopy and bridges notebooks or microscope devices to Tiled.
- The Tiled HTTP data server (`http://127.0.0.1:9091` unless the YAML's `tiled:` section says otherwise) indexes and serves files.

`startup_scripts/run_servers.py` starts the DATA device and its managed Tiled HTTP
server together. It also shuts down the managed Tiled server with the rest of
the server stack. `get_config()` reports what the managed server is doing:

```python
import json

config = json.loads(data.get_config())
config["tiled_server_serving"]   # the directory it serves
config["tiled_server_status"]    # e.g. "running; serving path", or why it is not
config["tiled_log"]              # Tiled's warnings and errors (outputs/logs/tiled/, newest five kept; no per-request lines)
config["restarts"], config["last_restart_reason"]
```

How DATA keeps Tiled usable:

- **Supervised.** A Tiled that DATA started is checked every 5 s. If it exits, stops answering three checks in a row, or another process takes its port, DATA restarts it (backing off, and giving up after 5 restarts in 10 minutes, with the reason in `tiled_server_status`; `start_tiled_server` starts it again and resets the count). A Tango `Init` on DATA starts it again too, if it had been started.
- **Files written while the supervised Tiled is down are not lost.** `register_path` then returns the key with the status `deferred: Tiled down`, and DATA runs `reconcile` on its next check once Tiled answers (and after each restart; a failed reconcile is retried), which registers every acquisition file under the save path that Tiled does not have yet (files changed in the last 10 s are left for a later pass, as they may still be being written). With an external Tiled, a stopped one, one DATA gave up on, or a file outside the save path, nothing would register the file later, so a Tiled outage is an error instead.
- **Calls fail in seconds, not minutes.** Each request to Tiled has bounded timeouts, and asyncroscopy limits the Tiled client's own retries to 2 attempts within 3 s (it would otherwise retry for 45 s; `TILED_RETRY_ATTEMPTS` and `TILED_RETRY_TIMEOUT` override this). `register_path` answers within about 10 s (a 2 s check that Tiled answers, then up to 8 s), inside the 12 s the instrument devices wait for DATA: a registration that takes longer carries on in the background, the status says "still registering in the background", and reconcile catches it if it fails. A collision, an unreadable file or a registration Tiled never finishes (20 s) is an error, never a deferral.
- **Registering twice is safe.** Tiled itself deletes an entry when the same key is registered again; `register_path` returns the key unchanged when that same file is already registered, and refuses a different file with a key that is taken.
- **Only files Tiled can read are registered:** `.h5`, `.hdf5`, `.hdf`, `.nxs`, `.tif`, `.tiff`. Anything else fails at once with that list.
- **`reconcile` registers files one by one, never the directory.** Registering a directory makes Tiled first delete everything under its root, including stitched maps and segmentations written through the Tiled API. `register_on_startup: true` in the YAML runs `reconcile` at startup.
- **It checks that what answers is Tiled, and whose it is.** Only a Tiled about-document at `/api/v1/` counts as "running". If another program holds the port, or a Tiled serving a different directory does, `start_tiled_server` refuses with the process name and pid. A Tiled orphaned by an earlier run (a `tiled serve` reading this save path whose starting process is gone) is stopped and replaced by a supervised one; if its starter still runs, it belongs to another stack or an operator, and DATA refuses with both pids instead of stopping it. If the operating system will not say which process holds the port (macOS without administrator rights, for example), DATA refuses rather than guess: stop the old server or change `tiled.port`.

If the DATA device connects to an already-running external Tiled HTTP server,
it manually registers acquisitions but does not terminate that external server
during shutdown.

The API key is `ASYNCROSCOPY_TILED_API_KEY` (default `secret`, which SciAgentGUI also uses). The default is fine while Tiled listens only on `127.0.0.1`; `run_servers.py` warns when Tiled listens on another address with the default key.

## Direct Tiled access

From a notebook:

```python
from tiled.client import from_uri

client = from_uri("http://127.0.0.1:9091", api_key="secret")
list(client)  # acquisition files and stitched maps
```

Code inside asyncroscopy reaches Tiled only through `asyncroscopy.data.tiled_client`
(`connect`, `node_for_key`, `uri_from_data_device`): bounded timeouts, two retries,
and errors that say whether Tiled is down (`TiledUnavailable`) or the key does not
exist (`DataKeyNotFound`, naming keys that start with it).
