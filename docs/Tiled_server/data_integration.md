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
- **`.tiff`**: returns a JSON list with one key per detector, e.g.
  `["stem_image_<stamp>_HAADF.tiff", "stem_image_<stamp>_BF-S.tiff"]`;
  `client[key].read()` returns the array directly (no nesting).

## Acquisition metadata

Every file carries the instrument state at acquisition time as root attributes
(Tiled node metadata), taken from `ElectronMicroscope.acquisition_metadata()`:
`instrument_class`, `device_name`, `acquisition_time`, `acquisition_id`,
`stem_mode`, `stage_{x,y,z,alpha,beta}`, `defocus_m`, `fov_m`,
`image_shift_{x,y}_m`, `beam_tilt_{x,y}_rad`, and the detector device settings
prefixed by device (`scan_dwell_time`, `scan_imsize`, `scan_scan_region`,
`camera_exposure_time`, `eds_exposure_time`, ...). Entries a backend cannot read
are omitted; a `file_attrs` argument to `save_acquisition` overlays them.
For TIFF the same dict is json-encoded into the `ImageDescription` tag.

CORRECTOR archives `acquire_tableau` and `measure_c1a1` results to Tiled when
its `data_device_address` property is set (`tableau_corrector_*.h5`,
`c1a1_corrector_*.h5`, dataset `coefficients` with the CEOS result as a
json-encoded `result` attribute); the command return values are unchanged and
`get_last_archived_key` gives the key.

Through MCP, `list_acquisitions(acquisition_type, since, limit)` lists recent
keys with this metadata and `get_data_from_key(key)` describes one.

## Notebook setup

Connect to the DATA Tango device once at the beginning of a workflow. The
`save_path` directory should be visible to the Tiled HTTP server, and the
microscope device should have `data_device_address` set to this Tango device.

```python
import json
import tango

data = tango.DeviceProxy("asyncroscopy/data/default")
data.set_timeout_millis(120_000)
data.host = "10.46.217.241"
data.port = 9091
data.save_path = "/path/served/by/tiled"
```

Changing `data.save_path` creates the directory and restarts a DATA-managed
Tiled HTTP server. Each acquisition is registered explicitly after it is
written; DATA does not run a filesystem watcher. If no Tiled server is
reachable the file is still saved and its key returned (the digital twin runs
without Tiled); `get_config()["tiled_server_status"]` counts the unregistered
files and `register_save_path` registers them once a server is up. `startup_scripts/run_servers.py` sets
the extended Tango timeout automatically.

Acquire as usual. With the default `.h5` the return value is the Tiled key; with
`.tiff` it is a JSON list of keys (see the format contract above):

```python
key = mic.acquire_scanned_image(["HAADF", "BF-S"])   # .h5  → client[key]["image"]["HAADF"]
# scan.output_format = ".tiff"                        # .tiff → client[json.loads(keys)[0]].read()

camera_key = mic.acquire_camera_image()               # .h5  → client[camera_key]["image"]
# camera.output_format = ".tiff"                      # .tiff → client[json.loads(keys)[0]].read()

# Select Flucam through the same CAMERA device and acquisition command:
# camera.camera_detector = "Flucam"
# flucam_key = mic.acquire_camera_image()
```

## Server Roles

There are two data-related servers:

- `asyncroscopy/data/default` is the DATA Tango device server. It belongs to asyncroscopy and bridges notebooks or microscope devices to Tiled.
- `http://10.46.217.241:9091` is the Tiled HTTP data server. It indexes and serves files.

`startup_scripts/run_servers.py` starts the DATA device and its managed Tiled HTTP
server together. It also shuts down the managed Tiled server with the rest of
the server stack. To inspect the active directory, use:

```python
import json

json.loads(data.get_config())["tiled_server_serving"]
```

If the DATA device connects to an already-running external Tiled HTTP server,
it manually registers acquisitions but does not terminate that external server
during shutdown.

## Direct Tiled access

`asyncroscopy.data.tiled_client` is the one place that knows the URI and API
key; every reader (DATA device, MCP bridge, instrument devices) goes through it.

```python
from asyncroscopy.data.tiled_client import open_client, list_acquisitions

client = open_client("http://10.46.217.241:9091")
list(client)                                   # keys
list_acquisitions(client, "stem_image", limit=5)  # newest first, with metadata
```

Environment: `ASYNCROSCOPY_TILED_URI` (default `http://10.46.217.241:9091`),
`ASYNCROSCOPY_TILED_API_KEY` (default `secret`, must match the key the DATA
device starts the server with), `ASYNCROSCOPY_ACQUISITION_DIR`.

GUIs and the LLM never use Tango for data: the LLM reads previews through the
MCP tools above, and a GUI reads arrays straight from the Tiled HTTP server
using the URI from the `DATA_get_config` tool.

## Browser access

A web GUI needs no client library; Tiled's REST API serves everything:

| Need | Request |
|---|---|
| acquisitions with metadata | `GET /api/v1/search/?fields=metadata&fields=structure_family&page[limit]=200` (sort client-side by the timestamp in the key) |
| filter by metadata | `GET /api/v1/search/?filter[eq][condition][key]=instrument_class&filter[eq][condition][value]="DigitalTwin"` |
| one key's metadata | `GET /api/v1/metadata/<key>` |
| datasets under a key | `GET /api/v1/search/<key>` (nested: `/search/<key>/image`) |
| image as PNG | `GET /api/v1/array/full/<key>/image/HAADF?format=image/png` |
| raw values | `GET /api/v1/array/full/<key>/spectrum?format=application/json` |

The managed Tiled server only answers cross-origin requests from
`ASYNCROSCOPY_TILED_ALLOW_ORIGINS` (space or comma separated; default is the
loopback SciAgentGUI dev origins, the same list the MCP bridge allows).
`DATA_get_config` reports the active list as `allow_origins`.
