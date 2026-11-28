# Data and Tiled

Acquisition commands save data to files on the instrument side and return a key. The DATA device (`asyncroscopy/data/data.py`) registers each file with a [Tiled](https://github.com/bluesky/tiled) HTTP server. Clients read the data from Tiled with the key.

There are two servers:

- `asyncroscopy/data/default` is the DATA Tango device. It starts Tiled and registers files.
- The Tiled HTTP server indexes and serves the files. Its address is `tiled.host:tiled.port` in the config.

`run_servers.py` starts DATA and its Tiled server together, and stops Tiled at shutdown. If DATA connects to a Tiled server that it did not start, it registers files there but does not stop that server.

## File formats

The default is one HDF5 file per acquisition. Each detector output is a dataset, and the vendor metadata is stored as HDF5 attributes.

Images can be saved as `.tiff` instead, with one Velox-compatible file per detector. Set `scan.output_format = ".tiff"` for scanned images or `camera.output_format = ".tiff"` for camera images. Spectra and STEM data are always HDF5.

`save_acquisition` (`asyncroscopy/data/data_writer.py`) accepts both vendor shapes:

- AutoScript returns adorned images that carry pixels and metadata together.
- PyJEM (JEOL) returns raw pixels. The metadata comes from `get_detectorsetting()` and goes in as `dataset_attrs`.

For `.tiff`, adorned images save their metadata natively. For raw arrays, the metadata is JSON-encoded into the TIFF `ImageDescription` tag.

## Reading data back

| Format | Command returns | Read it with |
|---|---|---|
| `.h5` | One Tiled key | `client[key]["image"]["HAADF"]`, `client[key]["spectrum"]` or `client[key]["stem_data"]`. A single camera frame is `client[key]["image"]`. |
| `.tiff` | A shared stem | `client[f"{stem}_{DET}.tiff"].read()`, one key per detector |

```python
import tango
from tiled.client import from_uri

mic = tango.DeviceProxy("asyncroscopy/instrument/default")
client = from_uri("http://<tiled-host>:9091")

key = mic.acquire_scanned_image(["HAADF", "BF-S"])
haadf = client[key]["image"]["HAADF"].read()

camera_key = mic.acquire_camera_image()
frame = client[camera_key]["image"].read()
```

To use Flucam, set `camera.camera_detector = "Flucam"` on the CAMERA device. Then call `acquire_camera_image()` as usual.

## Changing the save directory

```python
import json
import tango

data = tango.DeviceProxy("asyncroscopy/data/default")
data.set_timeout_millis(120_000)
data.save_path = "/path/served/by/tiled"

json.loads(data.get_config())["tiled_server_serving"]   # the directory Tiled serves
```

A new `save_path` creates the directory and restarts the DATA-managed Tiled server on it. So the server and the save path stay the same. DATA registers each file after it is written. It does not watch the file system. `run_servers.py` sets the long Tango timeout for you.
