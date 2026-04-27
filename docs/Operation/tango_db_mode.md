
```
workflow:

Start Tango DB
      ↓
Register device in DB
      ↓
Run device server
      ↓
Connect using DeviceProxy
```

## list of commands
"""
1) TANGO_HOST=localhost:11000 uv run python -m tango.databaseds.database 2
- returns:
    - Ready to accept request


2) in another terminal
export TANGO_HOST=localhost:11000 
uv run scripts/register_devices.py 
- returns:
    - Connected: MacBook-Pro-649.local:11000
    - registered: test/haadf/1
    - registered: test/microscope/1
    - property:   haadf_device_address = test/haadf/1

    Done!

3) in another terminal
export TANGO_HOST=localhost:11000 
uv run python -m asyncroscopy.detectors.HAADF haadf_instance
- returns:
    - Ready to accept request

4) in another terminal
export TANGO_HOST=localhost:11000 
uv run python -m asyncroscopy.ThermoMicroscope microscope_instance
- returns:
    - True
        Client connecting to [localhost:9090]...
        {'haadf': 'test/haadf/1', 'AdvancedAcquistion': None}
        Ready to accept request

5) client side
export TANGO_HOST=localhost:11000 
DeviceProxy("test/haadf/1")
"""


## Why use Tango Database Mode?

Using the **Tango database** provides several advantages over running devices in non-database mode.


### 1. Centralized device registry

All devices are registered in a single database. Clients only need the **device name**, and Tango resolves where the device server is running.

### 2. No manual port management

Clients do not need to know host addresses or ports for each device. The database handles device location, avoiding the need to open and manage multiple ports.

### 3. Deterministic system startup

The system can be brought up in a reproducible way:

```
Start Tango DB
Register devices
Start device servers
Connect via DeviceProxy
```

This makes it easy to automate instrument initialization.

### 4. Device discovery

Clients can query the database to discover available devices, classes, and servers, enabling dynamic workflows.

``` python
# List all devices
import tango

db = tango.Database()
devices = db.get_device_name("*", "*")

for d in devices:
    print(d)
```

``` python
# List devices of a specific class
db.get_device_name("HAADF", "*")

# List all device classes
db.get_class_list("*")

# List all device servers
db.get_server_list("*")
```

### 5. Configuration via database properties

Device configuration (e.g., dependencies between devices) can be stored as **database properties**, avoiding hardcoded relationships in code.

``` Python
db.put_device_property(MICRO_DEVICE,{"haadf_device_address": [HAADF_DEVICE]})

```

### 6. Supports distributed instruments

Device servers can run on different machines while clients connect using only the device name.

### 7. Scalable system architecture

Database mode enables building modular systems where higher-level devices orchestrate multiple lower-level devices (e.g., microscope → detectors → acquisition).

---

✔ In practice, this allows a workflow where the entire system can be initialized via scripts and then accessed from tools like **Jupyter notebooks** using simple `DeviceProxy` calls.
