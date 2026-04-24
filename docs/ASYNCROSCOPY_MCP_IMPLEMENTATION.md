# Asyncroscopy MCP Server Implementation

How the Asyncroscopy system bridges pyTango device control and LLM agents.

## Overview

The Asyncroscopy MCP server exposes microscopy hardware (via pyTango) to language models. This enables LLM-driven microscopy workflows without direct hardware knowledge.

## Tango Architecture

Asyncroscopy uses pyTango as the hardware abstraction layer:

```
LLM Agent
    ↓
MCP Client
    ↓
MCPServer (Asyncroscopy)
    ↓
Tango DeviceProxy
    ↓
Hardware (Microscope, Detectors, Stage, etc.)
```

PyTango decouples software from hardware through networked device objects. Each device exports commands and attributes.

## Device Classes

Asyncroscopy defines Tango Device subclasses for microscopy hardware:

### Base: `Microscope` (asyncroscopy/Microscope.py)

Core microscope control:
- `get_image()` - Acquire STEM image
- `get_spectrum()` - Acquire spectrum
- Attributes: voltage, magnification, probe_current

### Thermo Fisher Microscope: `ThermoMicroscope` (asyncroscopy/ThermoMicroscope.py)

Extends Microscope with multi-detector orchestration:
- Connects detector proxies (HAADF, EELS, EDS)
- Coordinates acquisition across detectors
- Manages state synchronization

### Detectors (asyncroscopy/detectors/)

Individual detector devices:
- `CAMERA.py` - Generic camera control
- `EDS.py` - Energy dispersive X-ray spectroscopy
- `EELS.py` - Electron energy loss spectroscopy

### Hardware (asyncroscopy/hardware/)

Low-level hardware control:
- `STAGE.py` - Specimen stage movement
- `CORRECTOR.py` - Aberration corrector
- `SCAN.py` - Beam scanning control

## MCP Server Discovery

When MCPServer starts, it performs discovery:

```python
server = MCPServer(
    name="Asyncroscopy",
    tango_host="microscope.lab",
    tango_port=9094,
    search_packages=["asyncroscopy"]
)
server.setup()
```

### Discovery Steps

1. **Connect to Tango Database**
   ```python
   self.database = Database(tango_host, tango_port)
   ```

2. **List All Exported Devices**
   ```python
   devices = self.database.get_device_exported("*")
   # Returns: ["test/microscope/1", "test/detector/eds", ...]
   ```

3. **Filter by Class**
   ```python
   # Skip infrastructure (DataBase, DServer)
   # Skip blocked classes (e.g., SimulatedStage)
   available = [d for d in devices if not blocked(d)]
   ```

4. **Extract Commands per Device**
   ```python
   dev = DeviceProxy(device_name)
   commands = dev.command_list_query()
   # Returns CommandInfo for each command
   ```

5. **Build Tool Wrappers**
   ```python
   for cmd in commands:
       if not blocked(cmd.name):
           wrapper = self._create_wrapper(cmd)
           tools[dev_class][cmd.name] = wrapper
   ```

6. **Find Source Code**
   ```python
   cls = self._get_tango_device_class("Microscope")
   # Searches asyncroscopy package for class definition
   ```

7. **Register with MCP**
   ```python
   for wrapper_func in all_wrappers:
       tool = Tool.from_function(wrapper_func)
       mcp.add_tool(tool)
   ```

## Example: Image Acquisition

How an LLM acquires a microscope image through MCP:

### Tango Device Definition

```python
# asyncroscopy/Microscope.py
class Microscope(Device):
    @command(dtype_in=int, dtype_out=str)
    def get_image(self, exposure_ms: int) -> str:
        """Acquire a STEM image with specified exposure time."""
        # Interact with AutoScript microscope API
        img = self._acquire_stem_image(exposure_ms)
        # Return as DevEncoded (binary image + metadata)
        return (json.dumps(metadata), img.tobytes())
```

### MCP Tool Registration

1. Server queries Tango: `Microscope.get_image` exists
2. Extracts parameter name from source: `exposure_ms`
3. Maps Tango type to Python: `int` → `int`
4. Builds function signature:
   ```python
   def Microscope_get_image(exposure_ms: int) -> dict:
       """Acquire a STEM image with specified exposure time.
       
       Tango Device Class: Microscope
       Tango Command: get_image
       """
       result = dev.get_image(exposure_ms)
       return {
           "encoding": "base64",
           "metadata": metadata,
           "payload": base64.b64encode(img).decode()
       }
   ```

### LLM Usage

```
Agent: "Acquire an image with 5ms exposure."

MCP Server invokes: Microscope_get_image(exposure_ms=5)

Result: {
    "encoding": "base64",
    "metadata": {"shape": [1024, 1024], "dtype": "uint8"},
    "payload": "iVBORw0KGgo..."
}

Agent: "Image acquired. Dimensions: 1024×1024."
```

## Multi-Device Coordination

ThermoMicroscope orchestrates multiple detector devices:

```python
class ThermoMicroscope(Microscope):
    def __init__(self, cl, name):
        super().__init__(cl, name)
        # Connect to detector device proxies
        self.haadf = DeviceProxy(self.haadf_device_address)
        self.eels = DeviceProxy(self.eels_device_address)
        self.eds = DeviceProxy(self.eds_device_address)
    
    @command(dtype_out=str)
    def acquire_multimodal(self) -> str:
        """Acquire STEM image + EELS spectrum simultaneously."""
        # Coordinate detector settings
        self.haadf.sync_dwell(self.scan_dwell_time)
        self.eels.sync_energy_range(self.energy_range)
        
        # Acquire data
        stem_data = self.haadf.acquire()
        eels_data = self.eels.acquire()
        
        # Return combined result
        return (metadata, combined_data)
```

The MCP server automatically exposes `acquire_multimodal` as a tool.

## Type Handling

### Scalar Types

```python
# Tango → MCP
DevInt32 → int
DevFloat64 → float
DevBoolean → bool
DevString → str
```

Tool parameter validates input type before sending to hardware.

### Array Types

```python
# Tango → MCP
DevVarULongArray → list[int]
DevVarFloatArray → list[float]
```

MCP converts JSON array to typed Python list.

### DevEncoded (Binary Data)

Used for images, spectra, and complex structures:

```python
# In Tango command
result = (metadata_string, image_bytes)
return result  # DevEncoded type

# In MCP tool
normalized = self._normalize_command_result(DevEncoded, result)
# Returns:
{
    "encoding": "base64",
    "metadata": metadata_string,
    "payload": base64_encode(image_bytes)
}
```

Agent receives JSON-safe structure. To use binary data, agent decodes base64:

```python
import base64
payload = base64.b64decode(result["payload"])
img_array = np.frombuffer(payload, dtype=np.uint8).reshape(...)
```

## Configuration for Custom Hardware

To add your own hardware to the Asyncroscopy MCP server:

### 1. Define a Tango Device

```python
# mymodule/my_detector.py
from tango.server import Device, command, attribute

class MyDetector(Device):
    @attribute(dtype=float)
    def signal_level(self):
        return self._get_signal()
    
    @command(dtype_in=int, dtype_out=str)
    def measure(self, duration_ms: int) -> str:
        """Measure signal for specified duration."""
        data = self._measure(duration_ms)
        return (json.dumps(metadata), data.tobytes())
```

### 2. Create a Device Server

```python
# mymodule/server.py
from tango.server import run

from mymodule.my_detector import MyDetector

if __name__ == "__main__":
    run([MyDetector])
```

### 3. Register Device in Tango Database

```bash
tango_admin --add-server MyServer/myinstance MyDetector test/detector/custom
```

### 4. Start Device Server

```bash
python mymodule/server.py
```

### 5. Create MCP Server with Custom Package

```python
from asyncroscopy.mcp.mcp_server import MCPServer

server = MCPServer(
    name="CustomMicroscopy",
    tango_host="localhost",
    tango_port=9094,
    search_packages=["mymodule", "asyncroscopy"]
)
server.start()
```

The MCP server now discovers and exposes your custom detector.

## Adding Custom MCP Tools

Extend MCPServer to add tools that coordinate or analyze:

```python
class EnhancedMicroscopyServer(MCPServer):
    @tool()
    def focus_iteratively(self, tolerance_nm: float = 1.0) -> dict:
        """Automatically focus microscope using iterative approach."""
        microscope = tango.DeviceProxy("test/microscope/1")
        stage = tango.DeviceProxy("test/stage/1")
        
        best_focus = None
        best_contrast = 0
        
        # Scan focus range
        for z in range(-100, 101, 10):
            stage.move_z(z)
            img = microscope.get_image(10)  # 10ms exposure
            contrast = self._calculate_contrast(img)
            
            if contrast > best_contrast:
                best_contrast = contrast
                best_focus = z
        
        return {"best_focus_um": best_focus / 1000.0, "contrast": best_contrast}
    
    @tool()
    def suggest_optimal_conditions(self, material: str) -> dict:
        """Suggest microscopy parameters for a material."""
        params = {
            "Si": {"voltage_kv": 200, "exposure_ms": 5, "magnification": 500000},
            "Au": {"voltage_kv": 100, "exposure_ms": 10, "magnification": 1000000},
        }
        return params.get(material, params["Si"])
```

## Blocking Commands

Exclude dangerous or irrelevant commands:

```python
server = MCPServer(
    name="SafeMicroscopy",
    tango_host="localhost",
    tango_port=9094,
    blocked_functions={
        "*": ["Init", "Status"],  # Skip lifecycle commands
        "Microscope": ["emergency_shutdown"],  # Class-specific
    }
)
```

Commands in the block list do not appear as MCP tools.

## Debugging

### Enable Verbose Output

```python
server = MCPServer(
    name="Debug",
    tango_host="localhost",
    tango_port=9094,
    verbose=True
)
server.setup()
```

Output shows:
- Discovered devices
- Available commands per device
- Registered tools
- Tool signatures

### Inspect Registered Tools

```python
# After setup()
for dev_class, commands in server.tools.items():
    print(f"{dev_class}:")
    for cmd_name, func in commands.items():
        print(f"  • {cmd_name}: {func.__doc__}")
```

### Test Tool Manually

```python
# Call the wrapped function directly
import asyncio
result = server.tools["Microscope"]["get_image"](exposure_ms=10)
print(result)
```

## Performance Considerations

### Caching Device Proxies

DeviceProxy creation is expensive. Cache them:

```python
class OptimizedServer(MCPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device_cache = {}
    
    def _get_device(self, name):
        if name not in self._device_cache:
            self._device_cache[name] = DeviceProxy(name)
        return self._device_cache[name]
```

### Avoid Blocking Operations

Use async patterns for long-running commands:

```python
from fastmcp.tools import tool

class AsyncMicroscopyServer(MCPServer):
    @tool()
    async def acquire_mosaic_async(self, tiles_x: int, tiles_y: int) -> dict:
        """Acquire mosaic (may take minutes)."""
        import asyncio
        results = []
        for i in range(tiles_x):
            for j in range(tiles_y):
                result = await asyncio.to_thread(
                    self._acquire_tile, i, j
                )
                results.append(result)
        return {"tiles": results}
```

## References

- [Tango Device Programming](http://www.tango-controls.org/developers/python-api/)
- [FastMCP Tools Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Protocol Specification](https://modelcontextprotocol.io/specification)
- Asyncroscopy Source: `asyncroscopy/`
