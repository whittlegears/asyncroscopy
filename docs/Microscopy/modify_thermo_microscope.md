If you’re editing this class, you’re usually doing one of these:


1. **Adding or modifying attributes**
   (Expose new device state that clients can read.)

2. **Updating attribute read/write methods**
   (Control how attribute values are validated, stored, or synchronized with AutoScript.)

3. **Adding or modifying commands**
   (Update input validation, settings retrieval via `DeviceProxy`, orchestration logic, caching behavior, or metadata packaging.)

4. **Adding a new detector**
   (Add a device property for its address, register it in `_connect_detector_proxies`, ensure naming normalization, and map it correctly in acquisition helpers.)
   ```python
   "newdet": self.newdet_device_address,
   ```

5. **ADD a new functionality pertaining to a microscope api - Adding Internal acquisition helpers**
- examples - already existing
   - `_acquire_stem_image`
   - `_acquire_stem_image_advanced`
   - `_acquire_spectrum`
-  **Adding or changing acquisition settings**
   (Extend what is read from detector devices—e.g., dwell time, resolution, scan region—and ensure they propagate correctly into acquisition helpers.)

