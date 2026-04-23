If you’re editing this class - `Microscope` located at asyncroscopy/Microscope.py, you’re usually doing one of these:

1. **Adding or modifying attributes of the base Microscope**
   (Expose new device state that clients can read.)

2. **Updating attribute read/write methods of the base Microscope**
   (Control how attribute values are validated, stored, or synchronized with AutoScript.)

3. **Adding or modifying commands**
- `get_image`
- `get_spectrum`

3. **ADD a new functionality pertaining to a microscope api - Adding Internal acquisition helpers in the base Microscope**
- examples - already existing
   - `_acquire_stem_image`
   - `_acquire_stem_image_advanced`
   - `_acquire_spectrum`

5. **Changing the transport format**
   (Modify DevEncoded usage, metadata schema, caching policy, or multi-image retrieval semantics.)

6. **Improving robustness**
   (Handle connection failures, missing proxies, AutoScript errors, simulation fallback logic, or state transitions like `FAULT`, `ON`, `OFF`.)

