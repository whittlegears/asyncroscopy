# Instrument

## Devices

- RamanTwin: confocal Raman microscope twin, 532 nm laser, 1800 l/mm grating, 100x objective, CCD spectrometer and a white-light camera. laser_on/laser_off, stage x/y in metres, focus offset, exposure and accumulations; acquire_raman_spectrum (Raman shift 100-3200 cm-1) and acquire_brightfield_image.

## Jobs

- None: no job device. A line or map of spectra is a loop of set_stage and acquire_raman_spectrum calls, each under 20 s.

## Conventions

- Units: metres for stage and focus, mW for laser power at the sample, seconds for exposure, cm-1 for Raman shift.
- Coordinates: stage (0, 0) is the wafer centre; the graphene flake covers x > 0.
- Data: keys in Tiled; look at them with view_data (text lists peaks in cm-1).

## Limits

- Laser at most 10 mW on graphene (it burns); 50 mW maximum on bare silicon.
- One spectrum (exposure_s x accumulations) at most 20 s.
- Switch the laser off at the end of every request.

## Samples

- Graphene flake exfoliated on Si/SiO2 (300 nm oxide). Si peak at 520.7 cm-1; graphene G 1582 and 2D 2690 cm-1.
