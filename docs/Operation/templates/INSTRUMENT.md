# Instrument

Copy this file to `outputs/agent_memory/INSTRUMENT.md` (or `$ASYNCROSCOPY_AGENT_MEMORY/INSTRUMENT.md`) and fill it in. Agents read the sections whose words match their task, a few hundred characters at a time, so keep each section short and use the words people will use in requests.

## Devices

- XRDTwin: powder X-ray diffractometer twin (tube power, 2theta range, step count, count time).
- (one line per device: what it is, what its main commands do)

## Jobs

- Job devices (run_job, then wait_for_job): (names)
- A long experiment is one job, not a loop of moves and acquisitions.

## Conventions

- Units: (metres, degrees, kV, mA, seconds...)
- Coordinates: (what a stage position means, where the origin is)
- Data: keys in Tiled; look at them with view_data (text or png).

## Limits

- (stage travel, field of view range, maximum dose or power, detectors to protect)
- (what must never be done, and what needs an operator)

## Samples

- (what is loaded, where, what it is made of, anything fragile)
