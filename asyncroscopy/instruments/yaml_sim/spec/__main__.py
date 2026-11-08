"""Check a YAML instrument file: ``python -m asyncroscopy.instruments.yaml_sim.spec my_instrument.yaml``."""

import json
import sys

from . import SpecError, load_spec, plan_example

# Check a file before starting it:  uv run python -m asyncroscopy.instruments.yaml_sim.spec my_instrument.yaml
if len(sys.argv) != 2:
    sys.exit("usage: python -m asyncroscopy.instruments.yaml_sim.spec <instrument.yaml>")
try:
    checked = load_spec(sys.argv[1])
except SpecError as exc:
    sys.exit(f"not valid: {exc}")
print(f"{checked.path}: valid. Commands (each one an agent tool, named <class_name>_<command>):")
for command_name in checked.command_names():
    print(f"  {command_name}")
for job_name, job_spec in checked.jobs.items():
    print(f"Job {job_name} (on the job device: run_job with plan {json.dumps(plan_example(checked, job_spec))})")
