"""Check that every sealed file is still what was written, and that the integrity manifest is intact.

    uv run startup_scripts/verify_integrity.py              # the manifest under outputs/integrity (or $ASYNCROSCOPY_INTEGRITY_LOG)
    uv run startup_scripts/verify_integrity.py FILE...      # only these files
    uv run startup_scripts/verify_integrity.py --seal FILE...  # seal files another program wrote, then check everything

``--seal`` is for what the analysis server writes, since it has no key: a ``.jsonl`` log (its placement log) is
sealed up to its last complete line and may keep growing; any other file (a stitched map saved as HDF5) is sealed
whole. Run it again later to seal the lines added since.

Prints what changed, what went missing and any break in the manifest's or the dose ledger's chain, then each chain's head: copy the
head somewhere else now and then (a lab notebook) so a manifest whose end was cut off is noticed too. Exits 1 when
anything is wrong. Needs the integrity key (asyncroscopy/data/integrity.py says where it lives).
"""

from __future__ import annotations

import json
import sys

from asyncroscopy.data.dose_ledger import ledger_path
from asyncroscopy.data.integrity import IntegrityError, manifest_path, seal, seal_log, verify


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args[:1] == ["--seal"]:
        for path in args[1:]:
            try:
                entry = (seal_log if path.endswith(".jsonl") else seal)(path, writer="verify_integrity --seal")
            except (IntegrityError, OSError) as exc:
                print(f"NOT SEALED: {path}: {exc}")
                return 1
            print(f"sealed {path}" + (f" ({entry['lines']} line(s))" if "lines" in entry else ""))
        args = []
    paths = args or None
    report = verify(paths, logs=(ledger_path(),))
    print(f"manifest: {manifest_path()}")
    print(f"dose ledger: {ledger_path()}")
    print(json.dumps(report, indent=2))
    if report["ok"]:
        print(f"OK: {report['checked']} sealed file(s) unchanged; chain head {report['head']}")
        for log, head in report.get("log_heads", {}).items():
            print(f"    {log} head {head}")
        return 0
    print("NOT OK: see changed, missing and log_problems above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
