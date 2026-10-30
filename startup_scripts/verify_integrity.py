"""Check that every sealed file is still what was written, and that the integrity manifest is intact.

    uv run startup_scripts/verify_integrity.py              # the manifest under outputs/integrity (or $ASYNCROSCOPY_INTEGRITY_LOG)
    uv run startup_scripts/verify_integrity.py FILE...      # only these files

Prints what changed, what went missing and any break in the manifest's chain, then the chain's head: copy the
head somewhere else now and then (a lab notebook) so a manifest whose end was cut off is noticed too. Exits 1 when
anything is wrong. Needs the integrity key (asyncroscopy/data/integrity.py says where it lives).
"""

from __future__ import annotations

import json
import sys

from asyncroscopy.data.integrity import manifest_path, verify


def main(argv: list[str] | None = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:]) or None
    report = verify(paths)
    print(f"manifest: {manifest_path()}")
    print(json.dumps(report, indent=2))
    if report["ok"]:
        print(f"OK: {report['checked']} sealed file(s) unchanged; chain head {report['head']}")
        return 0
    print("NOT OK: see changed, missing and log_problems above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
