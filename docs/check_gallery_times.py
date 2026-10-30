"""Fail when a gallery example ran longer than the budget: python docs/check_gallery_times.py

Reads docs/source/sg_execution_times.rst, which sphinx-gallery writes after
`make html` with one row per example, and exits 1 listing every example over
BUDGET seconds. docs.yml runs it after a build without numba, the slowest
install (and the one JupyterLite users get), so the budget holds everywhere.
"""

import re
import sys
from pathlib import Path

BUDGET = 5.0  # seconds per example, as CLAUDE.md asks of every gallery script

TIMES = Path(__file__).parent / "source" / "sg_execution_times.rst"

# A row is "* - :ref:`...` (``../../examples/<path>.py``)" followed by "- MM:SS.mmm".
ROW = re.compile(r"\(``(?P<path>[^`]+\.py)``\)\s*\n\s*- (?P<min>\d+):(?P<sec>\d+\.\d+)")


def main(path: Path = TIMES) -> int:
    if not path.exists():
        print(f"{path} not found: run `make html` first")
        return 1
    rows = [(m["path"], 60 * int(m["min"]) + float(m["sec"])) for m in ROW.finditer(path.read_text())]
    if not rows:
        print(f"no example timings found in {path}")
        return 1
    slow = sorted((t, p) for p, t in rows if t > BUDGET)
    for t, p in reversed(slow):
        print(f"{t:6.2f} s  {p.removeprefix('../../')}")
    if slow:
        print(f"{len(slow)} of {len(rows)} gallery examples exceed the {BUDGET:g} s budget")
        return 1
    slowest_t, slowest_p = max((t, p) for p, t in rows)
    print(f"all {len(rows)} gallery examples within {BUDGET:g} s (slowest {slowest_t:.2f} s, {slowest_p.removeprefix('../../')})")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else TIMES))
