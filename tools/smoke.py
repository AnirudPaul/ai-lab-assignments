"""Fast syntax/runtime check for one or more src/*.py question files.

Concatenates their code cells (always prefixed with 00_title.py's setup) and runs them
in a headless process.  Much faster than a full notebook execution, so it is the inner
loop while writing questions.

    python tools/smoke.py 01_dfs_bfs.py 02_astar.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# The Jupyter kernel writes UTF-8; a bare Windows console defaults to cp1252 and would
# raise UnicodeEncodeError on output the real notebook handles fine. Match the kernel.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import parse_cells  # noqa: E402


def main(argv: list[str]) -> int:
    names = argv[1:]
    if not names:
        print("usage: python tools/smoke.py <src file> [...]")
        return 2
    files = [ROOT / "src" / "00_title.py"] + [ROOT / "src" / n for n in names]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("missing:", ", ".join(str(m) for m in missing))
        return 2

    ns: dict = {"__name__": "__main__"}
    for f in files:
        for i, (kind, src) in enumerate(parse_cells(f)):
            if kind != "code":
                continue
            try:
                exec(compile(src, f"{f.name}:cell{i}", "exec"), ns)
            except Exception:
                print(f"\n{'='*70}\nFAILED in {f.name} cell {i}\n{'='*70}")
                print(src[:1500])
                print("-" * 70)
                traceback.print_exc()
                return 1
        print(f"OK  {f.name}")
    print("\nSMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
