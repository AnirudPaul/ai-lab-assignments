"""Update the executed notebook's prose from src/ without re-running anything.

Discussion sections quote measured figures, so they are often corrected *after* a run
finishes. Re-assembling from scratch would discard 80+ minutes of executed output just to
change some text.

This rewrites only the markdown cells, and refuses to do so unless every code cell in the
notebook is still byte-identical to its source. That guard is the whole point: if any code
changed, the stored outputs no longer correspond to the code that produced them, and the
notebook must genuinely be re-executed rather than patched.

    python tools/refresh_markdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from nbbuild import SRC, logo_data_uri, parse_cells  # noqa: E402


def main() -> int:
    nb_path = ROOT / "AI_Lab_Assignment.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    logo = logo_data_uri()
    expected: list[tuple[str, str]] = []
    for f in sorted(SRC.glob("[0-9][0-9]_*.py")):
        for kind, text in parse_cells(f):
            expected.append((kind, text.replace("{{LOGO_B64}}", logo) if kind == "markdown"
                             else text))

    if len(expected) != len(nb.cells):
        print(f"ABORT: src has {len(expected)} cells, notebook has {len(nb.cells)}. "
              f"The structure changed -- run `python build.py` instead.")
        return 1

    changed = 0
    for i, ((kind, text), cell) in enumerate(zip(expected, nb.cells)):
        if kind != cell.cell_type:
            print(f"ABORT: cell {i} is {cell.cell_type} in the notebook but {kind} in src.")
            return 1
        if kind == "code":
            if cell.source.strip() != text.strip():
                print(f"ABORT: code cell {i} differs from src. Its stored output would no "
                      f"longer match its code -- re-execute with `python build.py`.")
                print(f"  src      : {text.strip().splitlines()[0][:70]}")
                print(f"  notebook : {cell.source.strip().splitlines()[0][:70]}")
                return 1
        elif cell.source != text:
            cell.source = text
            changed += 1

    nbformat.write(nb, nb_path)
    print(f"all {sum(1 for k, _ in expected if k == 'code')} code cells verified identical")
    print(f"{changed} markdown cell(s) updated in {nb_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
