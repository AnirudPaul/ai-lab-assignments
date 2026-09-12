"""Build the lab record: src/*.py  ->  one executed notebook  ->  PDF + HTML.

    python build.py            # assemble, execute, export everything
    python build.py assemble   # just rebuild the .ipynb (no execution)
    python build.py execute    # execute the existing .ipynb in place
    python build.py export     # export PDF/HTML from the executed .ipynb

Datasets must be fetched first:

    python tools/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from nbbuild import ROOT, assemble, execute, export_html, export_pdf  # noqa: E402


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    nb = ROOT / "AI_Lab_Assignment.ipynb"

    if cmd in ("assemble", "all"):
        assemble(nb)
    if cmd in ("execute", "all"):
        execute(nb)
    if cmd in ("export", "all"):
        (ROOT / "docs").mkdir(exist_ok=True)
        export_html(nb, ROOT / "docs" / "notebook.html")
        ok = export_pdf(nb, ROOT / "AI_Lab_Assignment.pdf")
        if not ok:
            print("\nPDF export failed. The notebook and HTML are still usable.")
            return 1
    print("\nBuild complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
