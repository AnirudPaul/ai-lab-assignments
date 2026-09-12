"""Assemble src/*.py cell-marked sources into one notebook, execute, export.

Cell markers inside src/*.py:

    # %% [markdown]
    # rendered as markdown (leading '# ' stripped)

    # %%
    code cell

Everything here is deterministic file plumbing -- no notebook content lives
in this module.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

MARKER = re.compile(r"^#\s*%%\s*(\[markdown\])?\s*$")


def logo_data_uri() -> str:
    """Base64 data URI for the VIT logo, so the notebook/PDF stay standalone."""
    import base64

    p = ROOT / "assets" / "vit_logo.png"
    if not p.exists():
        print("  WARNING: assets/vit_logo.png missing, title page will have no logo")
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def parse_cells(path: Path) -> list[tuple[str, str]]:
    """Return [(kind, source)] where kind is 'markdown' or 'code'."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = MARKER.match(line)
        if m:
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if m.group(1) else "code"
            buf = []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, buf))

    out: list[tuple[str, str]] = []
    for k, lines in cells:
        if k == "markdown":
            stripped = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") if ln.strip() == "#" else ln for ln in lines]
            text = "\n".join(stripped).strip("\n")
        else:
            text = "\n".join(lines).strip("\n")
        if text.strip():
            out.append((k, text))
    return out


def assemble(out_path: Path) -> nbformat.NotebookNode:
    nb = new_notebook()
    sources = sorted(SRC.glob("[0-9][0-9]_*.py"))
    if not sources:
        raise SystemExit("no src/NN_*.py files found")
    n_md = n_code = 0
    logo = logo_data_uri()
    for f in sources:
        for kind, text in parse_cells(f):
            if kind == "markdown":
                nb.cells.append(new_markdown_cell(text.replace("{{LOGO_B64}}", logo)))
                n_md += 1
            else:
                nb.cells.append(new_code_cell(text))
                n_code += 1
        print(f"  + {f.name}")
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
    )
    nbformat.write(nb, out_path)
    print(f"assembled {out_path.name}: {len(nb.cells)} cells ({n_md} md, {n_code} code)")
    return nb


def execute(path: Path, timeout: int = 3600) -> None:
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    t = time.time()
    print(f"executing {len(nb.cells)} cells (cwd={ROOT}) ...", flush=True)
    client.execute()
    nbformat.write(nb, path)
    print(f"executed in {time.time()-t:.1f}s")


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p.returncode, (p.stdout + p.stderr)[-2500:]


def export_html(nb_path: Path, out: Path) -> bool:
    code, log = _run(
        [sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images",
         "--template", "lab", "--output", out.stem, "--output-dir", str(out.parent),
         str(nb_path)]
    )
    print(f"html export exit={code}")
    if code:
        print(log)
    return code == 0


def export_pdf(nb_path: Path, out: Path) -> bool:
    """Try webpdf (chromium) first, then LaTeX. Returns True on success."""
    attempts = [
        ("webpdf", [sys.executable, "-m", "nbconvert", "--to", "webpdf",
                    "--allow-chromium-download", "--output", out.stem,
                    "--output-dir", str(out.parent), str(nb_path)]),
        ("latex", [sys.executable, "-m", "nbconvert", "--to", "pdf",
                   "--output", out.stem, "--output-dir", str(out.parent), str(nb_path)]),
    ]
    for name, cmd in attempts:
        print(f"pdf export via {name} ...", flush=True)
        code, log = _run(cmd)
        if code == 0 and out.exists():
            print(f"pdf ok via {name}: {out.stat().st_size/1e6:.1f} MB")
            return True
        print(f"  {name} failed (exit={code})")
        print("  " + log.replace("\n", "\n  ")[-1200:])
    return False


if __name__ == "__main__":
    nb = ROOT / "AI_Lab_Assignment.ipynb"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("assemble", "all"):
        assemble(nb)
    if cmd in ("execute", "all"):
        execute(nb)
    if cmd in ("export", "all"):
        export_html(nb, ROOT / "docs" / "notebook.html")
        export_pdf(nb, ROOT / "AI_Lab_Assignment.pdf")
