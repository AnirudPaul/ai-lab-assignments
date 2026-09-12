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


def execute(path: Path, timeout: int = 3600, start: int = 0) -> None:
    """Execute the notebook cell by cell, reporting progress and saving as it goes.

    `client.execute()` runs the whole notebook as one opaque call: if the kernel dies
    part-way (this machine shares its GPU with other processes, and a CUDA crash takes
    the kernel with it) the run reports nothing about where it stopped and every
    completed cell is lost. Driving the loop here means progress is printed per cell and
    the partially executed notebook is written to disk, so a crash is diagnosable and the
    work up to it is not thrown away.
    """
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}},
                            allow_errors=False)
    total = len(nb.cells)
    code_total = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"executing {total} cells ({code_total} code) cwd={ROOT}", flush=True)

    t0 = time.time()
    done = 0
    with client.setup_kernel():
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            done += 1
            head = cell.source.strip().splitlines()[0][:58] if cell.source.strip() else ""
            print(f"  [{done:3d}/{code_total}] cell {idx:3d}  {head}", flush=True)
            try:
                client.execute_cell(cell, idx)
            except Exception as exc:
                nbformat.write(nb, path)
                print(f"\nFAILED at cell {idx} after {time.time()-t0:.0f}s: "
                      f"{type(exc).__name__}: {str(exc)[:900]}", flush=True)
                raise
            if done % 10 == 0:
                nbformat.write(nb, path)      # periodic checkpoint
    nbformat.write(nb, path)
    print(f"executed {code_total} code cells in {time.time()-t0:.1f}s", flush=True)


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


# Injected before </head> of the exported HTML so the PDF paginates sensibly.
PRINT_CSS = """
<style>
@page { size: A4; margin: 14mm 12mm; }
body { font-size: 10.5pt; }
/* Long console output and source lines must wrap instead of running off the page. */
pre, code, .jp-OutputArea-output pre, .highlight pre {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
}
.jp-OutputArea-output { overflow-x: visible !important; }
/* Do not split a figure, a table or a code cell across two pages. */
.jp-RenderedImage, .jp-OutputArea-child, table, .jp-Cell-inputWrapper {
  page-break-inside: avoid; break-inside: avoid;
}
img { max-width: 100% !important; height: auto !important; }
/* Start each experiment on a fresh page. Every experiment's markdown opens with a
   `---` rule, so breaking after each <hr> paginates exactly on those boundaries.
   Breaking before <h1> instead produced a blank leading page, because the template
   emits a heading of its own above the title page. */
hr { break-after: page; page-break-after: always;
     border: none; height: 0; margin: 0; visibility: hidden; }
.jp-InputPrompt, .jp-OutputPrompt { min-width: 0 !important; }
</style>
"""


def export_pdf(nb_path: Path, out: Path, html_path: Path | None = None) -> bool:
    """Render the notebook to PDF by printing its HTML with headless Chromium.

    nbconvert's own `--to webpdf` raises NotImplementedError on Windows: it drives
    Playwright through asyncio's subprocess API, which the event loop policy in use
    here does not implement. Playwright's *sync* API sidesteps that entirely, and
    printing the HTML ourselves also lets us inject print CSS for pagination.
    """
    html_path = html_path or (out.parent / f"{out.stem}_print.html")
    code, log = _run([sys.executable, "-m", "nbconvert", "--to", "html",
                      "--embed-images", "--template", "lab",
                      "--output", html_path.stem, "--output-dir", str(html_path.parent),
                      str(nb_path)])
    if code != 0 or not html_path.exists():
        print(f"  HTML step failed (exit={code})\n  {log[-1000:]}")
        return False

    html = html_path.read_text(encoding="utf-8")
    if "</head>" in html:
        html = html.replace("</head>", PRINT_CSS + "</head>", 1)
    else:
        html = PRINT_CSS + html
    html_path.write_text(html, encoding="utf-8")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  playwright not installed: pip install playwright && "
              "python -m playwright install chromium")
        return False

    print("  printing with headless chromium ...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=180_000)
        page.pdf(path=str(out), format="A4", print_background=True,
                 margin={"top": "14mm", "bottom": "14mm",
                         "left": "12mm", "right": "12mm"})
        browser.close()

    if out.exists():
        print(f"pdf ok: {out.name}, {out.stat().st_size/1e6:.1f} MB")
        return True
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
