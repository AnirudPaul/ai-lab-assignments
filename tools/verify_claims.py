"""Extract the headline numbers from the EXECUTED notebook.

Every discussion section quotes figures. Those figures must come from the run that is
actually in the notebook, not from an earlier run or from memory. This pulls the numbers
straight out of the stored cell outputs so they can be checked mechanically.

    python tools/verify_claims.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent

# (label, regex) -- first capture group is the value to report.
CHECKS = [
    ("Q1  BFS path to J (edges)", r"BFS path\s*:.*?\((\d+) edges\)"),
    ("Q1  DFS path to J (edges)", r"DFS path\s*:.*?\((\d+) edges\)"),
    ("Q2  optimum (moves)", r"All four agree the optimum is (\d+) moves"),
    ("Q2  work ratio", r"Work done differs by (\d+)x"),
    ("Q3  GA tour (km)", r"Best tour length\s*:\s*([\d,]+) km"),
    ("Q3  vs NN+2-opt", r"Versus NN \+ 2-opt\s*:\s*([+-][\d.]+)%"),
    ("Q3  seeds beating 2-opt", r"Seeds beating NN \+ 2-opt:\s*(\d+/\d+)"),
    ("Q6  MLP test accuracy", r"accuracy : (0\.\d+)\s+\(\d"),
    ("Q6  sklearn MLP accuracy", r"sklearn MLPClassifier.*?(0\.\d{4})"),
    ("Q7  SVM accuracy", r"accuracy\s+: (0\.\d+)\s*\n\s*balanced"),
    ("Q7  SVM balanced accuracy", r"balanced accuracy : (0\.\d+)"),
    ("Q7  majority baseline", r"majority baseline : (0\.\d+)"),
    ("Q7  best params", r"best parameters : (\{[^}]+\})"),
    ("Q8  purity", r"Purity\s+(0\.\d+)"),
    ("Q8  ARI", r"Adjusted Rand Index\s+(0\.\d+)"),
    ("Q8  digits unclaimed", r"digits NO cluster represents\s+: (\[[^\]]*\]|none)"),
    ("Q8  digits split", r"digits split across clusters\s+: (\[[^\]]*\]|none)"),
    ("Q9  CNN test accuracy", r"test accuracy : (0\.\d+)"),
    ("Q9  MLP baseline accuracy", r"MLP \(augmented\)\s+test acc = (0\.\d+)"),
    ("Q9  CNN no-aug accuracy", r"CNN \(no augmentation\)\s+test acc = (0\.\d+)"),
    ("Q10 BiLSTM test accuracy", r"2-layer BiLSTM\s+(0\.\d{4})"),
    ("Q10 TF-IDF accuracy", r"TF-IDF \(1-2 gram\) \+ LogReg\s+(0\.\d{4})"),
    ("Q10 baseline margin", r"Better test accuracy: (.*)"),
]


def main() -> int:
    nb_path = ROOT / "AI_Lab_Assignment.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    texts, n_out, n_err, n_fig = [], 0, 0, 0
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            n_out += 1
            if out.get("output_type") == "error":
                n_err += 1
                texts.append("ERROR: " + "\n".join(out.get("traceback", [])))
            if "text" in out:
                texts.append(out["text"])
            data = out.get("data", {})
            if "text/plain" in data:
                texts.append(data["text/plain"])
            if "image/png" in data:
                n_fig += 1
    blob = "\n".join(texts)

    executed = sum(1 for c in nb.cells
                   if c.cell_type == "code" and c.get("execution_count") is not None)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")

    print(f"notebook   : {nb_path.name}  ({nb_path.stat().st_size/1e6:.1f} MB)")
    print(f"cells      : {len(nb.cells)} total, {code_cells} code, {executed} executed")
    print(f"outputs    : {n_out} ({n_fig} inline figures)")
    print(f"errors     : {n_err}")
    if executed < code_cells:
        print(f"  WARNING: {code_cells - executed} code cells have no execution count")
    print("-" * 66)

    missing = 0
    for label, pattern in CHECKS:
        m = re.search(pattern, blob, re.S)
        if m:
            print(f"{label:32s} {m.group(1)}")
        else:
            missing += 1
            print(f"{label:32s} -- NOT FOUND --")
    print("-" * 66)
    print(f"{len(CHECKS) - missing}/{len(CHECKS)} headline numbers located")
    return 1 if (n_err or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
