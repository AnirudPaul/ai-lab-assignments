"""Generate the GitHub Pages site in docs/ from the figures produced by the notebook.

Deterministic: the experiment metadata lives in EXPERIMENTS, the figures are discovered
on disk by their `qN_` prefix, so the site cannot drift out of sync with what the
notebook actually produced.
"""
from __future__ import annotations

import html
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "fig"
DOCS = ROOT / "docs"

EXPERIMENTS = [
    (1, "Depth-First &amp; Breadth-First Search", "Graph traversal from scratch",
     "Recursive and iterative DFS, BFS with a queue, frontier tracing, and a measured "
     "comparison of search-tree shape, shortest-path optimality and peak memory."),
    (2, "8-Puzzle Solver with A*", "Informed search and admissible heuristics",
     "A* with four heuristics (uniform cost, misplaced tiles, Manhattan, Manhattan + "
     "linear conflict), verifying that all return the same optimal path while differing "
     "by ~100x in nodes expanded."),
    (3, "Travelling Salesman by Genetic Algorithm", "Evolutionary search",
     "Order crossover, inversion mutation and tournament selection on a 25-city tour of "
     "India, benchmarked honestly against nearest-neighbour and 2-opt local search."),
    (4, "FOIL on a Family Tree", "First-order rule induction",
     "A real FOIL implementation - sequential covering driven by information gain - "
     "inducing grandfather, grandmother, sibling and uncle from ground facts, plus beam "
     "search and post-pruning to fix where greedy search fails."),
    (5, "Expert System", "Forward chaining with certainty factors",
     "A 20-rule diagnostic knowledge base with MYCIN-style certainty factors, negated "
     "conditions, and how/why explanation of every conclusion."),
    (6, "MLP on MNIST", "Handwritten digit recognition",
     "A PyTorch multilayer perceptron reaching 98.2%, with a controlled test showing a "
     "parameter-identical network without activations collapses to 91%."),
    (7, "SVM Face Recognition on LFW", "Eigenfaces + support vector machine",
     "PCA eigenfaces feeding an RBF SVM, tuned by grid search, evaluated with balanced "
     "accuracy against the majority-class baseline rather than chance."),
    (8, "K-Means on MNIST", "Unsupervised clustering",
     "Lloyd's algorithm with k-means++ written from scratch, and an honest account of "
     "which digits clustering recovers and which it does not."),
    (9, "CNN on CIFAR-10", "Convolutional image classification",
     "A BatchNorm CNN with GPU-side augmentation reaching 86.9%, against a fully "
     "connected network given 4.7x more parameters that reaches only 54.7%."),
    (10, "Sentiment Analysis with RNNs", "Multilayer bidirectional LSTM",
     "A 2-layer BiLSTM on raw IMDB reviews, compared against a TF-IDF bigram baseline "
     "that turns out to be highly competitive."),
]

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202e;--muted:#5c6b80;--line:#dfe4ea;
      --accent:#2A4B9B;--accent2:#E4572E}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:var(--bg);color:var(--ink)}
a{color:var(--accent)}
.hero{background:linear-gradient(135deg,#22335c 0%,#2A4B9B 60%,#3d6fd0 100%);
      color:#fff;padding:54px 24px 46px}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
.hero h1{margin:0 0 6px;font-size:2.1rem;letter-spacing:-.5px}
.hero p{margin:6px 0 0;opacity:.92}
.badges{margin-top:18px;display:flex;gap:10px;flex-wrap:wrap}
.badge{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.28);
       padding:5px 12px;border-radius:999px;font-size:.82rem}
.actions{margin-top:22px;display:flex;gap:12px;flex-wrap:wrap}
.btn{background:#fff;color:var(--accent);padding:10px 18px;border-radius:8px;
     text-decoration:none;font-weight:600;font-size:.92rem}
.btn.alt{background:rgba(255,255,255,.16);color:#fff;border:1px solid rgba(255,255,255,.4)}
section{padding:34px 0}
h2{font-size:1.35rem;margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:22px;margin-bottom:22px}
.card h3{margin:0 0 4px;font-size:1.12rem}
.card .sub{color:var(--accent2);font-weight:600;font-size:.83rem;
           text-transform:uppercase;letter-spacing:.6px}
.card p{color:var(--muted);margin:10px 0 16px}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.figs a{display:block;border:1px solid var(--line);border-radius:8px;overflow:hidden;
        background:#fff}
.figs img{width:100%;display:block}
.figs span{display:block;padding:7px 10px;font-size:.76rem;color:var(--muted);
           border-top:1px solid var(--line);word-break:break-word}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);
      border-radius:10px;overflow:hidden}
th,td{padding:10px 13px;text-align:left;border-bottom:1px solid var(--line);font-size:.9rem}
th{background:#eef1f6}
tr:last-child td{border-bottom:none}
footer{padding:34px 0 50px;color:var(--muted);font-size:.87rem;text-align:center}
code{background:#eef1f6;padding:2px 6px;border-radius:4px;font-size:.86em}
@media (prefers-color-scheme:dark){
  :root{--bg:#11151b;--card:#171d26;--ink:#e6ecf3;--muted:#9aa8ba;--line:#26303d}
  th{background:#1d242f}
  .figs a{background:#fff}
}
"""


def figure_caption(stem: str) -> str:
    return stem.split("_", 1)[1].replace("_", " ") if "_" in stem else stem


def build() -> None:
    DOCS.mkdir(exist_ok=True)
    out_fig = DOCS / "fig"
    if out_fig.exists():
        shutil.rmtree(out_fig)
    out_fig.mkdir(parents=True)

    figures = sorted(FIG.glob("*.png"))
    for f in figures:
        shutil.copy2(f, out_fig / f.name)

    by_q: dict[int, list[Path]] = {}
    for f in figures:
        if f.stem.startswith("q") and "_" in f.stem:
            try:
                by_q.setdefault(int(f.stem[1:].split("_")[0]), []).append(f)
            except ValueError:
                pass

    parts = [f"<style>{CSS}</style>",
             '<div class="hero"><div class="wrap">',
             "<h1>Artificial Intelligence — Laboratory Record</h1>",
             "<p>Ten experiments: classical search, evolutionary algorithms, inductive "
             "logic programming, expert systems, and neural networks.</p>",
             '<div class="badges">',
             "<span class='badge'>Anirud Paul · 26MCF10001</span>",
             "<span class='badge'>VIT Bhopal University · SCAI</span>",
             "<span class='badge'>Python 3.11 · PyTorch · scikit-learn</span>",
             "</div>",
             '<div class="actions">',
             '<a class="btn" href="notebook.html">Read the full notebook</a>',
             '<a class="btn alt" href="https://github.com/AnirudPaul/ai-lab-assignments">'
             'GitHub repository</a>',
             '<a class="btn alt" href="https://github.com/AnirudPaul/ai-lab-assignments/'
             'raw/main/AI_Lab_Assignment.pdf">Download PDF</a>',
             "</div></div></div>",
             '<div class="wrap"><section>',
             "<h2>Experiments</h2>"]

    for num, title, sub, desc in EXPERIMENTS:
        figs = by_q.get(num, [])
        parts.append('<div class="card">')
        parts.append(f'<div class="sub">Experiment {num} — {html.escape(sub)}</div>')
        parts.append(f"<h3>{title}</h3>")
        parts.append(f"<p>{desc}</p>")
        if figs:
            parts.append('<div class="figs">')
            for f in figs:
                cap = html.escape(figure_caption(f.stem))
                parts.append(f'<a href="fig/{f.name}" target="_blank">'
                             f'<img src="fig/{f.name}" alt="{cap}" loading="lazy">'
                             f"<span>{cap}</span></a>")
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</section><section><h2>Reproducing this</h2>")
    parts.append("<table><tr><th>Step</th><th>Command</th></tr>"
                 "<tr><td>Install dependencies</td><td><code>pip install -r "
                 "requirements.txt</code></td></tr>"
                 "<tr><td>Download datasets (~500 MB)</td><td><code>python "
                 "tools/fetch_data.py</code></td></tr>"
                 "<tr><td>Build, execute and export</td><td><code>python "
                 "tools/nbbuild.py all</code></td></tr></table>")
    parts.append("</section>")
    parts.append('<footer>Anirud Paul (26MCF10001) · School of Computing Science '
                 "Engineering and Artificial Intelligence · VIT Bhopal University"
                 "</footer></div>")

    (DOCS / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"docs/index.html written: {len(figures)} figures across "
          f"{len(by_q)} experiments")


if __name__ == "__main__":
    build()
