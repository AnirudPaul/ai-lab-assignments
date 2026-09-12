"""Generate the GitHub Pages site in docs/ from the figures in assets/fig.

    python tools/build_site.py

Copies every generated figure into docs/fig/ and writes docs/index.html: one section
per experiment, each with its figures inline and links to the notebook and the PDF.
"""
from __future__ import annotations

import html
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "fig"
DOCS = ROOT / "docs"

GITHUB_USER = "AnirudPaul"
REPO = "ai-lab-assignments"

EXPERIMENTS = [
    (1, "Depth-First and Breadth-First Search",
     "DFS &amp; BFS from scratch", "Hand-built graph",
     "Graph class, recursive and iterative DFS, BFS with shortest paths, and a measured "
     "comparison of the time and memory each one uses."),
    (2, "8-Puzzle Solver with A*",
     "A* search, admissible heuristics", "Generated puzzle states",
     "A* with four heuristics of increasing strength, showing that all return the same "
     "optimal solution while the work done differs by two orders of magnitude."),
    (3, "Travelling Salesman by Evolutionary Search",
     "Genetic algorithm", "25 Indian cities",
     "Order crossover, inversion mutation and tournament selection, judged against "
     "nearest-neighbour and 2-opt baselines rather than against random tours alone."),
    (4, "FOIL — Learning Rules from a Family Tree",
     "First Order Inductive Learner", "Hand-built family tree",
     "Real FOIL with information gain and sequential covering. Learns grandfather, "
     "grandmother and sibling perfectly — then fails on uncle, which beam search fixes."),
    (5, "Expert System with Certainty Factors",
     "Forward chaining, MYCIN certainty factors", "Hand-built rule base",
     "A 20-rule diagnostic knowledge base with an inference engine, negated conditions "
     "and a full explanation facility that justifies every conclusion."),
    (6, "Handwritten Digit Recognition with an MLP",
     "Multilayer perceptron (PyTorch)", "MNIST",
     "98.2% test accuracy, plus a controlled test of whether the non-linearity actually "
     "matters: the same network with Identity activations scores 91.3%."),
    (7, "Face Recognition with an SVM",
     "PCA eigenfaces + SVM", "Labeled Faces in the Wild",
     "Eigenfaces and an RBF SVM tuned by grid search, evaluated against the majority-class "
     "baseline rather than against 1/7 chance."),
    (8, "K-Means Clustering of Digits",
     "K-Means from scratch, k-means++", "MNIST",
     "Lloyd's algorithm implemented from scratch. Recovers real digit structure without "
     "labels — but does not recover the ten digits, and the elbow method does not find k=10."),
    (9, "Image Classification with a CNN",
     "Convolutional neural network", "CIFAR-10",
     "A CNN with batch norm and GPU-side augmentation, compared against an MLP given "
     "4.7x more parameters, to isolate what the convolutional structure is worth."),
    (10, "Sentiment Analysis with Multilayer RNNs",
     "2-layer bidirectional LSTM", "IMDB movie reviews",
     "Tokeniser, vocabulary and embeddings all built from scratch, benchmarked honestly "
     "against a TF-IDF + logistic regression baseline."),
]

CAPTIONS = {
    "q1_traversal_trees": "The graph, and the search tree each algorithm produces",
    "q1_frontier_growth": "Frontier size per iteration, and the BFS level structure",
    "q1_space_comparison": "Peak memory on complete b-ary trees (log scale)",
    "q2_solution_path": "The optimal solution, move by move",
    "q2_heuristic_comparison": "Search effort, heuristic estimates and peak memory",
    "q2_scaling": "How search effort grows with puzzle difficulty",
    "q3_convergence": "GA convergence against the classical baselines",
    "q3_tours": "Random tour vs nearest neighbour vs the evolved tour",
    "q3_tour_evolution": "The best tour improving across generations",
    "q3_robustness": "Run-to-run variability and the mutation-rate sweep",
    "q4_family_tree": "The family tree FOIL learns from",
    "q4_gain_trace": "FOIL information gain at each step, chosen literal vs runners-up",
    "q4_rule_quality": "Precision and recall of every learned theory",
    "q5_expert_system": "The rule network, per-case conclusions, and CF combination",
    "q6_samples": "MNIST samples",
    "q6_class_distribution": "Class balance and the mean image per digit",
    "q6_training_curves": "Loss, accuracy and the generalisation gap",
    "q6_confusion_matrix": "Confusion matrix and per-class recall",
    "q6_misclassified": "The errors the model was most confident about",
    "q6_layer1_weights": "First-layer weights viewed as images — stroke detectors",
    "q6_architecture_comparison": "Architecture ablation, including the linear control",
    "q7_samples": "LFW samples",
    "q7_eigenfaces": "Eigenfaces, explained variance and reconstruction quality",
    "q7_grid_search": "Cross-validated balanced accuracy over the C-gamma grid",
    "q7_results": "Confusion matrix and predictions (blue correct, red wrong)",
    "q7_ablation": "PCA vs raw pixels, linear vs RBF",
    "q7_components": "Accuracy against the number of eigenfaces",
    "q8_centroids": "Cluster centroids as images, and what each cluster contains",
    "q8_confusion": "Cluster-to-digit mapping and the raw contingency table",
    "q8_choosing_k": "Elbow, silhouette and label-based agreement against k",
    "q8_projections": "PCA and t-SNE, coloured by true digit and by cluster",
    "q8_pca_comparison": "Clustering quality and cost in reduced feature spaces",
    "q9_samples": "CIFAR-10, ten classes",
    "q9_augmentation": "Random crop and flip augmentation, applied on the GPU",
    "q9_training_curves": "Loss, accuracy and the OneCycle learning-rate schedule",
    "q9_confusion_matrix": "Confusion matrix and per-class recall",
    "q9_predictions": "Correct predictions, and the most confident mistakes",
    "q9_filters_featuremaps": "Learned first-layer filters and their feature maps",
    "q9_cnn_vs_mlp": "CNN vs a larger MLP, and the effect of augmentation",
    "q10_data_exploration": "Review lengths, Zipf's law, and the most class-skewed words",
    "q10_training_and_results": "Training curves, confusion matrix and ROC curve",
    "q10_confidence": "Confidence distribution and calibration",
    "q10_custom_sentences": "Predictions on hand-written sentences, including negation",
    "q10_model_comparison": "RNN variants against a TF-IDF + logistic regression baseline",
    "q10_embeddings": "Learned word embeddings projected to two dimensions",
}

CSS = """
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#1a1f2b; --muted:#5c6675;
  --brand:#2A4B9B; --line:#e2e6ec; --accent:#E4572E;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--brand)}
header{background:linear-gradient(135deg,#22396f,#2A4B9B);color:#fff;padding:44px 20px 38px}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px}
header .wrap{display:flex;gap:26px;align-items:center;flex-wrap:wrap}
header img{width:120px;background:#fff;padding:10px 14px;border-radius:10px}
header h1{margin:0 0 6px;font-size:30px;letter-spacing:.3px}
header p{margin:2px 0;opacity:.92;font-size:15px}
.actions{margin-top:18px;display:flex;gap:12px;flex-wrap:wrap}
.btn{display:inline-block;background:#fff;color:var(--brand);text-decoration:none;
  font-weight:600;padding:10px 18px;border-radius:8px;font-size:14px}
.btn.ghost{background:rgba(255,255,255,.14);color:#fff;border:1px solid rgba(255,255,255,.4)}
.intro{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:22px 26px;margin:30px 0}
.toc{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;
  margin:22px 0 6px;padding:0;list-style:none}
.toc a{display:block;background:var(--card);border:1px solid var(--line);border-radius:9px;
  padding:11px 13px;text-decoration:none;color:var(--ink);font-size:14px}
.toc a:hover{border-color:var(--brand);box-shadow:0 2px 10px rgba(42,75,155,.10)}
.toc b{color:var(--brand);margin-right:7px}
section.exp{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:26px 28px;margin:22px 0;scroll-margin-top:16px}
section.exp h2{margin:0 0 4px;font-size:21px;color:var(--brand)}
.meta{color:var(--muted);font-size:13.5px;margin:0 0 12px}
.meta span{display:inline-block;background:#eef1f6;border-radius:5px;padding:2px 9px;
  margin-right:7px;color:#3b4759}
.desc{margin:0 0 18px}
figure{margin:0 0 20px}
figure img{width:100%;border:1px solid var(--line);border-radius:9px;background:#fff}
figcaption{color:var(--muted);font-size:13px;margin-top:7px}
footer{color:var(--muted);font-size:13.5px;padding:34px 0 60px;text-align:center}
code{background:#eef1f6;padding:2px 6px;border-radius:4px;font-size:13.5px}
@media (max-width:640px){header h1{font-size:23px}section.exp{padding:20px 17px}}
"""


def build() -> None:
    DOCS.mkdir(exist_ok=True)
    out_fig = DOCS / "fig"
    out_fig.mkdir(exist_ok=True)

    figures = sorted(FIG.glob("*.png"))
    if not figures:
        raise SystemExit(f"no figures in {FIG}; run the notebook first")
    for f in figures:
        shutil.copy2(f, out_fig / f.name)

    by_exp: dict[int, list[Path]] = {}
    for f in figures:
        if f.stem.startswith("q") and "_" in f.stem:
            try:
                n = int(f.stem[1:f.stem.index("_")])
            except ValueError:
                continue
            by_exp.setdefault(n, []).append(f)

    # Preserve the order captions are declared in, then anything unlisted.
    order = list(CAPTIONS)
    for n in by_exp:
        by_exp[n].sort(key=lambda p: order.index(p.stem) if p.stem in order else 999)

    has_pdf = (ROOT / "AI_Lab_Assignment.pdf").exists()
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Artificial Intelligence Laboratory Record — Anirud Paul</title>",
        f"<style>{CSS}</style></head><body>",
        "<header><div class='wrap'>",
        "<img src='fig/../vit_logo.png' alt='VIT Bhopal University'"
        " onerror=\"this.style.display='none'\">",
        "<div><h1>Artificial Intelligence — Laboratory Record</h1>",
        "<p><b>Anirud Paul</b> &nbsp;·&nbsp; 26MCF10001 &nbsp;·&nbsp; Fall Semester 2026-27</p>",
        "<p>School of Computing Science Engineering and Artificial Intelligence (SCAI),"
        " VIT Bhopal University</p>",
        "<div class='actions'>",
        "<a class='btn' href='notebook.html'>Read the full notebook</a>",
    ]
    if has_pdf:
        parts.append("<a class='btn ghost' href='AI_Lab_Assignment.pdf'>Download PDF</a>")
    parts.append(
        f"<a class='btn ghost' href='https://github.com/{GITHUB_USER}/{REPO}'>"
        "View source on GitHub</a>")
    parts.append("</div></div></div></header><div class='wrap'>")

    parts.append(
        "<div class='intro'><p>Ten experiments covering classical search, evolutionary "
        "computation, inductive logic programming, expert systems and modern machine "
        "learning. Every algorithm in experiments 1&ndash;5 and the K-Means of experiment 8 "
        "is implemented from first principles rather than called from a library, and each "
        "experiment states its theory, shows its results and discusses what actually "
        "happened &mdash; including the runs that did not work.</p>"
        f"<p>All {len(figures)} figures below were generated by executing the notebook. "
        "Datasets are downloaded by <code>python tools/fetch_data.py</code>; the record is "
        "rebuilt with <code>python build.py</code>.</p></div>")

    parts.append("<ul class='toc'>")
    for n, title, *_ in EXPERIMENTS:
        parts.append(f"<li><a href='#exp{n}'><b>{n}</b>{html.escape(title)}</a></li>")
    parts.append("</ul>")

    for n, title, technique, dataset, desc in EXPERIMENTS:
        parts.append(f"<section class='exp' id='exp{n}'>")
        parts.append(f"<h2>{n}. {html.escape(title)}</h2>")
        parts.append(f"<p class='meta'><span>{html.escape(technique)}</span>"
                     f"<span>{html.escape(dataset)}</span></p>")
        parts.append(f"<p class='desc'>{desc}</p>")
        for f in by_exp.get(n, []):
            cap = CAPTIONS.get(f.stem, f.stem.replace("_", " "))
            parts.append(f"<figure><img loading='lazy' src='fig/{f.name}' "
                         f"alt='{html.escape(cap)}'>"
                         f"<figcaption>{html.escape(cap)}</figcaption></figure>")
        parts.append("</section>")

    parts.append(
        "</div><footer>Artificial Intelligence Laboratory Record &middot; "
        "Anirud Paul &middot; VIT Bhopal University<br>"
        f"Source: <a href='https://github.com/{GITHUB_USER}/{REPO}'>"
        f"github.com/{GITHUB_USER}/{REPO}</a></footer></body></html>")

    (DOCS / "index.html").write_text("\n".join(parts), encoding="utf-8")
    logo = ROOT / "assets" / "vit_logo.png"
    if logo.exists():
        shutil.copy2(logo, DOCS / "vit_logo.png")

    print(f"docs/index.html written: {len(EXPERIMENTS)} sections, "
          f"{len(figures)} figures copied to docs/fig/")


if __name__ == "__main__":
    build()
