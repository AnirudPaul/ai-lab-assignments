# Artificial Intelligence — Laboratory Record

**Anirud Paul** · 26MCF10001 · Fall Semester 2026-27
School of Computing Science Engineering and Artificial Intelligence (SCAI), VIT Bhopal University

Ten experiments covering classical search, evolutionary computation, inductive logic
programming, expert systems and modern machine learning.

📄 **[Read the full notebook online →](https://anirudpaul.github.io/ai-lab-assignments/notebook.html)**
🖼️ **[Browse the figures →](https://anirudpaul.github.io/ai-lab-assignments/)**

---

## The experiments

| # | Experiment | Technique | Dataset | Headline result |
|---|---|---|---|---|
| 1 | Graph traversal | DFS & BFS, from scratch | Hand-built graph | BFS finds J in 2 edges, DFS in 5; BFS peak memory 93–216× DFS on b-ary trees |
| 2 | 8-puzzle solver | A\*, four heuristics | Generated states | All heuristics agree on the 12-move optimum; work differs **97×** |
| 3 | Travelling salesman | Genetic algorithm | 25 Indian cities | 9,026 km — beats nearest-neighbour by 30 %, 2-opt by 1.8 % (5/6 seeds) |
| 4 | Family-tree rule induction | FOIL, from scratch | Hand-built family tree | 3 relations learned at precision/recall **1.0**; `uncle` needed beam search + pruning |
| 5 | Expert system | Forward chaining + certainty factors | 20-rule knowledge base | 4 cases diagnosed with full explanation traces |
| 6 | Digit recognition | Multilayer perceptron | MNIST | **98.18 %** test accuracy; linear control scores 91.28 % |
| 7 | Face recognition | PCA eigenfaces + SVM | LFW | **84.78 %** accuracy, 81.48 % balanced (baseline 41.15 %) |
| 8 | Digit clustering | K-Means, from scratch | MNIST | 58.7 % purity, ARI 0.365 — without labels |
| 9 | Image classification | Convolutional neural network | CIFAR-10 | **86.98 %** test accuracy |
| 10 | Sentiment analysis | 2-layer bidirectional LSTM | IMDB reviews | see notebook |

Experiments 1–5 and the K-Means of experiment 8 are implemented **from first principles** —
no library call does the actual work. Experiments 6, 9 and 10 use PyTorch; 7 uses
scikit-learn.

## What this record tries to do differently

Every experiment states its theory, runs it, and then discusses what actually happened —
including the parts that did not work. A few examples:

- **Experiment 4** reports FOIL *failing* on `uncle` (precision 0.20), diagnoses the exact
  greedy decision that caused it, and then fixes it with beam search plus post-pruning.
- **Experiment 7** initially selected the corner of its hyperparameter grid. That is a
  warning sign, not a result, so the grid was widened — worth 4.4 points of accuracy.
- **Experiment 8** shows that neither the elbow method nor the silhouette score identifies
  k = 10 on MNIST, contrary to how those methods are usually taught.
- **Experiment 9** compares the CNN against an MLP given *more* parameters, so the
  comparison cannot be won by capacity alone.
- **Experiments 3 and 10** benchmark against strong classical baselines (2-opt, TF-IDF +
  logistic regression) rather than only against random guessing.

## Reproducing it

```bash
pip install -r requirements.txt
python tools/fetch_data.py      # downloads MNIST, CIFAR-10, IMDB, LFW (~700 MB)
python build.py                 # assemble -> execute -> PDF + HTML
python tools/build_site.py      # regenerate the GitHub Pages site
```

A CUDA GPU is used automatically when available and is not required — the PyTorch
experiments fall back to CPU, more slowly.

> **Note on dataset mirrors.** `tools/fetch_data.py` pulls MNIST, CIFAR-10 and IMDB from
> the HuggingFace CDN rather than their original academic hosts. The data is identical;
> the original hosts measured at 0.01–0.04 MB/s from this machine, which would have made
> CIFAR-10 alone a ~4.7 hour download against roughly 3 minutes from the mirror.

## Layout

```
src/                  one cell-marked .py per experiment — the actual source
├── 00_title.py       title page, index, shared setup and helpers
├── 01_dfs_bfs.py … 10_rnn_imdb_sentiment.py
tools/
├── nbbuild.py        assembles src/*.py into one notebook, executes, exports
├── fetch_data.py     downloads and normalises every dataset
├── build_site.py     generates the GitHub Pages site
└── smoke.py          fast per-experiment check while developing
build.py              the one command that builds everything
AI_Lab_Assignment.ipynb   the executed notebook  ← submission
AI_Lab_Assignment.pdf     the same thing as a PDF  ← submission
docs/                 GitHub Pages site
assets/fig/           every generated figure as a PNG
```

The notebook is **generated**, not hand-edited. Each experiment lives in its own `.py`
file with `# %%` cell markers, and `build.py` assembles them into a single notebook,
executes it, and exports the PDF. This keeps one experiment editable without touching a
multi-megabyte JSON blob.

## Reproducibility

A fixed seed (42) is set for Python, NumPy and PyTorch, and cuDNN is put in deterministic
mode, so the figures quoted in the discussion sections match the numbers the notebook
prints. Datasets are downloaded and normalised once, up front, so the notebook itself
makes no network calls.
