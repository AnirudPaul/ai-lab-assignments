# %% [markdown]
# <div style="text-align:center; font-family:'Times New Roman',Times,serif; page-break-after:always;">
# <img src="{{LOGO_B64}}" alt="VIT Bhopal University" style="width:230px; margin:18px auto 26px auto; display:block;">
# <div style="border-top:2px solid #111; border-bottom:2px solid #111; padding:10px 0; margin:0 40px 22px 40px;"><span style="font-size:30px; font-weight:bold; letter-spacing:2px;">LABORATORY RECORD</span></div>
# <div style="font-size:19px; font-weight:bold; margin-bottom:26px;">______________&nbsp;&nbsp;&nbsp;Artificial Intelligence</div>
# <div style="font-size:16px; line-height:2.0;"><b>Student Name:</b> Anirud Paul<br><b>Student Regn. No.:</b> 26MCF10001<br><b>Year:</b> 2026-27<br><b>Semester:</b> Fall Semester 26-27<br><b>ClassNbr:</b> ______________________<br><b>Slot:</b> ______________________<br><b>Class Room:</b> ______________________<br><b>Faculty Name:</b> ______________________</div>
# <div style="font-size:15px; font-weight:bold; margin-top:26px; line-height:1.7;">School of Computing Science Engineering and Artificial Intelligence (SCAI)<br>VIT Bhopal University</div>
# </div>

# %% [markdown]
# # Index
#
# | # | Experiment | Technique | Dataset |
# |---|------------|-----------|---------|
# | 1 | Graph traversal | Depth-First Search, Breadth-First Search | Hand-built graph |
# | 2 | 8-Puzzle solver | A\* search, admissible heuristics | Generated states |
# | 3 | Travelling Salesman Problem | Genetic algorithm (evolutionary search) | 25 Indian cities |
# | 4 | Family-tree rule induction | First Order Inductive Learner (FOIL) | Hand-built family tree |
# | 5 | Expert system | Forward chaining + certainty factors | Hand-built rule base |
# | 6 | Handwritten digit classifier | Multilayer Perceptron | MNIST |
# | 7 | Face recognition | PCA eigenfaces + Support Vector Machine | LFW |
# | 8 | Digit clustering | K-Means | MNIST |
# | 9 | Image classification | Convolutional Neural Network | CIFAR-10 |
# | 10 | Sentiment analysis | Multilayer bidirectional LSTM | IMDB movie reviews |
#
# Every experiment below is self-contained: theory, implementation, a run, results and
# discussion. Cells are meant to be executed top to bottom.

# %% [markdown]
# ---
# ## Environment setup
#
# One shared setup cell: imports, a fixed random seed so every number in this record is
# reproducible, a common plot style, and a `save_fig` helper that writes every figure to
# `assets/fig/` as a PNG as well as showing it inline.

# %%
from __future__ import annotations

import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path.cwd()
DATA = ROOT / "data"
FIG = ROOT / "assets" / "fig"
FIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SCIKIT_LEARN_DATA", str(DATA / "sklearn"))

# A single consistent look for all 40+ figures in this record.
PALETTE = ["#2A4B9B", "#E4572E", "#17A398", "#F2A65A", "#7D4F9E",
           "#3B8EA5", "#C44E52", "#8C8C8C", "#4C9F70", "#D6A419"]
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.frameon": False,
    "image.cmap": "viridis",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def save_fig(name: str, fig=None, tight: bool = True):
    """Save the current (or given) figure to assets/fig/<name>.png and show it."""
    fig = fig or plt.gcf()
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def banner(title: str) -> None:
    """Consistent section header for printed console output."""
    print("=" * 68)
    print(title.center(68))
    print("=" * 68)


def timed(fn, *args, **kwargs):
    """Run fn, return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def free_gpu() -> None:
    """Release cached GPU memory between experiments.

    PyTorch keeps a caching allocator, so memory freed by Python is not returned to the
    driver. This GPU is shared with other processes, so handing it back between the
    heavier experiments avoids an out-of-memory failure late in the notebook.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_dataset(name: str) -> dict:
    """Load a prepared dataset from data/prepared/<name>.npz.

    Datasets are downloaded and normalised once by `python tools/fetch_data.py`, which
    keeps this notebook free of network calls and makes every run reproducible.
    """
    path = DATA / "prepared" / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python tools/fetch_data.py {name}")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# %%
import sklearn
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # cuDNN picks convolution algorithms by benchmarking, and some are non-deterministic.
    # Left at the defaults, re-running this notebook shifts the reported accuracies by a
    # few tenths of a percent, which would make the figures quoted in the discussion
    # sections drift away from the numbers actually printed. Fixing the algorithm choice
    # costs a little speed and buys reproducible results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

env = pd.DataFrame(
    [
        ("Python", sys.version.split()[0]),
        ("NumPy", np.__version__),
        ("pandas", pd.__version__),
        ("scikit-learn", sklearn.__version__),
        ("PyTorch", torch.__version__),
        ("Matplotlib", matplotlib.__version__),
        ("Compute device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        ("Random seed", SEED),
    ],
    columns=["Component", "Version"],
)
banner("ENVIRONMENT")
print(env.to_string(index=False))
