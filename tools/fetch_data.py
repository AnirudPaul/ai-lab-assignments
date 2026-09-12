"""Download every dataset the notebook needs into ./data/prepared as .npz / .parquet.

The canonical academic hosts (cs.toronto.edu, ossci S3) measured at 0.01-0.04 MB/s
from this machine, which would have made CIFAR-10 alone a ~4.7 hour download.  The
HuggingFace CDN serves byte-identical data at ~1 MB/s, so MNIST / CIFAR-10 / IMDB come
from there and are normalised into plain numpy arrays once, up front.  LFW still comes
through scikit-learn's own fetcher.

Idempotent: anything already prepared is skipped.
"""
from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PREP = DATA / "prepared"
for d in (DATA, RAW, PREP):
    d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SCIKIT_LEARN_DATA", str(DATA / "sklearn"))

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "mnist_train": ("ylecun/mnist", "mnist/train-00000-of-00001.parquet"),
    "mnist_test": ("ylecun/mnist", "mnist/test-00000-of-00001.parquet"),
    "cifar_train": ("uoft-cs/cifar10", "plain_text/train-00000-of-00001.parquet"),
    "cifar_test": ("uoft-cs/cifar10", "plain_text/test-00000-of-00001.parquet"),
    "imdb_train": ("stanfordnlp/imdb", "plain_text/train-00000-of-00001.parquet"),
    "imdb_test": ("stanfordnlp/imdb", "plain_text/test-00000-of-00001.parquet"),
}


def log(m: str) -> None:
    print(f"[fetch] {m}", flush=True)


def download(key: str) -> Path:
    repo, path = SOURCES[key]
    out = RAW / f"{key}.parquet"
    if out.exists() and out.stat().st_size > 1000:
        log(f"{key}: cached ({out.stat().st_size/1e6:.1f} MB)")
        return out
    url = HF.format(repo=repo, path=path)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got, last = 0, time.time()
        while chunk := r.read(1 << 18):
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 10:
                pct = f"{100*got/total:.0f}%" if total else "?"
                log(f"  {key} {got/1e6:.0f}/{total/1e6:.0f} MB ({pct})")
                last = time.time()
    tmp.replace(out)
    log(f"{key}: downloaded {out.stat().st_size/1e6:.1f} MB")
    return out


def decode_images(table, img_col: str) -> np.ndarray:
    """Decode a HF parquet image column (PNG bytes) into a uint8 array."""
    from PIL import Image

    col = table.column(img_col).to_pylist()
    first = np.array(Image.open(io.BytesIO(col[0]["bytes"])))
    out = np.empty((len(col), *first.shape), dtype=np.uint8)
    for i, rec in enumerate(col):
        out[i] = np.array(Image.open(io.BytesIO(rec["bytes"])))
    return out


def prepare_images(name: str, train_key: str, test_key: str, img_col: str) -> None:
    import pyarrow.parquet as pq

    out = PREP / f"{name}.npz"
    if out.exists():
        log(f"{name}: already prepared")
        return
    arrays = {}
    for split, key in (("train", train_key), ("test", test_key)):
        t = pq.read_table(download(key))
        log(f"{name}/{split}: decoding {t.num_rows} images ...")
        arrays[f"X_{split}"] = decode_images(t, img_col)
        arrays[f"y_{split}"] = np.asarray(t.column("label").to_pylist(), dtype=np.int64)
    np.savez_compressed(out, **arrays)
    log(f"{name}: prepared {arrays['X_train'].shape} train / {arrays['X_test'].shape} test "
        f"-> {out.stat().st_size/1e6:.1f} MB")


def prepare_imdb() -> None:
    import pyarrow.parquet as pq

    out = PREP / "imdb.npz"
    if out.exists():
        log("imdb: already prepared")
        return
    arrays = {}
    for split, key in (("train", "imdb_train"), ("test", "imdb_test")):
        t = pq.read_table(download(key))
        arrays[f"X_{split}"] = np.asarray(t.column("text").to_pylist(), dtype=object)
        arrays[f"y_{split}"] = np.asarray(t.column("label").to_pylist(), dtype=np.int64)
        log(f"imdb/{split}: {len(arrays[f'X_{split}'])} reviews")
    np.savez_compressed(out, **arrays)
    log(f"imdb: prepared -> {out.stat().st_size/1e6:.1f} MB")


def prepare_lfw() -> None:
    from sklearn.datasets import fetch_lfw_people

    log("lfw: fetching via scikit-learn (~240 MB from a slow host, be patient) ...")
    p = fetch_lfw_people(data_home=str(DATA / "sklearn"), min_faces_per_person=70, resize=0.4)
    log(f"lfw: ready {p.images.shape}, {len(p.target_names)} people")


JOBS = {
    "mnist": lambda: prepare_images("mnist", "mnist_train", "mnist_test", "image"),
    "cifar10": lambda: prepare_images("cifar10", "cifar_train", "cifar_test", "img"),
    "imdb": prepare_imdb,
    "lfw": prepare_lfw,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(JOBS)
    failed = []
    for name in wanted:
        if name not in JOBS:
            log(f"unknown dataset {name!r}; choose from {list(JOBS)}")
            return 2
        try:
            t = time.time()
            JOBS[name]()
            log(f"{name}: OK in {time.time()-t:.1f}s")
        except Exception as exc:  # noqa: BLE001 - report all, fail at the end
            log(f"{name}: FAILED {type(exc).__name__}: {exc}")
            failed.append(name)
    if failed:
        log(f"FAILED: {', '.join(failed)}")
        return 1
    log("ALL REQUESTED DATASETS READY")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
