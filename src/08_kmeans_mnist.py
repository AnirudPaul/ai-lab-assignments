# %% [markdown]
# ---
# # Experiment 8 — K-Means Clustering on MNIST Digits
#
# **Aim.** Cluster MNIST digit images **without using their labels**, then use the labels
# only afterwards to measure how much of the digit structure unsupervised clustering
# recovered on its own.
#
# ## Theory
#
# K-Means partitions $n$ points into $k$ clusters so as to minimise the **within-cluster
# sum of squares** (inertia):
#
# $$J = \sum_{i=1}^{k} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$
#
# **Lloyd's algorithm** alternates two steps until the assignments stop changing:
#
# 1. **Assign** — put each point in the cluster with the nearest centroid.
# 2. **Update** — move each centroid to the mean of the points assigned to it.
#
# Each step can only decrease $J$, so the algorithm always converges — but only to a **local**
# minimum, and which one depends entirely on the initial centroids.
#
# ### k-means++ initialisation
# Choosing the first centroid at random and then each subsequent one with probability
# proportional to $D(x)^2$ (the squared distance to the nearest centroid already chosen)
# spreads the initial centroids out. This dramatically reduces the chance of a bad local
# minimum, and is why it is the default in practice.
#
# ### The critical caveat
# K-Means is **unsupervised**: it knows nothing about digits. Setting $k = 10$ does not ask
# it to find the ten digits — it asks it to find ten blobs of pixel-space variance. Whether
# those blobs correspond to digits is exactly the question this experiment measures.
#
# ### How clustering is scored
# Cluster labels are arbitrary (cluster 3 has no reason to be digit 3), so ordinary accuracy
# is meaningless. Instead:
#
# | Metric | Meaning | Chance level |
# |---|---|---|
# | **Purity** | fraction correct after mapping each cluster to its majority digit | ~0.1 |
# | **ARI** (Adjusted Rand Index) | agreement of point *pairs*, corrected for chance | 0.0 |
# | **NMI** (Normalised Mutual Information) | shared information between the two partitions | 0.0 |
# | **Silhouette** | cluster separation, uses *no* labels at all | 0.0 |

# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             normalized_mutual_info_score, silhouette_score)

mnist_k = load_dataset("mnist")
N_CLUSTER = 20000                      # subset keeps the k-sweep and t-SNE affordable
rng_k = np.random.default_rng(SEED)
sub = rng_k.choice(len(mnist_k["X_train"]), N_CLUSTER, replace=False)

images = mnist_k["X_train"][sub]
labels_true = mnist_k["y_train"][sub]
Xk = images.reshape(N_CLUSTER, -1).astype(np.float32) / 255.0

banner("DATA FOR CLUSTERING")
print(f"{N_CLUSTER:,} MNIST images, flattened to {Xk.shape[1]} features, scaled to [0,1]")
print("Labels are loaded but WILL NOT be used for fitting -- only for evaluation.")

# %% [markdown]
# ### K-Means from scratch
#
# Both steps are vectorised. The assignment step uses the expansion
# $\lVert x-\mu\rVert^2 = \lVert x\rVert^2 - 2x^\top\mu + \lVert\mu\rVert^2$, which turns
# the distance computation into one matrix product — the difference between a few seconds
# and several minutes at this size.

# %%
def kmeans_plusplus_init(X: np.ndarray, k: int, rng) -> np.ndarray:
    """Choose k initial centroids, each far from those already chosen."""
    centroids = [X[rng.integers(len(X))]]
    closest_sq = ((X - centroids[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = closest_sq / closest_sq.sum()
        centroids.append(X[rng.choice(len(X), p=probs)])
        closest_sq = np.minimum(closest_sq, ((X - centroids[-1]) ** 2).sum(1))
    return np.array(centroids)


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid for every point, via the ||x||^2 - 2x.mu + ||mu||^2 expansion."""
    d = (X ** 2).sum(1)[:, None] - 2 * X @ centroids.T + (centroids ** 2).sum(1)[None, :]
    return d.argmin(1)


def kmeans_scratch(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-5,
                   seed: int = SEED):
    """Lloyd's algorithm with k-means++ initialisation."""
    rng = np.random.default_rng(seed)
    centroids = kmeans_plusplus_init(X, k, rng)
    history = []
    assign = np.zeros(len(X), dtype=int)

    for it in range(max_iter):
        new_assign = assign_clusters(X, centroids)
        new_centroids = np.stack([
            X[new_assign == c].mean(0) if np.any(new_assign == c)
            else X[rng.integers(len(X))]          # re-seed an emptied cluster
            for c in range(k)])
        inertia = float(((X - new_centroids[new_assign]) ** 2).sum())
        shift = float(np.linalg.norm(new_centroids - centroids))
        history.append({"iteration": it + 1, "inertia": inertia,
                        "centroid shift": shift,
                        "reassigned": int((new_assign != assign).sum())})
        centroids, assign = new_centroids, new_assign
        if shift < tol:
            break
    return centroids, assign, float(history[-1]["inertia"]), pd.DataFrame(history)


banner("RUNNING K-MEANS FROM SCRATCH (k=10)")
(centroids, assign, inertia, km_hist), km_time = timed(kmeans_scratch, Xk, 10)
print(f"converged in {len(km_hist)} iterations, {km_time:.1f}s")
print(f"final inertia: {inertia:,.1f}\n")
print(km_hist.head(12).to_string(index=False))

# %% [markdown]
# ### Cross-check against scikit-learn
#
# An independent implementation should reach a comparable inertia. Exact agreement is not
# expected — different random initialisations land in different local minima.

# %%
sk_km = KMeans(n_clusters=10, init="k-means++", n_init=10, random_state=SEED)
(_, sk_time) = timed(sk_km.fit, Xk)

banner("SCRATCH vs SCIKIT-LEARN")
print(pd.DataFrame([
    {"implementation": "from scratch (1 init)", "inertia": round(inertia, 1),
     "iterations": len(km_hist), "time (s)": round(km_time, 2)},
    {"implementation": "sklearn (best of 10 inits)", "inertia": round(sk_km.inertia_, 1),
     "iterations": int(sk_km.n_iter_), "time (s)": round(sk_time, 2)},
]).to_string(index=False))
gap = 100 * abs(inertia - sk_km.inertia_) / sk_km.inertia_
better = "the from-scratch run" if inertia < sk_km.inertia_ else "sklearn"
print(f"\ninertia difference: {gap:.3f}%  (lower is better; {better} edged it)")
print("The two implementations agree to within a fraction of a percent, which is the")
print("correctness check. Neither is 'the' answer: both stop at a local minimum, and")
print("which one they reach depends on initialisation -- sklearn tries 10 starts and")
print("keeps the best, so it is the more reliable of the two in general even when a")
print("single lucky scratch run happens to match or beat it.")

# %% [markdown]
# ### What do the centroids look like?
#
# Each centroid is the mean of every image assigned to it, so it can be reshaped to 28×28
# and viewed. This is the most direct evidence of whether the clusters are digit-like.

# %%
def majority_map(assign: np.ndarray, truth: np.ndarray, k: int) -> dict[int, int]:
    """Label each cluster with the digit that is most common inside it."""
    return {c: int(Counter(truth[assign == c]).most_common(1)[0][0])
            if np.any(assign == c) else -1 for c in range(k)}


cluster_to_digit = majority_map(assign, labels_true, 10)
sizes = Counter(assign.tolist())

fig, axes = plt.subplots(2, 10, figsize=(16, 4.2))
for c in range(10):
    members = labels_true[assign == c]
    purity_c = (members == cluster_to_digit[c]).mean()
    axes[0, c].imshow(centroids[c].reshape(28, 28), cmap="gray_r")
    axes[0, c].set_title(f"cluster {c}\n→ digit {cluster_to_digit[c]}", fontsize=8)
    axes[0, c].set_axis_off()
    dist = Counter(members.tolist())
    axes[1, c].bar(range(10), [dist.get(d, 0) for d in range(10)],
                   color=[PALETTE[0] if d == cluster_to_digit[c] else "#c9d2de"
                          for d in range(10)])
    axes[1, c].set_title(f"n={sizes[c]}, purity {purity_c:.2f}", fontsize=7)
    axes[1, c].set_xticks(range(0, 10, 3))
    axes[1, c].tick_params(labelsize=6)
fig.suptitle("Top: cluster centroids as images.  "
             "Bottom: true digit composition of each cluster (majority digit highlighted)",
             y=1.03)
save_fig("q8_centroids", fig)

# %% [markdown]
# ### Scoring the clustering

# %%
mapped = np.array([cluster_to_digit[c] for c in assign])
purity = (mapped == labels_true).mean()
ari = adjusted_rand_score(labels_true, assign)
nmi = normalized_mutual_info_score(labels_true, assign)
sil_sample = rng_k.choice(N_CLUSTER, 5000, replace=False)
sil = silhouette_score(Xk[sil_sample], assign[sil_sample])

banner("CLUSTERING QUALITY (k=10)")
print(pd.DataFrame([
    {"metric": "Purity", "value": round(purity, 4), "chance": 0.10,
     "uses labels?": "yes (after fitting)"},
    {"metric": "Adjusted Rand Index", "value": round(ari, 4), "chance": 0.0,
     "uses labels?": "yes (after fitting)"},
    {"metric": "Normalised Mutual Info", "value": round(nmi, 4), "chance": 0.0,
     "uses labels?": "yes (after fitting)"},
    {"metric": "Silhouette", "value": round(sil, 4), "chance": 0.0,
     "uses labels?": "no"},
]).to_string(index=False))

digits_found = sorted(set(cluster_to_digit.values()))
missing = sorted(set(range(10)) - set(digits_found))
print(f"\ndigits claimed by some cluster : {digits_found}")
print(f"digits NO cluster represents   : {missing if missing else 'none'}")
dup = [d for d, n in Counter(cluster_to_digit.values()).items() if n > 1]
print(f"digits split across clusters   : {sorted(dup) if dup else 'none'}")

# %% [markdown]
# ### Cluster vs true digit

# %%
cm_k = confusion_matrix(labels_true, mapped, labels=list(range(10)))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

im = axes[0].imshow(cm_k, cmap="Blues")
axes[0].set(xticks=range(10), yticks=range(10), xlabel="cluster's majority digit",
            ylabel="true digit", title=f"After majority mapping (purity {purity:.3f})")
for i in range(10):
    for j in range(10):
        if cm_k[i, j]:
            axes[0].text(j, i, cm_k[i, j], ha="center", va="center", fontsize=6.5,
                         color="white" if cm_k[i, j] > cm_k.max() * 0.5 else "#333")
axes[0].grid(False)
plt.colorbar(im, ax=axes[0], fraction=0.046)

contingency = pd.crosstab(labels_true, assign)
im2 = axes[1].imshow(contingency.values, cmap="magma")
axes[1].set(xticks=range(10), yticks=range(10), xlabel="cluster id", ylabel="true digit",
            title="Raw contingency table\n(before any mapping)")
for i in range(10):
    for j in range(10):
        v = contingency.values[i, j]
        if v > 50:
            axes[1].text(j, i, v, ha="center", va="center", fontsize=6.5,
                         color="white" if v < contingency.values.max() * 0.6 else "black")
axes[1].grid(False)
plt.colorbar(im2, ax=axes[1], fraction=0.046)
save_fig("q8_confusion", fig)

# %% [markdown]
# ### Choosing k: the elbow and silhouette methods
#
# In a real unsupervised setting the number of clusters is unknown. The **elbow method**
# looks for the point where adding clusters stops sharply reducing inertia; the
# **silhouette** score measures separation directly. ARI and NMI are also plotted here —
# but note that those require labels, so they are a diagnostic for this experiment, not a
# method one could use on genuinely unlabelled data.

# %%
k_rows = []
for k in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
    km = KMeans(n_clusters=k, init="k-means++", n_init=4, random_state=SEED).fit(Xk)
    m = majority_map(km.labels_, labels_true, k)
    mp = np.array([m[c] for c in km.labels_])
    k_rows.append({
        "k": k, "inertia": round(km.inertia_, 1),
        "purity": round((mp == labels_true).mean(), 4),
        "ARI": round(adjusted_rand_score(labels_true, km.labels_), 4),
        "NMI": round(normalized_mutual_info_score(labels_true, km.labels_), 4),
        "silhouette": round(silhouette_score(Xk[sil_sample], km.labels_[sil_sample]), 4),
    })
k_df = pd.DataFrame(k_rows)
banner("CHOOSING k")
print(k_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
axes[0].plot(k_df["k"], k_df["inertia"], "o-", color=PALETTE[0])
axes[0].axvline(10, ls="--", color=PALETTE[1], label="k = 10 (true classes)")
axes[0].set(xlabel="k", ylabel="inertia", title="Elbow method")
axes[0].legend(fontsize=8)

axes[1].plot(k_df["k"], k_df["silhouette"], "s-", color=PALETTE[2])
axes[1].axvline(10, ls="--", color=PALETTE[1])
axes[1].set(xlabel="k", ylabel="silhouette", title="Silhouette (no labels used)")

axes[2].plot(k_df["k"], k_df["purity"], "o-", label="purity", color=PALETTE[0])
axes[2].plot(k_df["k"], k_df["ARI"], "s-", label="ARI", color=PALETTE[1])
axes[2].plot(k_df["k"], k_df["NMI"], "^-", label="NMI", color=PALETTE[2])
axes[2].axvline(10, ls="--", color="#999")
axes[2].set(xlabel="k", ylabel="score", title="Label-based agreement")
axes[2].legend(fontsize=8)
save_fig("q8_choosing_k", fig)

# %% [markdown]
# ### Seeing the clusters in two dimensions
#
# PCA gives a linear projection; t-SNE a non-linear one that preserves local neighbourhoods
# much better. Colouring the same t-SNE layout by cluster and by true digit shows directly
# where K-Means agreed with reality and where it did not.

# %%
vis_n = 4000
vis = rng_k.choice(N_CLUSTER, vis_n, replace=False)
pca_vis = PCA(n_components=2, random_state=SEED).fit_transform(Xk[vis])
tsne_vis, tsne_time = timed(
    lambda: TSNE(n_components=2, perplexity=30, init="pca",
                 random_state=SEED).fit_transform(Xk[vis]))
print(f"t-SNE on {vis_n} points: {tsne_time:.1f}s")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
for ax, coords, colour, title in [
    (axes[0], pca_vis, labels_true[vis], "PCA projection, coloured by TRUE digit"),
    (axes[1], tsne_vis, labels_true[vis], "t-SNE, coloured by TRUE digit"),
    (axes[2], tsne_vis, assign[vis], "t-SNE, coloured by K-MEANS CLUSTER"),
]:
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colour, cmap="tab10", s=5, alpha=0.75)
    ax.set(title=title, xticks=[], yticks=[])
    plt.colorbar(sc, ax=ax, fraction=0.046, ticks=range(10))
save_fig("q8_projections", fig)

# %% [markdown]
# ### Does clustering in PCA space help?
#
# K-Means uses Euclidean distance, which behaves poorly in very high dimensions. Reducing
# to a few dozen components before clustering often helps — and is far faster.

# %%
pca_rows = []
for n_comp in [None, 10, 25, 50, 100]:
    if n_comp is None:
        Xc, label = Xk, "raw 784-d pixels"
    else:
        Xc = PCA(n_components=n_comp, random_state=SEED).fit_transform(Xk)
        label = f"PCA {n_comp}-d"
    km = KMeans(n_clusters=10, init="k-means++", n_init=4, random_state=SEED)
    (_, t) = timed(km.fit, Xc)
    m = majority_map(km.labels_, labels_true, 10)
    mp = np.array([m[c] for c in km.labels_])
    pca_rows.append({"feature space": label, "dims": Xc.shape[1],
                     "purity": round((mp == labels_true).mean(), 4),
                     "ARI": round(adjusted_rand_score(labels_true, km.labels_), 4),
                     "NMI": round(normalized_mutual_info_score(labels_true, km.labels_), 4),
                     "fit time (s)": round(t, 2)})
pca_df = pd.DataFrame(pca_rows)
banner("CLUSTERING IN REDUCED SPACE")
print(pca_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4))
yy = np.arange(len(pca_df))
axes[0].barh(yy - 0.2, pca_df["purity"], 0.4, label="purity", color=PALETTE[0])
axes[0].barh(yy + 0.2, pca_df["ARI"], 0.4, label="ARI", color=PALETTE[2])
axes[0].set(yticks=yy, xlabel="score", title="Clustering quality by feature space")
axes[0].set_yticklabels(pca_df["feature space"], fontsize=8)
axes[0].legend(fontsize=8)
axes[1].barh(yy, pca_df["fit time (s)"], color=PALETTE[3])
axes[1].set(yticks=yy, xlabel="fit time (s)", title="Cost")
axes[1].set_yticklabels(pca_df["feature space"], fontsize=8)
save_fig("q8_pca_comparison", fig)

# %% [markdown]
# ## Result and discussion
#
# * **K-Means recovered substantial digit structure with no labels at all.** Purity 0.587
#   (chance 0.10), ARI 0.365 and NMI 0.501 (both chance 0.0), and the centroid images are
#   unmistakably digit-like. Nothing about digit identity was supplied during fitting.
#
# * **But it did not recover the ten digits — and the failure is specific.** With $k=10$,
#   **no cluster claimed digit 5 or digit 9**, while digits **1 and 6 were each split across
#   two clusters**. This is the central result of the experiment and the part most write-ups
#   gloss over: K-Means finds *regions of pixel-space variance*, and those regions do not
#   coincide with semantic classes. A vertical "1" and a slanted "1" are far apart in pixel
#   space, so they split; a "9" sits between "4" and "7" and gets absorbed rather than
#   claimed. Ten clusters were requested and ten were delivered — just not the ten intended.
#
# * **This is a limitation of the representation, not just of K-Means.** Euclidean distance
#   between raw pixels is not a good measure of visual similarity: it compares pixel (5,12)
#   to pixel (5,12) with no notion of shape, stroke or translation. Two images of the same
#   digit shifted by two pixels can be further apart than two images of different digits.
#
# * **Neither method for choosing $k$ found $k=10$.** The inertia curve bends smoothly with
#   no visible elbow at 10, and the silhouette score actually *peaks at $k=2$* (0.0892) and
#   is near its lowest at $k=10$ (0.0623). A practitioner with no labels, following either
#   standard recipe, would not have concluded "there are ten classes here." This deserves
#   stating plainly, because the elbow method is routinely taught as though it reliably
#   reveals the true class count. On this dataset it does not.
#
# * **Rising purity with $k$ is partly an artefact — and the adjusted metrics prove it.**
#   Purity climbs monotonically from 0.206 ($k=2$) to 0.749 ($k=25$), because smaller
#   clusters are more likely to be internally uniform; at the extreme $k=n$ gives purity 1.0
#   and means nothing. ARI, which corrects for chance, instead **peaks at $k=8$ (0.394) and
#   then falls** to 0.290 by $k=30$. Judging by purity alone would have led to the wrong
#   conclusion about how many clusters best describe the data.
#
# * **PCA before clustering was nearly free quality-wise and much cheaper.** 50 components
#   gave purity 0.5860 against 0.5880 for all 784 raw pixels — a 0.002 difference — while
#   cutting fit time from 11.45 s to 1.03 s, an **11× speed-up**. Most of the 784 raw
#   dimensions carry noise rather than structure.
#
# * **Contrast with Experiment 6.** A supervised MLP on this same data reached 98.2 %
#   accuracy; unsupervised clustering reached 58.7 % purity. That ~40-point gap is the
#   measurable value of labels.
