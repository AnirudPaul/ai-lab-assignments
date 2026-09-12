# %% [markdown]
# ---
# # Experiment 7 — Face Recognition with a Support Vector Machine (LFW)
#
# **Aim.** Recognise faces from the *Labeled Faces in the Wild* dataset using PCA for
# dimensionality reduction ("eigenfaces") followed by a support vector machine.
#
# ## Theory
#
# ### Support vector machines
#
# An SVM finds the hyperplane separating two classes with the **largest margin** — the
# widest gap to the nearest training point of either class. Only those nearest points, the
# **support vectors**, determine the boundary; the rest could be deleted without changing
# it. This is what makes SVMs work well when there are far more features than samples,
# exactly the situation with images.
#
# For non-separable data the soft-margin formulation solves
#
# $$\min_{w,b,\xi}\; \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i
#   \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0$$
#
# where $\xi_i$ are slack variables permitting violations and **$C$** sets the price of
# each one — small $C$ means a wide, tolerant margin (more regularisation); large $C$ means
# a narrow margin fitted hard to the training data.
#
# ### The kernel trick
#
# Faces are not linearly separable in pixel space. Rather than explicitly mapping to a
# higher-dimensional space, an SVM only ever needs *inner products*, which a kernel
# computes directly. The **RBF kernel**
#
# $$K(x, x') = \exp(-\gamma \lVert x - x'\rVert^2)$$
#
# corresponds to an infinite-dimensional feature space. **$\gamma$** sets how far a single
# training example's influence reaches: large $\gamma$ gives a tight, wiggly boundary that
# can overfit; small $\gamma$ gives a smooth one.
#
# ### Eigenfaces (PCA)
#
# Each image here is 1,850 pixels but there are only ~1,288 of them — more dimensions than
# samples. **PCA** projects onto the directions of greatest variance; applied to faces the
# resulting components are themselves face-like images, the classic **eigenfaces**. This
# cuts noise, speeds up training enormously, and reduces overfitting. `whiten=True`
# rescales each component to unit variance so no single direction dominates the RBF
# distance computation.

# %%
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

lfw = fetch_lfw_people(data_home=str(DATA / "sklearn"), min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw.images.shape
X = lfw.data
y = lfw.target
target_names = lfw.target_names

banner("LABELED FACES IN THE WILD")
print(f"images   : {n_samples} of size {h}x{w}  ->  {X.shape[1]} features each")
print(f"people   : {len(target_names)}")
print(f"\n{'person':25s} {'images':>7s}  {'share':>7s}")
counts = Counter(y.tolist())
for i, name in enumerate(target_names):
    print(f"{name:25s} {counts[i]:7d}  {100*counts[i]/n_samples:6.1f}%")

majority = max(counts.values()) / n_samples
print(f"\nThe dataset is strongly IMBALANCED. Always guessing "
      f"'{target_names[max(counts, key=counts.get)]}' would score {majority:.3f}.")
print("That figure -- not 1/7 = 0.143 -- is the baseline any model must beat.")

# %% [markdown]
# ### The faces

# %%
fig, axes = plt.subplots(3, 10, figsize=(15, 5.4))
rng_f = np.random.default_rng(SEED)
for ax, idx in zip(axes.ravel(), rng_f.choice(n_samples, 30, replace=False)):
    ax.imshow(lfw.images[idx], cmap="gray")
    ax.set_title(target_names[y[idx]].split()[-1], fontsize=7)
    ax.set_axis_off()
fig.suptitle("LFW samples — real photographs, varying pose, lighting and expression", y=1.02)
save_fig("q7_samples", fig)

# %% [markdown]
# ### Split first, then fit PCA
#
# PCA is fitted on the **training split only**. Fitting it on all the data before splitting
# would leak test-set structure into the transformation — a subtle and very common form of
# data leakage that quietly inflates reported accuracy.
#
# The split is **stratified**, so the class proportions above are preserved in both halves.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y)
print(f"train {X_train.shape}   test {X_test.shape}")

N_COMPONENTS = 150
pca, pca_time = timed(
    lambda: PCA(n_components=N_COMPONENTS, svd_solver="randomized", whiten=True,
                random_state=SEED).fit(X_train))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\nPCA: {X_train.shape[1]} features -> {N_COMPONENTS} components in {pca_time:.2f}s")
print(f"variance retained: {pca.explained_variance_ratio_.sum():.4f}")
print(f"compression: {X_train.shape[1]/N_COMPONENTS:.1f}x fewer dimensions")

# %% [markdown]
# ### The eigenfaces themselves

# %%
fig = plt.figure(figsize=(16, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], wspace=0.25, hspace=0.3)

ax = fig.add_subplot(gs[:, 0])
eigen_grid = np.vstack([np.hstack([pca.components_[r * 6 + c].reshape(h, w)
                                   for c in range(6)]) for r in range(3)])
ax.imshow(eigen_grid, cmap="gray")
ax.set_title("The first 18 eigenfaces (principal components)")
ax.set_axis_off()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(np.cumsum(pca.explained_variance_ratio_), color=PALETTE[0])
ax2.axhline(pca.explained_variance_ratio_.sum(), ls="--", color=PALETTE[1],
            label=f"{pca.explained_variance_ratio_.sum():.1%} at {N_COMPONENTS}")
ax2.set(xlabel="components", ylabel="cumulative variance", title="Variance explained")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(range(20), pca.explained_variance_ratio_[:20], color=PALETTE[0])
ax3.set(xlabel="component", ylabel="variance ratio", title="First 20 components")

# reconstruction quality at different component counts
ax4 = fig.add_subplot(gs[:, 2])
sample = X_test[0]
recons = [sample.reshape(h, w)]
labels = ["original"]
for k in [10, 50, 150]:
    p = PCA(n_components=k, svd_solver="randomized", random_state=SEED).fit(X_train)
    recons.append(p.inverse_transform(p.transform(sample[None]))[0].reshape(h, w))
    labels.append(f"{k} comps")
ax4.imshow(np.hstack(recons), cmap="gray")
ax4.set_title("Reconstruction: " + " | ".join(labels), fontsize=9)
ax4.set_axis_off()
save_fig("q7_eigenfaces", fig)

# %% [markdown]
# ### Tuning the SVM
#
# $C$ and $\gamma$ interact, so they are searched jointly by 5-fold cross-validation on the
# training set. `class_weight="balanced"` scales each class's penalty by the inverse of its
# frequency, which matters given how dominant one person is in this dataset.

# %%
# A first pass used C >= 1e2 and gamma >= 1e-4 and selected the extreme corner of that
# grid (C=100, gamma=1e-4). Selecting a boundary value means the true optimum may lie
# outside the range searched, so the grid is widened downwards in both parameters until
# the winner sits in the interior.
param_grid = {"C": [1, 10, 1e2, 1e3, 1e4],
              "gamma": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2]}
grid = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid,
                    cv=5, n_jobs=-1, scoring="balanced_accuracy")
(_, grid_time) = timed(grid.fit, X_train_pca, y_train)

banner("GRID SEARCH")
print(f"searched {len(param_grid['C']) * len(param_grid['gamma'])} combinations "
      f"x 5 folds in {grid_time:.1f}s")
print(f"best parameters : {grid.best_params_}")
print(f"best CV score   : {grid.best_score_:.4f} (balanced accuracy)")

clf = grid.best_estimator_
print(f"\nsupport vectors : {clf.n_support_.sum()} of {len(X_train)} training samples "
      f"({100*clf.n_support_.sum()/len(X_train):.0f}%)")
print("per class       :", dict(zip([n.split()[-1] for n in target_names],
                                    clf.n_support_.tolist())))

scores = grid.cv_results_["mean_test_score"].reshape(len(param_grid["C"]),
                                                     len(param_grid["gamma"]))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
im = ax.imshow(scores, cmap="viridis")
ax.set(xticks=range(len(param_grid["gamma"])), yticks=range(len(param_grid["C"])),
       xlabel="gamma", ylabel="C", title="Cross-validated balanced accuracy")
ax.set_xticklabels(param_grid["gamma"]); ax.set_yticklabels(param_grid["C"])
for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        ax.text(j, i, f"{scores[i,j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if scores[i, j] < scores.max() * 0.95 else "black")
ax.grid(False)
plt.colorbar(im, ax=ax, fraction=0.046)
save_fig("q7_grid_search", fig)

# %% [markdown]
# ### Test-set evaluation

# %%
y_pred = clf.predict(X_test_pca)
from sklearn.metrics import accuracy_score, balanced_accuracy_score

acc = accuracy_score(y_test, y_pred)
bal = balanced_accuracy_score(y_test, y_pred)

banner("TEST SET RESULTS")
print(f"accuracy          : {acc:.4f}")
print(f"balanced accuracy : {bal:.4f}   <- the fair measure on imbalanced data")
print(f"majority baseline : {majority:.4f}\n")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# %% [markdown]
# ### Confusion matrix and predictions

# %%
fig = plt.figure(figsize=(16.5, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.2)

ax = fig.add_subplot(gs[0])
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap="Blues")
short = [n.split()[-1] for n in target_names]
ax.set(xticks=range(len(short)), yticks=range(len(short)),
       xlabel="predicted", ylabel="true", title=f"Confusion matrix (accuracy {acc:.3f})")
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short, fontsize=8)
for i in range(len(short)):
    for j in range(len(short)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "#333")
ax.grid(False)

ax2 = fig.add_subplot(gs[1])
ax2.set_axis_off()
sub = fig.add_gridspec(3, 8, left=0.42, right=0.99, top=0.92, bottom=0.05,
                       wspace=0.08, hspace=0.45)
rng_p = np.random.default_rng(SEED)
show = rng_p.choice(len(y_test), 24, replace=False)
wrong_idx = np.where(y_pred != y_test)[0]
show = np.concatenate([wrong_idx[:8], show])[:24]
for k, idx in enumerate(show):
    a = fig.add_subplot(sub[k // 8, k % 8])
    a.imshow(X_test[idx].reshape(h, w), cmap="gray")
    ok = y_pred[idx] == y_test[idx]
    a.set_title(f"{short[y_test[idx]]}\n-> {short[y_pred[idx]]}", fontsize=6.5,
                color="#2A4B9B" if ok else "#C44E52")
    for s in a.spines.values():
        s.set_edgecolor("#2A4B9B" if ok else "#C44E52")
        s.set_linewidth(1.6)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Left: confusion matrix.  Right: predictions "
             "(blue = correct, red = wrong; the 8 errors are shown first)", y=1.0)
save_fig("q7_results", fig)

# %% [markdown]
# ### Was PCA worth it? Does the kernel matter?
#
# Four controlled comparisons, all on the same split: raw pixels versus eigenfaces, and
# linear versus RBF kernel.

# %%
ablation = []
configs = [
    ("RBF + PCA-150 (tuned)", clf, X_train_pca, X_test_pca),
    ("RBF + raw pixels", SVC(kernel="rbf", C=grid.best_params_["C"],
                             gamma="scale", class_weight="balanced"), X_train, X_test),
    ("Linear + PCA-150", SVC(kernel="linear", C=1.0, class_weight="balanced"),
     X_train_pca, X_test_pca),
    ("Linear + raw pixels", SVC(kernel="linear", C=1.0, class_weight="balanced"),
     X_train, X_test),
]
for name, model, Xtr, Xte in configs:
    (_, t) = timed(model.fit, Xtr, y_train)
    p = model.predict(Xte)
    ablation.append({"configuration": name, "features": Xtr.shape[1],
                     "accuracy": round(accuracy_score(y_test, p), 4),
                     "balanced acc": round(balanced_accuracy_score(y_test, p), 4),
                     "fit time (s)": round(t, 2)})

abl_df = pd.DataFrame(ablation)
banner("ABLATION")
print(abl_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
yy = np.arange(len(abl_df))
axes[0].barh(yy - 0.2, abl_df["accuracy"], 0.4, label="accuracy", color=PALETTE[0])
axes[0].barh(yy + 0.2, abl_df["balanced acc"], 0.4, label="balanced accuracy",
             color=PALETTE[2])
axes[0].axvline(majority, ls="--", color=PALETTE[1], label=f"majority baseline ({majority:.2f})")
axes[0].set(yticks=yy, xlim=(0, 1.05), xlabel="score", title="Accuracy by configuration")
axes[0].set_yticklabels(abl_df["configuration"], fontsize=8)
axes[0].legend(fontsize=7, loc="lower right")

axes[1].barh(yy, abl_df["fit time (s)"], color=PALETTE[3])
axes[1].set(yticks=yy, xlabel="fit time (s)", title="Training cost")
axes[1].set_yticklabels(abl_df["configuration"], fontsize=8)
for i, v in enumerate(abl_df["fit time (s)"]):
    axes[1].text(v, i, f" {v}s", va="center", fontsize=8)
save_fig("q7_ablation", fig)

# %% [markdown]
# ### How many components are actually needed?

# %%
comp_rows = []
for k in [10, 25, 50, 100, 150, 250]:
    p = PCA(n_components=k, svd_solver="randomized", whiten=True,
            random_state=SEED).fit(X_train)
    m = SVC(kernel="rbf", C=grid.best_params_["C"], gamma=grid.best_params_["gamma"],
            class_weight="balanced").fit(p.transform(X_train), y_train)
    pr = m.predict(p.transform(X_test))
    comp_rows.append({"components": k,
                      "variance": round(p.explained_variance_ratio_.sum(), 4),
                      "accuracy": round(accuracy_score(y_test, pr), 4),
                      "balanced acc": round(balanced_accuracy_score(y_test, pr), 4)})
comp_df = pd.DataFrame(comp_rows)
banner("EFFECT OF COMPONENT COUNT")
print(comp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(comp_df["components"], comp_df["accuracy"], "o-", label="accuracy", color=PALETTE[0])
ax.plot(comp_df["components"], comp_df["balanced acc"], "s-", label="balanced accuracy",
        color=PALETTE[2])
ax2 = ax.twinx()
ax2.plot(comp_df["components"], comp_df["variance"], "^--", color=PALETTE[3],
         label="variance retained")
ax2.set_ylabel("variance retained")
ax2.grid(False)
ax.set(xlabel="PCA components", ylabel="score", title="Accuracy vs number of eigenfaces")
ax.legend(loc="lower right", fontsize=8)
save_fig("q7_components", fig)

# %% [markdown]
# ## Result and discussion
#
# * **The pipeline recognises faces well above the honest baseline.** Test accuracy
#   **0.8478**, balanced accuracy **0.8148**, against a majority-class baseline of 0.4115.
#   That baseline is the one that matters: George W Bush supplies 41 % of this dataset, so a
#   model that learned nothing but "always say Bush" would already score 0.41. Quoting
#   accuracy against a 1/7 = 0.143 chance level, as is often done with LFW, roughly doubles
#   the apparent achievement.
#
# * **PCA is doing real work.** 1,850 pixels reduced to 150 eigenfaces retains 94.6 % of the
#   variance at 12.3× fewer dimensions, and cuts SVM fit time from 1.35 s to 0.19 s. The
#   reconstruction strip makes the trade-off visible: 10 components give a blurry archetype,
#   150 a recognisable individual.
#
# * **Both the kernel and the representation mattered — and only together.** The full
#   ablation:
#
#   | configuration | accuracy | balanced acc | fit time |
#   |---|---|---|---|
#   | RBF + PCA-150 (tuned) | **0.8478** | **0.8148** | 0.19 s |
#   | Linear + raw pixels | 0.8478 | 0.7894 | 0.70 s |
#   | RBF + raw pixels | 0.8261 | 0.7354 | 1.35 s |
#   | Linear + PCA-150 | 0.7578 | 0.7137 | 0.13 s |
#
#   Neither ingredient works alone. RBF on raw pixels is mediocre; a linear kernel on
#   eigenfaces is the *worst* configuration tested. Only the combination wins, and it wins
#   on balanced accuracy — the metric that counts the rare classes — by 2.5 points over the
#   nearest rival while training 3.7× faster.
#
# * **The grid search was necessary, and its first version was wrong.** The initial grid
#   ($C \ge 10^2$, $\gamma \ge 10^{-4}$) returned the extreme corner $C{=}100,
#   \gamma{=}10^{-4}$. Selecting a *boundary* value is a warning that the optimum lies
#   outside the range searched, so the grid was widened downwards. The interior optimum
#   $C{=}10$ then raised cross-validated balanced accuracy from 0.7338 to 0.7771 and test
#   accuracy from 0.8043 to **0.8478** — 4.4 points recovered purely by searching a wide
#   enough range. Reporting the first result without checking where it sat in the grid would
#   have understated the method.
#
# * **150 components is a genuine optimum, not an arbitrary choice.** The sweep shows
#   accuracy rising to 0.8478 at 150 and then *falling* to 0.8106 at 250, even though 250
#   components retain more variance (97.5 % vs 94.6 %). The extra components encode lighting
#   and background noise rather than identity. Ten components (0.19 accuracy) are
#   catastrophically too few — worse than the majority baseline.
#
# * **Performance tracks the number of training examples almost monotonically.** George W
#   Bush, with 530 images, scores F1 0.904; Gerhard Schroeder (109 images) scores 0.667 and
#   Hugo Chavez (71) scores 0.737. The model is not equally good at recognising everyone,
#   and the aggregate accuracy hides that entirely — which is the argument for reporting the
#   per-class table rather than a single number.
#
# * **99 % of training points became support vectors** (955 of 966). With a small $\gamma$
#   and a soft margin, almost every example sits inside the margin and contributes to the
#   boundary. The model is therefore not a sparse one, and the usual "SVMs give you a
#   compact model defined by a few points" claim simply does not apply at this setting.
#
# * **`class_weight="balanced"` was the right call.** Without it the decision boundary would
#   be pulled towards the dominant class; with it, balanced accuracy — which averages recall
#   over classes rather than over samples — reflects what the model actually learned. The
#   gap between accuracy (0.8478) and balanced accuracy (0.8148) is the residual effect of
#   imbalance that reweighting did not fully remove.
