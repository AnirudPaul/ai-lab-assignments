# %% [markdown]
# ---
# # Experiment 6 — Handwritten Digit Recognition with a Multilayer Perceptron
#
# **Aim.** Train a multilayer perceptron to classify handwritten digits from the MNIST
# dataset, and analyse where and why it fails.
#
# ## Theory
#
# A multilayer perceptron is a stack of fully connected layers separated by a non-linear
# activation. For input $\mathbf{x} \in \mathbb{R}^{784}$ (a flattened 28×28 image):
#
# $$\mathbf{h}_1 = \sigma(W_1\mathbf{x} + \mathbf{b}_1), \quad
#   \mathbf{h}_2 = \sigma(W_2\mathbf{h}_1 + \mathbf{b}_2), \quad
#   \mathbf{z} = W_3\mathbf{h}_2 + \mathbf{b}_3$$
#
# **Why the non-linearity is essential.** Without $\sigma$, the composition
# $W_3W_2W_1\mathbf{x}$ is itself a single linear map — a hundred stacked linear layers
# would have exactly the representational power of one. The activation is what makes depth
# meaningful. This experiment tests that claim directly rather than asserting it.
#
# We use **ReLU**, $\sigma(x) = \max(0, x)$: cheap, and its gradient is 1 for positive
# inputs, so it does not saturate the way a sigmoid does for large $|x|$.
#
# The output layer produces logits converted to probabilities by **softmax**, and training
# minimises **cross-entropy loss**:
#
# $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
#   \mathcal{L} = -\sum_i y_i \log p_i$$
#
# Weights are updated by **backpropagation** — the chain rule applied backwards through the
# network — using the **Adam** optimiser, which adapts a per-parameter learning rate from
# running estimates of the gradient's first and second moments.
#
# ### Regularisation
# **Dropout** randomly zeroes a fraction of activations during training, which prevents
# units from co-adapting and acts like averaging over many thinned networks. It is disabled
# at evaluation time.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

mnist = load_dataset("mnist")
X_train_full, y_train_full = mnist["X_train"], mnist["y_train"]
X_test, y_test = mnist["X_test"], mnist["y_test"]

banner("MNIST")
print(f"train images : {X_train_full.shape}  dtype={X_train_full.dtype}")
print(f"test images  : {X_test.shape}")
print(f"pixel range  : [{X_train_full.min()}, {X_train_full.max()}]")
print(f"classes      : {sorted(set(y_train_full.tolist()))}")
print("\nclass balance (train):")
counts = Counter(y_train_full.tolist())
for d in range(10):
    print(f"   digit {d}: {counts[d]:5d}  ({100*counts[d]/len(y_train_full):.1f}%)")

# %% [markdown]
# ### Looking at the data first

# %%
fig, axes = plt.subplots(4, 12, figsize=(14, 5))
rng_v = np.random.default_rng(SEED)
for i, ax in enumerate(axes.ravel()):
    j = rng_v.integers(len(X_train_full))
    ax.imshow(X_train_full[j], cmap="gray_r")
    ax.set_title(str(y_train_full[j]), fontsize=8, pad=2)
    ax.set_axis_off()
fig.suptitle("MNIST samples — 28×28 greyscale handwritten digits", y=1.01)
save_fig("q6_samples", fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
axes[0].bar(range(10), [counts[d] for d in range(10)], color=PALETTE[0])
axes[0].set(xticks=range(10), xlabel="digit", ylabel="training images",
            title="Class distribution (roughly balanced)")
mean_imgs = np.stack([X_train_full[y_train_full == d].mean(0) for d in range(10)])
axes[1].imshow(np.hstack(mean_imgs), cmap="magma")
axes[1].set(title="Mean image per class — the 'average' handwriting for each digit")
axes[1].set_axis_off()
save_fig("q6_class_distribution", fig)

# %% [markdown]
# ### Preprocessing
#
# Pixels are scaled to $[0,1]$ and standardised using the **training** mean and standard
# deviation only — computing those statistics over the test set too would leak information
# from the evaluation data into training. A 54k/6k train/validation split is carved out so
# that the test set is touched exactly once, at the end.

# %%
X_train_full_f = X_train_full.reshape(len(X_train_full), -1).astype(np.float32) / 255.0
X_test_f = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0

n_val = 6000
perm = np.random.default_rng(SEED).permutation(len(X_train_full_f))
val_idx, train_idx = perm[:n_val], perm[n_val:]

mu = X_train_full_f[train_idx].mean()
sd = X_train_full_f[train_idx].std()
standardise = lambda a: (a - mu) / sd

X_tr = standardise(X_train_full_f[train_idx])
y_tr = y_train_full[train_idx]
X_va = standardise(X_train_full_f[val_idx])
y_va = y_train_full[val_idx]
X_te = standardise(X_test_f)

print(f"train {X_tr.shape} | val {X_va.shape} | test {X_te.shape}")
print(f"normalisation from TRAIN split only: mean={mu:.4f}, sd={sd:.4f}")


def loader(X, y, batch_size=128, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_dl = loader(X_tr, y_tr, shuffle=True)
val_dl = loader(X_va, y_va)
test_dl = loader(X_te, y_test)

# %% [markdown]
# ### The model

# %%
class MLP(nn.Module):
    """784 -> 256 -> 128 -> 10 with ReLU and dropout."""

    def __init__(self, hidden=(256, 128), p_drop=0.2, activation=nn.ReLU):
        super().__init__()
        layers, prev = [], 784
        for h in hidden:
            layers += [nn.Linear(prev, h), activation(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate_model(model, dl) -> tuple[float, float, np.ndarray]:
    """Returns (loss, accuracy, predictions) with dropout disabled."""
    model.eval()
    total_loss, correct, preds = 0.0, 0, []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            total_loss += F.cross_entropy(out, yb, reduction="sum").item()
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            preds.append(p.cpu().numpy())
    n = len(dl.dataset)
    return total_loss / n, correct / n, np.concatenate(preds)


def train_model(model, train_dl, val_dl, epochs=20, lr=1e-3, quiet=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(xb)
            run_correct += (out.argmax(1) == yb).sum().item()
        n = len(train_dl.dataset)
        va_loss, va_acc, _ = evaluate_model(model, val_dl)
        history.append({"epoch": epoch, "train_loss": run_loss / n,
                        "train_acc": run_correct / n,
                        "val_loss": va_loss, "val_acc": va_acc})
        if not quiet and (epoch % 4 == 0 or epoch == 1):
            print(f"  epoch {epoch:2d}  train_loss={run_loss/n:.4f} "
                  f"train_acc={run_correct/n:.4f}  val_loss={va_loss:.4f} "
                  f"val_acc={va_acc:.4f}")
    return model, pd.DataFrame(history)


torch.manual_seed(SEED)
model = MLP()
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"\ntrainable parameters: {n_params:,}")
print(f"  layer 1: 784x256 + 256 = {784*256+256:,}")
print(f"  layer 2: 256x128 + 128 = {256*128+128:,}")
print(f"  layer 3: 128x10  + 10  = {128*10+10:,}")

banner("TRAINING")
model, hist = train_model(model, train_dl, val_dl, epochs=20)

# %% [markdown]
# ### Training curves
#
# The gap between the training and validation curves is the thing to watch: if training
# accuracy climbs while validation accuracy stalls, the network is memorising.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train", color=PALETTE[0])
axes[0].plot(hist["epoch"], hist["val_loss"], "s-", label="validation", color=PALETTE[1])
axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
axes[0].legend()

axes[1].plot(hist["epoch"], hist["train_acc"], "o-", label="train", color=PALETTE[0])
axes[1].plot(hist["epoch"], hist["val_acc"], "s-", label="validation", color=PALETTE[1])
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
axes[1].legend()

axes[2].plot(hist["epoch"], hist["train_acc"] - hist["val_acc"], "d-", color=PALETTE[4])
axes[2].axhline(0, color="#888", lw=0.8)
axes[2].set(xlabel="epoch", ylabel="train acc − val acc",
            title="Generalisation gap\n(positive and growing = overfitting)")
save_fig("q6_training_curves", fig)

# %% [markdown]
# ### Test-set evaluation
#
# The test set has been untouched until this point.

# %%
from sklearn.metrics import classification_report, confusion_matrix

test_loss, test_acc, test_preds = evaluate_model(model, test_dl)
banner("TEST SET RESULTS")
print(f"loss     : {test_loss:.4f}")
print(f"accuracy : {test_acc:.4f}  ({int(test_acc*len(y_test)):,}/{len(y_test):,} correct)")
print(f"error    : {100*(1-test_acc):.2f}%\n")
print(classification_report(y_test, test_preds, digits=4))

# %% [markdown]
# ### Where does it go wrong?

# %%
cm = confusion_matrix(y_test, test_preds)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

im = axes[0].imshow(cm, cmap="Blues")
axes[0].set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true",
            title=f"Confusion matrix (test accuracy {test_acc:.4f})")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                         color="white" if cm[i, j] > cm.max() * 0.5 else "#333")
axes[0].grid(False)
plt.colorbar(im, ax=axes[0], fraction=0.046)

per_class = cm.diagonal() / cm.sum(axis=1)
order = np.argsort(per_class)
axes[1].barh([str(d) for d in order], per_class[order], color=PALETTE[0])
axes[1].set(xlim=(0.9, 1.0), xlabel="recall", ylabel="digit",
            title="Per-class recall (hardest at the bottom)")
for i, d in enumerate(order):
    axes[1].text(per_class[d] + 0.001, i, f"{per_class[d]:.3f}", va="center", fontsize=8)
save_fig("q6_confusion_matrix", fig)

off = cm.copy()
np.fill_diagonal(off, 0)
pairs = [(off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j]]
pairs.sort(reverse=True)
print("Most frequent confusions (true -> predicted):")
for n, i, j in pairs[:8]:
    print(f"   {i} -> {j}: {n} times")

# %% [markdown]
# ### The actual mistakes
#
# Plotting the misclassified digits is more informative than any aggregate metric — many
# are genuinely ambiguous, which puts a ceiling on what any model can achieve.

# %%
wrong = np.where(test_preds != y_test)[0]
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_te[wrong]).to(DEVICE))
    conf = F.softmax(logits, dim=1).max(1).values.cpu().numpy()
most_confident_errors = wrong[np.argsort(-conf)]

fig, axes = plt.subplots(3, 10, figsize=(15, 5.2))
for ax, idx in zip(axes.ravel(), most_confident_errors[:30]):
    ax.imshow(X_test[idx], cmap="gray_r")
    c = conf[list(wrong).index(idx)]
    ax.set_title(f"true {y_test[idx]} / pred {test_preds[idx]}\nconf {c:.2f}", fontsize=7)
    ax.set_axis_off()
fig.suptitle(f"The {len(wrong)} test errors — 30 the model was most confident about",
             y=1.02)
save_fig("q6_misclassified", fig)
print(f"total errors: {len(wrong)} / {len(y_test)}")

# %% [markdown]
# ### What did the first layer learn?
#
# Each of the 256 first-layer units has a 784-dimensional weight vector, which can be
# reshaped back to 28×28 and viewed as the image patch that unit responds to. These are
# stroke and blob detectors — not whole digits.

# %%
W1 = model.net[0].weight.detach().cpu().numpy()
fig, axes = plt.subplots(4, 12, figsize=(14, 5))
lim = np.abs(W1).max()
for ax, i in zip(axes.ravel(), np.argsort(-np.abs(W1).sum(1))[:48]):
    ax.imshow(W1[i].reshape(28, 28), cmap="RdBu_r", vmin=-lim * 0.6, vmax=lim * 0.6)
    ax.set_axis_off()
fig.suptitle("First-layer weights as images (48 highest-magnitude units) — "
             "red positive, blue negative", y=1.02)
save_fig("q6_layer1_weights", fig)

# %% [markdown]
# ### Does depth and non-linearity actually matter?
#
# The theory section claimed a network without a non-linear activation collapses to a
# linear model no matter how many layers it has. That is a testable claim, so here it is
# tested — along with the effect of width, depth and dropout. Every variant is trained
# under identical conditions.

# %%
def build_linear_stack():
    """Same shape as the MLP but with Identity in place of ReLU -- mathematically a
    single linear map, however many layers are stacked."""
    return MLP(hidden=(256, 128), p_drop=0.0, activation=nn.Identity)


VARIANTS = {
    "Linear (no activation)": lambda: build_linear_stack(),
    "1 hidden layer (128)": lambda: MLP(hidden=(128,), p_drop=0.2),
    "2 hidden (256,128) [main]": lambda: MLP(hidden=(256, 128), p_drop=0.2),
    "3 hidden (512,256,128)": lambda: MLP(hidden=(512, 256, 128), p_drop=0.2),
    "2 hidden, no dropout": lambda: MLP(hidden=(256, 128), p_drop=0.0),
}

arch_rows, arch_hist = [], {}
for name, build in VARIANTS.items():
    torch.manual_seed(SEED)
    m, h = train_model(build(), train_dl, val_dl, epochs=12, quiet=True)
    _, acc, _ = evaluate_model(m, test_dl)
    arch_rows.append({
        "architecture": name,
        "parameters": sum(p.numel() for p in m.parameters()),
        "final train acc": round(h["train_acc"].iloc[-1], 4),
        "final val acc": round(h["val_acc"].iloc[-1], 4),
        "test acc": round(acc, 4),
        "gap": round(h["train_acc"].iloc[-1] - h["val_acc"].iloc[-1], 4),
    })
    arch_hist[name] = h
    free_gpu()
    print(f"  {name:28s} test acc = {acc:.4f}")

arch_df = pd.DataFrame(arch_rows)
banner("ARCHITECTURE COMPARISON (12 epochs each, identical settings)")
print(arch_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
for i, (name, h) in enumerate(arch_hist.items()):
    axes[0].plot(h["epoch"], h["val_acc"], "-o", ms=3, color=PALETTE[i], label=name)
axes[0].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy by architecture")
axes[0].legend(fontsize=7)

y = np.arange(len(arch_df))
axes[1].barh(y, arch_df["test acc"], color=[PALETTE[i] for i in range(len(arch_df))])
axes[1].set(yticks=y, xlim=(0.9, 1.0), xlabel="test accuracy", title="Final test accuracy")
axes[1].set_yticklabels(arch_df["architecture"], fontsize=8)
for i, v in enumerate(arch_df["test acc"]):
    axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
save_fig("q6_architecture_comparison", fig)

# %% [markdown]
# ### Cross-check against scikit-learn
#
# An independent implementation of the same idea. If the PyTorch model is roughly right,
# `MLPClassifier` with a comparable architecture should land in the same region.

# %%
from sklearn.neural_network import MLPClassifier

sk = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                   alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=20,
                   random_state=SEED, early_stopping=False, verbose=False)
(_, sk_time) = timed(sk.fit, X_tr, y_tr)
sk_acc = sk.score(X_te, y_test)

banner("PYTORCH vs SCIKIT-LEARN")
print(pd.DataFrame([
    {"implementation": "PyTorch MLP (this notebook)", "hidden": "256,128",
     "epochs": 20, "test accuracy": round(test_acc, 4), "device": str(DEVICE)},
    {"implementation": "sklearn MLPClassifier", "hidden": "256,128",
     "epochs": 20, "test accuracy": round(sk_acc, 4), "device": "cpu"},
]).to_string(index=False))
print(f"\nsklearn training time: {sk_time:.1f}s (CPU)")
print(f"difference in test accuracy: {abs(test_acc - sk_acc)*100:.2f} percentage points")

# %% [markdown]
# ## Result and discussion
#
# * **The MLP classifies MNIST well.** The 2-hidden-layer network reached **98.18 %** test
#   accuracy (182 errors in 10,000) with 235,146 parameters, and the training and validation
#   curves tracked each other closely throughout.
#
# * **The non-linearity claim was tested, not assumed — and it held decisively.** The
#   "Linear (no activation)" variant has *exactly* the same layer sizes and the same 235,146
#   parameters as the main model, but with `Identity` in place of `ReLU` it is algebraically
#   a single linear map. It scored **91.28 %** against 98.10 % for the identical
#   architecture with ReLU: a 6.8-point gap bought by nothing but the activation function.
#   91 % is precisely where multinomial logistic regression lands on MNIST, which is the
#   point — those extra layers contributed nothing at all.
#
# * **Depth helped, but with sharply diminishing returns.** 97.90 % (1 hidden) → 98.10 %
#   (2 hidden) → 98.26 % (3 hidden). The third layer *was* the best of the three, so it is
#   not true that depth stopped helping — but it bought +0.16 points for 2.4× the
#   parameters (567k vs 235k). On a saturated dataset that is a poor exchange rate.
#
# * **Dropout earned its place.** Removing it pushed training accuracy *up* to 99.49 % while
#   test accuracy fell to 97.64 %, widening the generalisation gap from 0.0104 to 0.0189.
#   That is textbook overfitting, visible in a single controlled comparison.
#
# * **The errors are mostly genuinely ambiguous.** The confusion matrix is dominated by
#   9→4 (11 cases), 4→9 (8), 3→9 (8), 3→5 (8) and 2→7 (8) — pairs separated by a single
#   stroke, which a human annotator would also hesitate over. This is the practical ceiling
#   of the dataset rather than a fixable modelling deficiency.
#
# * **The first layer learned strokes, not digits.** The weight images show localised
#   stroke and blob detectors rather than digit templates: the network builds its decision
#   from parts, which is exactly the compositional behaviour the architecture is meant to
#   encourage.
#
# * **Independent agreement.** scikit-learn's `MLPClassifier`, a completely separate
#   implementation with the same architecture, reached 97.82 % against PyTorch's 98.18 % —
#   a gap of 0.36 points. Good evidence the result is a property of the method rather than
#   of one implementation. It took 110 s on the CPU against a few seconds on the GPU, which
#   is the practical argument for the deep-learning stack even on a problem this small.
#
# * **The structural limitation.** An MLP flattens the image, discarding the fact that
#   neighbouring pixels are related: shuffle every image's pixels with one fixed
#   permutation and this model would score identically. That wasted spatial structure is
#   precisely what the convolutional network in Experiment 9 exploits.
