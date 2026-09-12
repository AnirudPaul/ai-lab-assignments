# %% [markdown]
# ---
# # Experiment 9 — Image Classification with a Convolutional Neural Network
#
# **Aim.** Classify CIFAR-10 colour images with a CNN, and measure directly how much the
# convolutional structure contributes by comparing against a fully connected network with a
# comparable parameter budget.
#
# ## Theory
#
# Experiment 6 ended on a specific complaint: an MLP flattens an image, so it never learns
# that neighbouring pixels are related. A convolutional layer fixes that with three ideas.
#
# **1. Local receptive fields.** Each unit sees only a small patch (here 3×3) rather than
# the whole image. Visual features — edges, corners, textures — are local, so this discards
# almost nothing useful.
#
# **2. Parameter sharing.** The *same* filter slides across every position. A 3×3 filter on
# a 32×32 input uses 9 weights instead of $32^2 \times 32^2$, and a fully connected layer
# over a 32×32×3 image would need ~3.1 M weights for a single 1024-unit layer. This is the
# main reason CNNs are tractable.
#
# **3. Translation equivariance.** Because the filter is shared, shifting the input shifts
# the feature map correspondingly — a cat detected in the corner uses the same weights as a
# cat in the centre. An MLP must learn each position independently.
#
# The 2-D convolution for output channel $o$ is
#
# $$y_{o,i,j} = b_o + \sum_{c}\sum_{u,v} w_{o,c,u,v}\; x_{c,\,i+u,\,j+v}$$
#
# ### Supporting components
#
# * **Max pooling** takes the maximum over a 2×2 window, halving spatial resolution. This
#   buys a degree of translation *invariance* and enlarges the effective receptive field of
#   later layers.
# * **Batch normalisation** standardises each channel's activations over the batch. It
#   stabilises training, permits higher learning rates, and has a mild regularising effect.
# * **Data augmentation** — random crops and horizontal flips — synthesises new training
#   images. A horizontally flipped cat is still a cat, so this is free label-preserving data
#   and directly attacks overfitting.
#
# CIFAR-10 is far harder than MNIST: colour, cluttered natural backgrounds, and enormous
# within-class variation in pose, scale and lighting.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

free_gpu()   # release anything Experiment 6 left cached

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

cifar = load_dataset("cifar10")
Xc_train, yc_train = cifar["X_train"], cifar["y_train"]
Xc_test, yc_test = cifar["X_test"], cifar["y_test"]

banner("CIFAR-10")
print(f"train : {Xc_train.shape}   test : {Xc_test.shape}")
print(f"classes: {len(CLASSES)} -> {', '.join(CLASSES)}")
print(f"per class in train: {Counter(yc_train.tolist())[0]:,} (perfectly balanced)")

fig, axes = plt.subplots(10, 10, figsize=(12, 12.4))
rng_c = np.random.default_rng(SEED)
for r, cls in enumerate(CLASSES):
    idxs = np.where(yc_train == r)[0]
    for c in range(10):
        ax = axes[r, c]
        ax.imshow(Xc_train[rng_c.choice(idxs)])
        ax.set_axis_off()
        if c == 0:
            ax.text(-0.6, 0.5, cls, transform=ax.transAxes, ha="right", va="center",
                    fontsize=9, fontweight="bold")
fig.suptitle("CIFAR-10 — ten classes, 32×32 colour, natural backgrounds", y=0.905)
save_fig("q9_samples", fig)

# %% [markdown]
# ### Preprocessing
#
# Images are converted to `(N, 3, 32, 32)` float tensors and standardised **per channel**
# using training-split statistics only.

# %%
def to_tensor(X):
    return torch.from_numpy(X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)


Xc_tr_t = to_tensor(Xc_train)
Xc_te_t = to_tensor(Xc_test)

CH_MEAN = Xc_tr_t.mean(dim=(0, 2, 3), keepdim=True)
CH_STD = Xc_tr_t.std(dim=(0, 2, 3), keepdim=True)
Xc_tr_t = (Xc_tr_t - CH_MEAN) / CH_STD
Xc_te_t = (Xc_te_t - CH_MEAN) / CH_STD

print("per-channel mean:", CH_MEAN.flatten().tolist())
print("per-channel std :", CH_STD.flatten().tolist())

n_val_c = 5000
perm_c = torch.randperm(len(Xc_tr_t), generator=torch.Generator().manual_seed(SEED))
val_i, tr_i = perm_c[:n_val_c], perm_c[n_val_c:]

y_tr_t = torch.from_numpy(yc_train)
cifar_train = TensorDataset(Xc_tr_t[tr_i], y_tr_t[tr_i])
cifar_val = TensorDataset(Xc_tr_t[val_i], y_tr_t[val_i])
cifar_test = TensorDataset(Xc_te_t, torch.from_numpy(yc_test))

train_c = DataLoader(cifar_train, batch_size=256, shuffle=True)
val_c = DataLoader(cifar_val, batch_size=512)
test_c = DataLoader(cifar_test, batch_size=512)
print(f"\ntrain {len(cifar_train):,} | val {len(cifar_val):,} | test {len(cifar_test):,}")

# %% [markdown]
# ### Augmentation, done on the GPU
#
# Random crop (with 4-pixel reflect padding) and random horizontal flip, applied to whole
# batches on the GPU. Doing this on-device avoids a CPU bottleneck that would otherwise
# dominate epoch time on this machine.

# %%
def augment_batch(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop + random horizontal flip, vectorised on the GPU."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = oy[:, None] + torch.arange(H, device=x.device)[None, :]
    cols = ox[:, None] + torch.arange(W, device=x.device)[None, :]
    out = xp[torch.arange(B, device=x.device)[:, None, None, None],
             torch.arange(C, device=x.device)[None, :, None, None],
             rows[:, None, :, None],
             cols[:, None, None, :]]
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


# Visual check that augmentation produces plausible images rather than garbage.
demo = Xc_tr_t[:8].to(DEVICE)
aug_demo = augment_batch(demo)
unnorm = lambda t: (t.cpu() * CH_STD + CH_MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()
fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
for i in range(8):
    axes[0, i].imshow(unnorm(demo)[i]); axes[0, i].set_axis_off()
    axes[1, i].imshow(unnorm(aug_demo)[i]); axes[1, i].set_axis_off()
axes[0, 0].set_title("original", loc="left", fontsize=9)
axes[1, 0].set_title("augmented", loc="left", fontsize=9)
save_fig("q9_augmentation", fig)

# %% [markdown]
# ### The network
#
# Three convolutional blocks with widening channels (32 → 64 → 128). Each block is
# `[Conv-BN-ReLU] × 2 → MaxPool → Dropout`, so spatial size falls 32 → 16 → 8 → 4 while
# channel depth rises — the standard pattern of trading resolution for semantic richness.

# %%
class CNN(nn.Module):
    def __init__(self, n_classes: int = 10, p_drop: float = 0.3):
        super().__init__()

        def block(cin, cout, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(p))

        self.features = nn.Sequential(
            block(3, 32, p_drop * 0.67),      # 32x32 -> 16x16
            block(32, 64, p_drop),            # 16x16 -> 8x8
            block(64, 128, p_drop),           # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(p_drop + 0.2), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epochs(model, train_dl, val_dl, epochs, lr=2e-3, augment=True,
               label="model", quiet=False):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl))
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            if augment:
                xb = augment_batch(xb)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(xb)
        vl, va, _ = eval_torch(model, val_dl)
        hist.append({"epoch": ep, "train_loss": loss_sum / tot, "train_acc": correct / tot,
                     "val_loss": vl, "val_acc": va, "lr": sched.get_last_lr()[0]})
        if not quiet and (ep % 4 == 0 or ep == 1):
            print(f"  [{label}] epoch {ep:2d}  train_acc={correct/tot:.4f}  "
                  f"val_acc={va:.4f}  val_loss={vl:.4f}")
    return model, pd.DataFrame(hist)


def eval_torch(model, dl):
    model.eval()
    loss_sum, correct, preds = 0.0, 0, []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            preds.append(p.cpu().numpy())
    n = len(dl.dataset)
    return loss_sum / n, correct / n, np.concatenate(preds)


torch.manual_seed(SEED)
cnn = CNN()
cnn_params = sum(p.numel() for p in cnn.parameters())
print(cnn)
print(f"\ntrainable parameters: {cnn_params:,}")

EPOCHS = 24
banner(f"TRAINING CNN ({EPOCHS} epochs on {DEVICE})")
(cnn, cnn_hist), cnn_time = timed(run_epochs, cnn, train_c, val_c, EPOCHS,
                                  augment=True, label="CNN")
print(f"\ntraining time: {cnn_time:.1f}s ({cnn_time/EPOCHS:.1f}s per epoch)")

# %% [markdown]
# ### Test evaluation

# %%
from sklearn.metrics import classification_report, confusion_matrix

cnn_loss, cnn_acc, cnn_preds = eval_torch(cnn, test_c)
banner("CNN TEST RESULTS")
print(f"test accuracy : {cnn_acc:.4f}")
print(f"test loss     : {cnn_loss:.4f}")
print(f"chance level  : 0.1000\n")
print(classification_report(yc_test, cnn_preds, target_names=CLASSES, digits=4))

# %%
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
axes[0].plot(cnn_hist["epoch"], cnn_hist["train_loss"], "o-", ms=3, label="train",
             color=PALETTE[0])
axes[0].plot(cnn_hist["epoch"], cnn_hist["val_loss"], "s-", ms=3, label="validation",
             color=PALETTE[1])
axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
axes[0].legend()

axes[1].plot(cnn_hist["epoch"], cnn_hist["train_acc"], "o-", ms=3, label="train",
             color=PALETTE[0])
axes[1].plot(cnn_hist["epoch"], cnn_hist["val_acc"], "s-", ms=3, label="validation",
             color=PALETTE[1])
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
axes[1].legend()

axes[2].plot(cnn_hist["epoch"], cnn_hist["lr"], color=PALETTE[4])
axes[2].set(xlabel="epoch", ylabel="learning rate",
            title="OneCycle schedule\n(warm-up then anneal)")
save_fig("q9_training_curves", fig)

# %% [markdown]
# ### Confusion matrix and per-class accuracy

# %%
cm_c = confusion_matrix(yc_test, cnn_preds)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
im = axes[0].imshow(cm_c, cmap="Blues")
axes[0].set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true",
            title=f"Confusion matrix (accuracy {cnn_acc:.4f})")
axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
axes[0].set_yticklabels(CLASSES, fontsize=8)
for i in range(10):
    for j in range(10):
        if cm_c[i, j] > 5:
            axes[0].text(j, i, cm_c[i, j], ha="center", va="center", fontsize=6.5,
                         color="white" if cm_c[i, j] > cm_c.max() * 0.5 else "#333")
axes[0].grid(False)
plt.colorbar(im, ax=axes[0], fraction=0.046)

recall_c = cm_c.diagonal() / cm_c.sum(1)
order_c = np.argsort(recall_c)
axes[1].barh(range(10), recall_c[order_c], color=PALETTE[0])
axes[1].set(yticks=range(10), xlabel="recall", title="Per-class recall")
axes[1].set_yticklabels([CLASSES[i] for i in order_c], fontsize=9)
for i, v in enumerate(recall_c[order_c]):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
save_fig("q9_confusion_matrix", fig)

off_c = cm_c.copy()
np.fill_diagonal(off_c, 0)
top_conf = sorted([(off_c[i, j], i, j) for i in range(10) for j in range(10)],
                  reverse=True)[:8]
print("Most frequent confusions (true -> predicted):")
for n, i, j in top_conf:
    print(f"   {CLASSES[i]:12s} -> {CLASSES[j]:12s} : {n}")

# %% [markdown]
# ### Predictions, right and wrong

# %%
probs_all = []
cnn.eval()
with torch.no_grad():
    for xb, _ in test_c:
        probs_all.append(F.softmax(cnn(xb.to(DEVICE)), dim=1).cpu().numpy())
probs_all = np.concatenate(probs_all)
conf_all = probs_all.max(1)

correct_mask = cnn_preds == yc_test
wrong_i = np.where(~correct_mask)[0]
right_i = np.where(correct_mask)[0]
pick = np.concatenate([rng_c.choice(right_i, 10, replace=False),
                       wrong_i[np.argsort(-conf_all[wrong_i])][:10]])

fig, axes = plt.subplots(2, 10, figsize=(16, 4.6))
for ax, idx in zip(axes.ravel(), pick):
    ax.imshow(Xc_test[idx])
    ok = cnn_preds[idx] == yc_test[idx]
    ax.set_title(f"{CLASSES[yc_test[idx]]}\n→{CLASSES[cnn_preds[idx]]} ({conf_all[idx]:.2f})",
                 fontsize=6.5, color="#2A4B9B" if ok else "#C44E52")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#2A4B9B" if ok else "#C44E52"); s.set_linewidth(1.6)
fig.suptitle("Top row: correct predictions.  Bottom row: the most confident mistakes.",
             y=1.03)
save_fig("q9_predictions", fig)

# %% [markdown]
# ### What the network sees
#
# The first-layer filters and the feature maps they produce. Early filters are colour and
# edge detectors; deeper maps become sparse and abstract, responding to parts rather than
# to raw intensity.

# %%
w0 = cnn.features[0][0].weight.detach().cpu().numpy()
w0n = (w0 - w0.min()) / (w0.max() - w0.min())

fig = plt.figure(figsize=(16, 6.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], wspace=0.15)

axf = fig.add_subplot(gs[0])
grid = np.vstack([np.hstack([w0n[r * 8 + c].transpose(1, 2, 0) for c in range(8)])
                  for r in range(4)])
axf.imshow(np.kron(grid, np.ones((12, 12, 1))))
axf.set(title="All 32 first-layer 3×3 filters (RGB)")
axf.set_axis_off()

sample_img = Xc_te_t[7:8].to(DEVICE)
acts = {}
h1 = cnn.features[0][:3].to(DEVICE)(sample_img)
h2 = cnn.features[1][:3].to(DEVICE)(cnn.features[0](sample_img))
axm = fig.add_subplot(gs[1])
axm.set_axis_off()
sub = fig.add_gridspec(3, 12, left=0.36, right=0.99, top=0.90, bottom=0.08,
                       wspace=0.05, hspace=0.15)
a0 = fig.add_subplot(sub[0, 0])
a0.imshow(unnorm(sample_img)[0]); a0.set_axis_off()
a0.set_title(f"input\n({CLASSES[yc_test[7]]})", fontsize=7)
for i in range(11):
    a = fig.add_subplot(sub[0, i + 1])
    a.imshow(h1[0, i].detach().cpu(), cmap="viridis"); a.set_axis_off()
for i in range(12):
    a = fig.add_subplot(sub[1, i])
    a.imshow(h1[0, i + 11].detach().cpu(), cmap="viridis"); a.set_axis_off()
for i in range(12):
    a = fig.add_subplot(sub[2, i])
    a.imshow(h2[0, i].detach().cpu(), cmap="magma"); a.set_axis_off()
fig.text(0.36, 0.93, "Rows 1–2: block-1 feature maps (32×32).   "
                     "Row 3: block-2 feature maps (16×16)", fontsize=9)
save_fig("q9_filters_featuremaps", fig)

# %% [markdown]
# ### The controlled comparison: does convolution actually help?
#
# This is the experiment that matters. An MLP is given a **larger** parameter budget than
# the CNN and trained under identical conditions on identical data. If the CNN still wins,
# the advantage comes from its structure, not from capacity.
#
# A CNN trained without augmentation is included to separate the two contributions.

# %%
class CifarMLP(nn.Module):
    """Fully connected baseline on flattened 3072-dim images."""

    def __init__(self, hidden=(1024, 512, 256), p_drop=0.3):
        super().__init__()
        layers, prev = [nn.Flatten()], 3072
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Matched to the main run's epoch count so that augmentation is the ONLY difference
# between the two CNN rows. A first version compared a 16-epoch no-augmentation run
# against the 24-epoch augmented one, which confounds augmentation with training length.
COMPARE_EPOCHS = EPOCHS

torch.manual_seed(SEED)
mlp_c = CifarMLP()
mlp_params = sum(p.numel() for p in mlp_c.parameters())
banner(f"BASELINE COMPARISON ({COMPARE_EPOCHS} epochs each)")
print(f"CNN parameters : {cnn_params:,}")
print(f"MLP parameters : {mlp_params:,}  ({mlp_params/cnn_params:.1f}x the CNN)\n")

mlp_c, mlp_hist = run_epochs(mlp_c, train_c, val_c, COMPARE_EPOCHS,
                             augment=True, label="MLP", quiet=True)
_, mlp_acc, mlp_preds = eval_torch(mlp_c, test_c)
print(f"  MLP (augmented)        test acc = {mlp_acc:.4f}")

torch.manual_seed(SEED)
cnn_noaug, noaug_hist = run_epochs(CNN(), train_c, val_c, COMPARE_EPOCHS,
                                   augment=False, label="CNN-noaug", quiet=True)
_, noaug_acc, _ = eval_torch(cnn_noaug, test_c)
print(f"  CNN (no augmentation)  test acc = {noaug_acc:.4f}")

# The augmented CNN at this epoch count is the main model already trained above --
# no need to retrain it.
short_hist, short_acc = cnn_hist, cnn_acc
print(f"  CNN (augmented)        test acc = {short_acc:.4f}   [the main model]")

comp_c = pd.DataFrame([
    {"model": "MLP (3072→1024→512→256)", "parameters": mlp_params,
     "augmented": "yes", "test acc": round(mlp_acc, 4),
     "final train acc": round(mlp_hist["train_acc"].iloc[-1], 4),
     "gap": round(mlp_hist["train_acc"].iloc[-1] - mlp_hist["val_acc"].iloc[-1], 4)},
    {"model": "CNN (no augmentation)", "parameters": cnn_params,
     "augmented": "no", "test acc": round(noaug_acc, 4),
     "final train acc": round(noaug_hist["train_acc"].iloc[-1], 4),
     "gap": round(noaug_hist["train_acc"].iloc[-1] - noaug_hist["val_acc"].iloc[-1], 4)},
    {"model": "CNN (augmented)", "parameters": cnn_params,
     "augmented": "yes", "test acc": round(short_acc, 4),
     "final train acc": round(short_hist["train_acc"].iloc[-1], 4),
     "gap": round(short_hist["train_acc"].iloc[-1] - short_hist["val_acc"].iloc[-1], 4)},
])
print()
print(comp_c.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
for name, h, col in [("MLP", mlp_hist, PALETTE[3]),
                     ("CNN no aug", noaug_hist, PALETTE[1]),
                     ("CNN + aug", short_hist, PALETTE[0])]:
    axes[0].plot(h["epoch"], h["val_acc"], "-o", ms=3, color=col, label=name)
axes[0].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
axes[0].legend(fontsize=8)

yy = np.arange(len(comp_c))
axes[1].barh(yy - 0.2, comp_c["final train acc"], 0.4, label="train acc", color="#c9d2de")
axes[1].barh(yy + 0.2, comp_c["test acc"], 0.4, label="test acc", color=PALETTE[0])
axes[1].set(yticks=yy, xlabel="accuracy", title="Train vs test — the overfitting gap")
axes[1].set_yticklabels(comp_c["model"], fontsize=8)
axes[1].legend(fontsize=8)
save_fig("q9_cnn_vs_mlp", fig)

# %% [markdown]
# ## Result and discussion
#
# * **The CNN classifies CIFAR-10 far above chance**, reaching **86.87 %** test accuracy
#   against a 10 % chance level, on a dataset with cluttered natural backgrounds and large
#   within-class variation.
#
# * **Convolution beat capacity — the central result.** The MLP was given **4.7× more
#   parameters** (3,805,450 vs 814,570) and trained on identical data with identical
#   augmentation, schedule and epoch count, yet reached only **54.69 %** against the CNN's
#   **86.87 %** — a gap of **32 points in the CNN's favour, while using far fewer weights**.
#   The advantage cannot be explained by model size, because the comparison deliberately
#   handed the size advantage to the opponent. What the CNN has instead is the right
#   *inductive bias*: locality, parameter sharing and translation equivariance. This is the
#   quantitative answer to the limitation flagged at the end of Experiment 6.
#
# * **Augmentation removed overfitting but did not improve accuracy here.** At a matched 24
#   epochs the two CNNs are indistinguishable on test accuracy — 0.8687 augmented versus
#   0.8684 not — a difference of 0.03 points, which is noise. What augmentation *did* change
#   is unambiguous: the train–validation gap went from **+0.0447 to −0.0201**, with training
#   accuracy falling from 0.9201 to 0.8537. So it worked exactly as a regulariser should
#   (the training set became harder to memorise) without that translating into better
#   generalisation at this budget.
#
#   The honest reading is that 24 epochs is too short for the benefit to materialise: the
#   un-augmented model is overfitting but has not yet been *hurt* by it, while the augmented
#   model is still underfit and had not finished learning when training stopped. Claiming
#   augmentation "improved accuracy" on this evidence would be wrong, and the earlier
#   16-epoch version of this comparison was worse still — it confounded augmentation with
#   training length.
#
# * **The errors are semantically sensible.** The largest confusions are cat→dog (134) and
#   dog→cat (93), followed by bird→frog (52), cat→frog (49) and bird→deer (49). These are
#   pairs sharing silhouette, texture and typical background. Vehicles, rigid and visually
#   distinctive, are recognised far more reliably — automobile and ship exceed 0.93 recall
#   while cat sits lowest. A model confusing cats with dogs is failing in a much more human
#   way than one confusing cats with trucks.
#
# * **The learned filters look like what the theory predicts.** The 32 first-layer filters
#   are colour-opponent blobs and oriented edge detectors — the same primitives found in
#   classical computer vision and in biological early vision — discovered here purely by
#   gradient descent on a classification objective.
#
# * **MNIST is not a hard problem; CIFAR-10 is.** An MLP managed 98.18 % on MNIST in
#   Experiment 6; the *same kind of model* manages 54.69 % here, and a purpose-built CNN
#   reaches 86.87 %. Useful calibration: a headline accuracy is meaningless without the
#   dataset attached to it.
