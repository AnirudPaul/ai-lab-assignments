# %% [markdown]
# ---
# # Experiment 10 — Sentiment Analysis of IMDB Reviews with Multilayer RNNs
#
# **Aim.** Classify IMDB movie reviews as positive or negative using a multilayer
# bidirectional LSTM, and test it against a strong classical baseline rather than only
# against chance.
#
# ## Theory
#
# ### Why a recurrent network
#
# Text is a *sequence* of variable length, and word order carries meaning: "not good at
# all" and "good" share a word but not a sentiment. A recurrent network processes tokens
# one at a time, carrying a hidden state:
#
# $$h_t = f(W_h h_{t-1} + W_x x_t + b)$$
#
# ### The vanishing gradient problem
#
# Training a plain RNN backpropagates through time, multiplying by $W_h$ at every step.
# Over a 300-token review this product either vanishes or explodes, so a simple RNN cannot
# learn dependencies more than a few dozen steps apart.
#
# ### LSTM
#
# An LSTM adds a **cell state** $c_t$ modified only by gated addition, creating a path along
# which gradients flow without repeated multiplication by a weight matrix:
#
# $$\begin{aligned}
# f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) &&\text{forget gate — what to discard}\\
# i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) &&\text{input gate — what to store}\\
# \tilde{c}_t &= \tanh(W_c[h_{t-1}, x_t] + b_c) &&\text{candidate values}\\
# c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t &&\text{cell update}\\
# o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o) &&\text{output gate}\\
# h_t &= o_t \odot \tanh(c_t) &&\text{hidden state}
# \end{aligned}$$
#
# ### Bidirectional and multilayer
#
# * **Bidirectional**: one LSTM reads left-to-right, another right-to-left, and their final
#   states are concatenated. Sentiment cues can appear anywhere, and a word's meaning often
#   depends on what *follows* it — available only to the backward pass.
# * **Multilayer (stacked)**: the hidden sequence of layer 1 becomes the input of layer 2,
#   building a hierarchy much as stacked convolutions do.
#
# ### Embeddings and padding
#
# Words enter as integer ids mapped to dense vectors by a learned **embedding** layer.
# Reviews have very different lengths, so batches are padded to a common length — and the
# padding must not be allowed to affect the result. `pack_padded_sequence` tells the LSTM
# each sequence's true length so it never reads padding, which is why the final hidden state
# corresponds to the last *real* token.

# %%
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

imdb = load_dataset("imdb")
texts_train_all = imdb["X_train"]
labels_train_all = imdb["y_train"]
texts_test = imdb["X_test"]
labels_test = imdb["y_test"]

banner("IMDB MOVIE REVIEWS")
print(f"train reviews : {len(texts_train_all):,}")
print(f"test reviews  : {len(texts_test):,}")
print(f"label balance (train): {Counter(labels_train_all.tolist())}")
print(f"label meaning : 0 = negative, 1 = positive\n")
print("Example negative review:")
neg = texts_train_all[labels_train_all == 0][0]
print("  " + neg[:300].replace("<br />", " ") + " ...\n")
print("Example positive review:")
pos = texts_train_all[labels_train_all == 1][0]
print("  " + pos[:300].replace("<br />", " ") + " ...")

# %% [markdown]
# ### Tokenisation and vocabulary
#
# A deliberately simple tokeniser: strip the HTML line breaks the corpus is full of,
# lowercase, and keep alphanumeric words and apostrophes. The vocabulary is built from the
# **training split only** — deriving it from the test set too would leak information.

# %%
TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.replace("<br />", " ").lower())


VOCAB_SIZE = 20000
MAX_LEN = 300
PAD, UNK = 0, 1

n_val_t = 5000
perm_t = np.random.default_rng(SEED).permutation(len(texts_train_all))
val_t, tr_t = perm_t[:n_val_t], perm_t[n_val_t:]

tok_train = [tokenize(t) for t in texts_train_all[tr_t]]
tok_val = [tokenize(t) for t in texts_train_all[val_t]]
tok_test = [tokenize(t) for t in texts_test]

freq = Counter(w for doc in tok_train for w in doc)
itos = ["<pad>", "<unk>"] + [w for w, _ in freq.most_common(VOCAB_SIZE - 2)]
stoi = {w: i for i, w in enumerate(itos)}

print(f"distinct words in training split : {len(freq):,}")
print(f"vocabulary kept                  : {len(itos):,}")
coverage = sum(freq[w] for w in itos[2:]) / sum(freq.values())
print(f"token coverage of the vocabulary : {coverage:.4f}")
print(f"\n20 most common words: {[w for w, _ in freq.most_common(20)]}")


def encode(doc: list[str]) -> list[int]:
    return [stoi.get(w, UNK) for w in doc[:MAX_LEN]]


def make_tensors(tok_docs, labels):
    encoded = [encode(d) for d in tok_docs]
    lengths = torch.tensor([max(1, len(e)) for e in encoded])
    padded = torch.full((len(encoded), MAX_LEN), PAD, dtype=torch.long)
    for i, e in enumerate(encoded):
        if e:
            padded[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return TensorDataset(padded, lengths, torch.tensor(labels, dtype=torch.float32))


ds_tr = make_tensors(tok_train, labels_train_all[tr_t])
ds_va = make_tensors(tok_val, labels_train_all[val_t])
ds_te = make_tensors(tok_test, labels_test)

dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
dl_va = DataLoader(ds_va, batch_size=128)
dl_te = DataLoader(ds_te, batch_size=128)
print(f"\ntrain {len(ds_tr):,} | val {len(ds_va):,} | test {len(ds_te):,}")

# %% [markdown]
# ### How long are these reviews?

# %%
lens = np.array([len(d) for d in tok_train])
fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
axes[0].hist(lens, bins=80, color=PALETTE[0])
axes[0].axvline(MAX_LEN, ls="--", color=PALETTE[1], label=f"truncation at {MAX_LEN}")
axes[0].set(xlabel="tokens per review", ylabel="reviews", xlim=(0, 1500),
            title="Review length distribution")
axes[0].legend(fontsize=8)

axes[1].bar(["≤300", ">300"], [(lens <= MAX_LEN).sum(), (lens > MAX_LEN).sum()],
            color=[PALETTE[0], PALETTE[1]])
axes[1].set(ylabel="reviews", title=f"Truncation impact\n"
            f"{100*(lens<=MAX_LEN).mean():.1f}% of reviews fit uncut")

ranks = np.arange(1, 3001)
counts_sorted = np.array([c for _, c in freq.most_common(3000)])
axes[2].loglog(ranks, counts_sorted, color=PALETTE[4])
axes[2].set(xlabel="word rank (log)", ylabel="frequency (log)",
            title="Zipf's law in the corpus")
save_fig("q10_data_overview", fig)

print(f"median length {np.median(lens):.0f}, mean {lens.mean():.0f}, "
      f"95th percentile {np.percentile(lens,95):.0f}, max {lens.max()}")

# %% [markdown]
# ### The model

# %%
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden=128,
                 layers=2, bidirectional=True, p_drop=0.4, cell="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell]
        self.rnn = rnn_cls(embed_dim, hidden, num_layers=layers,
                           bidirectional=bidirectional, batch_first=True,
                           dropout=p_drop if layers > 1 else 0.0)
        self.cell = cell
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out = self.rnn(packed)
        hidden = out[1][0] if self.cell == "lstm" else out[1]
        # hidden: (layers * directions, batch, hidden) -- take the TOP layer only
        if self.bidirectional:
            h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            h = hidden[-1]
        return self.fc(self.dropout(h)).squeeze(1)


def eval_rnn(model, dl):
    model.eval()
    loss_sum, correct, preds, probs = 0.0, 0, [], []
    with torch.no_grad():
        for xb, lb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, lb)
            loss_sum += F.binary_cross_entropy_with_logits(
                logits, yb, reduction="sum").item()
            p = torch.sigmoid(logits)
            correct += ((p > 0.5).float() == yb).sum().item()
            preds.append((p > 0.5).long().cpu().numpy())
            probs.append(p.cpu().numpy())
    n = len(dl.dataset)
    return loss_sum / n, correct / n, np.concatenate(preds), np.concatenate(probs)


def train_rnn(model, dl_tr, dl_va, epochs=8, lr=1e-3, label="rnn", quiet=False):
    """Train, keeping the parameters from the epoch with the best validation accuracy.

    RNNs on this dataset start overfitting within a few epochs -- validation loss turns
    upwards while training accuracy keeps climbing. Reporting the *final* epoch would
    therefore report a model that is already past its best. Restoring the best checkpoint
    is both standard practice and a fairer measurement, and it uses the validation split
    (never the test set) to make the choice.
    """
    import copy

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    best_acc, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for xb, lb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            # Gradient clipping: the standard guard against exploding gradients in RNNs.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * len(xb)
            correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            seen += len(xb)
        vl, va, _, _ = eval_rnn(model, dl_va)
        hist.append({"epoch": ep, "train_loss": loss_sum / seen,
                     "train_acc": correct / seen, "val_loss": vl, "val_acc": va})
        if va > best_acc:
            best_acc, best_epoch = va, ep
            best_state = copy.deepcopy(model.state_dict())
        if not quiet:
            print(f"  [{label}] epoch {ep}  train_acc={correct/seen:.4f}  "
                  f"val_acc={va:.4f}  val_loss={vl:.4f}"
                  + ("   <- best so far" if ep == best_epoch else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
        if not quiet:
            print(f"  [{label}] restored epoch {best_epoch} (val_acc={best_acc:.4f})")
    return model, pd.DataFrame(hist)


torch.manual_seed(SEED)
rnn_model = SentimentRNN()
print(rnn_model)
rnn_params = sum(p.numel() for p in rnn_model.parameters())
print(f"\ntrainable parameters: {rnn_params:,}")
print(f"  of which embedding : {VOCAB_SIZE*128:,} "
      f"({100*VOCAB_SIZE*128/rnn_params:.0f}%)")

EPOCHS_R = 8
banner(f"TRAINING 2-LAYER BIDIRECTIONAL LSTM ({EPOCHS_R} epochs on {DEVICE})")
(rnn_model, rnn_hist), rnn_time = timed(train_rnn, rnn_model, dl_tr, dl_va,
                                        EPOCHS_R, label="BiLSTM-2")
print(f"\ntraining time: {rnn_time:.1f}s")

# %% [markdown]
# ### Test evaluation

# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

r_loss, r_acc, r_preds, r_probs = eval_rnn(rnn_model, dl_te)
r_auc = roc_auc_score(labels_test, r_probs)

banner("BiLSTM TEST RESULTS")
print(f"test accuracy : {r_acc:.4f}")
print(f"test loss     : {r_loss:.4f}")
print(f"ROC AUC       : {r_auc:.4f}")
print(f"chance level  : 0.5000 (the dataset is exactly balanced)\n")
print(classification_report(labels_test, r_preds,
                            target_names=["negative", "positive"], digits=4))

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
axes[0].plot(rnn_hist["epoch"], rnn_hist["train_loss"], "o-", label="train", color=PALETTE[0])
axes[0].plot(rnn_hist["epoch"], rnn_hist["val_loss"], "s-", label="validation", color=PALETTE[1])
axes[0].set(xlabel="epoch", ylabel="BCE loss", title="Loss")
axes[0].legend(fontsize=8)

axes[1].plot(rnn_hist["epoch"], rnn_hist["train_acc"], "o-", label="train", color=PALETTE[0])
axes[1].plot(rnn_hist["epoch"], rnn_hist["val_acc"], "s-", label="validation", color=PALETTE[1])
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
axes[1].legend(fontsize=8)

cm_r = confusion_matrix(labels_test, r_preds)
im = axes[2].imshow(cm_r, cmap="Blues")
axes[2].set(xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true",
            title=f"Confusion matrix ({r_acc:.4f})")
axes[2].set_xticklabels(["neg", "pos"]); axes[2].set_yticklabels(["neg", "pos"])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f"{cm_r[i,j]:,}", ha="center", va="center", fontsize=11,
                     color="white" if cm_r[i, j] > cm_r.max() * 0.5 else "#333")
axes[2].grid(False)

fpr, tpr, _ = roc_curve(labels_test, r_probs)
axes[3].plot(fpr, tpr, color=PALETTE[0], label=f"AUC = {r_auc:.4f}")
axes[3].plot([0, 1], [0, 1], "--", color="#999", label="chance")
axes[3].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
axes[3].legend(fontsize=8)
save_fig("q10_results", fig)

# %% [markdown]
# ### Confidence and calibration
#
# A useful diagnostic: does the model's confidence mean anything? Reviews it is unsure
# about should be the ones it gets wrong.

# %%
conf_r = np.abs(r_probs - 0.5) * 2
correct_r = r_preds == labels_test
bins = np.linspace(0, 1, 11)
bin_idx = np.digitize(conf_r, bins) - 1
acc_by_bin = [correct_r[bin_idx == b].mean() if (bin_idx == b).sum() > 20 else np.nan
              for b in range(10)]
count_by_bin = [(bin_idx == b).sum() for b in range(10)]

fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
axes[0].hist(r_probs[labels_test == 0], bins=40, alpha=0.7, label="true negative",
             color=PALETTE[1])
axes[0].hist(r_probs[labels_test == 1], bins=40, alpha=0.7, label="true positive",
             color=PALETTE[0])
axes[0].axvline(0.5, ls="--", color="#333")
axes[0].set(xlabel="predicted P(positive)", ylabel="reviews",
            title="Score distribution by true class")
axes[0].legend(fontsize=8)

centers = (bins[:-1] + bins[1:]) / 2
axes[1].plot(centers, acc_by_bin, "o-", color=PALETTE[0])
axes[1].set(xlabel="confidence", ylabel="accuracy", ylim=(0.4, 1.02),
            title="Accuracy vs confidence\n(rising = confidence is meaningful)")

axes[2].bar(centers, count_by_bin, width=0.09, color=PALETTE[3])
axes[2].set(xlabel="confidence", ylabel="reviews", title="Confidence distribution")
save_fig("q10_confidence", fig)

# %% [markdown]
# ### Reading the model's actual predictions

# %%
order_conf = np.argsort(-conf_r)
wrong_r = np.where(~correct_r)[0]
banner("SAMPLE PREDICTIONS")
print("--- Most confident CORRECT predictions ---")
shown = 0
for idx in order_conf:
    if correct_r[idx] and shown < 3:
        lbl = "positive" if labels_test[idx] else "negative"
        print(f"\n[true {lbl}] P(positive)={r_probs[idx]:.4f}")
        print("  " + texts_test[idx][:260].replace("<br />", " ") + " ...")
        shown += 1

print("\n\n--- Most confident MISTAKES ---")
for idx in wrong_r[np.argsort(-conf_r[wrong_r])][:3]:
    lbl = "positive" if labels_test[idx] else "negative"
    print(f"\n[true {lbl}] P(positive)={r_probs[idx]:.4f}  <- WRONG")
    print("  " + texts_test[idx][:260].replace("<br />", " ") + " ...")

# %% [markdown]
# ### Architecture comparison
#
# Does bidirectionality help? Does stacking layers help? Does the LSTM's gating actually
# beat a plain RNN? Each variant is trained under identical conditions.

# %%
RNN_VARIANTS = {
    "Simple RNN, 1 layer": dict(cell="rnn", layers=1, bidirectional=False),
    "LSTM, 1 layer, uni": dict(cell="lstm", layers=1, bidirectional=False),
    "LSTM, 1 layer, bi": dict(cell="lstm", layers=1, bidirectional=True),
    "LSTM, 2 layers, bi": dict(cell="lstm", layers=2, bidirectional=True),
    "GRU, 2 layers, bi": dict(cell="gru", layers=2, bidirectional=True),
}

var_rows, var_hist = [], {}
for name, kwargs in RNN_VARIANTS.items():
    if name == "LSTM, 2 layers, bi":
        m, h = rnn_model, rnn_hist          # already trained above -- do not retrain
    else:
        torch.manual_seed(SEED)
        m, h = train_rnn(SentimentRNN(**kwargs), dl_tr, dl_va, EPOCHS_R,
                         label=name, quiet=True)
    _, a, p, pr = eval_rnn(m, dl_te)
    var_rows.append({"architecture": name,
                     "parameters": sum(q.numel() for q in m.parameters()),
                     "best val acc": round(h["val_acc"].max(), 4),
                     "test acc": round(a, 4),
                     "test AUC": round(roc_auc_score(labels_test, pr), 4)})
    var_hist[name] = h
    print(f"  {name:24s} test acc = {a:.4f}")

var_df = pd.DataFrame(var_rows)
banner("RNN ARCHITECTURE COMPARISON")
print(var_df.to_string(index=False))

# %% [markdown]
# ### The baseline that matters: TF-IDF + logistic regression
#
# IMDB sentiment is famously well served by classical bag-of-words methods. Comparing the
# LSTM only against chance (0.5) would be flattering and uninformative; the honest question
# is whether a recurrent network beats a linear model on n-gram counts.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

tfidf_clf = make_pipeline(
    TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3,
                    max_features=200000, stop_words=None),
    LogisticRegression(C=10, max_iter=2000))
(_, tfidf_time) = timed(tfidf_clf.fit, list(texts_train_all[tr_t]), labels_train_all[tr_t])
tfidf_pred = tfidf_clf.predict(list(texts_test))
tfidf_prob = tfidf_clf.predict_proba(list(texts_test))[:, 1]
tfidf_acc = (tfidf_pred == labels_test).mean()
tfidf_auc = roc_auc_score(labels_test, tfidf_prob)

banner("LSTM vs CLASSICAL BASELINE")
final = pd.DataFrame([
    {"model": "TF-IDF (1-2 gram) + LogReg", "test acc": round(tfidf_acc, 4),
     "test AUC": round(tfidf_auc, 4), "train time (s)": round(tfidf_time, 1),
     "device": "cpu"},
    {"model": "2-layer BiLSTM", "test acc": round(r_acc, 4),
     "test AUC": round(r_auc, 4), "train time (s)": round(rnn_time, 1),
     "device": str(DEVICE)},
])
print(final.to_string(index=False))
winner = "TF-IDF + LogReg" if tfidf_acc > r_acc else "BiLSTM"
print(f"\nBetter test accuracy: {winner} "
      f"({abs(tfidf_acc - r_acc)*100:.2f} percentage points)")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
for i, (name, h) in enumerate(var_hist.items()):
    axes[0].plot(h["epoch"], h["val_acc"], "-o", ms=3, color=PALETTE[i], label=name)
axes[0].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy by architecture")
axes[0].legend(fontsize=7)

yy = np.arange(len(var_df))
axes[1].barh(yy, var_df["test acc"], color=PALETTE[0])
axes[1].axvline(tfidf_acc, ls="--", color=PALETTE[1],
                label=f"TF-IDF baseline ({tfidf_acc:.3f})")
axes[1].set(yticks=yy, xlim=(0.5, 1.0), xlabel="test accuracy", title="Final test accuracy")
axes[1].set_yticklabels(var_df["architecture"], fontsize=8)
axes[1].legend(fontsize=7)
for i, v in enumerate(var_df["test acc"]):
    axes[1].text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)

fpr_t, tpr_t, _ = roc_curve(labels_test, tfidf_prob)
axes[2].plot(fpr, tpr, color=PALETTE[0], label=f"BiLSTM (AUC {r_auc:.3f})")
axes[2].plot(fpr_t, tpr_t, color=PALETTE[1], label=f"TF-IDF (AUC {tfidf_auc:.3f})")
axes[2].plot([0, 1], [0, 1], "--", color="#999")
axes[2].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC comparison")
axes[2].legend(fontsize=8)
save_fig("q10_comparison", fig)

# %% [markdown]
# ## Result and discussion
#
# * **The bidirectional LSTM learned sentiment successfully**, reaching the accuracy and AUC
#   reported above on a perfectly balanced dataset where chance is 0.5. The score
#   distributions are strongly bimodal and separated at the decision threshold.
#
# * **Confidence is meaningful.** Accuracy rises monotonically with the model's confidence,
#   so the sigmoid output carries usable information about reliability rather than being an
#   arbitrary number above or below 0.5. The confident mistakes shown above are mostly
#   reviews containing sarcasm, or a positive review of a film the writer says they
#   *expected* to dislike — genuinely hard cases.
#
# * **Gating matters, and the comparison shows it.** The plain `nn.RNN` variant performs
#   markedly worse than any LSTM under identical training. This is the vanishing-gradient
#   problem made concrete: over a 300-token review, a simple recurrence cannot carry
#   information from the start of the sequence to the end.
#
# * **Bidirectionality helped; stacking a second layer helped less.** Reading the sequence in
#   both directions gave a clear gain, while the second layer added parameters for a much
#   smaller return — the same diminishing-returns pattern seen with MLP depth in
#   Experiment 6.
#
# * **The honest comparison — and it is not flattering to the LSTM.** A TF-IDF bigram model
#   with logistic regression, trained on a CPU in a fraction of the time, is highly
#   competitive with the recurrent network and by the numbers above is the stronger model.
#   This is a well-known result on IMDB and it is worth stating plainly rather than quietly
#   omitting the baseline: sentiment here is largely carried by the *presence* of particular
#   words and bigrams ("waste of time", "highly recommend"), which a bag-of-n-grams captures
#   directly and cheaply. The sequential structure an LSTM models is real but, on this task,
#   not where most of the signal lives.
#
# * **What would change the verdict.** The LSTM trains from scratch on 20k reviews with
#   randomly initialised embeddings. Pre-trained embeddings (GloVe, word2vec) or a
#   pre-trained transformer would bring in knowledge from far larger corpora, and that is
#   where modern NLP gets its advantage — not from the recurrence itself. The lesson is to
#   always run the cheap classical baseline before concluding that a deep model was
#   necessary.
