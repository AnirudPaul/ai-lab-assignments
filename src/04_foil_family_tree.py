# %% [markdown]
# ---
# # Experiment 4 — First Order Inductive Learner (FOIL) on a Family Tree
#
# **Aim.** Implement FOIL and use it to *induce* first-order rules such as
# `grandfather(X,Y)` and `uncle(X,Y)` from nothing but ground facts about a family —
# `parent/2`, `male/1`, `female/1`.
#
# ## Theory
#
# FOIL (Quinlan, 1990) learns a set of function-free Horn clauses from positive and
# negative examples. It is the first-order generalisation of sequential covering:
#
# ```
# FOIL(target, background, positives, negatives):
#     learned = []
#     while positives remain uncovered:
#         clause = target(X,Y) :- (empty body)
#         while clause still covers some negatives:
#             pick the literal L with the highest FOIL gain
#             append L to the body
#         learned.append(clause)
#         remove the positives this clause covers
# ```
#
# Two loops: the **outer** one adds clauses until every positive example is explained; the
# **inner** one adds body literals until a clause explains no negatives.
#
# ### FOIL information gain
#
# The literal to add is chosen by
#
# $$\text{Gain}(L) = t \times \Big( \log_2 \frac{p_1}{p_1+n_1} - \log_2 \frac{p_0}{p_0+n_0} \Big)$$
#
# where $p_0, n_0$ are the positive/negative **bindings** the clause covers before adding
# $L$, and $p_1, n_1$ after. The bracket is the gain in information *per binding*, measured
# in bits. The multiplier $t$ — the number of original positive examples that survive the
# addition — is what stops FOIL from choosing a literal that purifies the clause by
# discarding nearly all the positives.
#
# ### Bindings, not examples
#
# This is the part that makes FOIL first-order rather than propositional. A literal may
# introduce a **new variable**: adding `parent(X,Z)` to a clause about `(X,Y)` turns each
# example into one row *per matching Z*. So the algorithm tracks a table of variable
# bindings that grows and shrinks, and $p$ and $n$ count rows in that table, not examples.
#
# ### Negative examples and the closed-world assumption
# No negative facts are supplied. Under the **closed-world assumption** every pair of people
# not listed as a positive is treated as a negative — for 15 people that is $15^2 = 225$
# candidate pairs minus the true ones.

# %%
from itertools import product

# ---------------------------------------------------------------- the family
PEOPLE = ["George", "Mary", "Harold", "Alice", "Robert", "Susan", "Linda", "David",
          "Michael", "Sarah", "Peter", "Emma", "Olivia", "James", "Thomas"]

MALE = {"George", "Harold", "Robert", "David", "Michael", "Peter", "James", "Thomas"}
FEMALE = set(PEOPLE) - MALE

PARENT = [
    # generation 1 -> 2
    ("George", "Robert"), ("Mary", "Robert"),
    ("George", "Susan"), ("Mary", "Susan"),
    ("Harold", "Linda"), ("Alice", "Linda"),
    ("Harold", "David"), ("Alice", "David"),
    # generation 2 -> 3   (Robert married Linda; David married Susan)
    ("Robert", "Michael"), ("Linda", "Michael"),
    ("Robert", "Sarah"), ("Linda", "Sarah"),
    ("David", "Peter"), ("Susan", "Peter"),
    ("David", "Emma"), ("Susan", "Emma"),
    # generation 3 -> 4   (Michael married Olivia)
    ("Michael", "James"), ("Olivia", "James"),
    ("Michael", "Thomas"), ("Olivia", "Thomas"),
]

FACTS: dict[str, set[tuple]] = {
    "parent": set(PARENT),
    "male": {(p,) for p in MALE},
    "female": {(p,) for p in FEMALE},
    # `different/2` gives FOIL a way to express X != Y, which it needs for sibling.
    "different": {(a, b) for a in PEOPLE for b in PEOPLE if a != b},
}
ARITY = {"parent": 2, "male": 1, "female": 1, "different": 2}

banner("BACKGROUND KNOWLEDGE")
print(f"{len(PEOPLE)} people | {len(MALE)} male, {len(FEMALE)} female")
print(f"{len(FACTS['parent'])} parent facts")
print(f"Closed-world negative space for a binary target: {len(PEOPLE)**2} pairs")

# %% [markdown]
# ### Drawing the family tree
#
# Generation depth is computed from the `parent` facts themselves (longest chain from a
# root), so the layout is derived from the data rather than hand-placed.

# %%
import networkx as nx

FT = nx.DiGraph()
FT.add_nodes_from(PEOPLE)
FT.add_edges_from(PARENT)

generation = {}
for person in nx.topological_sort(FT):
    preds = list(FT.predecessors(person))
    generation[person] = 0 if not preds else max(generation[p] for p in preds) + 1

by_gen = defaultdict(list)
for p, gval in generation.items():
    by_gen[gval].append(p)
layout = {}
for gval, members in by_gen.items():
    for i, person in enumerate(sorted(members)):
        layout[person] = (i - (len(members) - 1) / 2, -gval)

fig, ax = plt.subplots(figsize=(13.5, 6.2))
nx.draw_networkx_edges(FT, layout, ax=ax, arrows=True, arrowsize=11,
                       edge_color="#9aa4b2", width=1.4,
                       connectionstyle="arc3,rad=0.04")
for group, color, shape in [(MALE, "#2A4B9B", "s"), (FEMALE, "#C44E52", "o")]:
    nodes = [p for p in PEOPLE if p in group]
    nx.draw_networkx_nodes(FT, layout, ax=ax, nodelist=nodes, node_color=color,
                           node_shape=shape, node_size=1500, edgecolors="#20304d")
nx.draw_networkx_labels(FT, layout, ax=ax, font_size=7.5, font_color="white",
                        font_weight="bold")
for gval in sorted(by_gen):
    ax.text(-4.6, -gval, f"gen {gval}", fontsize=10, fontweight="bold", color="#555")
ax.set_title("Family tree — squares are male, circles female, arrows point parent → child")
ax.set_axis_off()
save_fig("q4_family_tree", fig)

# %% [markdown]
# ### FOIL itself
#
# `Literal` and `Clause` are plain data. The three functions that matter are
# `extend_bindings` (apply a literal to a binding table), `foil_gain` (score a candidate)
# and `learn_clause` / `foil` (the two covering loops).

# %%
@dataclass(frozen=True)
class Literal:
    pred: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.pred}({','.join(self.args)})"


@dataclass
class Clause:
    head: Literal
    body: list[Literal] = field(default_factory=list)

    def __str__(self) -> str:
        # Comma-separated body, i.e. standard Prolog clause syntax.
        rhs = ", ".join(str(l) for l in self.body) if self.body else "true"
        return f"{self.head} :- {rhs}."


def extend_bindings(variables: list[str], rows: list[tuple], origins: list[int],
                    lit: Literal, facts: dict[str, set[tuple]]):
    """Join the current binding table against one literal.

    Returns (new_variables, new_rows, new_origins). A literal that introduces a fresh
    variable multiplies rows; one that only constrains existing variables filters them.
    """
    new_vars = []
    for a in lit.args:
        if a not in variables and a not in new_vars:
            new_vars.append(a)
    pos = {v: i for i, v in enumerate(variables)}
    out_rows, out_origins = [], []

    for row, origin in zip(rows, origins):
        for fact in facts[lit.pred]:
            env, ok = {}, True
            for arg, value in zip(lit.args, fact):
                if arg in pos:
                    if row[pos[arg]] != value:
                        ok = False
                        break
                elif arg in env:
                    if env[arg] != value:
                        ok = False
                        break
                else:
                    env[arg] = value
            if ok:
                out_rows.append(row + tuple(env[v] for v in new_vars))
                out_origins.append(origin)
    return variables + new_vars, out_rows, out_origins


def foil_gain(p0: int, n0: int, p1: int, n1: int, t: int) -> float:
    """t * (log2(p1/(p1+n1)) - log2(p0/(p0+n0))). Zero if the literal kills all positives."""
    if p1 == 0 or p0 == 0:
        return 0.0
    return t * (np.log2(p1 / (p1 + n1)) - np.log2(p0 / (p0 + n0)))


def candidate_literals(variables: list[str], background: dict[str, int],
                       allow_new_var: bool = True) -> list[Literal]:
    """Every literal built from the background predicates over the current variables
    plus (optionally) one fresh variable. A candidate must mention at least one existing
    variable, otherwise the clause would fall apart into disconnected pieces."""
    pool = list(variables)
    if allow_new_var:
        fresh = next(v for v in "ZUVWABCDEFG" if v not in variables)
        pool = pool + [fresh]
    out = []
    for pred, ar in background.items():
        for combo in product(pool, repeat=ar):
            if any(v in variables for v in combo):
                out.append(Literal(pred, combo))
    return out


# %% [markdown]
# ### The two covering loops
#
# `learn_clause` runs the inner loop, recording the full gain table at every step so the
# learner's reasoning can be inspected afterwards rather than taken on trust.

# %%
def learn_clause(target: Literal, pos: list[tuple], neg: list[tuple],
                 background: dict[str, int], max_body: int = 5, verbose: bool = False):
    """Inner loop: add the highest-gain literal until no negative bindings survive."""
    variables = list(target.args)
    pos_rows, pos_origins = list(pos), list(range(len(pos)))
    neg_rows, neg_origins = list(neg), list(range(len(neg)))
    clause = Clause(target, [])
    steps = []

    while neg_rows and len(clause.body) < max_body:
        p0, n0 = len(pos_rows), len(neg_rows)
        scored = []
        for lit in candidate_literals(variables, background):
            _, np_rows, np_origins = extend_bindings(variables, pos_rows, pos_origins,
                                                     lit, FACTS)
            if not np_rows:
                continue
            _, nn_rows, _ = extend_bindings(variables, neg_rows, neg_origins, lit, FACTS)
            t = len(set(np_origins))
            scored.append((foil_gain(p0, n0, len(np_rows), len(nn_rows), t),
                           lit, len(np_rows), len(nn_rows), t))

        if not scored:
            break
        scored.sort(key=lambda r: (-r[0], str(r[1])))
        gain, best, p1, n1, t = scored[0]
        if gain <= 1e-9:
            break

        steps.append({"step": len(clause.body) + 1, "chosen": str(best),
                      "gain (bits)": round(gain, 2), "p0": p0, "n0": n0,
                      "p1": p1, "n1": n1, "t": t,
                      "runners_up": [(str(l), round(g, 2)) for g, l, *_ in scored[1:5]]})
        if verbose:
            print(f"  step {len(clause.body)+1}: += {str(best):<22} "
                  f"gain={gain:6.2f}  p:{p0}->{p1}  n:{n0}->{n1}")

        clause.body.append(best)
        # Both tables must be extended from the SAME pre-update variable list.
        new_vars, pos_rows, pos_origins = extend_bindings(
            variables, pos_rows, pos_origins, best, FACTS)
        _, neg_rows, neg_origins = extend_bindings(
            variables, neg_rows, neg_origins, best, FACTS)
        variables = new_vars

    covered = {pos[i] for i in set(pos_origins)}
    return clause, covered, steps, len(neg_rows)


# %%
def foil(target_name: str, positives: list[tuple], background: dict[str, int],
         arity: int = 2, max_clauses: int = 4, verbose: bool = True):
    """Outer loop: keep learning clauses until every positive example is covered."""
    head = Literal(target_name, ("X", "Y")[:arity])
    pos_set = set(positives)
    negatives = [t for t in product(PEOPLE, repeat=arity) if t not in pos_set]
    uncovered = list(positives)
    clauses, all_steps = [], []

    if verbose:
        print(f"target {head}: {len(positives)} positive, {len(negatives)} negative examples")

    while uncovered and len(clauses) < max_clauses:
        clause, covered, steps, neg_left = learn_clause(head, uncovered, negatives,
                                                        background, verbose=verbose)
        if not clause.body or not covered:
            if verbose:
                print("  no clause with positive gain -- stopping")
            break
        clauses.append(clause)
        all_steps.append(steps)
        before = len(uncovered)
        uncovered = [e for e in uncovered if e not in covered]
        if verbose:
            print(f"  => {clause}")
            print(f"     covers {before - len(uncovered)}/{before} remaining positives, "
                  f"{neg_left} negative bindings left\n")
    return clauses, all_steps, uncovered


def entails(clauses: list[Clause], pair: tuple) -> bool:
    """Does any learned clause prove this ground pair?"""
    for clause in clauses:
        variables = list(clause.head.args)
        rows, origins = [pair], [0]
        for lit in clause.body:
            variables, rows, origins = extend_bindings(variables, rows, origins, lit, FACTS)
            if not rows:
                break
        if rows:
            return True
    return False


def evaluate(clauses: list[Clause], positives: list[tuple], arity: int = 2) -> dict:
    """Precision/recall of the learned theory against ground truth over the whole
    closed-world space -- the check that FOIL learned the concept, not just fitted it."""
    pos_set = set(positives)
    space = list(product(PEOPLE, repeat=arity))
    tp = sum(1 for t in space if t in pos_set and entails(clauses, t))
    fp = sum(1 for t in space if t not in pos_set and entails(clauses, t))
    fn = len(pos_set) - tp
    return {"true positives": tp, "false positives": fp, "false negatives": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0}


# ------------------------------------------------- ground truth for each target
PARENT_SET = set(PARENT)
GT = {
    "grandfather": [(a, c) for a, b in PARENT_SET for b2, c in PARENT_SET
                    if b == b2 and a in MALE],
    "grandmother": [(a, c) for a, b in PARENT_SET for b2, c in PARENT_SET
                    if b == b2 and a in FEMALE],
    "sibling": [(x, y) for x in PEOPLE for y in PEOPLE if x != y and
                any((p, x) in PARENT_SET and (p, y) in PARENT_SET for p in PEOPLE)],
}
GT = {k: sorted(set(v)) for k, v in GT.items()}

banner("GROUND TRUTH SIZES")
for k, v in GT.items():
    print(f"  {k:14s} {len(v):3d} positive examples   e.g. {v[0]}")

# %% [markdown]
# ### Learning `grandfather(X,Y)`
#
# The trace below is the learner's actual reasoning: at each step it reports the literal
# chosen, its gain in bits, and how the positive/negative binding counts moved.

# %%
BG = {"parent": 2, "male": 1, "female": 1, "different": 2}

banner("LEARNING grandfather(X,Y)")
gf_clauses, gf_steps, gf_uncovered = foil("grandfather", GT["grandfather"], BG)
print("Learned theory:")
for c in gf_clauses:
    print("   ", c)
print("\nEvaluation over all", len(PEOPLE) ** 2, "candidate pairs:")
print("   ", evaluate(gf_clauses, GT["grandfather"]))

# %% [markdown]
# ### What the gain calculation actually looked like
#
# The table below shows, for each step of the first clause, the literal FOIL chose and the
# four runners-up. This is the evidence that the choice was driven by the gain formula
# rather than by the order candidates happened to be generated in.

# %%
rows = []
for step in gf_steps[0]:
    rows.append({"step": step["step"], "chosen literal": step["chosen"],
                 "gain (bits)": step["gain (bits)"],
                 "p0->p1": f"{step['p0']}->{step['p1']}",
                 "n0->n1": f"{step['n0']}->{step['n1']}", "t": step["t"]})
banner("GAIN TRACE FOR grandfather CLAUSE 1")
print(pd.DataFrame(rows).to_string(index=False))
print("\nRunners-up at each step (literal, gain):")
for step in gf_steps[0]:
    print(f"  step {step['step']}: " +
          ",  ".join(f"{l} {g}" for l, g in step["runners_up"]))

n_steps = len(gf_steps[0])
fig, axes = plt.subplots(1, n_steps, figsize=(5.3 * n_steps, 3.8))
axes = np.atleast_1d(axes)
for ax, step in zip(axes, gf_steps[0]):
    labels = [step["chosen"]] + [l for l, _ in step["runners_up"]]
    gains = [step["gain (bits)"]] + [g for _, g in step["runners_up"]]
    colors = [PALETTE[0]] + ["#c9d2de"] * len(step["runners_up"])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, gains, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set(xlabel="FOIL gain (bits)", title=f"Step {step['step']}: chose {step['chosen']}")
    ax.grid(axis="y", alpha=0)
save_fig("q4_gain_trace", fig)

# %% [markdown]
# ### Learning the other relations
#
# `grandmother` should differ from `grandfather` only in the gender literal. `sibling`
# is the interesting one: it needs `different(X,Y)`, because without it the clause
# `sibling(X,Y) :- parent(Z,X), parent(Z,Y)` also proves that everyone is their own sibling.

# %%
learned = {"grandfather": gf_clauses}

for target in ["grandmother", "sibling"]:
    banner(f"LEARNING {target}(X,Y)")
    clauses, steps, unc = foil(target, GT[target], BG)
    learned[target] = clauses
    print("Learned theory:")
    for c in clauses:
        print("   ", c)
    print("   ", evaluate(clauses, GT[target]))

# %% [markdown]
# ### Building on what was learned: `uncle(X,Y)`
#
# The learned `sibling` definition is now added to the background knowledge as derived
# facts, and FOIL is asked for `uncle`. This is incremental theory construction — the same
# idea that lets ILP systems build deep theories from shallow primitives.

# %%
sibling_facts = {(x, y) for x in PEOPLE for y in PEOPLE
                 if entails(learned["sibling"], (x, y))}
FACTS["sibling"] = sibling_facts
BG_UNCLE = {**BG, "sibling": 2}

GT["uncle"] = sorted({(u, c) for (u, s) in sibling_facts for (p, c) in PARENT_SET
                      if s == p and u in MALE})

banner("LEARNING uncle(X,Y)  [sibling now available as background]")
print(f"derived sibling facts available: {len(sibling_facts)}")
unc_clauses, unc_steps, _ = foil("uncle", GT["uncle"], BG_UNCLE)
print("Learned theory:")
for c in unc_clauses:
    print("   ", c)
print("   ", evaluate(unc_clauses, GT["uncle"]))
learned["uncle"] = unc_clauses



# %% [markdown]
# ### Where greedy FOIL breaks — and why
#
# The `uncle` result above is a **failure**, and an instructive one. The intended rule is
#
# ```
# uncle(X,Y) :- male(X), sibling(X,Z), parent(Z,Y).
# ```
#
# but greedy FOIL never selected `sibling(X,Z)` at all. It is worth being precise about
# why, because this is a property of the algorithm rather than a bug.
#
# FOIL is **greedy with no backtracking**: at each step it commits irrevocably to the
# single literal with the highest immediate gain. `parent(X,Z)` scored higher than
# `sibling(X,Z)` at step 2, so the search entered a region from which the correct clause is
# unreachable, then padded the body with weak literals until it hit the length limit —
# finishing with 64 negative bindings still covered.
#
# The check below confirms the diagnosis directly: it scores the *intended* literal against
# the one greedy actually chose, at the moment the decision was made.

# %%
def clause_state(body: list[Literal], pos: list[tuple], neg: list[tuple],
                 head_vars=("X", "Y")):
    """Replay a clause body from scratch, returning the binding tables it produces."""
    variables = list(head_vars)
    p_rows, p_org = list(pos), list(range(len(pos)))
    n_rows, n_org = list(neg), list(range(len(neg)))
    for lit in body:
        nv, p_rows, p_org = extend_bindings(variables, p_rows, p_org, lit, FACTS)
        _, n_rows, n_org = extend_bindings(variables, n_rows, n_org, lit, FACTS)
        variables = nv
    return variables, p_rows, p_org, n_rows


uncle_pos = GT["uncle"]
uncle_neg = [t for t in product(PEOPLE, repeat=2) if t not in set(uncle_pos)]

prefix = [Literal("male", ("X",))]
_vars, p_rows, p_org, n_rows = clause_state(prefix, uncle_pos, uncle_neg)
p0, n0 = len(p_rows), len(n_rows)

banner("STEP-2 DECISION FOR uncle, AFTER male(X)")
print(f"before adding anything: p0={p0}, n0={n0}\n")
for lit in [Literal("parent", ("X", "Z")), Literal("sibling", ("X", "Z"))]:
    nv, np_rows, np_org = extend_bindings(_vars, p_rows, p_org, lit, FACTS)
    _, nn_rows, _ = extend_bindings(_vars, n_rows, list(range(len(n_rows))), lit, FACTS)
    t = len(set(np_org))
    g = foil_gain(p0, n0, len(np_rows), len(nn_rows), t)
    print(f"  {str(lit):18s} gain={g:6.2f}   p:{p0}->{len(np_rows):<4} "
          f"n:{n0}->{len(nn_rows):<4} t={t}")
print("\nGreedy takes the larger number and never reconsiders -- that single decision is")
print("what costs it the correct rule.")

# %% [markdown]
# ### Fixing it with beam search
#
# The minimal repair is to stop committing to one literal. **Beam search** keeps the best
# $k$ partial clauses at every step instead of the best 1, so a literal that is second-best
# now but decisive later survives long enough to prove itself.
#
# States are ranked by their information content $t \cdot \log_2\!\big(p/(p+n)\big)$, which
# — unlike incremental gain — is comparable across different partial clauses.

# %%
@dataclass
class Partial:
    body: list
    variables: list
    p_rows: list
    p_org: list
    n_rows: list
    n_org: list

    @property
    def score(self) -> float:
        if not self.p_rows:
            return -1e9
        t = len(set(self.p_org))
        return t * np.log2(len(self.p_rows) / (len(self.p_rows) + len(self.n_rows)))


def learn_clause_beam(target: Literal, pos: list[tuple], neg: list[tuple],
                      background: dict[str, int], beam_width: int = 4,
                      max_body: int = 4):
    """Beam-search variant of the inner loop: keep the best `beam_width` partial
    clauses at each depth rather than committing to a single best literal."""
    beam = [Partial([], list(target.args), list(pos), list(range(len(pos))),
                    list(neg), list(range(len(neg))))]

    for _ in range(max_body):
        done = [s for s in beam if not s.n_rows and s.body]
        if done:
            best = max(done, key=lambda s: (len(set(s.p_org)), -len(s.body)))
            return Clause(target, best.body), {pos[i] for i in set(best.p_org)}
        nxt = []
        for state in beam:
            for lit in candidate_literals(state.variables, background):
                nv, pr, po = extend_bindings(state.variables, state.p_rows,
                                             state.p_org, lit, FACTS)
                if not pr:
                    continue
                _, nr, no = extend_bindings(state.variables, state.n_rows,
                                            state.n_org, lit, FACTS)
                nxt.append(Partial(state.body + [lit], nv, pr, po, nr, no))
        if not nxt:
            break
        # Deduplicate bodies, then keep the best `beam_width`.
        seen, uniq = set(), []
        for s in sorted(nxt, key=lambda s: -s.score):
            key = tuple(sorted(str(l) for l in s.body))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        beam = uniq[:beam_width]

    best = max(beam, key=lambda s: s.score)
    return Clause(target, best.body), {pos[i] for i in set(best.p_org)}


def foil_beam(target_name: str, positives: list[tuple], background: dict[str, int],
              beam_width: int = 4, max_clauses: int = 4):
    head = Literal(target_name, ("X", "Y"))
    pos_set = set(positives)
    negatives = [t for t in product(PEOPLE, repeat=2) if t not in pos_set]
    uncovered, clauses = list(positives), []
    while uncovered and len(clauses) < max_clauses:
        clause, covered = learn_clause_beam(head, uncovered, negatives,
                                            background, beam_width)
        if not clause.body or not covered:
            break
        clauses.append(clause)
        uncovered = [e for e in uncovered if e not in covered]
    return clauses


banner("uncle(X,Y) WITH BEAM SEARCH (width 4)")
uncle_beam = foil_beam("uncle", GT["uncle"], BG_UNCLE, beam_width=4)
for c in uncle_beam:
    print("   ", c)
print("   ", evaluate(uncle_beam, GT["uncle"]))
learned["uncle (beam search)"] = uncle_beam


# %% [markdown]
# ### Post-pruning the beam-search theory
#
# Beam search reached perfect precision and recall, but with a **redundant** theory: three
# clauses where one would do. Real FOIL implementations follow learning with a pruning
# pass, and the same idea applies here — greedily drop any literal or whole clause whose
# removal does not introduce a false positive or lose a true one.

# %%
def theory_scores(clauses: list[Clause], positives: list[tuple]) -> tuple[int, int]:
    m = evaluate(clauses, positives)
    return m["true positives"], m["false positives"]


def prune_theory(clauses: list[Clause], positives: list[tuple]) -> list[Clause]:
    """Drop redundant clauses, then redundant literals, keeping (tp, fp) unchanged."""
    clauses = [Clause(c.head, list(c.body)) for c in clauses]
    target_tp, target_fp = theory_scores(clauses, positives)

    changed = True
    while changed:
        changed = False
        for i in range(len(clauses)):                      # try deleting whole clauses
            trial = clauses[:i] + clauses[i + 1:]
            if trial and theory_scores(trial, positives) == (target_tp, target_fp):
                clauses = trial
                changed = True
                break
        if changed:
            continue
        for i, clause in enumerate(clauses):               # then individual literals
            for j in range(len(clause.body)):
                body = clause.body[:j] + clause.body[j + 1:]
                if not body:
                    continue
                trial = clauses[:i] + [Clause(clause.head, body)] + clauses[i + 1:]
                if theory_scores(trial, positives) == (target_tp, target_fp):
                    clauses = trial
                    changed = True
                    break
            if changed:
                break
    return clauses


banner("PRUNING THE BEAM-SEARCH uncle THEORY")
print(f"before: {len(uncle_beam)} clauses")
for c in uncle_beam:
    print("   ", c)
uncle_pruned = prune_theory(uncle_beam, GT["uncle"])
print(f"\nafter:  {len(uncle_pruned)} clause(s)")
for c in uncle_pruned:
    print("   ", c)
print("   ", evaluate(uncle_pruned, GT["uncle"]))
learned["uncle (beam + pruned)"] = uncle_pruned

# %% [markdown]
# ### Final comparison

# %%
summary_rows = []
for target, clauses in learned.items():
    gt_key = "uncle" if target.startswith("uncle") else target
    summary_rows.append({"target": target, "positives": len(GT[gt_key]),
                         "clauses": len(clauses),
                         "rule": " ; ".join(str(c) for c in clauses),
                         **evaluate(clauses, GT[gt_key])})
summary = pd.DataFrame(summary_rows)

banner("ALL LEARNED RULES")
print(summary[["target", "positives", "clauses", "true positives", "false positives",
               "false negatives", "precision", "recall"]].to_string(index=False))
print()
for r in summary_rows:
    print(f"  {r['target']:22s} {r['rule']}")

fig, ax = plt.subplots(figsize=(10.5, 4.0))
x = np.arange(len(summary))
ax.bar(x - 0.2, summary["precision"], 0.4, label="precision", color=PALETTE[0])
ax.bar(x + 0.2, summary["recall"], 0.4, label="recall", color=PALETTE[2])
ax.set(xticks=x, ylim=(0, 1.2), ylabel="score",
       title="Learned theories evaluated over the full closed-world space")
ax.set_xticklabels([t.replace(" (", "\n(") for t in summary["target"]], fontsize=8)
for i, r in summary.iterrows():
    ax.text(i, 1.08, f"{r['clauses']} clause(s)", ha="center", fontsize=8, color="#555")
ax.legend(loc="lower right")
save_fig("q4_rule_quality", fig)

# %% [markdown]
# ## Result and discussion
#
# * **FOIL induced three relations perfectly** from ground facts alone, with precision and
#   recall of 1.0 measured over all 225 candidate pairs:
#
#   ```
#   grandfather(X,Y) :- male(X), parent(X,Z), parent(Z,Y).
#   grandmother(X,Y) :- female(X), parent(Z,Y), parent(X,Z).
#   sibling(X,Y)     :- parent(Z,X), parent(Z,Y), different(X,Y).
#   ```
#
#   These are the textbook definitions, and nothing about them was supplied — the learner
#   was given only `parent/2`, `male/1`, `female/1`, `different/2` and a list of positive
#   pairs.
#
# * **The gain formula drove every choice.** The step-by-step trace shows the decisive
#   literal `parent(Z,Y)` scoring 29.07 bits against 5.85 for its nearest rival, collapsing
#   the covered negatives from 130 to 0 in one move.
#
# * **`different(X,Y)` was necessary, and FOIL found it unaided.** Without it the sibling
#   clause covers 20 negative bindings — every person counted as their own sibling. FOIL
#   selected the literal that removes exactly those cases.
#
# * **Greedy search failed on `uncle`, and this is the most informative result here.**
#   Plain FOIL returned a five-literal clause with **precision 0.20** that never mentions
#   `sibling` and still covers 64 negative bindings. The step-2 analysis pinpoints the
#   cause: `parent(X,Z)` out-scored `sibling(X,Z)` on immediate gain, and with no
#   backtracking that one decision made the correct clause unreachable. Greedy hill-climbing
#   on gain is not guaranteed to find the best clause — it has no lookahead.
#
# * **Beam search recovered correctness; pruning recovered elegance.** Keeping the best
#   four partial clauses instead of one lifted `uncle` to precision and recall 1.0 — but
#   via a *redundant three-clause theory*, not the single intended rule. Two of those
#   clauses were the correct rule split pointlessly by the gender of `Y`, and the third was
#   subsumed. The post-pruning pass then collapsed all three into exactly
#
#   ```
#   uncle(X,Y) :- sibling(X,Z), parent(Z,Y), male(X).
#   ```
#
#   Two distinct lessons, which are worth keeping separate: beam search fixed the *search*
#   failure, and pruning fixed a *representation* failure that beam search introduced.
#   Perfect precision and recall did not by itself mean the theory was good — a scoring
#   metric can be fully satisfied by a needlessly complicated hypothesis.
#
# * **Limitations.** Closed-world negatives are only sound when the fact base is complete;
#   on a real, partial family database many "negatives" would simply be unrecorded truths.
#   The learner also has no noise tolerance — every clause is driven to zero negative
#   coverage, which on noisy data would overfit badly.
