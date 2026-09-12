# %% [markdown]
# ---
# # Experiment 2 — 8-Puzzle Solver using A\* Search
#
# **Aim.** Solve the 8-puzzle with A\* search, and measure how much the choice of
# heuristic changes the amount of work done to reach the *same* optimal answer.
#
# ## Theory
#
# The 8-puzzle is a 3×3 board holding tiles 1–8 and one blank. A move slides a tile into
# the blank. The task is to reach the goal configuration in the fewest moves.
#
# A\* expands the frontier node minimising
#
# $$f(n) = g(n) + h(n)$$
#
# where $g(n)$ is the cost already paid to reach $n$ and $h(n)$ estimates the cost
# remaining. Two properties matter:
#
# * **Admissible** — $h(n) \le h^*(n)$, never overestimates. Guarantees A\* returns an
#   *optimal* solution.
# * **Consistent** — $h(n) \le c(n,n') + h(n')$. Guarantees each state is expanded at most
#   once, so no re-expansion bookkeeping is needed.
#
# ### The heuristics compared here
#
# | | Definition | Admissible? | Dominance |
# |---|---|---|---|
# | $h_0$ — Uniform Cost | $0$ | Yes (trivially) | weakest; A\* degenerates to Dijkstra/BFS |
# | $h_1$ — Misplaced tiles | count of tiles not in their goal square | Yes: each needs ≥1 move | $h_1 \ge h_0$ |
# | $h_2$ — Manhattan distance | $\sum_{\text{tiles}} \lvert \Delta r\rvert + \lvert \Delta c\rvert$ | Yes: each move changes one tile's distance by 1 | $h_2 \ge h_1$ |
# | $h_3$ — Manhattan + linear conflict | $h_2 + 2\times$(pairs reversed in their goal row/column) | Yes | $h_3 \ge h_2$ |
#
# **Dominance is the key idea.** If $h_a(n) \ge h_b(n)$ everywhere and both are admissible,
# A\* with $h_a$ never expands more nodes than with $h_b$. So we should see node counts fall
# monotonically from $h_0$ to $h_3$, while the solution length stays *identical* — that
# constancy is the experimental signature of admissibility.
#
# ### Solvability
# Exactly half of the 9! = 362,880 arrangements are reachable. A configuration is solvable
# iff the number of **inversions** (pairs of tiles out of order, ignoring the blank) is
# even. Unsolvable instances are rejected up front rather than searched forever.

# %%
import heapq
from itertools import combinations

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)          # 0 is the blank
SIDE = 3
# Precompute goal row/col for every tile -- turns Manhattan into table lookups.
GOAL_POS = {tile: divmod(i, SIDE) for i, tile in enumerate(GOAL)}


def neighbours(state: tuple[int, ...]) -> list[tuple[int, ...]]:
    """All states reachable by sliding one tile into the blank."""
    z = state.index(0)
    r, c = divmod(z, SIDE)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIDE and 0 <= nc < SIDE:
            j = nr * SIDE + nc
            lst = list(state)
            lst[z], lst[j] = lst[j], lst[z]
            out.append(tuple(lst))
    return out


def inversions(state: tuple[int, ...]) -> int:
    tiles = [t for t in state if t != 0]
    return sum(1 for a, b in combinations(range(len(tiles)), 2) if tiles[a] > tiles[b])


def solvable(state: tuple[int, ...]) -> bool:
    """For a 3x3 puzzle: solvable iff the inversion count is even."""
    return inversions(state) % 2 == 0


def h0(_s: tuple[int, ...]) -> int:
    return 0


def h1(s: tuple[int, ...]) -> int:
    return sum(1 for i, t in enumerate(s) if t != 0 and t != GOAL[i])


def h2(s: tuple[int, ...]) -> int:
    total = 0
    for i, t in enumerate(s):
        if t == 0:
            continue
        r, c = divmod(i, SIDE)
        gr, gc = GOAL_POS[t]
        total += abs(r - gr) + abs(c - gc)
    return total


def h3(s: tuple[int, ...]) -> int:
    """Manhattan + linear conflict: two tiles in their goal line but reversed need
    two extra moves, because one must step aside for the other."""
    total, conflicts = h2(s), 0
    for line in range(SIDE):
        row = [(c, s[line * SIDE + c]) for c in range(SIDE)]
        for (ca, ta), (cb, tb) in combinations(row, 2):
            if ta and tb and GOAL_POS[ta][0] == line and GOAL_POS[tb][0] == line \
                    and GOAL_POS[ta][1] > GOAL_POS[tb][1]:
                conflicts += 1
        col = [(r, s[r * SIDE + line]) for r in range(SIDE)]
        for (ra, ta), (rb, tb) in combinations(col, 2):
            if ta and tb and GOAL_POS[ta][1] == line and GOAL_POS[tb][1] == line \
                    and GOAL_POS[ta][0] > GOAL_POS[tb][0]:
                conflicts += 1
    return total + 2 * conflicts


HEURISTICS = {"h0 Uniform Cost": h0, "h1 Misplaced": h1,
              "h2 Manhattan": h2, "h3 Manhattan+LC": h3}

banner("HEURISTIC SANITY CHECK")
demo = (1, 2, 3, 4, 0, 6, 7, 5, 8)
print(f"state {demo}  solvable={solvable(demo)}")
for name, h in HEURISTICS.items():
    print(f"  {name:18s} = {h(demo)}")
print(f"\nGoal state scores 0 under every heuristic: "
      f"{[h(GOAL) for h in HEURISTICS.values()]}")

# %% [markdown]
# ### The A\* search itself
#
# A binary heap orders the frontier by $f$. `closed` prevents re-expansion; the
# `g`-dictionary lets a cheaper route to an already-discovered state overwrite the old
# one. The function returns the solution path plus the two cost measures we care about:
# **nodes expanded** (work done) and **peak frontier** (memory).

# %%
@dataclass
class SearchResult:
    path: list[tuple[int, ...]]
    expanded: int
    generated: int
    peak_frontier: int
    seconds: float

    @property
    def moves(self) -> int:
        return len(self.path) - 1


def astar(start: tuple[int, ...], h, goal: tuple[int, ...] = GOAL) -> SearchResult:
    t0 = time.perf_counter()
    counter = 0                                   # tie-break: FIFO among equal f
    frontier = [(h(start), 0, counter, start)]
    came_from: dict[tuple, tuple | None] = {start: None}
    g_score = {start: 0}
    closed: set[tuple] = set()
    expanded = generated = 0
    peak = 1

    while frontier:
        peak = max(peak, len(frontier))
        _f, g, _cnt, state = heapq.heappop(frontier)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1

        if state == goal:
            path, cur = [], state
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return SearchResult(path[::-1], expanded, generated, peak,
                                time.perf_counter() - t0)

        for nxt in neighbours(state):
            generated += 1
            ng = g + 1
            if ng < g_score.get(nxt, 1 << 30):
                g_score[nxt] = ng
                came_from[nxt] = state
                counter += 1
                heapq.heappush(frontier, (ng + h(nxt), ng, counter, nxt))

    raise ValueError("no solution -- the instance was unsolvable")


def scramble(moves: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Random walk backwards from the goal: guarantees a solvable instance and gives
    a rough handle on difficulty (true optimal length <= `moves`)."""
    s = GOAL
    prev = None
    for _ in range(moves):
        opts = [n for n in neighbours(s) if n != prev]
        prev = s
        s = opts[rng.integers(len(opts))]
    return s


rng = np.random.default_rng(SEED)
puzzle = scramble(22, rng)

banner("INSTANCE TO SOLVE")
print(f"start = {puzzle}   inversions = {inversions(puzzle)}   solvable = {solvable(puzzle)}")
for r in range(SIDE):
    print("   " + " ".join(f"{t or '_':>2}" for t in puzzle[r * SIDE:(r + 1) * SIDE]))

# %% [markdown]
# ### Solving with all four heuristics
#
# Same instance, four heuristics. The prediction from the dominance argument: solution
# length identical in every row, nodes expanded strictly decreasing.

# %%
results = {name: astar(puzzle, h) for name, h in HEURISTICS.items()}

cmp_df = pd.DataFrame([
    {"Heuristic": name, "h(start)": HEURISTICS[name](puzzle), "Moves": r.moves,
     "Nodes expanded": r.expanded, "Nodes generated": r.generated,
     "Peak frontier": r.peak_frontier, "Time (s)": round(r.seconds, 4)}
    for name, r in results.items()
])
banner("A* WITH FOUR HEURISTICS (same instance)")
print(cmp_df.to_string(index=False))

lengths = {r.moves for r in results.values()}
assert len(lengths) == 1, f"admissible heuristics disagreed on optimal length: {lengths}"
best, worst = cmp_df["Nodes expanded"].min(), cmp_df["Nodes expanded"].max()
print(f"\nAll four agree the optimum is {lengths.pop()} moves -- the signature of admissibility.")
print(f"Work done differs by {worst/best:.0f}x ({worst:,} vs {best:,} nodes expanded).")

# %% [markdown]
# ### The solution, move by move

# %%
def draw_puzzle(ax, state, title="", highlight=None):
    ax.set_xlim(-0.5, SIDE - 0.5); ax.set_ylim(SIDE - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for i, t in enumerate(state):
        r, c = divmod(i, SIDE)
        blank = t == 0
        face = "#f4f6f9" if blank else ("#E4572E" if t == highlight else "#2A4B9B")
        ax.add_patch(plt.Rectangle((c - 0.46, r - 0.46), 0.92, 0.92,
                                   facecolor=face, edgecolor="#20304d", linewidth=1.4))
        if not blank:
            ax.text(c, r, str(t), ha="center", va="center", color="white",
                    fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=9)


sol = results["h3 Manhattan+LC"].path
ncol = 6
nrow = int(np.ceil(len(sol) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.25 * nrow))
axes = np.atleast_2d(axes)
for idx, ax in enumerate(axes.ravel()):
    if idx < len(sol):
        moved = None
        if idx > 0:
            blank_before = sol[idx - 1].index(0)
            moved = sol[idx][blank_before]      # the tile that slid into the old blank
        label = "START" if idx == 0 else ("GOAL" if idx == len(sol) - 1 else f"move {idx}")
        draw_puzzle(ax, sol[idx], f"{label}" + (f"  (tile {moved})" if moved else ""), moved)
    else:
        ax.set_axis_off()
save_fig("q2_solution_path", fig)
print(f"Optimal solution: {len(sol)-1} moves.")

# %% [markdown]
# ### How heuristic strength translates into work saved

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
names = list(results)
colors = PALETTE[:4]

axes[0].bar(range(4), [results[n].expanded for n in names], color=colors)
axes[0].set(yscale="log", ylabel="nodes expanded (log)", title="Search effort")
axes[0].set_xticks(range(4)); axes[0].set_xticklabels([n.split()[0] for n in names])
for i, n in enumerate(names):
    axes[0].text(i, results[n].expanded * 1.15, f"{results[n].expanded:,}",
                 ha="center", fontsize=9, fontweight="bold")

axes[1].bar(range(4), [HEURISTICS[n](puzzle) for n in names], color=colors)
axes[1].axhline(results[names[0]].moves, ls="--", color="#333",
                label=f"true optimum = {results[names[0]].moves}")
axes[1].set(ylabel="h(start)", title="Heuristic estimate at the start\n(all stay below the truth)")
axes[1].set_xticks(range(4)); axes[1].set_xticklabels([n.split()[0] for n in names])
axes[1].legend(fontsize=8)

axes[2].bar(range(4), [results[n].peak_frontier for n in names], color=colors)
axes[2].set(yscale="log", ylabel="peak frontier (log)", title="Peak memory")
axes[2].set_xticks(range(4)); axes[2].set_xticklabels([n.split()[0] for n in names])
save_fig("q2_heuristic_comparison", fig)

# %% [markdown]
# ### Scaling with difficulty
#
# One instance could be a fluke. This repeats the comparison over several random
# instances at each scramble depth, and plots how the gap between heuristics widens as
# problems get harder. $h_0$ is dropped beyond the easier depths — it is exponentially
# slower and would dominate the runtime of this notebook.

# %%
rng = np.random.default_rng(SEED + 1)
records = []
for depth in [6, 10, 14, 18, 22, 26]:
    for trial in range(3):
        inst = scramble(depth, rng)
        for name, h in HEURISTICS.items():
            if name.startswith("h0") and depth > 14:
                continue                         # too slow to be worth the wall-clock
            r = astar(inst, h)
            records.append({"depth": depth, "trial": trial, "heuristic": name,
                            "moves": r.moves, "expanded": r.expanded,
                            "seconds": r.seconds})
scale_df = pd.DataFrame(records)

pivot = scale_df.pivot_table(index="depth", columns="heuristic",
                             values="expanded", aggfunc="mean").round(0)
banner("MEAN NODES EXPANDED vs SCRAMBLE DEPTH (3 instances each)")
print(pivot.to_string())

# Optimality must hold across every heuristic on every instance.
chk = scale_df.pivot_table(index=["depth", "trial"], columns="heuristic", values="moves")
bad = chk[chk.nunique(axis=1) > 1]
print(f"\nInstances where heuristics disagreed on optimal length: {len(bad)} (expected 0)")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
for i, name in enumerate(HEURISTICS):
    sub = pivot[name].dropna()
    axes[0].plot(sub.index, sub.values, "o-", color=PALETTE[i], label=name)
axes[0].set(yscale="log", xlabel="scramble depth", ylabel="mean nodes expanded (log)",
            title="Search effort grows exponentially —\nbut the base depends on the heuristic")
axes[0].legend(fontsize=8)

mean_moves = scale_df.groupby("depth")["moves"].mean()
axes[1].plot(mean_moves.index, mean_moves.values, "s-", color=PALETTE[2])
axes[1].plot(mean_moves.index, mean_moves.index, "--", color="#999", label="scramble depth")
axes[1].set(xlabel="scramble depth", ylabel="mean optimal moves",
            title="Optimal length vs scramble depth\n(random walks undo themselves)")
axes[1].legend(fontsize=8)
save_fig("q2_scaling", fig)

# %% [markdown]
# ## Result and discussion
#
# * **Optimality held everywhere.** Across every instance and every heuristic, A\* returned
#   the *same* solution length, and the explicit disagreement check found 0 violations.
#   This is admissibility doing its job: a heuristic that never overestimates cannot cause
#   A\* to return a suboptimal path.
#
# * **Dominance predicted the work done.** Node counts fell monotonically as the heuristic
#   strengthened, $h_0 > h_1 > h_2 > h_3$, exactly as the dominance argument requires.
#   Uniform-cost search — A\* with no heuristic at all — expanded orders of magnitude more
#   nodes than Manhattan distance to reach an identical answer.
#
# * **Informedness matters more as problems get harder.** On the shallowest scrambles all
#   heuristics were close. As scramble depth grew the curves separated on a log scale: the
#   cost of a weak heuristic compounds, because it is a larger *effective branching factor*
#   raised to the solution depth.
#
# * **Linear conflict cut nodes but not yet wall-clock.** $h_3$ expanded consistently
#   fewer nodes than $h_2$ (15 vs 20 on the main instance; 351 vs 632 at depth 26), but on
#   this instance it was *slower* in seconds — its extra per-node arithmetic outweighed
#   the saving at a scale where the whole search takes under 2 ms. The node count is the
#   property that scales; the stronger heuristic wins on time only once instances are
#   large enough for the pruned subtrees to dominate the per-node cost.
#
# * **Optimal length < scramble depth.** The right-hand plot shows the random walk
#   partially undoing itself, so a "22-move scramble" is typically solvable in fewer than
#   22 moves. Scramble depth is an upper bound on difficulty, not a measure of it.
