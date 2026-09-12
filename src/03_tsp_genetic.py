# %% [markdown]
# ---
# # Experiment 3 — Travelling Salesman Problem by Evolutionary Search
#
# **Aim.** Solve a 25-city Travelling Salesman Problem with a genetic algorithm, and
# judge the result honestly against classical construction and local-search baselines.
#
# ## Theory
#
# Given $n$ cities and pairwise distances, find the shortest closed tour visiting each
# exactly once. TSP is NP-hard: there are $(n-1)!/2$ distinct tours, so for $n = 25$ that
# is roughly $3.1 \times 10^{23}$ — exhaustive search is permanently out of reach.
#
# A **genetic algorithm** is a population-based metaheuristic that mimics natural
# selection. It makes no guarantee of optimality; it trades that guarantee for the ability
# to search an enormous space in reasonable time.
#
# | Component | Choice here | Why |
# |---|---|---|
# | Representation | permutation of city indices | every genome is a valid tour by construction |
# | Fitness | $1/\text{tour length}$ | shorter tours are fitter |
# | Selection | tournament ($k=5$) | cheap, and selection pressure is tunable via $k$ |
# | Crossover | **Order Crossover (OX1)** | preserves relative city order *and* yields a valid permutation |
# | Mutation | **inversion** of a random segment | reverses a sub-path — exactly a 2-opt move |
# | Elitism | best 2 copied forward | guarantees the best tour never degrades |
#
# **Why not single-point crossover?** Splicing two permutations at a point generally
# produces a genome with repeated and missing cities — an invalid tour. OX1 exists
# precisely to avoid this: it copies a contiguous slice from one parent, then fills the
# remaining positions with the cities of the other parent *in that parent's order*,
# skipping any already present. The result is always a valid permutation.
#
# **Inversion mutation is a 2-opt move.** Reversing the segment between positions $i$ and
# $j$ removes two edges from the tour and reconnects them the other way round. This is the
# single most effective local move for Euclidean TSP, which is why it is preferred over a
# naive swap.

# %%
from math import asin, cos, factorial, radians, sin, sqrt

CITIES = {
    "Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49),
    "Ahmedabad": (23.02, 72.57), "Pune": (18.52, 73.86), "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95), "Kanpur": (26.45, 80.33), "Nagpur": (21.15, 79.09),
    "Indore": (22.72, 75.86), "Bhopal": (23.26, 77.41), "Patna": (25.59, 85.14),
    "Surat": (21.17, 72.83), "Vadodara": (22.31, 73.18), "Visakhapatnam": (17.69, 83.22),
    "Kochi": (9.93, 76.27), "Guwahati": (26.14, 91.74), "Chandigarh": (30.73, 76.78),
    "Amritsar": (31.63, 74.87), "Jodhpur": (26.24, 73.02), "Raipur": (21.25, 81.63),
    "Bhubaneswar": (20.30, 85.82),
}
NAMES = list(CITIES)
COORDS = np.array([CITIES[c] for c in NAMES])      # (lat, lon)
N_CITIES = len(NAMES)


def haversine(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle distance in km -- real road-network geometry is closer to this
    than to plain Euclidean distance on latitude/longitude."""
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


DIST = np.array([[haversine(COORDS[i], COORDS[j]) for j in range(N_CITIES)]
                 for i in range(N_CITIES)])


def tour_length(tour: np.ndarray) -> float:
    """Closed-tour length: sum of consecutive distances, wrapping back to the start."""
    return float(DIST[tour, np.roll(tour, -1)].sum())


def tour_lengths(pop: np.ndarray) -> np.ndarray:
    """Vectorised: length of every tour in a (pop_size, n) population at once.

    Scoring the population in one indexing operation rather than one call per
    individual is what makes the parameter sweeps below affordable.
    """
    return DIST[pop, np.roll(pop, -1, axis=1)].sum(axis=1)


banner("PROBLEM INSTANCE")
print(f"{N_CITIES} cities, distances in km (haversine).")
print(f"Search space: (n-1)!/2 = {factorial(N_CITIES-1)/2:.2e} distinct tours")
print(f"Nearest pair : {DIST[DIST > 0].min():.0f} km")
print(f"Farthest pair: {DIST.max():.0f} km")

# %% [markdown]
# ### Baselines to beat
#
# A GA result means nothing without a reference point. Three baselines:
#
# 1. **Random tour** — what "no algorithm at all" looks like.
# 2. **Nearest neighbour** — the classic greedy construction: repeatedly hop to the closest
#    unvisited city. Fast, typically 15–25 % above optimal.
# 3. **Nearest neighbour + 2-opt** — greedily apply segment reversals until no single
#    reversal improves the tour. A genuinely strong local-search baseline.

# %%
def nearest_neighbour(start: int = 0) -> np.ndarray:
    unvisited = set(range(N_CITIES)) - {start}
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: DIST[last, c])
        tour.append(nxt)
        unvisited.remove(nxt)
    return np.array(tour)


def two_opt(tour: np.ndarray) -> np.ndarray:
    """Repeatedly reverse the segment that most improves the tour, until none does."""
    best = tour.copy()
    best_len = tour_length(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, N_CITIES - 1):
            for j in range(i + 1, N_CITIES):
                cand = best.copy()
                cand[i:j + 1] = cand[i:j + 1][::-1]
                cand_len = tour_length(cand)
                if cand_len < best_len - 1e-9:
                    best, best_len, improved = cand, cand_len, True
    return best


rng = np.random.default_rng(SEED)
random_tour = rng.permutation(N_CITIES)
nn_tour = nearest_neighbour(0)
opt_tour = two_opt(nn_tour)

baselines = {
    "Random tour": tour_length(random_tour),
    "Nearest neighbour": tour_length(nn_tour),
    "NN + 2-opt": tour_length(opt_tour),
}
banner("BASELINES")
for k, v in baselines.items():
    print(f"  {k:20s} {v:10,.0f} km")

# %% [markdown]
# ### The genetic algorithm

# %%
def order_crossover(p1: np.ndarray, p2: np.ndarray, a: int, b: int) -> np.ndarray:
    """OX1: copy the slice p1[a..b], then fill the remaining positions with the cities
    of p2 that are not already present, in p2's order.

    Written with a boolean membership mask rather than a Python set + loop: the result
    is identical but it runs entirely in numpy, which matters because this is called
    ~300 times per generation for thousands of generations.
    """
    n = len(p1)
    child = np.empty(n, dtype=np.int64)
    child[a:b + 1] = p1[a:b + 1]
    in_slice = np.zeros(n, dtype=bool)
    in_slice[p1[a:b + 1]] = True
    fill = p2[~in_slice[p2]]                 # p2's cities, minus the copied slice
    child[:a] = fill[:a]                     # two slice assignments beat building an
    child[b + 1:] = fill[a:]                 # index array with np.r_
    return child


def inversion_mutation(tour: np.ndarray, a: int, b: int) -> np.ndarray:
    """Reverse the segment a..b -- i.e. apply one 2-opt move."""
    tour[a:b + 1] = tour[a:b + 1][::-1]
    return tour


def tournament_batch(lengths: np.ndarray, n: int, k: int, rng) -> np.ndarray:
    """Run n independent k-way tournaments at once, returning the winners' indices.

    Selection is the same as picking k individuals at random and keeping the fittest;
    doing all n tournaments in one vectorised step rather than n Python calls is what
    makes the multi-seed and parameter-sweep studies below affordable.
    """
    cand = rng.integers(0, len(lengths), (n, k))
    return cand[np.arange(n), np.argmin(lengths[cand], axis=1)]


def genetic_algorithm(pop_size=300, generations=600, tournament_k=5,
                      mutation_rate=0.25, elitism=2, seed=SEED, record_every=None):
    rng = np.random.default_rng(seed)
    pop = np.array([rng.permutation(N_CITIES) for _ in range(pop_size)])
    lengths = tour_lengths(pop)
    history, snapshots = [], {}

    for gen in range(generations):
        order = np.argsort(lengths)
        pop, lengths = pop[order], lengths[order]
        history.append({"generation": gen, "best": lengths[0],
                        "mean": lengths.mean(), "worst": lengths[-1]})
        if record_every and gen % record_every == 0:
            snapshots[gen] = pop[0].copy()

        n_children = pop_size - elitism
        # Draw every random decision for this generation in one batch each.
        parents = pop[tournament_batch(lengths, 2 * n_children, tournament_k, rng)]
        cuts = np.sort(rng.integers(0, N_CITIES, (n_children, 2)), axis=1)
        mut_seg = np.sort(rng.integers(0, N_CITIES, (n_children, 2)), axis=1)
        do_mutate = rng.random(n_children) < mutation_rate

        new_pop = [pop[i].copy() for i in range(elitism)]       # elitism
        for c in range(n_children):
            child = order_crossover(parents[2 * c], parents[2 * c + 1],
                                    cuts[c, 0], cuts[c, 1])
            if do_mutate[c]:
                child = inversion_mutation(child, mut_seg[c, 0], mut_seg[c, 1])
            new_pop.append(child)

        pop = np.array(new_pop)
        lengths = tour_lengths(pop)

    best_i = int(np.argmin(lengths))
    snapshots["final"] = pop[best_i].copy()
    return pop[best_i], float(lengths[best_i]), pd.DataFrame(history), snapshots


(ga_tour, ga_len, hist, snaps), ga_time = timed(
    genetic_algorithm, record_every=50)

banner("GENETIC ALGORITHM RESULT")
print(f"Best tour length : {ga_len:,.0f} km")
print(f"Generations      : {len(hist)}   wall-clock: {ga_time:.1f}s")
print(f"Improvement over random : {100*(1-ga_len/baselines['Random tour']):.1f}%")
print(f"Improvement over NN     : {100*(1-ga_len/baselines['Nearest neighbour']):.1f}%")
print(f"Versus NN + 2-opt       : {100*(ga_len/baselines['NN + 2-opt']-1):+.1f}%")
print("\nTour:", " -> ".join(NAMES[i] for i in ga_tour[:6]), "-> ...")

# %% [markdown]
# ### Convergence

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))

axes[0].plot(hist["generation"], hist["worst"], color="#c9d2de", label="worst in population")
axes[0].plot(hist["generation"], hist["mean"], color=PALETTE[3], label="population mean")
axes[0].plot(hist["generation"], hist["best"], color=PALETTE[0], lw=2, label="best so far")
axes[0].axhline(baselines["NN + 2-opt"], ls="--", color=PALETTE[1],
                label=f"NN + 2-opt ({baselines['NN + 2-opt']:,.0f} km)")
axes[0].axhline(baselines["Nearest neighbour"], ls=":", color=PALETTE[2],
                label=f"Nearest neighbour ({baselines['Nearest neighbour']:,.0f} km)")
axes[0].set(xlabel="generation", ylabel="tour length (km)", title="GA convergence")
axes[0].legend(fontsize=8)

axes[1].plot(hist["generation"], hist["mean"] - hist["best"], color=PALETTE[4])
axes[1].set(xlabel="generation", ylabel="mean − best (km)",
            title="Population diversity\n(collapses as the GA converges)")
save_fig("q3_convergence", fig)

# %% [markdown]
# ### The tours themselves

# %%
def plot_tour(ax, tour, title, color=PALETTE[0], annotate=False):
    pts = COORDS[np.append(tour, tour[0])]
    ax.plot(pts[:, 1], pts[:, 0], "-", color=color, lw=1.6, zorder=1)
    ax.scatter(COORDS[:, 1], COORDS[:, 0], s=42, color="#20304d", zorder=3)
    ax.scatter(COORDS[tour[0], 1], COORDS[tour[0], 0], s=150, marker="*",
               color=PALETTE[1], zorder=4)
    if annotate:
        for i, n in enumerate(NAMES):
            ax.text(COORDS[i, 1] + 0.25, COORDS[i, 0] + 0.15, n, fontsize=6.5, zorder=5)
    ax.set(xlabel="longitude (°E)", ylabel="latitude (°N)")
    ax.set_title(f"{title}\n{tour_length(tour):,.0f} km", fontsize=10)


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
plot_tour(axes[0], random_tour, "Random tour", "#c0392b")
plot_tour(axes[1], nn_tour, "Nearest neighbour", PALETTE[3])
plot_tour(axes[2], ga_tour, "Genetic algorithm", PALETTE[0], annotate=True)
save_fig("q3_tours", fig)

# %% [markdown]
# ### How the best tour evolves

# %%
gens = [g for g in snaps if g != "final"][:5] + ["final"]
fig, axes = plt.subplots(1, len(gens), figsize=(3.0 * len(gens), 3.4))
for ax, g in zip(axes, gens):
    plot_tour(ax, snaps[g], "final" if g == "final" else f"generation {g}")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
save_fig("q3_tour_evolution", fig)

# %% [markdown]
# ### Does the GA reliably beat the baselines?
#
# A single GA run is one sample from a stochastic process. This repeats the run across
# several seeds, and separately sweeps the mutation rate — the parameter that controls the
# exploration/exploitation balance.

# %%
seed_results = []
for s in range(6):
    _, L, _, _ = genetic_algorithm(generations=400, seed=SEED + s)
    seed_results.append(L)
seed_results = np.array(seed_results)

sweep = []
for rate in [0.05, 0.15, 0.25, 0.40, 0.60, 0.85]:
    lens = [genetic_algorithm(generations=300, mutation_rate=rate, seed=SEED + s)[1]
            for s in range(3)]
    sweep.append({"mutation_rate": rate, "mean_km": np.mean(lens), "std_km": np.std(lens)})
sweep_df = pd.DataFrame(sweep)

banner("ROBUSTNESS")
print(f"6 seeds, 400 generations: mean {seed_results.mean():,.0f} km, "
      f"sd {seed_results.std():,.0f} km, best {seed_results.min():,.0f}, "
      f"worst {seed_results.max():,.0f}")
print(f"NN + 2-opt reference    : {baselines['NN + 2-opt']:,.0f} km")
print(f"Seeds beating NN + 2-opt: {(seed_results < baselines['NN + 2-opt']).sum()}/6")
print()
print(sweep_df.to_string(index=False,
                         formatters={"mutation_rate": "{:.2f}".format,
                                     "mean_km": "{:,.0f}".format,
                                     "std_km": "{:,.0f}".format}))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))
axes[0].bar(range(len(seed_results)), seed_results, color=PALETTE[0])
axes[0].axhline(baselines["NN + 2-opt"], ls="--", color=PALETTE[1], label="NN + 2-opt")
axes[0].axhline(baselines["Nearest neighbour"], ls=":", color=PALETTE[2], label="NN")
axes[0].set(xlabel="seed", ylabel="tour length (km)", title="Run-to-run variability")
axes[0].set_ylim(min(seed_results.min(), baselines["NN + 2-opt"]) * 0.97,
                 baselines["Nearest neighbour"] * 1.03)
axes[0].legend(fontsize=8)

axes[1].errorbar(sweep_df["mutation_rate"], sweep_df["mean_km"], yerr=sweep_df["std_km"],
                 fmt="o-", capsize=4, color=PALETTE[4])
axes[1].set(xlabel="mutation rate", ylabel="tour length (km)",
            title="Mutation rate sweep\n(too low = premature convergence)")
save_fig("q3_robustness", fig)

# %% [markdown]
# ## Result and discussion
#
# * **The GA works.** From random tours it reached **9,026 km**, against 25,417 km for a
#   random tour (−64.5 %) and 12,940 km for greedy nearest neighbour (−30.2 %) — a
#   sensible circuit of India recovered from nothing but a distance matrix and selection
#   pressure.
#
# * **2-opt is the honest yardstick, and it is close.** Nearest-neighbour-plus-2-opt
#   scored 9,192 km, so the GA's margin was only **1.8 %** — not the landslide the
#   random-tour comparison suggests. Across six seeds the GA averaged 9,014 km (sd 123)
#   and beat 2-opt in **5 of 6** runs. So the GA is genuinely better here, but modestly,
#   and not on every run. Comparing a metaheuristic only against a random baseline
#   flatters it; this is the comparison that actually tests it.
#
# * **Convergence has two phases.** The best-so-far curve drops steeply for the first
#   ~100 generations, then flattens. The diversity plot explains why: `mean − best`
#   collapses as the population fills with near-copies of the elite tour, and once
#   diversity is gone crossover produces little new material — only mutation can escape.
#
# * **Mutation rate mattered less than expected.** The sweep from 0.05 to 0.85 moved the
#   mean tour by only ~2 %, and the spread between rates is comparable to the
#   seed-to-seed standard deviation — so at this problem size the sweep does *not*
#   support a strong claim about an optimal rate. The theoretical story (low rates
#   converge prematurely, high rates never consolidate) is visible at best as a shallow
#   trend. A 25-city instance is simply too easy for the parameter to bite; the effect
#   would need a larger instance, or more seeds per rate, to be demonstrated properly.
#
# * **No optimality guarantee.** Unlike A* in Experiment 2, nothing here certifies the
#   answer. The GA returns the best tour *it happened to find*. For 25 cities the true
#   optimum is unknown to us, so every claim above is stated relative to measured
#   baselines rather than to an optimum.
