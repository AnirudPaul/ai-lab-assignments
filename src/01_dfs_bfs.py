# %% [markdown]
# ---
# # Experiment 1 — Depth-First Search and Breadth-First Search
#
# **Aim.** Implement DFS and BFS from first principles and use them to traverse a graph,
# comparing the order in which they discover vertices and the shape of the search tree
# each one produces.
#
# ## Theory
#
# Both algorithms explore a graph by maintaining a **frontier** of discovered-but-not-yet-
# expanded vertices. The *only* difference between them is the discipline of that frontier:
#
# | | DFS | BFS |
# |---|---|---|
# | Frontier | **Stack** (LIFO) | **Queue** (FIFO) |
# | Explores | One branch as deep as possible, then backtracks | All vertices at distance *d* before any at *d+1* |
# | Time | $O(V + E)$ | $O(V + E)$ |
# | Space | $O(b\,m)$ — $m$ = deepest level, $b$ = branching factor | $O(b^{\,d})$ — $d$ = depth of the shallowest goal |
# | Shortest path (unweighted)? | **No** — finds *a* path, not the shortest | **Yes** — first time a vertex is dequeued it is at minimum distance |
# | Typical use | Cycle detection, topological sort, connectivity | Shortest hop count, level-order processing |
#
# The BFS shortest-path guarantee is worth stating precisely, because it is the reason BFS
# is used for unweighted shortest paths: since BFS dequeues vertices in non-decreasing order
# of distance from the source, the first time it reaches a vertex it has done so along a path
# with the fewest possible edges.
#
# ### Implementation note on determinism
# A graph's adjacency list has no inherent order, so two correct DFS implementations can
# produce different (equally valid) visit orders. To make every number in this record
# reproducible, `neighbours()` below returns neighbours in sorted order.

# %%
class Graph:
    """Undirected/directed graph on an adjacency list, built from scratch."""

    def __init__(self, directed: bool = False):
        self.adj: dict[str, list[str]] = defaultdict(list)
        self.directed = directed

    def add_edge(self, u: str, v: str) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u) if not self.directed else self.adj.setdefault(v, [])

    def neighbours(self, u: str) -> list[str]:
        # Sorted => deterministic, reproducible traversal order.
        return sorted(self.adj[u])

    @property
    def nodes(self) -> list[str]:
        return sorted(self.adj)

    @property
    def edges(self) -> list[tuple[str, str]]:
        seen = set()
        for u in self.nodes:
            for v in self.adj[u]:
                key = (u, v) if self.directed else tuple(sorted((u, v)))
                seen.add(key)
        return sorted(seen)

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        return f"<Graph {kind}: {len(self.nodes)} vertices, {len(self.edges)} edges>"


# The graph used throughout this experiment. It is built so DFS and BFS visibly
# disagree: there is a SHORT route A-C-J (2 edges) and a LONG route
# A-B-D-E-F-J (5 edges). Because neighbours are visited in sorted order, DFS
# commits to B first and finds J the long way round, while BFS finds the short one.
EDGES = [("A", "B"), ("A", "C"), ("C", "J"),
         ("B", "D"), ("D", "E"), ("E", "F"), ("F", "J"),
         ("B", "G"), ("G", "H"), ("H", "E")]

g = Graph()
for u, v in EDGES:
    g.add_edge(u, v)

banner("GRAPH UNDER TEST")
print(g)
for n in g.nodes:
    print(f"  {n} -> {g.neighbours(n)}")

# %% [markdown]
# ### The two algorithms
#
# Each returns the visit order, the **search tree** (the `parent` map — which vertex
# discovered which), and a step-by-step trace of the frontier so we can plot how the two
# frontiers grow. DFS is given twice: recursively, which is the textbook form, and
# iteratively with an explicit stack, which makes the duality with BFS obvious.

# %%
def dfs_recursive(graph: Graph, start: str):
    """Textbook recursive DFS. The call stack *is* the frontier."""
    visited, order, parent = set(), [], {start: None}

    def visit(u: str) -> None:
        visited.add(u)
        order.append(u)
        for v in graph.neighbours(u):
            if v not in visited:
                parent[v] = u
                visit(v)

    visit(start)
    return order, parent


def dfs_iterative(graph: Graph, start: str):
    """DFS with an explicit LIFO stack. Traces frontier size at every step."""
    visited, order, parent, trace = set(), [], {start: None}, []
    stack = [start]
    while stack:
        trace.append((len(order), list(stack)))
        u = stack.pop()                      # LIFO -- the only line that differs from BFS
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        # Reversed so that sorted neighbours are popped in ascending order.
        for v in reversed(graph.neighbours(u)):
            if v not in visited:
                parent.setdefault(v, u)
                stack.append(v)
    return order, parent, trace


def bfs(graph: Graph, start: str):
    """BFS with a FIFO queue. Also returns each vertex's distance from the source."""
    visited, order, parent, trace = {start}, [], {start: None}, []
    dist = {start: 0}
    queue = deque([start])
    while queue:
        trace.append((len(order), list(queue)))
        u = queue.popleft()                  # FIFO -- the only line that differs from DFS
        order.append(u)
        for v in graph.neighbours(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                dist[v] = dist[u] + 1
                queue.append(v)
    return order, parent, dist, trace


dfs_order_rec, dfs_parent_rec = dfs_recursive(g, "A")
dfs_order, dfs_parent, dfs_trace = dfs_iterative(g, "A")
bfs_order, bfs_parent, bfs_dist, bfs_trace = bfs(g, "A")

banner("TRAVERSAL FROM VERTEX A")
print(f"DFS (recursive) : {' -> '.join(dfs_order_rec)}")
print(f"DFS (iterative) : {' -> '.join(dfs_order)}")
print(f"BFS             : {' -> '.join(bfs_order)}")
print()
print("BFS distance from A:", {k: bfs_dist[k] for k in sorted(bfs_dist)})
assert set(dfs_order) == set(bfs_order) == set(g.nodes), "a traversal missed a vertex"
print("\nBoth traversals visited all", len(g.nodes), "vertices.")

# %% [markdown]
# ### Step-by-step frontier trace
#
# This is the heart of the difference. At every iteration we print the frontier *before*
# a vertex is removed. Watch the DFS stack stay narrow and deep while the BFS queue
# fans out level by level.

# %%
rows = []
for i in range(max(len(dfs_trace), len(bfs_trace))):
    d = dfs_trace[i] if i < len(dfs_trace) else ("", [])
    b = bfs_trace[i] if i < len(bfs_trace) else ("", [])
    rows.append({
        "step": i + 1,
        "DFS stack (top at right)": " ".join(d[1]),
        "#visited": d[0],
        "BFS queue (front at left)": " ".join(b[1]),
        "#visited ": b[0],
    })
trace_df = pd.DataFrame(rows)
banner("FRONTIER EVOLUTION")
print(trace_df.to_string(index=False))
print("\nNote: a vertex can appear twice on the DFS stack -- it was pushed by two different")
print("neighbours before being popped. The `if u in visited: continue` guard discards the")
print("stale copy, which is why the stack is longer than the number of distinct vertices.")

# %% [markdown]
# ### Shortest path via BFS
#
# Because BFS labels each vertex with its true hop-distance, walking the `parent` map
# backwards from any target reconstructs a minimum-edge path. DFS's parent map does not
# have this property — shown below, where DFS finds a longer A→J route.

# %%
def path_from_parents(parent: dict[str, str | None], target: str) -> list[str]:
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


bfs_path = path_from_parents(bfs_parent, "J")
dfs_path = path_from_parents(dfs_parent, "J")

banner("PATH FROM A TO J")
print(f"BFS path : {' -> '.join(bfs_path)}   ({len(bfs_path)-1} edges)  <- guaranteed minimum")
print(f"DFS path : {' -> '.join(dfs_path)}   ({len(dfs_path)-1} edges)")
print(f"\nBFS distance to J = {bfs_dist['J']}, and the reconstructed path has "
      f"{len(bfs_path)-1} edges -- consistent.")

# %% [markdown]
# ### Visualising the two search trees
#
# Three panels: the graph itself; the DFS tree with vertices numbered and coloured by
# visit order; and the BFS tree, where colour encodes distance from the source, making
# the level structure obvious.

# %%
import networkx as nx

G = nx.Graph()
G.add_edges_from(g.edges)
pos = nx.spring_layout(G, seed=SEED, k=0.9)

def tree_edges(parent: dict[str, str | None]) -> list[tuple[str, str]]:
    return [(p, c) for c, p in parent.items() if p is not None]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

# Panel 1 -- the raw graph
nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="#9aa4b2", width=1.8)
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color="#dfe5ee",
                       edgecolors="#2A4B9B", linewidths=1.8, node_size=900)
nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=12, font_weight="bold")
axes[0].set_title(f"The graph\n{len(g.nodes)} vertices, {len(g.edges)} edges")

# Panels 2 and 3 -- the search trees
for ax, order, parent, name, cmap, label in [
    (axes[1], dfs_order, dfs_parent, "DFS", "Oranges", "visit order"),
    (axes[2], bfs_order, bfs_parent, "BFS", "Blues", "distance from A"),
]:
    values = ([order.index(n) for n in G.nodes()] if name == "DFS"
              else [bfs_dist[n] for n in G.nodes()])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#d7dce3", width=1.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=tree_edges(parent),
                           edge_color="#20304d", width=3.0)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=values, cmap=plt.get_cmap(cmap),
                                   edgecolors="#20304d", linewidths=1.8, node_size=980,
                                   vmin=min(values) - 0.6, vmax=max(values) + 0.2)
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=11, font_weight="bold",
        labels={n: f"{n}\n{order.index(n)+1}" if name == "DFS" else f"{n}\nd={bfs_dist[n]}"
                for n in G.nodes()})
    ax.set_title(f"{name} tree (bold edges)\n{' '.join(order)}")
    plt.colorbar(nodes, ax=ax, fraction=0.04, pad=0.02, label=label)

for ax in axes:
    ax.set_axis_off()
save_fig("q1_traversal_trees", fig)

# %% [markdown]
# ### Frontier growth
#
# Plotting frontier size per step quantifies the space trade-off from the theory table:
# DFS's stack stays shallow, BFS's queue bulges as it holds an entire level at once.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

axes[0].plot([len(f) for _, f in dfs_trace], "o-", color=PALETTE[1], label="DFS stack")
axes[0].plot([len(f) for _, f in bfs_trace], "s-", color=PALETTE[0], label="BFS queue")
axes[0].set(xlabel="iteration", ylabel="frontier size",
            title="Frontier size per iteration")
axes[0].legend()

levels = Counter(bfs_dist.values())
axes[1].bar([f"d={d}" for d in sorted(levels)], [levels[d] for d in sorted(levels)],
            color=PALETTE[0])
axes[1].set(xlabel="distance from A", ylabel="vertices",
            title="BFS level structure\n(max level width = max queue size)")
for i, d in enumerate(sorted(levels)):
    axes[1].text(i, levels[d] + 0.05, str(levels[d]), ha="center", fontweight="bold")
save_fig("q1_frontier_growth", fig)

# %% [markdown]
# ### Scaling check on a larger random graph
#
# Two separate questions, measured separately.
#
# **(a) Time.** On a 4,000-vertex random graph, does runtime track $O(V+E)$?
#
# **(b) Space.** The textbook claim is that on a *balanced tree* BFS needs $O(b^d)$ memory
# while DFS needs only $O(b\,m)$. A random graph is the wrong shape to show this — its
# levels are irregular, and DFS can pile up a large stack of pushed-but-unvisited
# vertices. A complete $b$-ary tree is the shape the claim is actually about, so that is
# what is measured here.

# %%
rng = np.random.default_rng(SEED)
big = Graph()
N = 4000
for i in range(1, N):                       # random tree => connected
    big.add_edge(f"n{rng.integers(0, i)}", f"n{i}")
for _ in range(2000):                       # extra edges => cycles
    a, b = rng.integers(0, N, 2)
    if a != b:
        big.add_edge(f"n{a}", f"n{b}")

(do, dp, dt), t_dfs = timed(dfs_iterative, big, "n0")
(bo, bp, bd, bt), t_bfs = timed(bfs, big, "n0")

banner(f"(a) TIME on a random graph: {len(big.nodes)} vertices, {len(big.edges)} edges")
print(pd.DataFrame([
    {"Algorithm": "DFS (iterative)", "Visited": len(do), "Time (ms)": round(t_dfs * 1000, 2)},
    {"Algorithm": "BFS", "Visited": len(bo), "Time (ms)": round(t_bfs * 1000, 2)},
]).to_string(index=False))
print(f"\nBoth visited every vertex; deepest BFS level = {max(bd.values())}.")


def bary_tree(b: int, depth: int) -> Graph:
    """Complete b-ary tree, the shape the O(b^d) vs O(b*m) claim is about."""
    t, frontier, nid = Graph(), ["r"], 0
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for _ in range(b):
                nid += 1
                t.add_edge(u, f"v{nid}")
                nxt.append(f"v{nid}")
        frontier = nxt
    return t

space_rows = []
for b, depth in [(2, 10), (3, 7), (4, 6)]:
    t = bary_tree(b, depth)
    _, _, dtr = dfs_iterative(t, "r")
    _, _, _, btr = bfs(t, "r")
    space_rows.append({
        "tree": f"b={b}, depth={depth}", "vertices": len(t.nodes),
        "DFS peak stack": max(len(f) for _, f in dtr),
        "BFS peak queue": max(len(f) for _, f in btr),
        "ratio BFS/DFS": round(max(len(f) for _, f in btr) / max(len(f) for _, f in dtr), 1),
    })
space_df = pd.DataFrame(space_rows)
banner("(b) SPACE on complete b-ary trees")
print(space_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(space_df))
ax.bar(x - 0.2, space_df["DFS peak stack"], 0.4, label="DFS peak stack", color=PALETTE[1])
ax.bar(x + 0.2, space_df["BFS peak queue"], 0.4, label="BFS peak queue", color=PALETTE[0])
ax.set(xticks=x, yscale="log", ylabel="peak frontier (log scale)",
       title="Peak memory on complete $b$-ary trees")
ax.set_xticklabels(space_df["tree"])
for i, r in space_df.iterrows():
    ax.text(i + 0.2, r["BFS peak queue"] * 1.1, f"{r['ratio BFS/DFS']}x", ha="center",
            fontweight="bold", fontsize=9)
ax.legend()
save_fig("q1_space_comparison", fig)

# %% [markdown]
# ## Result and discussion
#
# * **Correctness.** Both traversals visited all 9 vertices. The recursive and the
#   iterative DFS produced *identical* visit orders
#   (`A B D E F J C H G`), which is the check that an explicit stack faithfully
#   reproduces the call stack.
#
# * **Order.** DFS went `A B D E F J C H G`: it committed to B and drove all the way down
#   the B–D–E–F chain before ever looking at C. BFS went `A B C D G J E H F` — strictly
#   non-decreasing in distance from A (0, 1, 1, 2, 2, 2, 3, 3, 3).
#
# * **Shortest path — the headline result.** BFS reached J in **2 edges** (A→C→J); DFS
#   reached the same vertex in **5** (A→B→D→E→F→J). Both are valid paths; only BFS's is
#   minimal. This is the predicted failure of DFS, and it is why BFS underlies unweighted
#   shortest-path routing while DFS does not.
#
# * **Space.** On complete $b$-ary trees the BFS queue peaked at **93× to 216×** the DFS
#   stack (1,024 vs 11 at $b{=}2$; 4,096 vs 19 at $b{=}4$), and the gap widened as $b$
#   grew — the $O(b^d)$ versus $O(b\,m)$ distinction, measured.
#
#   Worth recording honestly: on the 4,000-vertex *random* graph this ordering reverses —
#   DFS's stack peaked higher than BFS's queue, because that graph has no clean level
#   structure and DFS accumulates pushed-but-unvisited vertices. The textbook bound
#   describes balanced trees, not arbitrary graphs, and only the tree measurement is
#   evidence for it.
#
# **Conclusion.** Use BFS when the fewest hops matter and the frontier fits in memory;
# use DFS when the graph is deep, memory is tight, and any path (or mere reachability)
# will do.
