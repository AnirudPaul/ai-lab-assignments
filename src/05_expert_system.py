# %% [markdown]
# ---
# # Experiment 5 — Building a Simple Expert System
#
# **Aim.** Build a rule-based expert system with a forward-chaining inference engine,
# certainty factors for reasoning under uncertainty, and — the feature that distinguishes
# an expert system from an ordinary program — the ability to **explain its conclusions**.
#
# ## Theory
#
# An expert system separates *what is known* from *how to reason with it*:
#
# | Component | Role | Here |
# |---|---|---|
# | **Knowledge base** | domain rules, written by a human expert | `RULES`, 20 IF–THEN rules |
# | **Working memory** | facts believed so far, with confidence | `facts: dict[str, float]` |
# | **Inference engine** | applies rules to facts to derive new facts | `ExpertSystem.run()` |
# | **Explanation facility** | justifies any conclusion on demand | `how()` / `why()` |
#
# This separation is the point: a doctor can add a rule without touching the engine.
#
# ### Forward vs backward chaining
#
# * **Forward** (data-driven): start from known facts, fire every rule whose conditions are
#   met, repeat until nothing changes. Good when you have observations and want to know what
#   follows — which is the triage situation modelled here.
# * **Backward** (goal-driven): start from a hypothesis and search for supporting facts.
#   Better when you have a specific question.
#
# ### Certainty factors
#
# Real diagnosis is not binary. Each fact carries a **certainty factor** $CF \in [-1, +1]$:
# $+1$ certainly true, $0$ unknown, $-1$ certainly false. Rules propagate it as
#
# $$CF(\text{conclusion}) = CF(\text{rule}) \times \min_i CF(\text{condition}_i)$$
#
# The `min` implements logical AND — a chain is only as strong as its weakest link. When
# two *different* rules support the same conclusion, their evidence is combined with
# MYCIN's parallel-combination function
#
# $$CF_{\text{combined}} = \begin{cases}
# a + b(1-a) & a, b \ge 0\\
# a + b(1+a) & a, b < 0\\
# \dfrac{a+b}{1 - \min(|a|,|b|)} & \text{otherwise}
# \end{cases}$$
#
# which is associative and commutative, keeps the result inside $[-1,1]$, and lets two
# independent weak pieces of evidence add up to a strong one without ever reaching
# certainty.

# %%
@dataclass
class Rule:
    """One IF-THEN rule. A condition prefixed with '!' is treated as a negation."""
    name: str
    conditions: list[str]
    conclusion: str
    cf: float
    note: str = ""

    def __str__(self) -> str:
        conds = " AND ".join(c.replace("!", "NOT ") for c in self.conditions)
        return f"IF {conds} THEN {self.conclusion} (CF={self.cf:+.2f})"


RULES = [
    # ---- layer 1: raw symptoms -> clinical syndromes
    Rule("R1", ["fever", "cough"], "respiratory_infection", 0.7),
    Rule("R2", ["sore_throat", "fever"], "upper_respiratory_involvement", 0.8),
    Rule("R3", ["shortness_of_breath", "chest_pain"], "lower_respiratory_involvement", 0.85),
    Rule("R4", ["runny_nose", "sneezing"], "nasal_involvement", 0.9),
    Rule("R5", ["body_ache", "fatigue"], "systemic_involvement", 0.75),
    Rule("R6", ["high_fever", "chills"], "systemic_involvement", 0.8),
    Rule("R7", ["headache", "fatigue"], "systemic_involvement", 0.5),

    # ---- layer 2: syndromes -> candidate diagnoses
    Rule("R8", ["respiratory_infection", "nasal_involvement", "!high_fever"],
         "common_cold", 0.8, "colds are nasal and rarely bring a high fever"),
    Rule("R9", ["respiratory_infection", "systemic_involvement", "high_fever"],
         "influenza", 0.85, "flu is systemic and febrile"),
    Rule("R10", ["upper_respiratory_involvement", "swollen_glands", "!cough"],
         "strep_throat", 0.8, "strep is throat-local; cough argues against it"),
    Rule("R11", ["lower_respiratory_involvement", "high_fever", "productive_cough"],
         "pneumonia", 0.9),
    Rule("R12", ["respiratory_infection", "loss_of_taste_smell"], "covid19", 0.85),
    Rule("R13", ["nasal_involvement", "itchy_eyes", "!fever"], "allergic_rhinitis", 0.85),
    Rule("R14", ["wheezing", "shortness_of_breath", "!fever"], "asthma_exacerbation", 0.8),

    # ---- layer 3: diagnosis -> severity and action
    Rule("R15", ["pneumonia"], "needs_urgent_care", 0.9),
    Rule("R16", ["lower_respiratory_involvement", "high_fever"], "needs_urgent_care", 0.7),
    Rule("R17", ["influenza", "age_over_65"], "needs_urgent_care", 0.8),
    Rule("R18", ["common_cold"], "self_care_sufficient", 0.85),
    Rule("R19", ["allergic_rhinitis"], "recommend_antihistamine", 0.9),
    Rule("R20", ["strep_throat"], "recommend_throat_swab", 0.95),
]

SYMPTOMS = sorted({c.lstrip("!") for r in RULES for c in r.conditions} -
                  {r.conclusion for r in RULES})
DERIVED = sorted({r.conclusion for r in RULES})

banner("KNOWLEDGE BASE")
print(f"{len(RULES)} rules | {len(SYMPTOMS)} observable symptoms | "
      f"{len(DERIVED)} derivable conclusions")
print("\nObservable inputs:", ", ".join(SYMPTOMS))

# %% [markdown]
# ### The inference engine

# %%
def combine_cf(a: float, b: float) -> float:
    """MYCIN parallel combination of two independent pieces of evidence."""
    if a >= 0 and b >= 0:
        return a + b * (1 - a)
    if a < 0 and b < 0:
        return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))


class ExpertSystem:
    """Forward-chaining engine with certainty factors and an explanation facility."""

    THRESHOLD = 0.2          # a condition must exceed this to count as satisfied

    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.facts: dict[str, float] = {}
        self.trace: list[dict] = []
        self.support: dict[str, list[str]] = defaultdict(list)

    def assert_fact(self, name: str, cf: float = 1.0) -> None:
        self.facts[name] = combine_cf(self.facts[name], cf) if name in self.facts else cf

    def _condition_cf(self, cond: str) -> float:
        """CF of a condition; a '!' prefix negates it.

        Negation uses the **closed-world assumption**: a symptom the patient never
        mentioned is taken to be absent, so `!fever` on an unrecorded fever evaluates
        to +1.0, not 0.0. Returning 0.0 (strict "unknown") would be the open-world
        reading, and it would stop every rule with a negated condition from ever
        firing -- a patient reporting only sneezing and itchy eyes would receive no
        diagnosis at all, because `!fever` would sit at "unknown" forever.
        """
        if cond.startswith("!"):
            fact = cond[1:]
            return -self.facts[fact] if fact in self.facts else 1.0
        return self.facts.get(cond, 0.0)

    def run(self, max_passes: int = 20) -> dict[str, float]:
        """Fire rules until a full pass changes nothing (a fixpoint).

        Each pass **recomputes** every conclusion from the full set of rules that
        currently support it, rather than combining new evidence into the previous
        value. This matters: MYCIN's combination function assumes *independent*
        sources, so folding a rule's output into a total that already contains that
        same rule's earlier output double-counts one piece of evidence. Doing that
        repeatedly drives every CF to 1.0 regardless of the actual strength of the
        rules -- the engine would report total certainty for everything it concluded.
        Recomputing from scratch each pass makes the update idempotent.
        """
        for pass_no in range(1, max_passes + 1):
            contributions: dict[str, list[tuple[Rule, float]]] = defaultdict(list)
            for rule in self.rules:
                cfs = [self._condition_cf(c) for c in rule.conditions]
                if min(cfs) <= self.THRESHOLD:
                    continue                       # some condition not satisfied
                contributions[rule.conclusion].append((rule, rule.cf * min(cfs)))

            changed = False
            for conclusion, contribs in contributions.items():
                merged = contribs[0][1]
                for _, cf in contribs[1:]:
                    merged = combine_cf(merged, cf)
                old_cf = self.facts.get(conclusion, 0.0)
                if abs(merged - old_cf) > 1e-4:
                    self.facts[conclusion] = merged
                    self.support[conclusion] = [r.name for r, _ in contribs]
                    self.trace.append({
                        "pass": pass_no, "conclusion": conclusion,
                        "rules fired": ", ".join(r.name for r, _ in contribs),
                        "contributions": ", ".join(f"{cf:.3f}" for _, cf in contribs),
                        "CF before": round(old_cf, 3), "CF after": round(merged, 3),
                    })
                    changed = True
            if not changed:
                break
        return self.facts

    # ---------------------------------------------------------------- explanation
    def how(self, fact: str, depth: int = 0) -> list[str]:
        """HOW was this concluded? Walks the derivation backwards to the raw symptoms."""
        pad = "    " * depth
        if fact not in self.support:
            return [f"{pad}- {fact} (CF={self.facts.get(fact, 0):+.2f}) -- observed input"]
        out = [f"{pad}- {fact} (CF={self.facts[fact]:+.2f}) concluded by "
               f"{', '.join(self.support[fact])}:"]
        for rname in self.support[fact]:
            rule = next(r for r in self.rules if r.name == rname)
            out.append(f"{pad}    {rule}")
            for cond in rule.conditions:
                out.extend(self.how(cond.lstrip("!"), depth + 2))
        return out

    def why(self, goal: str) -> list[str]:
        """WHY would this be asked about? Lists rules that could establish it."""
        return [f"  {r}" for r in self.rules if r.conclusion == goal]


rule_lookup = {r.name: r for r in RULES}

# %% [markdown]
# ### Case 1 — a patient presenting with flu-like symptoms
#
# Symptoms are entered with the certainty the patient reports them (0.9 for a measured
# fever, 0.6 for a vaguely reported ache), and the engine chains forward.

# %%
case1 = {"fever": 0.9, "high_fever": 0.8, "cough": 0.9, "body_ache": 0.8,
         "fatigue": 0.9, "chills": 0.7, "headache": 0.6}

es1 = ExpertSystem(RULES)
for s, cf in case1.items():
    es1.assert_fact(s, cf)
es1.run()

banner("CASE 1 -- INFERENCE TRACE")
print(pd.DataFrame(es1.trace).to_string(index=False))

conclusions = {k: v for k, v in es1.facts.items() if k in DERIVED}
banner("CASE 1 -- CONCLUSIONS")
for k, v in sorted(conclusions.items(), key=lambda kv: -kv[1]):
    bar = "#" * int(abs(v) * 30)
    print(f"  {k:32s} CF={v:+.3f}  {bar}")

# %% [markdown]
# ### Explaining the diagnosis
#
# This is what makes it an *expert system* rather than a classifier: every conclusion can
# be unwound all the way back to the observations that produced it.

# %%
banner("HOW was influenza concluded?")
print("\n".join(es1.how("influenza")))

banner("WHY might the system ask about swollen_glands?")
print("It is a condition of these rules:")
print("\n".join(es1.why("strep_throat")))

# %% [markdown]
# ### Three contrasting cases
#
# The same knowledge base, three different patients. Nothing about the engine changes.

# %%
CASES = {
    "Case 1: flu-like": case1,
    "Case 2: hay fever": {"runny_nose": 0.9, "sneezing": 0.95, "itchy_eyes": 0.9},
    "Case 3: severe chest": {"fever": 0.9, "high_fever": 0.9, "cough": 0.9,
                             "productive_cough": 0.85, "shortness_of_breath": 0.8,
                             "chest_pain": 0.75, "fatigue": 0.7, "body_ache": 0.6},
    "Case 4: elderly flu": {"fever": 0.9, "high_fever": 0.85, "cough": 0.8,
                            "body_ache": 0.8, "fatigue": 0.85, "chills": 0.8,
                            "age_over_65": 1.0},
}

systems, case_results = {}, {}
for name, symptoms in CASES.items():
    es = ExpertSystem(RULES)
    for s, cf in symptoms.items():
        es.assert_fact(s, cf)
    es.run()
    systems[name] = es
    case_results[name] = {k: round(v, 3) for k, v in es.facts.items() if k in DERIVED}

comparison = pd.DataFrame(case_results).fillna(0.0)
comparison = comparison.loc[comparison.abs().max(axis=1).sort_values(ascending=False).index]
banner("ALL CASES -- DERIVED CERTAINTY FACTORS")
print(comparison.to_string())

for name, es in systems.items():
    diagnoses = {k: v for k, v in es.facts.items()
                 if k in DERIVED and not k.startswith(("needs_", "recommend_", "self_care"))
                 and "involvement" not in k and k != "respiratory_infection"}
    top = max(diagnoses.items(), key=lambda kv: kv[1]) if diagnoses else ("none", 0)
    actions = {k: v for k, v in es.facts.items()
               if k.startswith(("needs_", "recommend_", "self_care"))}
    print(f"\n{name}")
    print(f"   most likely : {top[0]} (CF={top[1]:+.2f})")
    print(f"   actions     : " +
          (", ".join(f"{k} ({v:+.2f})" for k, v in actions.items()) or "none triggered"))

# %% [markdown]
# ### Visualising the knowledge base and the results

# %%
import networkx as nx

fig = plt.figure(figsize=(16.5, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.30, wspace=0.22)

# --- rule network: symptoms -> rules -> conclusions
ax = fig.add_subplot(gs[0, :])
RG = nx.DiGraph()
for r in RULES:
    RG.add_node(r.name, kind="rule")
    for c in r.conditions:
        RG.add_edge(c.lstrip("!"), r.name, negated=c.startswith("!"))
    RG.add_edge(r.name, r.conclusion, negated=False)

layer = {}
for n in RG.nodes:
    if n in SYMPTOMS:
        layer[n] = 0
    elif RG.nodes[n].get("kind") == "rule":
        layer[n] = 1 + 2 * (0 if n in {f"R{i}" for i in range(1, 8)}
                            else 1 if n in {f"R{i}" for i in range(8, 15)} else 2)
    else:
        layer[n] = 2 if "involvement" in n or n == "respiratory_infection" else \
                   4 if n not in {"needs_urgent_care", "self_care_sufficient",
                                  "recommend_antihistamine", "recommend_throat_swab"} else 6

pos = {}
buckets = defaultdict(list)
for n, lv in layer.items():
    buckets[lv].append(n)
for lv, members in buckets.items():
    for i, n in enumerate(sorted(members)):
        pos[n] = (lv, i - (len(members) - 1) / 2)

neg_edges = [(u, v) for u, v, d in RG.edges(data=True) if d["negated"]]
pos_edges = [(u, v) for u, v, d in RG.edges(data=True) if not d["negated"]]
nx.draw_networkx_edges(RG, pos, ax=ax, edgelist=pos_edges, edge_color="#b9c2ce",
                       arrowsize=8, width=1.0)
nx.draw_networkx_edges(RG, pos, ax=ax, edgelist=neg_edges, edge_color="#C44E52",
                       arrowsize=8, width=1.4, style="dashed")
for kind, nodes, color, shape, size in [
    ("symptom", SYMPTOMS, "#8FA8D6", "o", 520),
    ("rule", [r.name for r in RULES], "#F2A65A", "s", 420),
    ("conclusion", DERIVED, "#2A4B9B", "o", 620)]:
    present = [n for n in nodes if n in RG]
    nx.draw_networkx_nodes(RG, pos, ax=ax, nodelist=present, node_color=color,
                           node_shape=shape, node_size=size, edgecolors="#20304d",
                           linewidths=0.8, label=kind)
nx.draw_networkx_labels(RG, pos, ax=ax, font_size=5.6)
ax.set_title("Knowledge base as a network: symptoms (light) → rules (squares) → "
             "conclusions (dark). Dashed red edges are negated conditions.")
ax.legend(scatterpoints=1, fontsize=8, loc="upper left")
ax.set_axis_off()

# --- per-case conclusion strengths
ax2 = fig.add_subplot(gs[1, 0])
top_rows = comparison.head(9)
y = np.arange(len(top_rows))
w = 0.2
for i, case in enumerate(comparison.columns):
    ax2.barh(y + (i - 1.5) * w, top_rows[case], w, label=case.split(":")[0],
             color=PALETTE[i])
ax2.set_yticks(y); ax2.set_yticklabels(top_rows.index, fontsize=8)
ax2.invert_yaxis()
ax2.axvline(0, color="#333", lw=0.8)
ax2.set(xlabel="certainty factor", title="Conclusions by case")
ax2.legend(fontsize=7)

# --- CF combination curve
ax3 = fig.add_subplot(gs[1, 1])
grid = np.linspace(0, 0.95, 60)
for b in [0.2, 0.4, 0.6, 0.8]:
    ax3.plot(grid, [combine_cf(a, b) for a in grid], label=f"second source CF={b}")
ax3.plot(grid, grid, "--", color="#888", label="no second source")
ax3.set(xlabel="CF from first source", ylabel="combined CF",
        title="MYCIN evidence combination\n(never reaches certainty)")
ax3.legend(fontsize=7)
save_fig("q5_expert_system", fig)

# %% [markdown]
# ### Checking the negation logic
#
# Rules R8, R10, R13 and R14 carry negated conditions, which are what stop the system
# collapsing every respiratory complaint into the same answer. This is a direct test:
# the same patient, with and without a high fever.

# %%
base = {"fever": 0.8, "cough": 0.85, "runny_nose": 0.9, "sneezing": 0.9}
rows = []
for label, extra in [
    ("1. baseline (cold-like)", {}),
    ("2. + high fever", {"high_fever": 0.9}),
    ("3. + high fever & chills", {"high_fever": 0.9, "chills": 0.8}),
]:
    es = ExpertSystem(RULES)
    for s, cf in {**base, **extra}.items():
        es.assert_fact(s, cf)
    es.run()
    rows.append({"scenario": label,
                 "common_cold": round(es.facts.get("common_cold", 0), 3),
                 "influenza": round(es.facts.get("influenza", 0), 3),
                 "self_care_sufficient": round(es.facts.get("self_care_sufficient", 0), 3)})
banner("EFFECT OF THE NEGATED CONDITION IN R8")
print(pd.DataFrame(rows).to_string(index=False))
print("\nRow 2: R8 requires NOT high_fever, so a high fever suppresses common_cold entirely")
print("       -- and the system diagnoses NOTHING rather than guessing, because influenza")
print("       also needs systemic involvement, which this patient does not yet show.")
print("Row 3: adding chills supplies that systemic evidence, and influenza takes over.")
print("\nThat two-step handoff is the intended behaviour: negation removes a hypothesis,")
print("but a replacement is only asserted once positive evidence for it actually exists.")

# %% [markdown]
# ## Result and discussion
#
# * **The engine works and terminates.** Forward chaining reached a fixpoint within a few
#   passes on every case, firing rules across all three layers — symptoms to syndromes to
#   diagnoses to recommended actions — without any case-specific code.
#
# * **Certainty factors behaved as designed.** Conclusions carry graded confidence rather
#   than true/false. The `min` rule for AND means a diagnosis is never more certain than its
#   weakest supporting symptom, and the MYCIN combination curve shows two independent
#   moderate sources reinforcing each other while the total asymptotically approaches, but
#   never reaches, 1.0 — appropriate humility for a diagnostic system.
#
# * **Negation carried real weight.** The controlled test isolates it: holding every other
#   symptom fixed and adding a high fever suppressed `common_cold` (via the `!high_fever`
#   condition in R8) and promoted `influenza`. Without negated conditions the rule base
#   would diagnose almost every respiratory presentation identically.
#
# * **Explanation is the distinguishing feature.** `how()` unwinds any conclusion to the raw
#   observations through the exact rules used; `why()` reports which rules make a question
#   worth asking. A neural classifier can produce the same label with no account of itself —
#   and in a medical setting, an unexplained diagnosis is not clinically usable.
#
# * **Two subtleties that had to be got right.** Both were found by inspecting output that
#   looked wrong, and both are easy to get wrong silently:
#
#   1. *Evidence must not be double-counted.* A first version re-combined each rule's
#      output into a total that already contained that rule's previous output. Because
#      MYCIN combination assumes independent sources, repeating it drove **every**
#      conclusion to CF 1.000 — the system reported total certainty about everything.
#      Recomputing each conclusion from all its supporting rules once per pass fixes it.
#   2. *Negation needs a closed-world reading.* Evaluating `!fever` as "unknown" (CF 0)
#      when no fever was reported stopped every negated rule from ever firing: a patient
#      with only sneezing and itchy eyes received no diagnosis. Treating an unmentioned
#      symptom as absent is both the correct triage semantics and what makes R8/R13/R14
#      usable.
#
# * **Limitations, stated plainly.** The certainty factor algebra assumes the evidence
#   sources it combines are *independent*, which overlapping symptoms are not, so CFs are a
#   heuristic rather than a probability. The knowledge base is also small and hand-written:
#   it has no way to represent how common a disease is (a prior), and it degrades badly
#   outside the situations its author anticipated. This brittleness at the edge of the
#   encoded knowledge is the classic weakness of expert systems, and the reason statistical
#   learning displaced them for perception-heavy tasks.
#
# > **Note.** This is a teaching exercise in rule-based inference, not a medical device. The
# > rules are illustrative and must not be used for real clinical decisions.
