# This script isolates the Moran selection effect by decoupling treatment
# choice from agent beliefs. Agents administer treatments with 50/50
# probability regardless of their credence. Bayesian updating and the
# Moran process operate as in the main model.
#
# Two tests:
#   Test A: Does the peripheral incubation effect persist when treatment
#           choice is randomized?
#   Test B: Is there a funding level so high that the peripheral advantage
#           disappears?
# ============================================================================

import subprocess
subprocess.run(['pip', 'install', 'networkx', 'tqdm', 'pingouin', 'joblib', '-q'])

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from dataclasses import dataclass
from enum import Enum
import time
from joblib import Parallel, delayed
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Reviewer #2 ablation setup complete!")

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_P7 = {
    "prior_alpha": 1,
    "prior_beta": 1,
    "convergence_threshold": 0.99,
}

ASYMMETRIC_TOPOLOGIES = ["star", "wheel", "line", "hierarchical",
                         "clustered", "scale_free", "small_world", "random"]
SYMMETRIC_TOPOLOGIES = ["complete", "cycle"]

DEFAULT_METH_BIAS_VARIANCE = 0.02
DEFAULT_EXIT_PROB = 0.20
DEFAULT_INDUSTRY_THRESHOLD = 0.10
DEFAULT_BASE_PRODUCTIVITY_MIN = 1
DEFAULT_BASE_PRODUCTIVITY_MAX = 5
DEFAULT_N_ROUNDS = 2000
DEFAULT_EFFICACY_DIFF = 0.05

# Test A: Standard funding levels
FUNDING_LEVELS_STANDARD = [0, 5, 10, 15, 20, 30, 50, 75, 100]

# Test B: Extreme funding levels to find the disappearance threshold
FUNDING_LEVELS_EXTREME = [100, 200, 500, 750, 1000]

SEEDS = 200

# ============================================================================
# AGENT TYPES & DATA CLASSES
# ============================================================================
class P7AgentType(Enum):
    TRUTH_SEEKER = "truth_seeker"
    INDUSTRIAL_SELECTION = "industrial_selection"

@dataclass
class P7Agent:
    agent_id: int
    agent_type: P7AgentType
    alpha: float = 1.0
    beta: float = 1.0
    personal_p_A: float = 0.55
    personal_p_B: float = 0.50
    base_productivity: int = 10
    funding_boost: int = 0
    is_funded: bool = False

    @property
    def credence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def brier_score(self) -> float:
        return (self.credence - 1.0)**2

    @property
    def effective_productivity(self) -> int:
        return self.base_productivity + self.funding_boost

# ============================================================================
# NETWORK FUNCTIONS
# ============================================================================
def create_network(topology, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if topology == "star":
        return nx.star_graph(n - 1)
    elif topology == "wheel":
        return nx.wheel_graph(n)
    elif topology == "cycle":
        return nx.cycle_graph(n)
    elif topology == "complete":
        return nx.complete_graph(n)
    elif topology == "line":
        return nx.path_graph(n)
    elif topology == "hierarchical":
        G = nx.Graph(); G.add_nodes_from(range(n))
        for i in range(1, n): G.add_edge((i - 1) // 2, i)
        return G
    elif topology == "clustered":
        G = nx.Graph(); G.add_nodes_from(range(n)); half = n // 2
        for i in range(half):
            for j in range(i + 1, half): G.add_edge(i, j)
        for i in range(half, n):
            for j in range(i + 1, n): G.add_edge(i, j)
        G.add_edge(half - 1, half); return G
    elif topology == "scale_free":
        return nx.barabasi_albert_graph(n, m=min(2, n - 1), seed=seed)
    elif topology == "small_world":
        k = min(4, n - 1)
        if k % 2 == 1: k = max(2, k - 1)
        return nx.watts_strogatz_graph(n, k, p=0.3, seed=seed)
    elif topology == "random":
        G = nx.erdos_renyi_graph(n, 2 * np.log(n) / n, seed=seed)
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(list(comps[i])[0], list(comps[i + 1])[0])
        return G
    raise ValueError(f"Unknown topology: {topology}")

def get_centrality_measures(G):
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        eigenvector = {n: 0.0 for n in G.nodes()}
    return {n: {'degree': degree[n], 'betweenness': betweenness[n],
                'closeness': closeness[n], 'eigenvector': eigenvector[n]}
            for n in G.nodes()}

def get_network_properties(G):
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = -1
    return {'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'diameter': diameter}

def get_high_centrality_positions(G, n_positions=1):
    degree = nx.degree_centrality(G)
    return sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[:n_positions]

def get_low_centrality_positions(G, n_positions=1, exclude=None):
    degree = nx.degree_centrality(G)
    exclude = exclude or []
    candidates = [n for n in G.nodes() if n not in exclude]
    return sorted(candidates, key=lambda x: degree[x])[:n_positions]

# ============================================================================
# CORE MECHANICS
# ============================================================================
def assign_methodological_biases(agents, p_A_true, p_B_true, meth_bias_variance, rng):
    sigma = np.sqrt(meth_bias_variance)
    for agent in agents:
        agent.personal_p_A = np.clip(rng.normal(p_A_true, sigma), 0.01, 0.99)
        agent.personal_p_B = np.clip(rng.normal(p_B_true, sigma), 0.01, 0.99)

def apply_industry_funding(agents, is_agent_positions, industry_threshold, funding_boost, p_B_true):
    for agent in agents:
        if agent.agent_id in is_agent_positions:
            agent.funding_boost = funding_boost
            agent.is_funded = True
            if agent.personal_p_B <= p_B_true + industry_threshold:
                agent.personal_p_B = p_B_true + industry_threshold + 0.01
        else:
            if agent.personal_p_B > p_B_true + industry_threshold:
                agent.funding_boost = funding_boost
                agent.is_funded = True
            else:
                agent.funding_boost = 0
                agent.is_funded = False

def moran_replacement_position_sensitive(agents, adj_list, is_agent_positions, p_A_true, p_B_true, meth_bias_variance, rng):
    replaceable = [a for a in agents if a.agent_id not in is_agent_positions]
    if not replaceable:
        return
    leaving = rng.choice(replaceable)
    leaving_id = leaving.agent_id
    neighbor_ids = set(adj_list[leaving_id])
    neighbor_agents = [a for a in agents if a.agent_id in neighbor_ids]
    if not neighbor_agents:
        neighbor_agents = [a for a in agents if a.agent_id != leaving_id]
    productivities = np.array([a.effective_productivity for a in neighbor_agents], dtype=float)
    if productivities.sum() > 0:
        probs = productivities / productivities.sum()
    else:
        probs = np.ones(len(productivities)) / len(productivities)
    mentor = neighbor_agents[rng.choice(len(neighbor_agents), p=probs)]
    leaving.alpha = CONFIG_P7["prior_alpha"]
    leaving.beta = CONFIG_P7["prior_beta"]
    sigma = np.sqrt(meth_bias_variance) * 0.1
    leaving.personal_p_A = np.clip(rng.normal(mentor.personal_p_A, sigma), 0.01, 0.99)
    leaving.personal_p_B = np.clip(rng.normal(mentor.personal_p_B, sigma), 0.01, 0.99)
    leaving.funding_boost = 0
    leaving.is_funded = False
    leaving.base_productivity = rng.integers(DEFAULT_BASE_PRODUCTIVITY_MIN, DEFAULT_BASE_PRODUCTIVITY_MAX + 1)

# ============================================================================
# STRIPPED-DOWN SIMULATION ENGINE
# ============================================================================
# CHANGE: Treatment choice is randomized (50/50) instead of
# credence-based.
# ============================================================================
def run_stripped_simulation(
    G, topology, n_agents, n_rounds,
    is_agent_positions,
    intended_condition,
    efficacy_difference=DEFAULT_EFFICACY_DIFF,
    funding_boost=50,
    meth_bias_variance=DEFAULT_METH_BIAS_VARIANCE,
    industry_threshold=DEFAULT_INDUSTRY_THRESHOLD,
    exit_prob=DEFAULT_EXIT_PROB,
    enable_moran=True,
    seed=None,
):
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    p_A_true = 0.50 + efficacy_difference
    p_B_true = 0.50
    topology_class = ("symmetric" if topology in SYMMETRIC_TOPOLOGIES else "asymmetric")

    # Optimization 1: Pre-compute adjacency as a boolean matrix
    adj_matrix = nx.to_numpy_array(G, dtype=bool)
    adj_list = [list(G.neighbors(i)) for i in range(n_agents)]  # kept for Moran process

    agents = []
    for i in range(n_agents):
        atype = (P7AgentType.INDUSTRIAL_SELECTION if i in is_agent_positions else P7AgentType.TRUTH_SEEKER)
        base_prod = rng.integers(DEFAULT_BASE_PRODUCTIVITY_MIN, DEFAULT_BASE_PRODUCTIVITY_MAX + 1)
        agents.append(P7Agent(
            agent_id=i, agent_type=atype,
            alpha=CONFIG_P7["prior_alpha"], beta=CONFIG_P7["prior_beta"],
            base_productivity=base_prod,
        ))

    assign_methodological_biases(agents, p_A_true, p_B_true, meth_bias_variance, rng)
    apply_industry_funding(agents, is_agent_positions, industry_threshold, funding_boost, p_B_true)

    moran_events = 0

    for round_num in range(n_rounds):
        # Optimization 2: Vectorized evidence gathering
        # Build arrays from agent state
        productivities = np.array([a.effective_productivity for a in agents])
        p_A_arr = np.array([a.personal_p_A for a in agents])
        p_B_arr = np.array([a.personal_p_B for a in agents])

        # ============================================================
        # KEY CHANGE: 50/50 randomized treatment choice
        # Instead of: if agent.credence >= 0.5 → test A, else → test B
        # Now: each agent randomly picks A or B with equal probability
        # ============================================================
        chose_A = rng.random(n_agents) < 0.5  # boolean array
        efficacies = np.where(chose_A, p_A_arr, p_B_arr)
        successes = rng.binomial(productivities, efficacies)

        # Build per-treatment arrays for updating
        A_successes = np.where(chose_A, successes, 0)
        A_trials = np.where(chose_A, productivities, 0)
        B_successes = np.where(~chose_A, successes, 0)
        B_trials = np.where(~chose_A, productivities, 0)

        # Aggregate neighbor evidence via matrix multiply
        neighbor_A_succ = adj_matrix @ A_successes
        neighbor_A_tri = adj_matrix @ A_trials
        neighbor_B_succ = adj_matrix @ B_successes
        neighbor_B_tri = adj_matrix @ B_trials

        # Total evidence = own + neighbors
        total_A_succ = A_successes + neighbor_A_succ
        total_A_tri = A_trials + neighbor_A_tri
        total_B_succ = B_successes + neighbor_B_succ
        total_B_tri = B_trials + neighbor_B_tri

        # Bayesian updating
        for i, agent in enumerate(agents):
            if total_A_tri[i] > 0:
                agent.alpha += total_A_succ[i]
                agent.beta += (total_A_tri[i] - total_A_succ[i])
            if total_B_tri[i] > 0:
                agent.beta += total_B_succ[i]
                agent.alpha += (total_B_tri[i] - total_B_succ[i])

        # Moran process
        if enable_moran and rng.random() < exit_prob:
            moran_replacement_position_sensitive(agents, adj_list, is_agent_positions, p_A_true, p_B_true, meth_bias_variance, rng)
            moran_events += 1
            apply_industry_funding(agents, is_agent_positions, industry_threshold, funding_boost, p_B_true)

    ts_briers = [a.brier_score for a in agents if a.agent_type == P7AgentType.TRUTH_SEEKER]
    if not ts_briers:
        ts_briers = [a.brier_score for a in agents]

    if "high" in intended_condition:
        bc_label = "high"
    elif "low" in intended_condition:
        bc_label = "low"
    else:
        bc_label = "none"

    return {
        'topology': topology, 'topology_class': topology_class,
        'n_agents': n_agents, 'n_rounds': n_rounds,
        'n_biased': len(is_agent_positions),
        'intended_condition': intended_condition,
        'biased_centrality': bc_label,
        'final_mean_brier': np.mean(ts_briers),
        'community_captured': np.mean(ts_briers) > 0.25,
        'funding_boost': funding_boost,
        'enable_moran': enable_moran,
        'total_moran_events': moran_events,
    }

# ============================================================================
# SPLIT-PRODUCTIVITY SIMULATION ENGINE
# ============================================================================
# VARIANT 2: Each agent tests BOTH treatments every round, splitting their
# productivity equally (N/2 trials of A, N/2 trials of B).
# Everything else is identical to the main model.
# ============================================================================
def run_split_simulation(
    G, topology, n_agents, n_rounds,
    is_agent_positions,
    intended_condition,
    efficacy_difference=DEFAULT_EFFICACY_DIFF,
    funding_boost=50,
    meth_bias_variance=DEFAULT_METH_BIAS_VARIANCE,
    industry_threshold=DEFAULT_INDUSTRY_THRESHOLD,
    exit_prob=DEFAULT_EXIT_PROB,
    enable_moran=True,
    seed=None,
):
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    p_A_true = 0.50 + efficacy_difference
    p_B_true = 0.50
    topology_class = ("symmetric" if topology in SYMMETRIC_TOPOLOGIES else "asymmetric")

    adj_matrix = nx.to_numpy_array(G, dtype=bool)
    adj_list = [list(G.neighbors(i)) for i in range(n_agents)]

    agents = []
    for i in range(n_agents):
        atype = (P7AgentType.INDUSTRIAL_SELECTION if i in is_agent_positions else P7AgentType.TRUTH_SEEKER)
        base_prod = rng.integers(DEFAULT_BASE_PRODUCTIVITY_MIN, DEFAULT_BASE_PRODUCTIVITY_MAX + 1)
        agents.append(P7Agent(
            agent_id=i, agent_type=atype,
            alpha=CONFIG_P7["prior_alpha"], beta=CONFIG_P7["prior_beta"],
            base_productivity=base_prod,
        ))

    assign_methodological_biases(agents, p_A_true, p_B_true, meth_bias_variance, rng)
    apply_industry_funding(agents, is_agent_positions, industry_threshold, funding_boost, p_B_true)

    moran_events = 0

    for round_num in range(n_rounds):
        productivities = np.array([a.effective_productivity for a in agents])
        p_A_arr = np.array([a.personal_p_A for a in agents])
        p_B_arr = np.array([a.personal_p_B for a in agents])

        # ============================================================
        # KEY CHANGE: Split productivity equally between both treatments
        # Each agent administers N/2 trials of A and N/2 trials of B
        # ============================================================
        trials_A = productivities // 2
        trials_B = productivities - trials_A  # handles odd productivity

        A_successes = rng.binomial(trials_A, p_A_arr)
        B_successes = rng.binomial(trials_B, p_B_arr)
        A_trials = trials_A
        B_trials = trials_B

        # Aggregate neighbor evidence via matrix multiply
        neighbor_A_succ = adj_matrix @ A_successes
        neighbor_A_tri = adj_matrix @ A_trials
        neighbor_B_succ = adj_matrix @ B_successes
        neighbor_B_tri = adj_matrix @ B_trials

        total_A_succ = A_successes + neighbor_A_succ
        total_A_tri = A_trials + neighbor_A_tri
        total_B_succ = B_successes + neighbor_B_succ
        total_B_tri = B_trials + neighbor_B_tri

        # Bayesian updating
        for i, agent in enumerate(agents):
            if total_A_tri[i] > 0:
                agent.alpha += total_A_succ[i]
                agent.beta += (total_A_tri[i] - total_A_succ[i])
            if total_B_tri[i] > 0:
                agent.beta += total_B_succ[i]
                agent.alpha += (total_B_tri[i] - total_B_succ[i])

        # Moran process
        if enable_moran and rng.random() < exit_prob:
            moran_replacement_position_sensitive(agents, adj_list, is_agent_positions, p_A_true, p_B_true, meth_bias_variance, rng)
            moran_events += 1
            apply_industry_funding(agents, is_agent_positions, industry_threshold, funding_boost, p_B_true)

    ts_briers = [a.brier_score for a in agents if a.agent_type == P7AgentType.TRUTH_SEEKER]
    if not ts_briers:
        ts_briers = [a.brier_score for a in agents]

    if "high" in intended_condition:
        bc_label = "high"
    elif "low" in intended_condition:
        bc_label = "low"
    else:
        bc_label = "none"

    return {
        'topology': topology, 'topology_class': topology_class,
        'n_agents': n_agents, 'n_rounds': n_rounds,
        'n_biased': len(is_agent_positions),
        'intended_condition': intended_condition,
        'biased_centrality': bc_label,
        'final_mean_brier': np.mean(ts_briers),
        'community_captured': np.mean(ts_briers) > 0.25,
        'funding_boost': funding_boost,
        'enable_moran': enable_moran,
        'total_moran_events': moran_events,
    }

# ============================================================================
# WORKER FUNCTIONS
# ============================================================================
def _worker_split(topo, n_agents, cond_name, centrality, fb, seed):
    G = create_network(topo, n_agents, seed)
    is_positions = get_high_centrality_positions(G, 1) if centrality == 'high' else \
                   (get_low_centrality_positions(G, 1, exclude=get_high_centrality_positions(G, 1)) if centrality == 'low' else [])
    enable_m = centrality is not None
    return run_split_simulation(
        G=G, topology=topo, n_agents=n_agents, n_rounds=DEFAULT_N_ROUNDS,
        is_agent_positions=is_positions, intended_condition=cond_name,
        funding_boost=fb, enable_moran=enable_m, seed=seed)

def _worker_stripped(topo, n_agents, cond_name, centrality, fb, seed):
    G = create_network(topo, n_agents, seed)
    is_positions = get_high_centrality_positions(G, 1) if centrality == 'high' else \
                   (get_low_centrality_positions(G, 1, exclude=get_high_centrality_positions(G, 1)) if centrality == 'low' else [])
    enable_m = centrality is not None
    return run_stripped_simulation(
        G=G, topology=topo, n_agents=n_agents, n_rounds=DEFAULT_N_ROUNDS,
        is_agent_positions=is_positions, intended_condition=cond_name,
        funding_boost=fb, enable_moran=enable_m, seed=seed)

# ============================================================================
# TEST A: Stripped-down model with standard funding levels
# Does the peripheral incubation effect persist?
# ============================================================================
def run_test_a(seeds=SEEDS):
    print("\n" + "=" * 70)
    print("TEST A: STRIPPED-DOWN MODEL (50/50 Treatment Choice)")
    print("Does the peripheral incubation effect persist?")
    print("=" * 70)
    tasks = [(topo, n_agents, cond_name, centrality, fb, seed)
             for topo in ASYMMETRIC_TOPOLOGIES for n_agents in [6, 10, 15, 20, 30]
             for cond_name, centrality, fb in
                [("control", None, 0)] +
                [(f"F{fb}_high", "high", fb) for fb in FUNDING_LEVELS_STANDARD] +
                [(f"F{fb}_low", "low", fb) for fb in FUNDING_LEVELS_STANDARD]
             for seed in range(seeds)]
    print(f"  Total simulations: {len(tasks)}")
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_worker_stripped)(*t) for t in tqdm(tasks, desc="Test A"))
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results

# ============================================================================
# TEST B: Extreme funding levels
# Is there a funding level where the peripheral advantage disappears?
# ============================================================================
def run_test_b(seeds=SEEDS):
    print("\n" + "=" * 70)
    print("TEST B: EXTREME FUNDING LEVELS")
    print("Does the peripheral advantage disappear at very high funding?")
    print("=" * 70)
    tasks = [(topo, n_agents, cond_name, centrality, fb, seed)
             for topo in ASYMMETRIC_TOPOLOGIES for n_agents in [6, 10, 15, 20, 30]
             for cond_name, centrality, fb in
                [(f"F{fb}_high", "high", fb) for fb in FUNDING_LEVELS_EXTREME] +
                [(f"F{fb}_low", "low", fb) for fb in FUNDING_LEVELS_EXTREME]
             for seed in range(seeds)]
    print(f"  Total simulations: {len(tasks)}")
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_worker_stripped)(*t) for t in tqdm(tasks, desc="Test B"))
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results

# ============================================================================
# TEST C: Split-productivity model with standard funding levels
# Does the peripheral incubation effect persist under the alternative variant?
# ============================================================================
def run_test_c(seeds=SEEDS):
    print("\n" + "=" * 70)
    print("TEST C: SPLIT-PRODUCTIVITY MODEL (Both Treatments Every Round)")
    print("Does the peripheral incubation effect persist?")
    print("=" * 70)
    tasks = [(topo, n_agents, cond_name, centrality, fb, seed)
             for topo in ASYMMETRIC_TOPOLOGIES for n_agents in [6, 10, 15, 20, 30]
             for cond_name, centrality, fb in
                [("control", None, 0)] +
                [(f"F{fb}_high", "high", fb) for fb in FUNDING_LEVELS_STANDARD] +
                [(f"F{fb}_low", "low", fb) for fb in FUNDING_LEVELS_STANDARD]
             for seed in range(seeds)]
    print(f"  Total simulations: {len(tasks)}")
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_worker_split)(*t) for t in tqdm(tasks, desc="Test C"))
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results

# ============================================================================
# ANALYSIS
# ============================================================================
def cohens_d(g1, g2):
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    return (g2.mean() - g1.mean()) / pooled if pooled > 0 else 0

def analyze_results(results_a, results_b, results_c=None):
    print("\n" + "=" * 70)
    print("ANALYSIS: STRIPPED-DOWN MODEL RESULTS")
    print("=" * 70)

    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)
    df_c = pd.DataFrame(results_c) if results_c is not None else None

    # --- Test A: Does the effect persist? ---
    print("\n" + "─" * 70)
    print("TEST A: PERIPHERAL INCUBATION EFFECT IN STRIPPED-DOWN MODEL")
    print("─" * 70)

    asym = df_a[(df_a['topology_class'] == 'asymmetric') & (df_a['n_biased'] == 1)]
    funding_vals = sorted(asym['funding_boost'].unique())

    print(f"\n  {'Funding':>10} {'Hub M':>10} {'Leaf M':>10} {'Diff':>10} {'d':>8} {'p':>14} {'Sig':>5}")
    print("  " + "─" * 70)
    for fb in funding_vals:
        fsub = asym[asym['funding_boost'] == fb]
        hub = fsub[fsub['biased_centrality'] == 'high']['final_mean_brier']
        leaf = fsub[fsub['biased_centrality'] == 'low']['final_mean_brier']
        if len(hub) > 1 and len(leaf) > 1:
            d = cohens_d(hub, leaf)
            t, p = stats.ttest_ind(hub, leaf)
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            print(f"  {fb:>10} {hub.mean():>10.4f} {leaf.mean():>10.4f} {leaf.mean()-hub.mean():>+10.4f} {d:>8.3f} {p:>14.2e} {sig:>5}")

    # Overall effect across all funding levels > 0
    active = asym[asym['funding_boost'] > 0]
    hub_all = active[active['biased_centrality'] == 'high']['final_mean_brier']
    leaf_all = active[active['biased_centrality'] == 'low']['final_mean_brier']
    if len(hub_all) > 1 and len(leaf_all) > 1:
        d_all = cohens_d(hub_all, leaf_all)
        t_all, p_all = stats.ttest_ind(hub_all, leaf_all)
        print(f"\n  OVERALL (F>0): d={d_all:.3f}, t={t_all:.3f}, p={p_all:.2e}")
        if d_all > 0:
            print("  → PERIPHERAL INCUBATION EFFECT PERSISTS in stripped-down model")
        else:
            print("  → PERIPHERAL INCUBATION EFFECT DOES NOT PERSIST")

    # By topology
    print(f"\n  {'Topology':<16} {'d':>8} {'p':>14}")
    print("  " + "─" * 40)
    for topo in sorted(asym['topology'].unique()):
        tsub = active[active['topology'] == topo]
        hub_t = tsub[tsub['biased_centrality'] == 'high']['final_mean_brier']
        leaf_t = tsub[tsub['biased_centrality'] == 'low']['final_mean_brier']
        if len(hub_t) > 1 and len(leaf_t) > 1:
            d_t = cohens_d(hub_t, leaf_t)
            _, p_t = stats.ttest_ind(hub_t, leaf_t)
            print(f"  {topo:<16} {d_t:>8.3f} {p_t:>14.2e}")

    # --- Test B: Extreme funding levels ---
    print("\n" + "─" * 70)
    print("TEST B: EXTREME FUNDING LEVELS")
    print("─" * 70)

    asym_b = df_b[(df_b['topology_class'] == 'asymmetric') & (df_b['n_biased'] == 1)]
    extreme_vals = sorted(asym_b['funding_boost'].unique())

    print(f"\n  {'Funding':>10} {'Hub M':>10} {'Leaf M':>10} {'Diff':>10} {'d':>8} {'p':>14} {'Direction':>12}")
    print("  " + "─" * 80)
    for fb in extreme_vals:
        fsub = asym_b[asym_b['funding_boost'] == fb]
        hub = fsub[fsub['biased_centrality'] == 'high']['final_mean_brier']
        leaf = fsub[fsub['biased_centrality'] == 'low']['final_mean_brier']
        if len(hub) > 1 and len(leaf) > 1:
            d = cohens_d(hub, leaf)
            t, p = stats.ttest_ind(hub, leaf)
            direction = "Leaf > Hub" if d > 0 else "Hub > Leaf"
            print(f"  {fb:>10} {hub.mean():>10.4f} {leaf.mean():>10.4f} {leaf.mean()-hub.mean():>+10.4f} {d:>8.3f} {p:>14.2e} {direction:>12}")

    # --- Test C: Split-productivity variant ---
    if df_c is not None:
        print("\n" + "─" * 70)
        print("TEST C: SPLIT-PRODUCTIVITY MODEL (Both Treatments Every Round)")
        print("─" * 70)

        asym_c = df_c[(df_c['topology_class'] == 'asymmetric') & (df_c['n_biased'] == 1)]
        funding_vals_c = sorted(asym_c['funding_boost'].unique())

        print(f"\n  {'Funding':>10} {'Hub M':>10} {'Leaf M':>10} {'Diff':>10} {'d':>8} {'p':>14} {'Sig':>5}")
        print("  " + "─" * 70)
        for fb in funding_vals_c:
            fsub = asym_c[asym_c['funding_boost'] == fb]
            hub = fsub[fsub['biased_centrality'] == 'high']['final_mean_brier']
            leaf = fsub[fsub['biased_centrality'] == 'low']['final_mean_brier']
            if len(hub) > 1 and len(leaf) > 1:
                d = cohens_d(hub, leaf)
                t, p = stats.ttest_ind(hub, leaf)
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
                print(f"  {fb:>10} {hub.mean():>10.4f} {leaf.mean():>10.4f} {leaf.mean()-hub.mean():>+10.4f} {d:>8.3f} {p:>14.2e} {sig:>5}")

        active_c = asym_c[asym_c['funding_boost'] > 0]
        hub_c = active_c[active_c['biased_centrality'] == 'high']['final_mean_brier']
        leaf_c = active_c[active_c['biased_centrality'] == 'low']['final_mean_brier']
        if len(hub_c) > 1 and len(leaf_c) > 1:
            d_c = cohens_d(hub_c, leaf_c)
            t_c, p_c = stats.ttest_ind(hub_c, leaf_c)
            print(f"\n  OVERALL (F>0): d={d_c:.3f}, t={t_c:.3f}, p={p_c:.2e}")
            if d_c > 0:
                print("  → PERIPHERAL INCUBATION EFFECT PERSISTS in split-productivity model")
            else:
                print("  → PERIPHERAL INCUBATION EFFECT DOES NOT PERSIST")

        # By topology
        print(f"\n  {'Topology':<16} {'d':>8} {'p':>14}")
        print("  " + "─" * 40)
        for topo in sorted(asym_c['topology'].unique()):
            tsub = active_c[active_c['topology'] == topo]
            hub_t = tsub[tsub['biased_centrality'] == 'high']['final_mean_brier']
            leaf_t = tsub[tsub['biased_centrality'] == 'low']['final_mean_brier']
            if len(hub_t) > 1 and len(leaf_t) > 1:
                d_t = cohens_d(hub_t, leaf_t)
                _, p_t = stats.ttest_ind(hub_t, leaf_t)
                print(f"  {topo:<16} {d_t:>8.3f} {p_t:>14.2e}")

        df_c.to_csv('reviewer2_test_c_results.csv', index=False)

    # Save results
    df_a.to_csv('reviewer2_test_a_results.csv', index=False)
    df_b.to_csv('reviewer2_test_b_results.csv', index=False)
    saved = "reviewer2_test_a_results.csv, reviewer2_test_b_results.csv"
    if df_c is not None:
        saved += ", reviewer2_test_c_results.csv"
    print(f"\n✓ Saved: {saved}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REVIEWER #2 ABLATION EXPERIMENT")
    print("Two variants: 50/50 randomized choice & split productivity")
    print("=" * 70)

    results_a = run_test_a()
    results_b = run_test_b()
    results_c = run_test_c()
    analyze_results(results_a, results_b, results_c)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)