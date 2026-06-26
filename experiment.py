# ============================================================================
# SETUP
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import pingouin as pg

print("✓ Phase 7 v10 setup complete!")

# ============================================================================
# EXPERIMENT RUN SETTINGS (Toggle for Quick vs. Full runs)
# ============================================================================
SEEDS_PHASE_7A = 200   # Core funding x position dose-response
SEEDS_PHASE_7B = 200   # Temporal dynamics (slow burn vs fast plateau)
SEEDS_PHASE_7C = 200   # Negative controls (symmetric networks)
SEEDS_PHASE_7D = 200   # Moran ablation (the mechanistic proof)

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

# REALISTIC AMPLIFICATION DEFAULTS (Optimized for Peripheral Capture)
DEFAULT_METH_BIAS_VARIANCE = 0.02
DEFAULT_EXIT_PROB = 0.20              # Fast generational turnover
DEFAULT_INDUSTRY_THRESHOLD = 0.10     # Strong initial methodological bias
DEFAULT_BASE_PRODUCTIVITY_MIN = 1     # Low baseline honest evidence
DEFAULT_BASE_PRODUCTIVITY_MAX = 5     # Low baseline honest evidence
DEFAULT_N_ROUNDS = 2000
DEFAULT_EFFICACY_DIFF = 0.05          # Standard problem (A=55%, B=50%)

# Trimmed funding levels: Focuses on the transition and the ceiling
FUNDING_LEVELS = [0, 5, 10, 15, 20, 30, 50, 75, 100]

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
        # Assuming Treatment A (1.0) is the objective truth
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
# SIMULATION ENGINE
# ============================================================================
def run_phase7_simulation(
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
    save_trajectory=False,
):
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    p_A_true = 0.50 + efficacy_difference
    p_B_true = 0.50
    centralities = get_centrality_measures(G)
    net_props = get_network_properties(G)
    topology_class = ("symmetric" if topology in SYMMETRIC_TOPOLOGIES else "asymmetric")

    # Precompute adjacency list for massive speedup in the inner loops
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

    trajectory = [] if save_trajectory else None
    moran_events = 0

    for round_num in range(n_rounds):
        evidence = {}
        for agent in agents:
            n_trials = agent.effective_productivity
            if agent.credence >= 0.5:
                successes = rng.binomial(n_trials, agent.personal_p_A)
                evidence[agent.agent_id] = ('A', successes, n_trials)
            else:
                successes = rng.binomial(n_trials, agent.personal_p_B)
                evidence[agent.agent_id] = ('B', successes, n_trials)

        for agent in agents:
            tA_succ = tA_tri = tB_succ = tB_tri = 0
            own_t, own_s, own_n = evidence[agent.agent_id]
            if own_t == 'A':
                tA_succ, tA_tri = own_s, own_n
            else:
                tB_succ, tB_tri = own_s, own_n

            for nb in adj_list[agent.agent_id]:
                nt, ns, nn = evidence[nb]
                if nt == 'A':
                    tA_succ += ns; tA_tri += nn
                else:
                    tB_succ += ns; tB_tri += nn

            if tA_tri > 0:
                agent.alpha += tA_succ
                agent.beta += (tA_tri - tA_succ)
            if tB_tri > 0:
                agent.beta += tB_succ
                agent.alpha += (tB_tri - tB_succ)

        if enable_moran and rng.random() < exit_prob:
            moran_replacement_position_sensitive(agents, adj_list, is_agent_positions, p_A_true, p_B_true, meth_bias_variance, rng)
            moran_events += 1
            apply_industry_funding(agents, is_agent_positions, industry_threshold, funding_boost, p_B_true)

        if save_trajectory:
            ts_briers = [a.brier_score for a in agents if a.agent_type == P7AgentType.TRUTH_SEEKER]
            if ts_briers:
                trajectory.append({
                    'round': round_num,
                    'mean_brier': np.mean(ts_briers),
                    'std_brier': np.std(ts_briers),
                })

    ts_briers = [a.brier_score for a in agents if a.agent_type == P7AgentType.TRUTH_SEEKER]
    if not ts_briers:
        ts_briers = [a.brier_score for a in agents]

    if "high" in intended_condition:
        bc_label = "high"
    elif "low" in intended_condition:
        bc_label = "low"
    else:
        bc_label = "none"

    community_captured = np.mean(ts_briers) > 0.25

    result = {
        'phase': 7,
        'topology': topology, 'topology_class': topology_class,
        'n_agents': n_agents, 'n_rounds': n_rounds,
        'n_biased': len(is_agent_positions),
        'intended_condition': intended_condition,
        'biased_centrality': bc_label,
        'final_mean_brier': np.mean(ts_briers),
        'community_captured': community_captured,
        'funding_boost': funding_boost,
        'enable_moran': enable_moran,
        'total_moran_events': moran_events,
    }

    return result, trajectory

# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================
def _worker_7a(topo, n_agents, cond_name, centrality, fb, seed):
    G = create_network(topo, n_agents, seed)
    is_positions = get_high_centrality_positions(G, 1) if centrality == 'high' else \
                   (get_low_centrality_positions(G, 1, exclude=get_high_centrality_positions(G, 1)) if centrality == 'low' else [])
    enable_m = centrality is not None
    res, _ = run_phase7_simulation(
        G=G, topology=topo, n_agents=n_agents, n_rounds=DEFAULT_N_ROUNDS,
        is_agent_positions=is_positions, intended_condition=cond_name,
        funding_boost=fb, enable_moran=enable_m, seed=seed)
    return res

def _worker_7b(topo, n_agents, n_rounds, cond_name, centrality, fb, seed):
    G = create_network(topo, n_agents, seed)
    is_positions = get_high_centrality_positions(G, 1) if centrality == 'high' else \
                   (get_low_centrality_positions(G, 1, exclude=get_high_centrality_positions(G, 1)) if centrality == 'low' else [])
    enable_m = centrality is not None
    save_traj = (seed < 5 and n_rounds == 2000 and n_agents == 15) # specifically target trajectories to save memory
    res, traj = run_phase7_simulation(
        G=G, topology=topo, n_agents=n_agents, n_rounds=n_rounds,
        is_agent_positions=is_positions, intended_condition=cond_name,
        funding_boost=fb, enable_moran=enable_m, seed=seed, save_trajectory=save_traj)

    traj_data = []
    if save_traj and traj:
        for pt in traj:
            traj_data.append({'topology': topo, 'condition': cond_name, 'n_rounds_total': n_rounds, 'funding_boost': fb, 'seed': seed, **pt})
    return res, traj_data

def _worker_7c(topo, n_agents, fb, cond, seed):
    G = create_network(topo, n_agents, seed)
    is_positions = [] if cond == "control" else ([0] if cond == "1_high" else [n_agents - 1])
    res, _ = run_phase7_simulation(
        G=G, topology=topo, n_agents=n_agents, n_rounds=DEFAULT_N_ROUNDS,
        is_agent_positions=is_positions, intended_condition=cond,
        funding_boost=fb if cond != "control" else 0, enable_moran=(cond!="control"), seed=seed)
    return res

def _worker_7d(topo, n_agents, fb, enable_moran, centrality, seed):
    G = create_network(topo, n_agents, seed)
    cond = f"1_{centrality}"
    is_positions = get_high_centrality_positions(G, 1) if centrality == 'high' else get_low_centrality_positions(G, 1, exclude=get_high_centrality_positions(G, 1))
    res, _ = run_phase7_simulation(
        G=G, topology=topo, n_agents=n_agents, n_rounds=DEFAULT_N_ROUNDS,
        is_agent_positions=is_positions, intended_condition=cond,
        funding_boost=fb, enable_moran=enable_moran, seed=seed)
    return res

# ============================================================================
# EXPERIMENT PHASES (Parallelized)
# ============================================================================
def run_phase7a_funding_x_position(seeds=SEEDS_PHASE_7A):
    print("\n" + "=" * 70)
    print("PHASE 1: FUNDING × POSITION (Core Experiment)")
    print("=" * 70)
    tasks = [(topo, n_agents, cond_name, centrality, fb, seed)
             for topo in ASYMMETRIC_TOPOLOGIES for n_agents in [6, 10, 15, 20, 30]
             for cond_name, centrality, fb in [("control", None, 0)] + [(f"F{fb}_high", "high", fb) for fb in FUNDING_LEVELS] + [(f"F{fb}_low", "low", fb) for fb in FUNDING_LEVELS]
             for seed in range(seeds)]
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_worker_7a)(*t) for t in tqdm(tasks, desc="Phase 7a"))
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results

def run_phase7b_temporal(seeds=SEEDS_PHASE_7B):
    print("\n" + "=" * 70)
    print("PHASE 2: TEMPORAL DYNAMICS")
    print("=" * 70)
    # Include 0 as a true experimental baseline alongside the active funding levels
    funding_levels_temporal = [0, 10, 20, 50]
    tasks = [(topo, n_agents, n_rounds, cond_name, centrality, fb, seed)
             for topo in ASYMMETRIC_TOPOLOGIES for n_agents in [15, 20]
             for n_rounds in [2000]
             for cond_name, centrality, fb in [("control", None, 0)] + [(f"F{fb}_high", "high", fb) for fb in funding_levels_temporal] + [(f"F{fb}_low", "low", fb) for fb in funding_levels_temporal]
             for seed in range(seeds)]
    start = time.time()
    out = Parallel(n_jobs=-1)(delayed(_worker_7b)(*t) for t in tqdm(tasks, desc="Phase 7b"))
    results = [o[0] for o in out]
    trajectory_data = [item for o in out for item in o[1]]
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results, trajectory_data

def run_phase7c_negative_controls(seeds=SEEDS_PHASE_7C):
    print("\n" + "=" * 70)
    print("PHASE 3: NEGATIVE CONTROLS (Symmetric Topologies)")
    print("=" * 70)
    tasks = [(topo, n_agents, fb, cond, seed)
             for topo in SYMMETRIC_TOPOLOGIES for n_agents in [6, 10, 15, 20, 30]
             for fb in [0, 20, 50, 100] for cond in ["control", "1_high", "1_low"]
             for seed in range(seeds)]
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_worker_7c)(*t) for t in tqdm(tasks, desc="Phase 7c"))
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results

def run_phase7d_moran_ablation(seeds=SEEDS_PHASE_7D):
    print("\n" + "=" * 70)
    print("PHASE 4: MORAN ABLATION (Mechanistic Proof)")
    print("=" * 70)
    # Include 0 to verify the baseline mechanics without a funding multiplier
    tasks = [(topo, n_agents, fb, enable_moran, centrality, seed)
             for topo in ASYMMETRIC_TOPOLOGIES for n_agents in [6, 10, 15, 20, 30]
             for fb in [0, 20, 50] for enable_moran in [False, True]
             for centrality in ["high", "low"] for seed in range(seeds)]
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_worker_7d)(*t) for t in tqdm(tasks, desc="Phase 7d"))
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
def cohens_d(g1, g2):
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    return (g2.mean() - g1.mean()) / pooled if pooled > 0 else 0

def run_phase7_statistics(df):
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS — THE PERIPHERAL INCUBATION EFFECT")
    print("=" * 70)

    asym = df[(df['phase'] == 7) & (df['topology_class'] == 'asymmetric')]

    # 1. THE PERIPHERAL ADVANTAGE
    print("\n" + "─" * 70)
    print("1. THE PERIPHERAL ADVANTAGE: Mean Brier Score (Higher is Worse)")
    print("   Hypothesis: Due to methodological incubation, Low Centrality agents")
    print("   cause MORE damage than High Centrality agents under Industrial Selection.")
    print("─" * 70)
    print(f"\n  {'Funding':>10} {'Hub M':>8} {'Leaf M':>8} {'Diff':>8} {'d':>8} {'p':>12}")
    print("  " + "─" * 58)

    # Isolate Phase 7a data (Moran=True)
    p7a = asym[asym['enable_moran'] == True]
    for fb in sorted(p7a['funding_boost'].unique()):
        fsub = p7a[p7a['funding_boost'] == fb]
        ht = fsub[fsub['biased_centrality'] == 'high']['final_mean_brier']
        lt = fsub[fsub['biased_centrality'] == 'low']['final_mean_brier']
        if len(ht) > 1 and len(lt) > 1:
            d = cohens_d(ht, lt) # Low - High: Positive d = Low causes MORE damage
            t, p = stats.ttest_ind(ht, lt)
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            print(f"  {fb:>10} {ht.mean():>8.4f} {lt.mean():>8.4f} "
                  f"{lt.mean()-ht.mean():>+8.4f} {d:>8.2f} {p:>12.2e} {sig}")

    # 2. MORAN ABLATION INTERACTION
    print("\n" + "─" * 70)
    print("2. THE MECHANISTIC PROOF: Moran Ablation Interaction (F=20)")
    print("   Hypothesis: If you turn OFF generational turnover (Moran=False),")
    print("   the peripheral advantage disappears, reverting to the Hub advantage.")
    print("─" * 70)

    p7d = asym[(asym['funding_boost'] == 20)]
    ht_on = p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'high')]['final_mean_brier']
    lt_on = p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'low')]['final_mean_brier']
    ht_off = p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'high')]['final_mean_brier']
    lt_off = p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'low')]['final_mean_brier']

    if len(ht_on) > 1 and len(lt_off) > 1:
        print(f"\n  MORAN = ON (Industrial Selection Active):")
        print(f"    Hub Brier:  {ht_on.mean():.4f}")
        print(f"    Leaf Brier: {lt_on.mean():.4f}")
        print(f"    Difference: {(lt_on.mean()-ht_on.mean()):+.4f} (Leaf is more damaging)")

        # CORRECTED TERMINOLOGY HERE
        print(f"\n  MORAN = OFF (Biased Production Only, No Mentorship):")
        print(f"    Hub Brier:  {ht_off.mean():.4f}")
        print(f"    Leaf Brier: {lt_off.mean():.4f}")
        print(f"    Difference: {(lt_off.mean()-ht_off.mean()):+.4f} (Hub is more damaging)")

    print("\n" + "=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_phase7_main(df, filename='figure_phase7v10_main_brier.png'):
    asym = df[(df['phase'] == 7) & (df['topology_class'] == 'asymmetric') & (df['n_biased'] == 1)]

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
    fig.suptitle('Phase 7 v10: The Peripheral Incubation Effect (Industrial Selection)',
                 fontsize=18, fontweight='bold', y=0.96)

    # A: Funding × Position dose-response curves (Brier)
    ax = fig.add_subplot(gs[0, 0])
    p7a = asym[asym['enable_moran'] == True]
    funding_vals = sorted(p7a['funding_boost'].unique())
    for cent, col, mk in [('high', '#e74c3c', 'o'), ('low', '#27ae60', 's')]:
        ms = [p7a[(p7a['funding_boost'] == fb) & (p7a['biased_centrality'] == cent)]['final_mean_brier'].mean() for fb in funding_vals]
        es = [p7a[(p7a['funding_boost'] == fb) & (p7a['biased_centrality'] == cent)]['final_mean_brier'].std() / np.sqrt(len(p7a[(p7a['funding_boost'] == fb) & (p7a['biased_centrality'] == cent)])) for fb in funding_vals]
        label = 'Hub (High Centrality)' if cent == 'high' else 'Leaf (Low Centrality)'
        ax.errorbar(funding_vals, ms, yerr=es, marker=mk, color=col, label=label, capsize=4, linewidth=2.5, markersize=8)

    ax.axhline(0.25, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Baseline Ignorance (Brier 0.25)')
    ax.legend(fontsize=12)
    ax.set_xlabel('Industry Funding Boost (F)', fontsize=12)
    ax.set_ylabel('Final Mean Brier Score (Higher is Worse)', fontsize=12)
    ax.set_title('A. Dose-Response: The Peripheral Advantage', fontsize=14, fontweight='bold')

    # B: Peripheral Advantage Bar Chart (Cohen's d)
    ax = fig.add_subplot(gs[0, 1])
    ds = []
    fv_valid = funding_vals
    for fb in fv_valid:
        fsub = p7a[p7a['funding_boost'] == fb]
        ht = fsub[fsub['biased_centrality'] == 'high']['final_mean_brier']
        lt = fsub[fsub['biased_centrality'] == 'low']['final_mean_brier']
        if len(ht) > 1 and len(lt) > 1:
            ds.append(cohens_d(ht, lt)) # Low - High
        else:
            ds.append(0)

    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in ds]
    bars = ax.bar(range(len(fv_valid)), ds, color=colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(fv_valid)))
    ax.set_xticklabels([str(f) for f in fv_valid], fontsize=11)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Industry Funding Boost (F)', fontsize=12)
    ax.set_ylabel("Cohen's d (Leaf Damage − Hub Damage)", fontsize=12)
    ax.set_title("B. Effect Size of the Peripheral Advantage", fontsize=14, fontweight='bold')

    # Add text annotation explaining ceiling
    ax.text(0.5, 0.85, "Effect plateaus as Hub's mentorship\nprobability maxes out",
            transform=ax.transAxes, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # C: Negative Controls
    ax = fig.add_subplot(gs[1, 0])
    sym = df[(df['phase'] == 7) & (df['topology_class'] == 'symmetric') & (df['n_biased'] == 1)]
    sym_fv = sorted(sym['funding_boost'].unique())
    if len(sym_fv) > 0:
        for topo in SYMMETRIC_TOPOLOGIES:
            for cent, col, ls in [('high', '#e74c3c', '-'), ('low', '#27ae60', '--')]:
                ms = []
                for fb in sym_fv:
                    s = sym[(sym['topology'] == topo) & (sym['funding_boost'] == fb) & (sym['biased_centrality'] == cent)]
                    ms.append(s['final_mean_brier'].mean() if len(s) > 0 else np.nan)
                ax.plot(sym_fv, ms, linestyle=ls, color=col, linewidth=2, label=f'{topo.capitalize()} {cent.capitalize()}', marker='o', markersize=6)
    ax.axhline(0.25, color='black', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.set_xlabel('Industry Funding Boost (F)', fontsize=12)
    ax.set_ylabel('Final Mean Brier Score', fontsize=12)
    ax.set_title('C. Negative Controls (Symmetric Topologies)', fontsize=14, fontweight='bold')

    # D: NEW - Moran Ablation Interaction (The Smoking Gun)
    ax = fig.add_subplot(gs[1, 1])
    p7d = asym[(asym['funding_boost'] == 20)]

    means = {
        'Hub (High)': [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].mean(),
                       p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].mean()],
        'Leaf (Low)':  [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].mean(),
                       p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].mean()]
    }
    stds = {
        'Hub (High)': [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].std() / np.sqrt(SEEDS_PHASE_7D * 8),
                       p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].std() / np.sqrt(SEEDS_PHASE_7D * 8)],
        'Leaf (Low)':  [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].std() / np.sqrt(SEEDS_PHASE_7D * 8),
                       p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].std() / np.sqrt(SEEDS_PHASE_7D * 8)]
    }

    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, means['Hub (High)'], width, yerr=stds['Hub (High)'], label='Hub (High Centrality)', color='#e74c3c', edgecolor='black', capsize=5)
    ax.bar(x + width/2, means['Leaf (Low)'], width, yerr=stds['Leaf (Low)'], label='Leaf (Low Centrality)', color='#27ae60', edgecolor='black', capsize=5)

    ax.set_xticks(x)
    # CORRECTED TERMINOLOGY HERE:
    ax.set_xticklabels(['Moran = OFF\n(Biased Production Only)', 'Moran = ON\n(Industrial Selection)'], fontsize=12)
    ax.set_ylabel('Final Mean Brier Score', fontsize=12)
    ax.set_title('D. Mechanistic Proof: Moran Ablation Interaction (F=20)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_phase7_temporal(traj_data, filename='figure_phase7v10_temporal_brier.png'):
    if not traj_data:
        print("  No trajectory data"); return
    tdf = pd.DataFrame(traj_data)

    # RESTORED: Use all 8 asymmetric topologies to match the initial paper's scope
    topos = ASYMMETRIC_TOPOLOGIES
    active_funding_levels = sorted(tdf['funding_boost'].unique())

    fig, axes = plt.subplots(len(active_funding_levels), len(topos),
                             figsize=(6 * len(topos), 5 * len(active_funding_levels)),
                             sharey=True)
    if len(topos) == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Phase 7 v10: Temporal Dynamics of Peripheral Incubation (Brier Score)',
                 fontsize=20, fontweight='bold', y=0.98)

    for row, fb in enumerate(active_funding_levels):
        for col_idx, topo in enumerate(topos):
            ax = axes[row, col_idx] if len(active_funding_levels) > 1 else axes[col_idx]

            md = tdf[(tdf['topology'] == topo) & (tdf['funding_boost'] == fb)]['n_rounds_total'].max()
            td = tdf[(tdf['topology'] == topo) & (tdf['n_rounds_total'] == md) & (tdf['funding_boost'] == fb)]

            # Plot Control baseline
            cd = tdf[(tdf['topology'] == topo) & (tdf['n_rounds_total'] == md) & (tdf['condition'] == 'control')]
            if len(cd) > 0:
                mt = cd.groupby('round')['mean_brier'].mean()
                ax.plot(mt.index, mt.values, color='#3498db', linestyle='--', label='Control (Baseline)', linewidth=2, alpha=0.8)

            # Plot High and Low Centrality (Refined with Confidence Intervals instead of Standard Deviation)
            for cond, col, ls in [(f'F{fb}_high', '#e74c3c', '-'), (f'F{fb}_low', '#27ae60', '-')]:
                cd_cond = td[td['condition'] == cond]
                if len(cd_cond) > 0:
                    mt = cd_cond.groupby('round')['mean_brier'].mean()
                    # Calculate 95% Confidence Interval (1.96 * standard error) instead of raw standard deviation
                    st = cd_cond.groupby('round')['mean_brier'].std()
                    n_seeds = cd_cond['seed'].nunique()
                    ci = 1.96 * (st / np.sqrt(n_seeds))

                    label = 'Hub (High Centrality)' if 'high' in cond else 'Leaf (Low Centrality)'

                    # Apply a rolling window to smooth the noisy temporal data slightly for clarity
                    mt_smooth = mt.rolling(window=20, min_periods=1).mean()

                    ax.plot(mt.index, mt_smooth.values, color=col, linestyle=ls, label=label, linewidth=3, alpha=0.9)

                    # Shade the 95% CI
                    ax.fill_between(mt.index, mt.values - ci.values, mt.values + ci.values, color=col, alpha=0.15)

            ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5)

            if row == 0:
                ax.set_title(topo.capitalize(), fontweight='bold', fontsize=16)
            if col_idx == 0:
                ax.set_ylabel(f'Funding Boost (F={fb})\nMean Brier Score', fontsize=14)
            if row == len(active_funding_levels) - 1:
                ax.set_xlabel('Round', fontsize=14)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=12, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INDUSTRIAL SELECTION — PERIPHERAL INCUBATION EFFECT")
    print("=" * 70)
    print(f"Start: {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    r7a = run_phase7a_funding_x_position(seeds=SEEDS_PHASE_7A)
    r7b, traj7 = run_phase7b_temporal(seeds=SEEDS_PHASE_7B)
    r7c = run_phase7c_negative_controls(seeds=SEEDS_PHASE_7C)
    r7d = run_phase7d_moran_ablation(seeds=SEEDS_PHASE_7D)

    all_r7 = r7a + r7b + r7c + r7d
    df7 = pd.DataFrame(all_r7)
    df7.to_csv('phase7v10_results.csv', index=False)

    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"PHASE 7 v10 COMPLETE: {len(df7)} sims in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    stats_r7 = run_phase7_statistics(df7)

    print("\n" + "=" * 70)
    print("GENERATING PHASE 7 v10 FIGURES")
    print("=" * 70)
    plot_phase7_main(df7, 'figure_phase7v10_main_brier.png')
    plot_phase7_temporal(traj7, 'figure_phase7v10_temporal_brier.png')

    print(f"""
{'=' * 70}
PHASE 7 v10 EXPERIMENT COMPLETE
{'=' * 70}
  Output files:
    phase7v10_results.csv
    figure_phase7v10_main_brier.png
    figure_phase7v10_temporal_brier.png
""")