import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load the results
df = pd.read_csv('phase7v10_results.csv')

ASYMMETRIC_TOPOLOGIES = ["star", "wheel", "line", "hierarchical",
                         "clustered", "scale_free", "small_world", "random"]
SYMMETRIC_TOPOLOGIES = ["complete", "cycle"]

def cohens_d(g1, g2):
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    return (g2.mean() - g1.mean()) / pooled if pooled > 0 else 0

# ── FIGURE 1: Main Brier ──
asym = df[(df['phase'] == 7) & (df['topology_class'] == 'asymmetric') & (df['n_biased'] == 1)]

fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
fig.suptitle('The Peripheral Incubation Effect (Industrial Selection)',
             fontsize=18, fontweight='bold', y=0.96)

# A: Dose-response
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

# B: Cohen's d
ax = fig.add_subplot(gs[0, 1])
ds = []
for fb in funding_vals:
    fsub = p7a[p7a['funding_boost'] == fb]
    ht = fsub[fsub['biased_centrality'] == 'high']['final_mean_brier']
    lt = fsub[fsub['biased_centrality'] == 'low']['final_mean_brier']
    ds.append(cohens_d(ht, lt) if len(ht) > 1 and len(lt) > 1 else 0)
colors = ['#27ae60' if d > 0 else '#e74c3c' for d in ds]
ax.bar(range(len(funding_vals)), ds, color=colors, edgecolor='black', linewidth=1)
ax.set_xticks(range(len(funding_vals)))
ax.set_xticklabels([str(f) for f in funding_vals], fontsize=11)
ax.axhline(0, color='black', linewidth=1.5)
ax.set_xlabel('Industry Funding Boost (F)', fontsize=12)
ax.set_ylabel("Cohen's d (Leaf Damage − Hub Damage)", fontsize=12)
ax.set_title("B. Effect Size of the Peripheral Advantage", fontsize=14, fontweight='bold')
ax.text(0.5, 0.85, "Effect plateaus as Hub's mentorship\nprobability maxes out",
        transform=ax.transAxes, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# C: Negative Controls
ax = fig.add_subplot(gs[1, 0])
sym = df[(df['phase'] == 7) & (df['topology_class'] == 'symmetric') & (df['n_biased'] == 1)]
sym_fv = sorted(sym['funding_boost'].unique())
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

# D: Moran Ablation
ax = fig.add_subplot(gs[1, 1])
p7d = asym[(asym['funding_boost'] == 20)]
means = {
    'Hub (High)': [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].mean(),
                   p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].mean()],
    'Leaf (Low)':  [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].mean(),
                   p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].mean()]
}
n_d = len(p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'high')])
stds = {
    'Hub (High)': [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].std() / np.sqrt(n_d),
                   p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'high')]['final_mean_brier'].std() / np.sqrt(n_d)],
    'Leaf (Low)':  [p7d[(p7d['enable_moran'] == False) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].std() / np.sqrt(n_d),
                   p7d[(p7d['enable_moran'] == True) & (p7d['biased_centrality'] == 'low')]['final_mean_brier'].std() / np.sqrt(n_d)]
}
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, means['Hub (High)'], width, yerr=stds['Hub (High)'], label='Hub (High Centrality)', color='#e74c3c', edgecolor='black', capsize=5)
ax.bar(x + width/2, means['Leaf (Low)'], width, yerr=stds['Leaf (Low)'], label='Leaf (Low Centrality)', color='#27ae60', edgecolor='black', capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(['Moran = OFF\n(Biased Production Only)', 'Moran = ON\n(Industrial Selection)'], fontsize=12)
ax.set_ylabel('Final Mean Brier Score', fontsize=12)
ax.set_title('D. Mechanistic Proof: Moran Ablation Interaction (F=20)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5)

plt.savefig('figure_phase7v10_main_brier.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: figure_phase7v10_main_brier.png")


# ── FIGURE 2: Temporal ──
# Reload trajectory data from the traj7 variable (must still be in memory)
# If traj7 is not in memory, you'll need to re-run phase 7b or skip this figure.

try:
    tdf = pd.DataFrame(traj7)
    topos = ASYMMETRIC_TOPOLOGIES
    active_funding_levels = sorted(tdf['funding_boost'].unique())

    fig, axes = plt.subplots(len(active_funding_levels), len(topos),
                             figsize=(6 * len(topos), 5 * len(active_funding_levels)),
                             sharey=True)
    if len(topos) == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Temporal Dynamics of Peripheral Incubation (Brier Score)',
                 fontsize=20, fontweight='bold', y=0.98)

    for row, fb in enumerate(active_funding_levels):
        for col_idx, topo in enumerate(topos):
            ax = axes[row, col_idx] if len(active_funding_levels) > 1 else axes[col_idx]
            md = tdf[(tdf['topology'] == topo) & (tdf['funding_boost'] == fb)]['n_rounds_total'].max()
            td = tdf[(tdf['topology'] == topo) & (tdf['n_rounds_total'] == md) & (tdf['funding_boost'] == fb)]

            cd = tdf[(tdf['topology'] == topo) & (tdf['n_rounds_total'] == md) & (tdf['condition'] == 'control')]
            if len(cd) > 0:
                mt = cd.groupby('round')['mean_brier'].mean()
                ax.plot(mt.index, mt.values, color='#3498db', linestyle='--', label='Control (Baseline)', linewidth=2, alpha=0.8)

            for cond, col, ls in [(f'F{fb}_high', '#e74c3c', '-'), (f'F{fb}_low', '#27ae60', '-')]:
                cd_cond = td[td['condition'] == cond]
                if len(cd_cond) > 0:
                    mt = cd_cond.groupby('round')['mean_brier'].mean()
                    st = cd_cond.groupby('round')['mean_brier'].std()
                    n_seeds = cd_cond['seed'].nunique()
                    ci = 1.96 * (st / np.sqrt(n_seeds))
                    label = 'Hub (High Centrality)' if 'high' in cond else 'Leaf (Low Centrality)'
                    mt_smooth = mt.rolling(window=20, min_periods=1).mean()
                    ax.plot(mt.index, mt_smooth.values, color=col, linestyle=ls, label=label, linewidth=3, alpha=0.9)
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
    plt.savefig('figure_phase7v10_temporal_brier.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: figure_phase7v10_temporal_brier.png")

except NameError:
    print("⚠ traj7 not in memory — temporal figure skipped. Re-run phase 7b first if needed.")