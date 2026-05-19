import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\tripa\OneDrive\Desktop\diabetic_patients_350.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART A – SEVERITY ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Ordinal encodings
df['Retinopathy_Ord']  = df['Retinopathy'].map({'None':0,'Mild':1,'Moderate':2,'Severe':3})
df['Neuropathy_Ord']   = df['Neuropathy'].map({'None':0,'Peripheral':1,'Autonomic':2,'Both':3})
df['Nephropathy_Ord']  = df['Nephropathy'].map({'None':0,'Microalbuminuria':1,'Macroalbuminuria':2,'CKD':3})
df['CV_Ord']           = df['Cardiovascular_Complication'].map(
    {'None':0,'Hypertension':1,'CAD':2,'Heart Failure':3,'Stroke':3})

# Composite burden score (0–12)
df['Burden'] = (df['Retinopathy_Ord'].fillna(0) + df['Neuropathy_Ord'].fillna(0) +
                df['Nephropathy_Ord'].fillna(0) + df['CV_Ord'].fillna(0))

NUM_COLS = ['Age','Duration_Years','BMI','HbA1c_Percent','Fasting_Glucose_mgdL',
            'Systolic_BP_mmHg','eGFR_mLmin','Urine_Albumin_mgL','Triglycerides_mgdL']

# ── Fig A1: Severity distributions per complication ──────────────
fig_a1, axes = plt.subplots(2, 4, figsize=(20, 9))

comp_configs = [
    ('Retinopathy',              'Retinopathy_Ord',
     ['None','Mild','Moderate','Severe'],         ['#d0e4f7','#7fb3d3','#2980b9','#1a5276']),
    ('Neuropathy',               'Neuropathy_Ord',
     ['None','Peripheral','Autonomic','Both'],     ['#d5f5e3','#7dcea0','#27ae60','#145a32']),
    ('Nephropathy',              'Nephropathy_Ord',
     ['None','Microalb.','Macroalb.','CKD'],      ['#fdf2e9','#f0a27a','#e67e22','#a04000']),
    ('CV Complication',          'CV_Ord',
     ['None','HTN','CAD/HF/Stroke'],              ['#f9ebea','#e07070','#c0392b']),
]

for col_idx, (comp, ord_col, levels, colors) in enumerate(comp_configs):
    ax = axes[0, col_idx]
    if comp == 'CV Complication':
        counts = df[ord_col].fillna(0).value_counts().sort_index()
        pcts   = counts / len(df) * 100
        idx    = [0, 1, 2]
        vals   = [pcts.get(i, 0) for i in idx]
        bars   = ax.bar(levels, vals, color=colors, edgecolor='white', width=0.6)
    else:
        raw = df[comp].fillna('None').value_counts()
        vals = [raw.get(l, 0) / len(df) * 100 for l in levels]
        bars = ax.bar(levels, vals, color=colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_title(comp, fontweight='bold', fontsize=10)
    ax.set_ylabel('% patients')
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Fig A1 bottom row: clinical markers by severity ──────────────
sev_pairs = [
    ('Retinopathy', 'Retinopathy_Ord', 'HbA1c_Percent',         'HbA1c (%)'),
    ('Neuropathy',  'Neuropathy_Ord',  'Age',                    'Age (yrs)'),
    ('Nephropathy', 'Nephropathy_Ord', 'eGFR_mLmin',             'eGFR (mL/min)'),
    ('CV',          'CV_Ord',          'Systolic_BP_mmHg',       'Systolic BP (mmHg)'),
]

for col_idx, (comp, ord_col, metric, metric_label) in enumerate(sev_pairs):
    ax = axes[1, col_idx]
    tmp = df[[ord_col, metric]].dropna()
    tmp[ord_col] = tmp[ord_col].astype(int)

    sev_groups = sorted(tmp[ord_col].unique())
    box_data   = [tmp[tmp[ord_col] == g][metric].values for g in sev_groups]
    level_map  = {0:'None',1:'Mild/Low',2:'Moderate',3:'Severe/High'}
    x_labels   = [level_map.get(g, str(g)) for g in sev_groups]

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
    palette = sns.color_palette('Blues', len(sev_groups))
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)

    # Kruskal-Wallis (non-parametric)
    H, p = stats.kruskal(*box_data)
    sig   = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.set_title(f'{metric_label} by {comp} Severity\nKruskal-Wallis: {sig} (p={p:.3f})',
                 fontsize=9, fontweight='bold')
    ax.set_xticklabels(x_labels, rotation=20, fontsize=8)
    ax.set_ylabel(metric_label, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_a1.suptitle('Complication Severity Distribution & Clinical Markers',
                fontsize=14, fontweight='bold')
plt.tight_layout()
fig_a1.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig5_severity_analysis.png", dpi=150, bbox_inches='tight')
plt.close(fig_a1)
print("Saved: fig5_severity_analysis.png")

# ── Fig A2: Composite burden vs key markers ───────────────────────
fig_a2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

scatter_pairs = [
    ('HbA1c_Percent',     'HbA1c (%)'),
    ('Age',               'Age (yrs)'),
    ('Urine_Albumin_mgL', 'Urine Albumin (mg/L)'),
]
for ax, (metric, label) in zip(axes2, scatter_pairs):
    tmp = df[['Burden', metric]].dropna()
    rho, p = stats.spearmanr(tmp['Burden'], tmp[metric])
    ax.scatter(tmp['Burden'], tmp[metric], alpha=0.35, s=18,
               c=tmp['Burden'], cmap='YlOrRd', edgecolors='none')
    # Regression line
    m, b = np.polyfit(tmp['Burden'], tmp[metric], 1)
    xr = np.linspace(tmp['Burden'].min(), tmp['Burden'].max(), 100)
    ax.plot(xr, m*xr + b, color='#c0392b', lw=2)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.set_title(f'Burden Score vs {label}\nSpearman rho={rho:.2f} ({sig})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Composite Burden Score (0–12)', fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_a2.suptitle('Composite Complication Burden Score vs Clinical Markers',
                fontsize=13, fontweight='bold')
plt.tight_layout()
fig_a2.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig6_burden_score.png", dpi=150, bbox_inches='tight')
plt.close(fig_a2)
print("Saved: fig6_burden_score.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART B – CLUSTER ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cluster_cols = ['Age','Duration_Years','BMI','HbA1c_Percent',
                'Fasting_Glucose_mgdL','Systolic_BP_mmHg',
                'eGFR_mLmin','Urine_Albumin_mgL','Triglycerides_mgdL',
                'Retinopathy_Ord','Neuropathy_Ord','Nephropathy_Ord','CV_Ord']

df_c = df[cluster_cols].copy()
for col in ['Retinopathy_Ord','Neuropathy_Ord','Nephropathy_Ord','CV_Ord']:
    df_c[col] = df_c[col].fillna(0)
df_c = df_c.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_c)

# Elbow + silhouette to choose k
inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = K_range[np.argmax(silhouettes)]
print(f"Best k by silhouette: {best_k}  (score={max(silhouettes):.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df.loc[df_c.index, 'Cluster'] = km_final.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype('Int64')

# PCA for 2-D visualisation
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

# ── Fig B1: Elbow + silhouette ────────────────────────────────────
fig_b1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(list(K_range), inertias, 'o-', color='#4C72B0', lw=2)
ax1.set_title('Elbow Plot (K-Means Inertia)', fontweight='bold')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.plot(list(K_range), silhouettes, 's-', color='#DD8452', lw=2)
ax2.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
ax2.set_title('Silhouette Score by k', fontweight='bold')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig_b1.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig7_cluster_selection.png", dpi=150, bbox_inches='tight')
plt.close(fig_b1)
print("Saved: fig7_cluster_selection.png")

# ── Fig B2: PCA scatter + cluster profiles ────────────────────────
CLUSTER_PALETTE = sns.color_palette('tab10', best_k)

fig_b2 = plt.figure(figsize=(18, 7))
gs2 = gridspec.GridSpec(1, 2, figure=fig_b2, wspace=0.35)

# PCA scatter
ax_pca = fig_b2.add_subplot(gs2[0, 0])
for ci in range(best_k):
    mask = (df.loc[df_c.index, 'Cluster'] == ci).values
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=CLUSTER_PALETTE[ci], s=25, alpha=0.65,
                   label=f'Cluster {ci+1} (n={mask.sum()})', edgecolors='none')
ax_pca.set_title(f'PCA Projection of {best_k} Clusters\n'
                 f'PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}% variance explained',
                 fontweight='bold')
ax_pca.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
ax_pca.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
ax_pca.legend(fontsize=8, markerscale=1.5)
ax_pca.spines['top'].set_visible(False); ax_pca.spines['right'].set_visible(False)

# Cluster profile radar / heatmap
profile_cols = ['Age','Duration_Years','BMI','HbA1c_Percent',
                'Fasting_Glucose_mgdL','Systolic_BP_mmHg',
                'eGFR_mLmin','Burden']
df_profile = df.loc[df_c.index].copy()
cluster_means = df_profile.groupby('Cluster')[profile_cols].mean()
cluster_means_z = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-9)

ax_heat = fig_b2.add_subplot(gs2[0, 1])
short_names = {
    'Age':'Age','Duration_Years':'Duration','BMI':'BMI',
    'HbA1c_Percent':'HbA1c','Fasting_Glucose_mgdL':'Fast Gluc',
    'Systolic_BP_mmHg':'Sys BP','eGFR_mLmin':'eGFR','Burden':'Burden'
}
cluster_means_z.columns = [short_names[c] for c in cluster_means_z.columns]
cluster_means_z.index   = [f'Cluster {int(i)+1}' for i in cluster_means_z.index]

sns.heatmap(cluster_means_z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax_heat, cbar_kws={'label': 'Z-score vs grand mean'})
ax_heat.set_title('Cluster Profiles (Z-scores)', fontweight='bold')
ax_heat.tick_params(axis='x', rotation=40, labelsize=9)
ax_heat.tick_params(axis='y', rotation=0, labelsize=9)

fig_b2.suptitle(f'K-Means Clustering  (k={best_k})  —  Patient Subgroups',
                fontsize=14, fontweight='bold')
plt.tight_layout()
fig_b2.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig8_cluster_profiles.png", dpi=150, bbox_inches='tight')
plt.close(fig_b2)
print("Saved: fig8_cluster_profiles.png")

# ── Print cluster summary table ───────────────────────────────────
print("\n" + "=" * 65)
print(f"CLUSTER SUMMARY  (k={best_k})")
print("=" * 65)
summary_cols = ['Age','Duration_Years','BMI','HbA1c_Percent',
                'Fasting_Glucose_mgdL','Systolic_BP_mmHg','eGFR_mLmin','Burden']
for ci in range(best_k):
    grp = df_profile[df_profile['Cluster'] == ci]
    print(f"\nCluster {ci+1}  (n={len(grp)}, {100*len(grp)/len(df_profile):.1f}%)")
    for col in summary_cols:
        m, s = grp[col].mean(), grp[col].std()
        print(f"  {col:<30s}: {m:6.1f} +/- {s:.1f}")
    # Diabetes type composition
    tc = grp['Diabetes_Type'].value_counts()
    print(f"  {'Diabetes Type mix':<30s}: " + ", ".join(f"{t}={c}" for t, c in tc.items()))
    # Top complication
    for cc in ['Retinopathy','Neuropathy','Nephropathy','Cardiovascular_Complication']:
        pct = (grp[cc].notna() & (grp[cc] != 'None')).mean() * 100
        print(f"  {cc:<30s}: {pct:.1f}%")
