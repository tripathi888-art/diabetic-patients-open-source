import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\tripa\OneDrive\Desktop\diabetic_patients_350.csv")

PALETTE = {
    'Type 1': '#4C72B0',
    'Type 2': '#DD8452',
    'LADA':   '#55A868',
    'Gestational': '#C44E52',
}
TYPE_ORDER = ['Type 1', 'Type 2', 'LADA', 'Gestational']

# ── helper ────────────────────────────────────────────────────────
def annotate_sig(ax, p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 1 – Correlation Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
num_cols = [
    'Age', 'Duration_Years', 'BMI', 'Systolic_BP_mmHg', 'Diastolic_BP_mmHg',
    'Fasting_Glucose_mgdL', 'HbA1c_Percent', 'Total_Cholesterol_mgdL',
    'LDL_mgdL', 'HDL_mgdL', 'Triglycerides_mgdL',
    'Creatinine_mgdL', 'eGFR_mLmin', 'Urine_Albumin_mgL',
]
short = {
    'Age': 'Age', 'Duration_Years': 'Duration', 'BMI': 'BMI',
    'Systolic_BP_mmHg': 'Sys BP', 'Diastolic_BP_mmHg': 'Dia BP',
    'Fasting_Glucose_mgdL': 'Fast Gluc', 'HbA1c_Percent': 'HbA1c',
    'Total_Cholesterol_mgdL': 'Tot Chol', 'LDL_mgdL': 'LDL',
    'HDL_mgdL': 'HDL', 'Triglycerides_mgdL': 'TG',
    'Creatinine_mgdL': 'Creat', 'eGFR_mLmin': 'eGFR',
    'Urine_Albumin_mgL': 'Urine Alb',
}

corr = df[num_cols].rename(columns=short).corr()

# p-value mask
n = len(df[num_cols].dropna())
t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
p_mat  = pd.DataFrame(
    2 * (1 - stats.t.cdf(np.abs(t_stat.values), df=n - 2)),
    index=corr.index, columns=corr.columns
)
sig_mask = p_mat > 0.05

fig1, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
    annot=True, fmt='.2f', annot_kws={'size': 8},
    square=True, linewidths=0.5, ax=ax,
    cbar_kws={'shrink': 0.8, 'label': 'Pearson r'}
)

# Cross out non-significant cells
for i in range(len(corr)):
    for j in range(i):
        if sig_mask.iloc[i, j]:
            ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center',
                    fontsize=9, color='grey', alpha=0.6)

ax.set_title('Correlation Heatmap — Clinical Variables\n(X = not significant, p > .05)',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
fig1.tight_layout()
fig1.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig1_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved: fig1_correlation_heatmap.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 2 – Box plots by Diabetes Type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
box_vars = [
    ('BMI',                      'BMI (kg/m²)'),
    ('HbA1c_Percent',            'HbA1c (%)'),
    ('Fasting_Glucose_mgdL',     'Fasting Glucose (mg/dL)'),
    ('Systolic_BP_mmHg',         'Systolic BP (mmHg)'),
    ('eGFR_mLmin',               'eGFR (mL/min)'),
    ('Urine_Albumin_mgL',        'Urine Albumin (mg/L)'),
    ('Duration_Years',           'Diabetes Duration (yrs)'),
    ('Triglycerides_mgdL',       'Triglycerides (mg/dL)'),
]

fig2, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, (col, label) in enumerate(box_vars):
    ax = axes[idx]
    data_plot = df[df['Diabetes_Type'].isin(TYPE_ORDER)][[col, 'Diabetes_Type']].dropna()

    sns.boxplot(
        data=data_plot, x='Diabetes_Type', y=col,
        order=TYPE_ORDER,
        palette=PALETTE, width=0.55,
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
        ax=ax
    )
    sns.stripplot(
        data=data_plot, x='Diabetes_Type', y=col,
        order=TYPE_ORDER,
        palette=PALETTE, size=2.5, alpha=0.35, jitter=True, ax=ax
    )

    # ANOVA p-value
    groups = [data_plot[data_plot['Diabetes_Type'] == t][col].values for t in TYPE_ORDER]
    _, p = stats.f_oneway(*groups)
    sig = annotate_sig(ax, p)
    ax.set_title(f'{label}\nANOVA: {sig} (p={p:.3f})', fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(axis='x', labelsize=8, rotation=20)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reference lines
    ref = {'HbA1c_Percent': 7.0, 'Fasting_Glucose_mgdL': 100,
           'Systolic_BP_mmHg': 130, 'eGFR_mLmin': 60, 'BMI': 25}
    if col in ref:
        ax.axhline(ref[col], color='red', linestyle='--', linewidth=0.9, alpha=0.7)

fig2.suptitle('Key Clinical Metrics by Diabetes Type', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig2_boxplots_by_type.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: fig2_boxplots_by_type.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 3 – ROC Curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
outcomes = {
    'Retinopathy'   : lambda d: (d['Retinopathy'].notna()  & (d['Retinopathy']  != 'None')).astype(int),
    'Neuropathy'    : lambda d: (d['Neuropathy'].notna()   & (d['Neuropathy']   != 'None')).astype(int),
    'Nephropathy'   : lambda d: (d['Nephropathy'].notna()  & (d['Nephropathy']  != 'None')).astype(int),
    'CV Complication': lambda d: (d['Cardiovascular_Complication'].notna() &
                                   (d['Cardiovascular_Complication'] != 'None')).astype(int),
}

predictors_roc = ['Age', 'Duration_Years', 'BMI', 'HbA1c_Percent',
                  'Systolic_BP_mmHg', 'Fasting_Glucose_mgdL', 'eGFR_mLmin']

PRED_COLORS = {
    'Age':                  '#1f77b4',
    'Duration_Years':       '#ff7f0e',
    'BMI':                  '#2ca02c',
    'HbA1c_Percent':        '#d62728',
    'Systolic_BP_mmHg':     '#9467bd',
    'Fasting_Glucose_mgdL': '#8c564b',
    'eGFR_mLmin':           '#e377c2',
}
PRED_LABELS = {
    'Age': 'Age', 'Duration_Years': 'Duration',
    'BMI': 'BMI', 'HbA1c_Percent': 'HbA1c',
    'Systolic_BP_mmHg': 'Systolic BP',
    'Fasting_Glucose_mgdL': 'Fasting Glucose',
    'eGFR_mLmin': 'eGFR',
}

fig3, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (outcome_name, outcome_fn) in zip(axes, outcomes.items()):
    y = outcome_fn(df)

    for pred in predictors_roc:
        sub = df[[pred]].copy()
        sub['y'] = y
        sub = sub.dropna()
        if sub['y'].nunique() < 2:
            continue
        X = sub[[pred]].values
        yv = sub['y'].values
        Xz = (X - X.mean()) / (X.std() + 1e-9)

        lr = LogisticRegression(max_iter=500, solver='lbfgs')
        lr.fit(Xz, yv)
        probs = lr.predict_proba(Xz)[:, 1]
        fpr, tpr, _ = roc_curve(yv, probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=PRED_COLORS[pred], lw=1.8,
                label=f'{PRED_LABELS[pred]} (AUC={roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    prev = y.mean()
    ax.set_title(f'{outcome_name}\n(prevalence {prev*100:.0f}%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

fig3.suptitle('ROC Curves — Complication Predictors (single-predictor logistic regression)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
fig3.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig3_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Saved: fig3_roc_curves.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 4 – Demographics + Complication overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4 = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.45, wspace=0.35)

# A – Age distribution by gender
ax_age = fig4.add_subplot(gs[0, 0])
for gender, color in [('Male', '#4C72B0'), ('Female', '#DD8452')]:
    subset = df[df['Gender'] == gender]['Age']
    ax_age.hist(subset, bins=14, alpha=0.6, color=color, label=gender, edgecolor='white')
ax_age.set_title('Age Distribution by Gender', fontweight='bold')
ax_age.set_xlabel('Age (years)')
ax_age.set_ylabel('Count')
ax_age.legend()
ax_age.spines['top'].set_visible(False)
ax_age.spines['right'].set_visible(False)

# B – Ethnicity donut
ax_eth = fig4.add_subplot(gs[0, 1])
eth_counts = df['Ethnicity'].value_counts()
eth_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#9467BD']
wedges, texts, autotexts = ax_eth.pie(
    eth_counts.values, labels=eth_counts.index,
    autopct='%1.1f%%', colors=eth_colors[:len(eth_counts)],
    startangle=90, pctdistance=0.82,
    wedgeprops=dict(width=0.55, edgecolor='white')
)
for t in autotexts: t.set_fontsize(8)
for t in texts:     t.set_fontsize(9)
ax_eth.set_title('Ethnicity Distribution', fontweight='bold')

# C – HbA1c histogram with reference lines
ax_hba = fig4.add_subplot(gs[0, 2])
ax_hba.hist(df['HbA1c_Percent'].dropna(), bins=25, color='#4C72B0', edgecolor='white', alpha=0.8)
ax_hba.axvline(7.0, color='green', linestyle='--', linewidth=1.5, label='Target (<7%)')
ax_hba.axvline(9.0, color='red',   linestyle='--', linewidth=1.5, label='Poor (>9%)')
ax_hba.set_title('HbA1c Distribution', fontweight='bold')
ax_hba.set_xlabel('HbA1c (%)')
ax_hba.set_ylabel('Count')
ax_hba.legend(fontsize=8)
ax_hba.spines['top'].set_visible(False)
ax_hba.spines['right'].set_visible(False)

# D – Complication prevalence bar chart
ax_comp = fig4.add_subplot(gs[1, 0])
comp_data = {
    'Retinopathy':    (df['Retinopathy'].notna()  & (df['Retinopathy']  != 'None')).mean() * 100,
    'Neuropathy':     (df['Neuropathy'].notna()   & (df['Neuropathy']   != 'None')).mean() * 100,
    'Nephropathy':    (df['Nephropathy'].notna()  & (df['Nephropathy']  != 'None')).mean() * 100,
    'CV Complication':(df['Cardiovascular_Complication'].notna() &
                       (df['Cardiovascular_Complication'] != 'None')).mean() * 100,
}
bars = ax_comp.barh(list(comp_data.keys()), list(comp_data.values()),
                    color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'], edgecolor='white')
for bar, val in zip(bars, comp_data.values()):
    ax_comp.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9)
ax_comp.set_xlim(0, 80)
ax_comp.set_title('Complication Prevalence (%)', fontweight='bold')
ax_comp.set_xlabel('% of patients')
ax_comp.spines['top'].set_visible(False)
ax_comp.spines['right'].set_visible(False)

# E – Medication use
ax_med = fig4.add_subplot(gs[1, 1])
meds = ['Metformin', 'Statin', 'Antihypertensive', 'Insulin', 'SGLT2_Inhibitor', 'GLP1_Agonist']
med_pct = [(df[m] == 'Yes').mean() * 100 for m in meds]
colors_m = sns.color_palette('muted', len(meds))
bars2 = ax_med.barh(meds, med_pct, color=colors_m, edgecolor='white')
for bar, val in zip(bars2, med_pct):
    ax_med.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
ax_med.set_xlim(0, 90)
ax_med.set_title('Medication Use (%)', fontweight='bold')
ax_med.set_xlabel('% of patients')
ax_med.spines['top'].set_visible(False)
ax_med.spines['right'].set_visible(False)

# F – Physical activity + diet stacked bar
ax_life = fig4.add_subplot(gs[1, 2])
act_counts  = df['Physical_Activity'].value_counts()
diet_counts = df['Diet_Adherence'].value_counts()
act_order  = ['Sedentary', 'Light', 'Moderate', 'Active']
diet_order = ['Poor', 'Fair', 'Good', 'Excellent']
act_pct  = [act_counts.get(a, 0) / len(df) * 100 for a in act_order]
diet_pct = [diet_counts.get(d, 0) / len(df) * 100 for d in diet_order]
x = np.arange(2)
bar_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bottom_a = bottom_d = 0
for i, (ac, dc, color) in enumerate(zip(act_pct, diet_pct, bar_colors)):
    ax_life.bar(0, ac,  bottom=bottom_a, color=color, edgecolor='white', label=act_order[i])
    ax_life.bar(1, dc,  bottom=bottom_d, color=color, edgecolor='white')
    bottom_a += ac
    bottom_d += dc
ax_life.set_xticks([0, 1])
ax_life.set_xticklabels(['Physical\nActivity', 'Diet\nAdherence'])
ax_life.set_ylabel('% of patients')
ax_life.set_title('Lifestyle: Activity & Diet', fontweight='bold')
ax_life.legend(fontsize=7, loc='upper right', title='Level')
ax_life.spines['top'].set_visible(False)
ax_life.spines['right'].set_visible(False)

fig4.suptitle('Diabetic Patients (N=350) — Demographic & Clinical Overview',
              fontsize=14, fontweight='bold')
fig4.savefig(r"C:\Users\tripa\OneDrive\Desktop\fig4_overview_dashboard.png", dpi=150, bbox_inches='tight')
plt.close(fig4)
print("Saved: fig4_overview_dashboard.png")

print("\nAll 4 figures saved to Desktop.")
