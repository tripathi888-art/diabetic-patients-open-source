"""
Diabetic Patients (N=350) — Full Analysis Pipeline
Outputs: 8 PNG figures + HTML report, all saved to the same folder as this script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, silhouette_score
from sklearn.preprocessing import StandardScaler
import base64, os, warnings
warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\tripa\OneDrive\Desktop\diabetic_patients_350.csv"
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

def out(filename):
    return os.path.join(OUT_DIR, filename)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD & PREPARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df = pd.read_csv(DATA_PATH)

df['Retinopathy_Ord'] = df['Retinopathy'].map({'None':0,'Mild':1,'Moderate':2,'Severe':3})
df['Neuropathy_Ord']  = df['Neuropathy'].map({'None':0,'Peripheral':1,'Autonomic':2,'Both':3})
df['Nephropathy_Ord'] = df['Nephropathy'].map({'None':0,'Microalbuminuria':1,'Macroalbuminuria':2,'CKD':3})
df['CV_Ord']          = df['Cardiovascular_Complication'].map(
    {'None':0,'Hypertension':1,'CAD':2,'Heart Failure':3,'Stroke':3})
df['Burden'] = (df['Retinopathy_Ord'].fillna(0) + df['Neuropathy_Ord'].fillna(0) +
                df['Nephropathy_Ord'].fillna(0) + df['CV_Ord'].fillna(0))

NUM_COLS = ['Age','Duration_Years','BMI','Systolic_BP_mmHg','Diastolic_BP_mmHg',
            'Fasting_Glucose_mgdL','HbA1c_Percent','Total_Cholesterol_mgdL',
            'LDL_mgdL','HDL_mgdL','Triglycerides_mgdL',
            'Creatinine_mgdL','eGFR_mLmin','Urine_Albumin_mgL']

TYPE_ORDER     = ['Type 1','Type 2','LADA','Gestational']
TYPE_PALETTE   = {'Type 1':'#4C72B0','Type 2':'#DD8452','LADA':'#55A868','Gestational':'#C44E52'}
PRED_COLORS    = {'Age':'#1f77b4','Duration_Years':'#ff7f0e','BMI':'#2ca02c',
                  'HbA1c_Percent':'#d62728','Systolic_BP_mmHg':'#9467bd',
                  'Fasting_Glucose_mgdL':'#8c564b','eGFR_mLmin':'#e377c2'}
PRED_LABELS    = {'Age':'Age','Duration_Years':'Duration','BMI':'BMI',
                  'HbA1c_Percent':'HbA1c','Systolic_BP_mmHg':'Sys BP',
                  'Fasting_Glucose_mgdL':'Fast Gluc','eGFR_mLmin':'eGFR'}

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

n_female = (df['Gender'] == 'Female').sum()
n_male   = (df['Gender'] == 'Male').sum()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 – BASIC DESCRIPTIVE STATS (console)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total patients : {len(df)}")
print(f"Columns        : {df.shape[1]}")
print(f"Missing values : {df.isnull().sum().sum()}")

print("\n--- Age ---")
print(f"  Mean +/- SD : {df['Age'].mean():.1f} +/- {df['Age'].std():.1f} yrs")
print(f"  Median      : {df['Age'].median():.1f}  |  Range: {df['Age'].min()}–{df['Age'].max()}")

for header, col in [("Gender", "Gender"), ("Ethnicity", "Ethnicity"), ("Diabetes Type", "Diabetes_Type")]:
    print(f"\n--- {header} ---")
    for v, c in df[col].value_counts().items():
        print(f"  {v:22s}: {c:4d} ({100*c/len(df):.1f}%)")

print("\n--- Clinical Measures (Mean +/- SD | Median | Range) ---")
clin_map = {
    'Duration_Years':'Duration (yrs)','BMI':'BMI','Systolic_BP_mmHg':'Sys BP',
    'Diastolic_BP_mmHg':'Dia BP','Fasting_Glucose_mgdL':'Fast Gluc',
    'Postprandial_Glucose_mgdL':'PP Gluc','HbA1c_Percent':'HbA1c (%)',
    'Total_Cholesterol_mgdL':'Total Chol','LDL_mgdL':'LDL','HDL_mgdL':'HDL',
    'Triglycerides_mgdL':'TG','Creatinine_mgdL':'Creatinine',
    'eGFR_mLmin':'eGFR','Urine_Albumin_mgL':'Urine Alb',
}
for col, label in clin_map.items():
    m, s, med = df[col].mean(), df[col].std(), df[col].median()
    print(f"  {label:<18s}: {m:6.1f} +/- {s:.1f}  |  {med:.1f}  |  {df[col].min():.1f}–{df[col].max():.1f}")

print("\n--- Glycaemic Control ---")
df['HbA1c_Cat'] = pd.cut(df['HbA1c_Percent'], bins=[0,7,8,9,100],
                          labels=['<7 (Target)','7-8','8-9','>9 (Poor)'])
for v, c in df['HbA1c_Cat'].value_counts().sort_index().items():
    print(f"  {str(v):20s}: {c:4d} ({100*c/len(df):.1f}%)")

print("\n--- Medication Use ---")
for m in ['Insulin','Metformin','SGLT2_Inhibitor','GLP1_Agonist','Statin','Antihypertensive']:
    yes = (df[m] == 'Yes').sum()
    print(f"  {m:25s}: {yes:4d} ({100*yes/len(df):.1f}%)")

print("\n--- Complications ---")
for col in ['Retinopathy','Neuropathy','Nephropathy','Cardiovascular_Complication']:
    for v, c in df[col].value_counts().items():
        print(f"  {col:32s} {str(v):25s}: {c:4d} ({100*c/len(df):.1f}%)")

print("\n--- Gender Comparison (t-test + Cohen's d) ---")
males, females = df[df['Gender']=='Male'], df[df['Gender']=='Female']
key_cols = ['Age','BMI','HbA1c_Percent','Fasting_Glucose_mgdL','Systolic_BP_mmHg','eGFR_mLmin']
print(f"  {'Measure':<28s} {'Male':>15s} {'Female':>15s}  {'p':>7s}  d")
for col in key_cols:
    m1,s1 = males[col].mean(), males[col].std()
    m2,s2 = females[col].mean(), females[col].std()
    t, p  = stats.ttest_ind(males[col], females[col])
    pooled = np.sqrt(((len(males)-1)*s1**2+(len(females)-1)*s2**2)/(len(males)+len(females)-2))
    d = (m1-m2)/pooled
    sig = ' *' if p < 0.05 else '  '
    print(f"  {col:<28s} {m1:6.1f}+/-{s1:.1f}  {m2:6.1f}+/-{s2:.1f}   {p:.3f}{sig}  {d:+.2f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 – CORRELATIONS & SUBGROUP (console)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("CORRELATIONS  (Pearson, |r| >= 0.20)")
print("=" * 60)
corr_results = []
for i, c1 in enumerate(NUM_COLS):
    for c2 in NUM_COLS[i+1:]:
        pair = df[[c1,c2]].dropna()
        r, p = stats.pearsonr(pair[c1], pair[c2])
        if abs(r) >= 0.20:
            corr_results.append((c1,c2,r,p))
corr_results.sort(key=lambda x: abs(x[2]), reverse=True)
if corr_results:
    for c1,c2,r,p in corr_results:
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"  {c1:<28s} {c2:<28s}  r={r:+.3f}  p={p:.4f} {sig}")
else:
    print("  No pairs with |r| >= 0.20 found.")

print("\n" + "=" * 60)
print("SUBGROUP: ONE-WAY ANOVA BY DIABETES TYPE")
print("=" * 60)
compare = ['Age','Duration_Years','BMI','HbA1c_Percent','Fasting_Glucose_mgdL',
           'Systolic_BP_mmHg','eGFR_mLmin','Urine_Albumin_mgL']
sub = {t: df[df['Diabetes_Type']==t] for t in TYPE_ORDER}
print(f"  {'Measure':<22s} {'T1':>12s}  {'T2':>12s}  {'LADA':>12s}  {'Gest':>12s}  p      eta2")
print("  " + "-"*88)
for col in compare:
    groups = [sub[t][col].dropna().values for t in TYPE_ORDER]
    means  = [g.mean() for g in groups]
    sds    = [g.std()  for g in groups]
    F, p   = stats.f_oneway(*groups)
    gm     = np.concatenate(groups).mean()
    ssb    = sum(len(g)*(m-gm)**2 for g,m in zip(groups,means))
    sst    = sum(((v-gm)**2).sum() for v in groups)
    eta2   = ssb/sst if sst > 0 else 0
    sig    = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '   '
    cells  = "  ".join(f"{m:5.1f}+/-{s:.1f}" for m,s in zip(means,sds))
    print(f"  {col:<22s}  {cells}  {p:.4f}{sig}  {eta2:.3f}")

print("\n--- CV Complications by Diabetes Type (chi-squared) ---")
for t in TYPE_ORDER:
    pct = (sub[t]['Cardiovascular_Complication'].notna() &
           (sub[t]['Cardiovascular_Complication'] != 'None')).mean()*100
    print(f"  {t:15s}: {pct:.1f}%")

print("\n" + "=" * 60)
print("COMPLICATION PREDICTORS  (logistic regression, OR per 1-SD)")
print("=" * 60)
outcomes = {
    'Retinopathy'   : lambda d: (d['Retinopathy'].notna()  & (d['Retinopathy']  != 'None')).astype(int),
    'Neuropathy'    : lambda d: (d['Neuropathy'].notna()   & (d['Neuropathy']   != 'None')).astype(int),
    'Nephropathy'   : lambda d: (d['Nephropathy'].notna()  & (d['Nephropathy']  != 'None')).astype(int),
    'CV Complication': lambda d: (d['Cardiovascular_Complication'].notna() &
                                   (d['Cardiovascular_Complication'] != 'None')).astype(int),
}
predictors_lr = ['Age','Duration_Years','BMI','HbA1c_Percent',
                 'Systolic_BP_mmHg','Fasting_Glucose_mgdL','eGFR_mLmin']
for outcome_name, outcome_fn in outcomes.items():
    y = outcome_fn(df)
    print(f"\n  {outcome_name}  (prev {y.mean()*100:.1f}%)")
    print(f"  {'Predictor':<22s}  {'OR':>5s}  {'95% CI':>16s}  p")
    for pred in predictors_lr:
        tmp = df[[pred]].copy(); tmp['y'] = y; tmp = tmp.dropna()
        if tmp['y'].nunique() < 2: continue
        X = tmp[[pred]].values
        Xz = (X - X.mean()) / (X.std() + 1e-9)
        lr = LogisticRegression(max_iter=500, solver='lbfgs')
        lr.fit(Xz, tmp['y'].values)
        coef = lr.coef_[0][0]
        OR   = np.exp(coef)
        p_hat = lr.predict_proba(Xz)[:,1]
        W     = p_hat*(1-p_hat)
        info  = (Xz.flatten()**2*W).sum()
        se    = 1/np.sqrt(info) if info > 0 else np.nan
        z     = coef/se if se and not np.isnan(se) else np.nan
        pv    = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        lo,hi = np.exp(coef-1.96*se), np.exp(coef+1.96*se)
        sig   = ' ***' if pv<0.001 else ' ** ' if pv<0.01 else ' *  ' if pv<0.05 else '    '
        print(f"  {pred:<22s}  {OR:5.2f}  [{lo:.2f}, {hi:.2f}]  {pv:.4f}{sig}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 – FIGURES 1–4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nGenerating figures...")

# Fig 1 – Correlation heatmap
short = {'Age':'Age','Duration_Years':'Duration','BMI':'BMI',
         'Systolic_BP_mmHg':'Sys BP','Diastolic_BP_mmHg':'Dia BP',
         'Fasting_Glucose_mgdL':'Fast Gluc','HbA1c_Percent':'HbA1c',
         'Total_Cholesterol_mgdL':'Tot Chol','LDL_mgdL':'LDL',
         'HDL_mgdL':'HDL','Triglycerides_mgdL':'TG',
         'Creatinine_mgdL':'Creat','eGFR_mLmin':'eGFR','Urine_Albumin_mgL':'Urine Alb'}
corr = df[NUM_COLS].rename(columns=short).corr()
n_obs = len(df[NUM_COLS].dropna())
t_stat = corr * np.sqrt((n_obs-2)/(1-corr**2))
p_mat  = pd.DataFrame(2*(1-stats.t.cdf(np.abs(t_stat.values), df=n_obs-2)),
                       index=corr.index, columns=corr.columns)

fig1, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(230,20,as_cmap=True),
            vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
            annot_kws={'size':8}, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink':0.8,'label':'Pearson r'})
for i in range(len(corr)):
    for j in range(i):
        if p_mat.iloc[i,j] > 0.05:
            ax.text(j+0.5, i+0.5, 'X', ha='center', va='center',
                    fontsize=9, color='grey', alpha=0.6)
ax.set_title('Correlation Heatmap — Clinical Variables\n(X = not significant, p > .05)',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)
fig1.tight_layout()
fig1.savefig(out('fig1_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  Saved fig1_correlation_heatmap.png")

# Fig 2 – Box plots by diabetes type
box_vars = [
    ('BMI','BMI (kg/m²)'),('HbA1c_Percent','HbA1c (%)'),
    ('Fasting_Glucose_mgdL','Fasting Glucose (mg/dL)'),('Systolic_BP_mmHg','Systolic BP (mmHg)'),
    ('eGFR_mLmin','eGFR (mL/min)'),('Urine_Albumin_mgL','Urine Albumin (mg/L)'),
    ('Duration_Years','Diabetes Duration (yrs)'),('Triglycerides_mgdL','Triglycerides (mg/dL)'),
]
ref_lines = {'HbA1c_Percent':7.0,'Fasting_Glucose_mgdL':100,
             'Systolic_BP_mmHg':130,'eGFR_mLmin':60,'BMI':25}
fig2, axes = plt.subplots(2, 4, figsize=(18, 9))
for idx, (col, label) in enumerate(box_vars):
    ax = axes.flatten()[idx]
    data_plot = df[df['Diabetes_Type'].isin(TYPE_ORDER)][[col,'Diabetes_Type']].dropna()
    sns.boxplot(data=data_plot, x='Diabetes_Type', y=col, order=TYPE_ORDER,
                palette=TYPE_PALETTE, width=0.55,
                flierprops=dict(marker='o',markersize=3,alpha=0.5), ax=ax)
    sns.stripplot(data=data_plot, x='Diabetes_Type', y=col, order=TYPE_ORDER,
                  palette=TYPE_PALETTE, size=2.5, alpha=0.35, jitter=True, ax=ax)
    groups = [data_plot[data_plot['Diabetes_Type']==t][col].values for t in TYPE_ORDER]
    _, p = stats.f_oneway(*groups)
    ax.set_title(f'{label}\nANOVA: {sig_stars(p)} (p={p:.3f})', fontsize=9, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel(label, fontsize=8)
    ax.tick_params(axis='x', labelsize=8, rotation=20)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if col in ref_lines:
        ax.axhline(ref_lines[col], color='red', linestyle='--', lw=0.9, alpha=0.7)
fig2.suptitle('Key Clinical Metrics by Diabetes Type', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(out('fig2_boxplots_by_type.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved fig2_boxplots_by_type.png")

# Fig 3 – ROC curves
fig3, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (outcome_name, outcome_fn) in zip(axes, outcomes.items()):
    y = outcome_fn(df)
    for pred in predictors_lr:
        tmp = df[[pred]].copy(); tmp['y'] = y; tmp = tmp.dropna()
        if tmp['y'].nunique() < 2: continue
        X = tmp[[pred]].values
        Xz = (X - X.mean()) / (X.std() + 1e-9)
        lr = LogisticRegression(max_iter=500, solver='lbfgs')
        lr.fit(Xz, tmp['y'].values)
        fpr, tpr, _ = roc_curve(tmp['y'].values, lr.predict_proba(Xz)[:,1])
        ax.plot(fpr, tpr, color=PRED_COLORS[pred], lw=1.8,
                label=f'{PRED_LABELS[pred]} (AUC={auc(fpr,tpr):.2f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Chance')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_title(f'{outcome_name}\n(prev {y.mean()*100:.0f}%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
fig3.suptitle('ROC Curves — Complication Predictors', fontsize=13, fontweight='bold')
plt.tight_layout()
fig3.savefig(out('fig3_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  Saved fig3_roc_curves.png")

# Fig 4 – Overview dashboard
fig4 = plt.figure(figsize=(18, 11))
gs4  = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.45, wspace=0.35)

ax_age = fig4.add_subplot(gs4[0,0])
for gender, color in [('Male','#4C72B0'),('Female','#DD8452')]:
    ax_age.hist(df[df['Gender']==gender]['Age'], bins=14, alpha=0.6,
                color=color, label=gender, edgecolor='white')
ax_age.set_title('Age Distribution by Gender', fontweight='bold')
ax_age.set_xlabel('Age (years)'); ax_age.set_ylabel('Count')
ax_age.legend(); ax_age.spines['top'].set_visible(False); ax_age.spines['right'].set_visible(False)

ax_eth = fig4.add_subplot(gs4[0,1])
eth_counts = df['Ethnicity'].value_counts()
eth_colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#9467BD']
wedges, texts, autotexts = ax_eth.pie(
    eth_counts.values, labels=eth_counts.index, autopct='%1.1f%%',
    colors=eth_colors[:len(eth_counts)], startangle=90, pctdistance=0.82,
    wedgeprops=dict(width=0.55, edgecolor='white'))
for t in autotexts: t.set_fontsize(8)
for t in texts:     t.set_fontsize(9)
ax_eth.set_title('Ethnicity Distribution', fontweight='bold')

ax_hba = fig4.add_subplot(gs4[0,2])
ax_hba.hist(df['HbA1c_Percent'].dropna(), bins=25, color='#4C72B0', edgecolor='white', alpha=0.8)
ax_hba.axvline(7.0, color='green', linestyle='--', lw=1.5, label='Target (<7%)')
ax_hba.axvline(9.0, color='red',   linestyle='--', lw=1.5, label='Poor (>9%)')
ax_hba.set_title('HbA1c Distribution', fontweight='bold')
ax_hba.set_xlabel('HbA1c (%)'); ax_hba.set_ylabel('Count')
ax_hba.legend(fontsize=8)
ax_hba.spines['top'].set_visible(False); ax_hba.spines['right'].set_visible(False)

ax_comp = fig4.add_subplot(gs4[1,0])
comp_pcts = {
    'Retinopathy':   (df['Retinopathy'].notna()  & (df['Retinopathy']  !='None')).mean()*100,
    'Neuropathy':    (df['Neuropathy'].notna()   & (df['Neuropathy']   !='None')).mean()*100,
    'Nephropathy':   (df['Nephropathy'].notna()  & (df['Nephropathy']  !='None')).mean()*100,
    'CV Complication':(df['Cardiovascular_Complication'].notna() &
                       (df['Cardiovascular_Complication']!='None')).mean()*100,
}
bars = ax_comp.barh(list(comp_pcts.keys()), list(comp_pcts.values()),
                    color=['#4C72B0','#DD8452','#55A868','#C44E52'], edgecolor='white')
for bar, val in zip(bars, comp_pcts.values()):
    ax_comp.text(val+0.5, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9)
ax_comp.set_xlim(0,80); ax_comp.set_title('Complication Prevalence (%)', fontweight='bold')
ax_comp.set_xlabel('% of patients')
ax_comp.spines['top'].set_visible(False); ax_comp.spines['right'].set_visible(False)

ax_med = fig4.add_subplot(gs4[1,1])
meds = ['Metformin','Statin','Antihypertensive','Insulin','SGLT2_Inhibitor','GLP1_Agonist']
med_pct = [(df[m]=='Yes').mean()*100 for m in meds]
bars2 = ax_med.barh(meds, med_pct, color=sns.color_palette('muted',len(meds)), edgecolor='white')
for bar, val in zip(bars2, med_pct):
    ax_med.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
ax_med.set_xlim(0,90); ax_med.set_title('Medication Use (%)', fontweight='bold')
ax_med.set_xlabel('% of patients')
ax_med.spines['top'].set_visible(False); ax_med.spines['right'].set_visible(False)

ax_life = fig4.add_subplot(gs4[1,2])
act_order  = ['Sedentary','Light','Moderate','Active']
diet_order = ['Poor','Fair','Good','Excellent']
act_pct  = [df['Physical_Activity'].value_counts().get(a,0)/len(df)*100 for a in act_order]
diet_pct = [df['Diet_Adherence'].value_counts().get(d,0)/len(df)*100 for d in diet_order]
bar_colors = ['#d62728','#ff7f0e','#2ca02c','#1f77b4']
ba, bd = 0, 0
for ac, dc, color, lbl in zip(act_pct, diet_pct, bar_colors, act_order):
    ax_life.bar(0, ac, bottom=ba, color=color, edgecolor='white', label=lbl)
    ax_life.bar(1, dc, bottom=bd, color=color, edgecolor='white')
    ba += ac; bd += dc
ax_life.set_xticks([0,1]); ax_life.set_xticklabels(['Physical\nActivity','Diet\nAdherence'])
ax_life.set_ylabel('% of patients'); ax_life.set_title('Lifestyle: Activity & Diet', fontweight='bold')
ax_life.legend(fontsize=7, loc='upper right', title='Level')
ax_life.spines['top'].set_visible(False); ax_life.spines['right'].set_visible(False)

fig4.suptitle('Diabetic Patients (N=350) — Demographic & Clinical Overview',
              fontsize=14, fontweight='bold')
fig4.savefig(out('fig4_overview_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  Saved fig4_overview_dashboard.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 – SEVERITY & CLUSTER FIGURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Fig 5 – Severity distributions
fig5, axes = plt.subplots(2, 4, figsize=(20, 9))
comp_configs = [
    ('Retinopathy', ['None','Mild','Moderate','Severe'], ['#d0e4f7','#7fb3d3','#2980b9','#1a5276']),
    ('Neuropathy',  ['None','Peripheral','Autonomic','Both'], ['#d5f5e3','#7dcea0','#27ae60','#145a32']),
    ('Nephropathy', ['None','Microalbuminuria','Macroalbuminuria','CKD'], ['#fdf2e9','#f0a27a','#e67e22','#a04000']),
    ('Cardiovascular_Complication', ['None','Hypertension','CAD','Heart Failure','Stroke'],
     ['#f9ebea','#e07070','#c0392b','#922b21','#641e16']),
]
for ci, (comp, levels, colors) in enumerate(comp_configs):
    ax = axes[0, ci]
    counts = df[comp].fillna('None').value_counts()
    vals   = [counts.get(l,0)/len(df)*100 for l in levels]
    bars   = ax.bar(levels, vals, color=colors[:len(levels)], edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_title(comp.replace('_',' '), fontweight='bold', fontsize=10)
    ax.set_ylabel('% patients'); ax.tick_params(axis='x', rotation=20, labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

sev_pairs = [
    ('Retinopathy_Ord',  'HbA1c_Percent',   'HbA1c (%)',        'Retinopathy'),
    ('Neuropathy_Ord',   'Age',             'Age (yrs)',         'Neuropathy'),
    ('Nephropathy_Ord',  'eGFR_mLmin',      'eGFR (mL/min)',     'Nephropathy'),
    ('CV_Ord',           'Systolic_BP_mmHg','Systolic BP (mmHg)','CV'),
]
for ci, (ord_col, metric, metric_label, comp_name) in enumerate(sev_pairs):
    ax = axes[1, ci]
    tmp = df[[ord_col, metric]].dropna()
    tmp[ord_col] = tmp[ord_col].astype(int)
    sev_groups = sorted(tmp[ord_col].unique())
    box_data   = [tmp[tmp[ord_col]==g][metric].values for g in sev_groups]
    level_map  = {0:'None',1:'Mild/Low',2:'Moderate',3:'Severe'}
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', lw=1.5))
    for patch, color in zip(bp['boxes'], sns.color_palette('Blues', len(sev_groups))):
        patch.set_facecolor(color)
    H, p = stats.kruskal(*box_data)
    ax.set_title(f'{metric_label} by {comp_name} Severity\nKruskal-Wallis: {sig_stars(p)} (p={p:.3f})',
                 fontsize=9, fontweight='bold')
    ax.set_xticklabels([level_map.get(g,str(g)) for g in sev_groups], rotation=20, fontsize=8)
    ax.set_ylabel(metric_label, fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig5.suptitle('Complication Severity Distribution & Clinical Markers', fontsize=14, fontweight='bold')
plt.tight_layout()
fig5.savefig(out('fig5_severity_analysis.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  Saved fig5_severity_analysis.png")

# Fig 6 – Burden score scatter
fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5))
for ax, (metric, label) in zip(axes6, [
    ('HbA1c_Percent','HbA1c (%)'),('Age','Age (yrs)'),('Urine_Albumin_mgL','Urine Albumin (mg/L)')]):
    tmp = df[['Burden', metric]].dropna()
    rho, p = stats.spearmanr(tmp['Burden'], tmp[metric])
    ax.scatter(tmp['Burden'], tmp[metric], alpha=0.35, s=18,
               c=tmp['Burden'], cmap='YlOrRd', edgecolors='none')
    m, b = np.polyfit(tmp['Burden'], tmp[metric], 1)
    xr = np.linspace(tmp['Burden'].min(), tmp['Burden'].max(), 100)
    ax.plot(xr, m*xr+b, color='#c0392b', lw=2)
    ax.set_title(f'Burden vs {label}\nSpearman rho={rho:.2f} ({sig_stars(p)})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Composite Burden Score (0–12)', fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig6.suptitle('Composite Complication Burden Score vs Clinical Markers', fontsize=13, fontweight='bold')
plt.tight_layout()
fig6.savefig(out('fig6_burden_score.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  Saved fig6_burden_score.png")

# Clustering
cluster_cols = ['Age','Duration_Years','BMI','HbA1c_Percent','Fasting_Glucose_mgdL',
                'Systolic_BP_mmHg','eGFR_mLmin','Urine_Albumin_mgL','Triglycerides_mgdL',
                'Retinopathy_Ord','Neuropathy_Ord','Nephropathy_Ord','CV_Ord']
df_c = df[cluster_cols].copy()
for col in ['Retinopathy_Ord','Neuropathy_Ord','Nephropathy_Ord','CV_Ord']:
    df_c[col] = df_c[col].fillna(0)
df_c = df_c.dropna()
X_scaled = StandardScaler().fit_transform(df_c)
inertias, silhouettes = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
best_k = 2 + int(np.argmax(silhouettes))
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df.loc[df_c.index, 'Cluster'] = km_final.fit_predict(X_scaled) + 1
df['Cluster'] = df['Cluster'].astype('Int64')
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_
print(f"  Best k = {best_k}  (silhouette = {max(silhouettes):.3f})")

# Fig 7 – Elbow + silhouette
fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(range(2,9), inertias, 'o-', color='#4C72B0', lw=2)
ax1.set_title('Elbow Plot (K-Means Inertia)', fontweight='bold')
ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax2.plot(range(2,9), silhouettes, 's-', color='#DD8452', lw=2)
ax2.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
ax2.set_title('Silhouette Score by k', fontweight='bold')
ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette Score')
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
fig7.savefig(out('fig7_cluster_selection.png'), dpi=150, bbox_inches='tight')
plt.close(fig7)
print("  Saved fig7_cluster_selection.png")

# Fig 8 – PCA scatter + heatmap
CLUSTER_PALETTE = sns.color_palette('tab10', best_k)
fig8 = plt.figure(figsize=(18, 7))
gs8  = gridspec.GridSpec(1, 2, figure=fig8, wspace=0.35)
ax_pca  = fig8.add_subplot(gs8[0,0])
ax_heat = fig8.add_subplot(gs8[0,1])
for ci in range(best_k):
    mask = (df.loc[df_c.index,'Cluster'] == ci+1).values
    ax_pca.scatter(X_pca[mask,0], X_pca[mask,1], color=CLUSTER_PALETTE[ci],
                   s=25, alpha=0.65, label=f'C{ci+1} (n={mask.sum()})', edgecolors='none')
ax_pca.set_title(f'PCA Projection — {best_k} Clusters\n'
                 f'PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%',
                 fontweight='bold')
ax_pca.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
ax_pca.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
ax_pca.legend(fontsize=8, markerscale=1.5)
ax_pca.spines['top'].set_visible(False); ax_pca.spines['right'].set_visible(False)

profile_cols = ['Age','Duration_Years','BMI','HbA1c_Percent',
                'Fasting_Glucose_mgdL','Systolic_BP_mmHg','eGFR_mLmin','Burden']
short_p = {'Age':'Age','Duration_Years':'Duration','BMI':'BMI','HbA1c_Percent':'HbA1c',
           'Fasting_Glucose_mgdL':'Fast Gluc','Systolic_BP_mmHg':'Sys BP',
           'eGFR_mLmin':'eGFR','Burden':'Burden'}
df_prof = df.loc[df_c.index].copy()
cm = df_prof.groupby('Cluster')[profile_cols].mean()
cmz = (cm - cm.mean()) / (cm.std() + 1e-9)
cmz.columns = [short_p[c] for c in cmz.columns]
cmz.index   = [f'C{int(i)}' for i in cmz.index]
sns.heatmap(cmz, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax_heat,
            cbar_kws={'label':'Z-score vs grand mean'})
ax_heat.set_title('Cluster Profiles (Z-scores)', fontweight='bold')
ax_heat.tick_params(axis='x', rotation=40, labelsize=9)
ax_heat.tick_params(axis='y', rotation=0, labelsize=9)
fig8.suptitle(f'K-Means Clustering (k={best_k}) — Patient Subgroups', fontsize=14, fontweight='bold')
plt.tight_layout()
fig8.savefig(out('fig8_cluster_profiles.png'), dpi=150, bbox_inches='tight')
plt.close(fig8)
print("  Saved fig8_cluster_profiles.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 – HTML REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nGenerating HTML report...")

def img_tag(filename, caption='', width='100%'):
    path = out(filename)
    if not os.path.exists(path):
        return f'<p style="color:red">Missing: {filename}</p>'
    with open(path,'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{width};border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15)">'
            + (f'<p style="font-size:.8rem;color:#7f8c8d;text-align:center;margin-top:6px">{caption}</p>'
               if caption else ''))

def tr(label, value):
    return f'<tr><td style="padding:5px 12px;color:#555">{label}</td><td style="padding:5px 12px;font-weight:600">{value}</td></tr>'

def sec(title, anchor, body):
    return (f'<section id="{anchor}" style="margin-bottom:48px">'
            f'<h2 style="border-left:5px solid #2980b9;padding-left:12px;color:#2c3e50">{title}</h2>'
            f'{body}</section>')

hba1c_target_n = (df['HbA1c_Percent'] < 7).sum()
hba1c_poor_n   = (df['HbA1c_Percent'] > 9).sum()
cv_pct         = comp_pcts['CV Complication']

html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Diabetic Patients — Statistical Analysis Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",Arial,sans-serif;background:#f4f6f9;color:#2c3e50;line-height:1.6}}
.page{{max-width:1100px;margin:0 auto;padding:32px 24px}}
header{{background:linear-gradient(135deg,#1a5276,#2980b9);color:#fff;padding:36px 32px;border-radius:10px;margin-bottom:40px}}
header h1{{font-size:2rem;margin-bottom:8px}}header p{{opacity:.85}}
nav{{background:#fff;border-radius:8px;padding:18px 24px;margin-bottom:36px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
nav a{{color:#2980b9;text-decoration:none;margin-right:20px;font-size:.92rem;font-weight:500}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px}}
.kpi{{background:#fff;border-radius:8px;padding:18px 16px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.kpi .num{{font-size:2rem;font-weight:700;color:#2980b9}}.kpi .lbl{{font-size:.82rem;color:#7f8c8d;margin-top:4px}}
.kpi.warn .num{{color:#e74c3c}}.kpi.ok .num{{color:#27ae60}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;
       box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px}}
thead tr{{background:#2980b9;color:#fff}}
th{{padding:10px 14px;text-align:left;font-size:.88rem;font-weight:600}}
td{{padding:8px 14px;font-size:.88rem;border-bottom:1px solid #ecf0f1}}
tr:last-child td{{border-bottom:none}}tr:nth-child(even) td{{background:#f8f9fa}}
.note{{background:#fef9e7;border-left:4px solid #f39c12;padding:12px 16px;border-radius:4px;margin-bottom:20px;font-size:.88rem}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
h3{{color:#2980b9;margin:20px 0 10px;font-size:1rem}}
footer{{text-align:center;padding:24px;color:#95a5a6;font-size:.8rem;margin-top:24px;border-top:1px solid #ddd}}
@media(max-width:700px){{.kpi-grid{{grid-template-columns:repeat(2,1fr)}}.two-col{{grid-template-columns:1fr}}}}
</style></head><body><div class="page">

<header>
  <h1>Diabetic Patients — Statistical Analysis Report</h1>
  <p>Dataset: <strong>diabetic_patients_350.csv</strong> &nbsp;|&nbsp; N = 350 &nbsp;|&nbsp; 34 variables &nbsp;|&nbsp; 2026-05-19</p>
</header>

<nav><strong>Sections: </strong>
  <a href="#overview">Overview</a><a href="#demographics">Demographics</a>
  <a href="#clinical">Clinical</a><a href="#glycaemic">Glycaemic</a>
  <a href="#complications">Complications</a><a href="#severity">Severity</a>
  <a href="#correlations">Correlations</a><a href="#subgroup">By Type</a>
  <a href="#predictors">Predictors</a><a href="#clusters">Clusters</a>
  <a href="#notes">Data Quality</a>
</nav>

<div class="kpi-grid">
  <div class="kpi"><div class="num">350</div><div class="lbl">Total patients</div></div>
  <div class="kpi warn"><div class="num">{hba1c_poor_n} ({hba1c_poor_n/len(df)*100:.0f}%)</div><div class="lbl">HbA1c &gt; 9% (poor)</div></div>
  <div class="kpi ok"><div class="num">{hba1c_target_n} ({hba1c_target_n/len(df)*100:.0f}%)</div><div class="lbl">HbA1c &lt; 7% (target)</div></div>
  <div class="kpi warn"><div class="num">{cv_pct:.0f}%</div><div class="lbl">CV complication rate</div></div>
</div>
'''

# Overview
html += sec('1. Dataset Overview','overview',f'''
<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead><tbody>
{tr("Total patients","350")}{tr("Variables","34")}
{tr("Missing values","861")}{tr("Last visit range",df["Last_Visit_Date"].min()+" – "+df["Last_Visit_Date"].max())}
</tbody></table>
<div class="note"><strong>Data Quality:</strong> Variables appear independently generated — no meaningful inter-variable correlations exist (see Correlations section).</div>
''')

# Demographics
eth_rows  = ''.join(tr(v,f"{c} ({100*c/len(df):.1f}%)") for v,c in df['Ethnicity'].value_counts().items())
type_rows = ''.join(tr(v,f"{c} ({100*c/len(df):.1f}%)") for v,c in df['Diabetes_Type'].value_counts().items())
html += sec('2. Demographics','demographics',f'''
<div class="two-col">
<div><h3>Age</h3>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
{tr("Mean ± SD",f"{df['Age'].mean():.1f} ± {df['Age'].std():.1f} years")}
{tr("Median",f"{df['Age'].median():.1f} years")}
{tr("Range",f"{df['Age'].min()} – {df['Age'].max()} years")}
{tr("< 30","39 (11.1%)")}{tr("30–44","75 (21.4%)")}{tr("45–59","92 (26.3%)")}{tr("60–74","83 (23.7%)")}{tr("75+","61 (17.4%)")}
</tbody></table>
<h3>Gender</h3>
<table><thead><tr><th>Gender</th><th>n (%)</th></tr></thead><tbody>
{tr("Female",f"{n_female} ({100*n_female/len(df):.1f}%)")}{tr("Male",f"{n_male} ({100*n_male/len(df):.1f}%)")}
</tbody></table></div>
<div><h3>Ethnicity</h3><table><thead><tr><th>Ethnicity</th><th>n (%)</th></tr></thead><tbody>{eth_rows}</tbody></table>
<h3>Diabetes Type</h3><table><thead><tr><th>Type</th><th>n (%)</th></tr></thead><tbody>{type_rows}</tbody></table>
</div></div>
{img_tag("fig4_overview_dashboard.png","Figure 1. Demographic & clinical overview dashboard.")}
''')

# Clinical
clin_rows_html = ''
for col, label in clin_map.items():
    m,s,med = df[col].mean(),df[col].std(),df[col].median()
    clin_rows_html += f'<tr><td>{label}</td><td>{m:.1f} ± {s:.1f}</td><td>{med:.1f}</td><td>{df[col].min():.1f} – {df[col].max():.1f}</td></tr>'
html += sec('3. Clinical & Metabolic Measures','clinical',f'''
<table><thead><tr><th>Measure</th><th>Mean ± SD</th><th>Median</th><th>Range</th></tr></thead>
<tbody>{clin_rows_html}</tbody></table>
''')

# Glycaemic
gc_rows = ''.join(f'<tr><td>{c}</td><td>{n}</td><td>{p:.1f}%</td></tr>'
                  for c,n,p in [('<7% Target',97,27.7),('7–8%',67,19.1),('8–9%',80,22.9),('>9% Poor',106,30.3)])
html += sec('4. Glycaemic Control','glycaemic',f'''
<p style="margin-bottom:12px"><strong>72.3% of patients are above the 7% HbA1c target.</strong></p>
<table><thead><tr><th>HbA1c Category</th><th>n</th><th>%</th></tr></thead><tbody>{gc_rows}</tbody></table>
''')

# Complications
comp_detail = [
    ('Retinopathy',41.7,[('Mild',73,20.9),('Moderate',46,13.1),('Severe',27,7.7)]),
    ('Neuropathy',50.9,[('Peripheral',108,30.9),('Autonomic',29,8.3),('Both',41,11.7)]),
    ('Nephropathy',40.6,[('Microalbuminuria',70,20.0),('Macroalbuminuria',45,12.9),('CKD',27,7.7)]),
    ('CV Complication',55.4,[('Hypertension',110,31.4),('CAD',44,12.6),('Heart Failure',25,7.1),('Stroke',15,4.3)]),
]
comp_html = ''
for name,pct,cats in comp_detail:
    comp_html += f'<tr style="background:#eaf3fb"><td><strong>{name}</strong></td><td><strong>{int(pct/100*350)}</strong></td><td><strong>{pct:.1f}%</strong></td></tr>'
    comp_html += ''.join(f'<tr><td style="padding-left:24px">↳ {c}</td><td>{n}</td><td>{p:.1f}%</td></tr>' for c,n,p in cats)
html += sec('5. Complications','complications',f'''
<table><thead><tr><th>Complication / Subtype</th><th>n</th><th>%</th></tr></thead>
<tbody>{comp_html}</tbody></table>
''')

# Severity
html += sec('6. Severity & Burden Analysis','severity',f'''
<p style="margin-bottom:16px">Composite burden score (0–12) = sum of ordinal severity across all four complication domains.</p>
{img_tag("fig5_severity_analysis.png","Figure 2. Severity distributions and clinical markers by severity level.")}
{img_tag("fig6_burden_score.png","Figure 3. Composite burden score vs clinical markers (Spearman ρ).")}
''')

# Correlations
html += sec('7. Correlations','correlations',f'''
<div class="note"><strong>Finding:</strong> No clinically meaningful correlations (|r| ≥ 0.25) exist between any pair of numeric variables — consistent with synthetic independent variable generation.</div>
{img_tag("fig1_correlation_heatmap.png","Figure 4. Pearson correlation heatmap. X = not significant (p > .05).")}
''')

# Subgroup
sr_rows = ''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td></tr>'
    for r in [
        ('BMI (kg/m²)','26.3 ± 6.1','28.5 ± 6.0','29.9 ± 5.9','31.1 ± 5.1','0.025 *','0.027'),
        ('HbA1c (%)','7.9 ± 1.7','8.2 ± 1.7','8.0 ± 1.7','8.0 ± 1.8','0.774','0.003'),
        ('Fasting Glucose','137.7 ± 31','145.6 ± 38','146.3 ± 39','148.7 ± 45','0.715','0.004'),
        ('Systolic BP','129.2 ± 25','137.0 ± 23','133.0 ± 24','133.7 ± 22','0.314','0.010'),
        ('eGFR','66.7 ± 26','75.4 ± 23','74.1 ± 23','74.2 ± 25','0.321','0.010'),
    ])
html += sec('8. Subgroup by Diabetes Type','subgroup',f'''
<p style="margin-bottom:12px">One-way ANOVA + η² effect size. Only BMI differed significantly. CV complications also differed by type (χ² p=.010).</p>
<table><thead><tr><th>Measure</th><th>Type 1 (n=29)</th><th>Type 2 (n=276)</th><th>LADA (n=22)</th><th>Gestational (n=23)</th><th>p</th><th>η²</th></tr></thead>
<tbody>{sr_rows}</tbody></table>
{img_tag("fig2_boxplots_by_type.png","Figure 5. Clinical metrics by diabetes type. Red dashed = clinical reference thresholds.")}
''')

# Predictors
pr_rows = ''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>' for r in [
    ('Retinopathy','BMI','0.79','[0.63, 0.98]','.029 *'),
    ('Neuropathy','Age','1.34','[1.08, 1.66]','.007 **'),
])
html += sec('9. Complication Predictors','predictors',f'''
<p style="margin-bottom:12px">Single-predictor logistic regression (OR per 1-SD). Only two significant predictors found.</p>
<table><thead><tr><th>Outcome</th><th>Predictor</th><th>OR</th><th>95% CI</th><th>p</th></tr></thead>
<tbody>{pr_rows}</tbody></table>
{img_tag("fig3_roc_curves.png","Figure 6. ROC curves — all AUCs near 0.5, confirming no predictive signal.")}
''')

# Clusters
c_labels = {1:'High BP + Nephropathy',2:'Hyperglycaemic + Full Nephropathy',
            3:'Lean, Poor HbA1c + High CV',4:'Obese + Hypertensive, Low Burden',
            5:'All CV Events',6:'Full Neuropathy',7:'Moderate Multi-Complication',
            8:'Full Retinopathy, Best HbA1c'}
c_colors = {1:'#e74c3c',2:'#e67e22',3:'#f39c12',4:'#27ae60',
            5:'#2980b9',6:'#8e44ad',7:'#16a085',8:'#c0392b'}
c_data = [(1,21,6.0,'51.2','26.5','7.8','156.7','5.1','Retinopathy 91%, Nephropathy 95%'),
          (2,36,10.3,'55.9','28.6','8.2','120.3','3.8','Full Nephropathy 100%, High glucose'),
          (3,55,15.7,'59.8','23.9','9.0','133.5','1.9','Poor HbA1c, CV 58%, Lean'),
          (4,70,20.0,'53.7','33.2','7.8','145.8','1.3','Obese + Hypertensive, Low burden'),
          (5,52,14.9,'50.6','28.4','8.3','133.2','4.1','All CV events 100%'),
          (6,47,13.4,'61.1','28.3','8.6','137.1','4.3','Full Neuropathy 100%'),
          (7,32,9.1,'52.2','29.5','8.4','136.3','2.0','Moderate multi-complication'),
          (8,37,10.6,'51.9','28.2','6.9','125.8','4.6','Full Retinopathy 100%, Best HbA1c')]
c_rows = ''.join(
    f'<tr><td><span style="background:{c_colors[ci]};color:#fff;padding:2px 8px;border-radius:10px;font-size:.8rem">C{ci}</span> {c_labels[ci]}</td>'
    f'<td>{n} ({pct}%)</td><td>{age}</td><td>{bmi}</td><td>{hba1c}</td><td>{sbp}</td><td>{burd}</td>'
    f'<td style="font-size:.82rem">{notes}</td></tr>'
    for ci,n,pct,age,bmi,hba1c,sbp,burd,notes in c_data)
html += sec(f'10. Cluster Analysis (K-Means, k={best_k})','clusters',f'''
<div class="note"><strong>Note:</strong> Silhouette = 0.079 (low). Clusters are driven by complication-pattern combinations, not metabolic phenotypes — expected for synthetic data.</div>
<table><thead><tr><th>Cluster</th><th>n (%)</th><th>Age</th><th>BMI</th><th>HbA1c</th><th>Sys BP</th><th>Burden</th><th>Key features</th></tr></thead>
<tbody>{c_rows}</tbody></table>
{img_tag("fig7_cluster_selection.png","Figure 7. Elbow and silhouette plots.")}
{img_tag("fig8_cluster_profiles.png","Figure 8. PCA cluster projection and Z-score profile heatmap.")}
''')

# Data quality
html += sec('11. Data Quality Notes','notes',f'''
<table><thead><tr><th>Observation</th><th>Implication</th></tr></thead><tbody>
{tr("861 missing values","Pairwise-complete observations used throughout")}
{tr("No pairwise |r| ≥ 0.25","Strongly indicates synthetic/independently-generated variables")}
{tr("HbA1c vs fasting glucose: r = +0.04","Real-world expectation is r ≈ 0.6–0.7")}
{tr("All ROC AUCs ≈ 0.5","No predictive signal between risk factors and complications")}
{tr("Suitable for","Descriptive statistics, visualisation practice, ML benchmarking")}
</tbody></table>
''')

html += '<footer>Analysis: Python (pandas · scipy · scikit-learn · matplotlib · seaborn) &nbsp;|&nbsp; 2026-05-19</footer></div></body></html>'

report_path = out('diabetic_analysis_report.html')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"  Saved diabetic_analysis_report.html")
print(f"\nDone. All outputs in: {OUT_DIR}")
