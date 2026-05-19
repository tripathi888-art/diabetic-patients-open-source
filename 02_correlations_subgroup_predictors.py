import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\tripa\OneDrive\Desktop\diabetic_patients_350.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: CORRELATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("PART 1: CORRELATIONS (Pearson r, p-value)")
print("=" * 65)

num_cols = [
    'Age', 'Duration_Years', 'BMI', 'Systolic_BP_mmHg', 'Diastolic_BP_mmHg',
    'Fasting_Glucose_mgdL', 'HbA1c_Percent', 'Total_Cholesterol_mgdL',
    'LDL_mgdL', 'HDL_mgdL', 'Triglycerides_mgdL',
    'Creatinine_mgdL', 'eGFR_mLmin', 'Urine_Albumin_mgL'
]

# Key pairwise correlations sorted by |r|
print("\nKey pairwise correlations (|r| >= 0.25, sorted by strength):\n")
print(f"  {'Variable 1':<30s} {'Variable 2':<30s}  {'r':>6s}  {'p':>7s}")
print("  " + "-" * 76)

results = []
for i, c1 in enumerate(num_cols):
    for c2 in num_cols[i+1:]:
        pair = df[[c1, c2]].dropna()
        if len(pair) < 10:
            continue
        r, p = stats.pearsonr(pair[c1], pair[c2])
        results.append((c1, c2, r, p))

results.sort(key=lambda x: abs(x[2]), reverse=True)
shown = 0
for c1, c2, r, p in results:
    if abs(r) < 0.25:
        break
    sig = ' ***' if p < 0.001 else ' ** ' if p < 0.01 else ' *  ' if p < 0.05 else '    '
    print(f"  {c1:<30s} {c2:<30s}  {r:+6.3f}  {p:7.4f}{sig}")
    shown += 1

print(f"\n  * p<.05  ** p<.01  *** p<.001  |  showing {shown} pairs with |r|>=0.25")

# HbA1c specific correlations
print("\n\nCorrelations with HbA1c (all variables, sorted by |r|):\n")
print(f"  {'Variable':<35s}  {'r':>6s}  {'p':>7s}")
print("  " + "-" * 52)
hba1c_corrs = []
for col in num_cols:
    if col == 'HbA1c_Percent':
        continue
    pair = df[['HbA1c_Percent', col]].dropna()
    r, p = stats.pearsonr(pair['HbA1c_Percent'], pair[col])
    hba1c_corrs.append((col, r, p))
hba1c_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for col, r, p in hba1c_corrs:
    sig = ' ***' if p < 0.001 else ' ** ' if p < 0.01 else ' *  ' if p < 0.05 else '    '
    print(f"  {col:<35s}  {r:+6.3f}  {p:7.4f}{sig}")

# BMI vs BP
print("\n\nSpearman correlations (robust, for skewed variables):\n")
spearman_pairs = [
    ('BMI', 'Systolic_BP_mmHg'),
    ('BMI', 'Triglycerides_mgdL'),
    ('BMI', 'Urine_Albumin_mgL'),
    ('Duration_Years', 'Urine_Albumin_mgL'),
    ('Duration_Years', 'eGFR_mLmin'),
    ('Age', 'eGFR_mLmin'),
    ('Age', 'Urine_Albumin_mgL'),
]
print(f"  {'Variable 1':<30s} {'Variable 2':<30s}  {'rho':>6s}  {'p':>7s}")
print("  " + "-" * 76)
for c1, c2 in spearman_pairs:
    pair = df[[c1, c2]].dropna()
    rho, p = stats.spearmanr(pair[c1], pair[c2])
    sig = ' ***' if p < 0.001 else ' ** ' if p < 0.01 else ' *  ' if p < 0.05 else '    '
    print(f"  {c1:<30s} {c2:<30s}  {rho:+6.3f}  {p:7.4f}{sig}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: SUBGROUP COMPARISON BY DIABETES TYPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 65)
print("PART 2: SUBGROUP COMPARISON BY DIABETES TYPE (one-way ANOVA)")
print("=" * 65)

types = ['Type 1', 'Type 2', 'LADA', 'Gestational']
sub = {t: df[df['Diabetes_Type'] == t] for t in types}
ns   = {t: len(sub[t]) for t in types}
print(f"\n  Sample sizes: " + "  |  ".join(f"{t}: n={ns[t]}" for t in types))

compare_cols = {
    'Age'                      : 'Age (yrs)',
    'Duration_Years'           : 'Duration (yrs)',
    'BMI'                      : 'BMI (kg/m2)',
    'HbA1c_Percent'            : 'HbA1c (%)',
    'Fasting_Glucose_mgdL'     : 'Fasting Glucose',
    'Systolic_BP_mmHg'         : 'Systolic BP',
    'eGFR_mLmin'               : 'eGFR',
    'Urine_Albumin_mgL'        : 'Urine Albumin',
    'Total_Cholesterol_mgdL'   : 'Total Cholesterol',
    'Triglycerides_mgdL'       : 'Triglycerides',
}

header = f"  {'Measure':<22s} " + "  ".join(f"{'T1':>12s}  {'T2':>12s}  {'LADA':>12s}  {'Gest':>12s}")
print("\n  " + "-" * 90)
print(f"  {'Measure':<22s}  {'Type 1 (n=29)':>14s}  {'Type 2 (n=276)':>14s}  {'LADA (n=22)':>12s}  {'Gest (n=23)':>12s}  {'F-p':>7s}  eta2")
print("  " + "-" * 90)

for col, label in compare_cols.items():
    groups = [sub[t][col].dropna().values for t in types]
    means  = [g.mean() for g in groups]
    sds    = [g.std()  for g in groups]
    F, p   = stats.f_oneway(*groups)
    # eta-squared
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (m - grand_mean)**2 for g, m in zip(groups, means))
    ss_total   = sum(((v - grand_mean)**2).sum() for v in groups)
    eta2       = ss_between / ss_total if ss_total > 0 else 0
    sig        = '***' if p < 0.001 else '** ' if p < 0.01 else '*  ' if p < 0.05 else '   '
    cells = "  ".join(f"{m:6.1f}+/-{s:.1f}" for m, s in zip(means, sds))
    print(f"  {label:<22s}  {cells}  {p:7.4f}{sig}  {eta2:.3f}")

print("\n  * p<.05  ** p<.01  *** p<.001  |  eta2: 0.01 small, 0.06 medium, 0.14 large")

# Medication use by type
print("\n\nMedication use by diabetes type (%):\n")
meds = ['Insulin', 'Metformin', 'SGLT2_Inhibitor', 'GLP1_Agonist', 'Statin']
print(f"  {'Medication':<20s}  {'T1':>8s}  {'T2':>8s}  {'LADA':>8s}  {'Gest':>8s}  chi2-p")
print("  " + "-" * 65)
for m in meds:
    pcts = []
    obs  = []
    for t in types:
        g = sub[t]
        yes = (g[m] == 'Yes').sum()
        no  = (g[m] == 'No').sum()
        pcts.append(100 * yes / len(g) if len(g) > 0 else 0)
        obs.append([yes, no])
    ct = np.array(obs)
    try:
        chi2, p, _, _ = stats.chi2_contingency(ct)
        sig = '***' if p < 0.001 else '** ' if p < 0.01 else '*  ' if p < 0.05 else '   '
        pstr = f"{p:7.4f}{sig}"
    except Exception:
        pstr = "   N/A  "
    row = "  ".join(f"{pc:7.1f}%" for pc in pcts)
    print(f"  {m:<20s}  {row}  {pstr}")

# Complication rates by type
print("\n\nComplication prevalence by diabetes type (%):\n")
comp_cols = ['Retinopathy', 'Neuropathy', 'Nephropathy', 'Cardiovascular_Complication']
print(f"  {'Complication':<28s}  {'T1':>8s}  {'T2':>8s}  {'LADA':>8s}  {'Gest':>8s}  chi2-p")
print("  " + "-" * 72)
for col in comp_cols:
    pcts = []
    obs  = []
    for t in types:
        g = sub[t]
        has = (g[col].notna() & (g[col] != 'None')).sum()
        no  = len(g) - has
        pcts.append(100 * has / len(g) if len(g) > 0 else 0)
        obs.append([has, no])
    ct = np.array(obs)
    try:
        chi2, p, _, _ = stats.chi2_contingency(ct)
        sig = '***' if p < 0.001 else '** ' if p < 0.01 else '*  ' if p < 0.05 else '   '
        pstr = f"{p:7.4f}{sig}"
    except Exception:
        pstr = "   N/A  "
    row = "  ".join(f"{pc:7.1f}%" for pc in pcts)
    print(f"  {col:<28s}  {row}  {pstr}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: COMPLICATION PREDICTORS (Logistic Regression + Odds Ratios)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 65)
print("PART 3: COMPLICATION PREDICTORS")
print("Logistic regression — OR (95% CI) per 1-SD increase in predictor")
print("=" * 65)

predictors = ['Age', 'Duration_Years', 'BMI', 'HbA1c_Percent',
              'Systolic_BP_mmHg', 'Fasting_Glucose_mgdL', 'eGFR_mLmin']

outcomes = {
    'Any_Retinopathy'  : lambda d: (d['Retinopathy'].notna() & (d['Retinopathy'] != 'None')).astype(int),
    'Any_Neuropathy'   : lambda d: (d['Neuropathy'].notna()  & (d['Neuropathy']  != 'None')).astype(int),
    'Any_Nephropathy'  : lambda d: (d['Nephropathy'].notna() & (d['Nephropathy'] != 'None')).astype(int),
    'Any_CV_Complication': lambda d: (d['Cardiovascular_Complication'].notna() &
                                      (d['Cardiovascular_Complication'] != 'None')).astype(int),
}

for outcome_name, outcome_fn in outcomes.items():
    print(f"\n  Outcome: {outcome_name}")
    print(f"  {'Predictor':<25s}  {'OR':>6s}  {'95% CI':>18s}  {'p':>7s}")
    print("  " + "-" * 62)

    y = outcome_fn(df)
    prev = y.mean()
    print(f"  Prevalence: {prev*100:.1f}%  (n={y.sum()} / {len(y)})\n")

    for pred in predictors:
        sub2 = df[[pred]].copy()
        sub2['y'] = y
        sub2 = sub2.dropna()
        if sub2['y'].nunique() < 2:
            continue
        X = sub2[[pred]].values
        yv = sub2['y'].values

        # Standardize
        sd = X.std()
        Xz = (X - X.mean()) / sd if sd > 0 else X

        # Logistic regression
        lr = LogisticRegression(max_iter=500, solver='lbfgs')
        lr.fit(Xz, yv)
        coef = lr.coef_[0][0]
        OR   = np.exp(coef)

        # Wald CI approximation via manual Hessian
        p_hat = lr.predict_proba(Xz)[:, 1]
        W     = p_hat * (1 - p_hat)
        info  = (Xz.flatten()**2 * W).sum()
        se    = 1.0 / np.sqrt(info) if info > 0 else np.nan
        z     = coef / se if se and not np.isnan(se) else np.nan
        p_val = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        lo    = np.exp(coef - 1.96 * se) if se else np.nan
        hi    = np.exp(coef + 1.96 * se) if se else np.nan

        sig   = ' ***' if p_val < 0.001 else ' ** ' if p_val < 0.01 else ' *  ' if p_val < 0.05 else '    '
        ci    = f"[{lo:.2f}, {hi:.2f}]" if not (np.isnan(lo) or np.isnan(hi)) else "N/A"
        pstr  = f"{p_val:.4f}" if not np.isnan(p_val) else " N/A  "
        print(f"  {pred:<25s}  {OR:6.2f}  {ci:>18s}  {pstr}{sig}")

print("\n  OR > 1 = increases odds; OR < 1 = reduces odds")
print("  * p<.05  ** p<.01  *** p<.001")
print("  All predictors standardized (per 1-SD increase)")
