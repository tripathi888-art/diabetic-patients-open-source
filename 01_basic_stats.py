import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r"C:\Users\tripa\OneDrive\Desktop\diabetic_patients_350.csv")

# ── DATASET OVERVIEW ──────────────────────────────────────────────
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total patients : {len(df)}")
print(f"Columns        : {df.shape[1]}")
print(f"Missing values : {df.isnull().sum().sum()}")

# ── DEMOGRAPHICS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEMOGRAPHICS")
print("=" * 60)

print("\n--- Age ---")
print(f"  Mean +/- SD : {df['Age'].mean():.1f} +/- {df['Age'].std():.1f} years")
print(f"  Median      : {df['Age'].median():.1f}")
print(f"  Range       : {df['Age'].min()}-{df['Age'].max()}")
bins = [0, 30, 45, 60, 75, 200]
labels = ['<30', '30-44', '45-59', '60-74', '75+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
ag = df['AgeGroup'].value_counts().sort_index()
for g, c in ag.items():
    print(f"  {g:8s}: {c:4d} ({100*c/len(df):.1f}%)")

print("\n--- Gender ---")
for v, c in df['Gender'].value_counts().items():
    print(f"  {v:20s}: {c:4d} ({100*c/len(df):.1f}%)")

print("\n--- Ethnicity ---")
for v, c in df['Ethnicity'].value_counts().items():
    print(f"  {v:20s}: {c:4d} ({100*c/len(df):.1f}%)")

print("\n--- Diabetes Type ---")
for v, c in df['Diabetes_Type'].value_counts().items():
    print(f"  {v:20s}: {c:4d} ({100*c/len(df):.1f}%)")

# ── CLINICAL MEASURES ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLINICAL / METABOLIC MEASURES  (Mean +/- SD  |  Median  |  Range)")
print("=" * 60)

clin = {
    'Duration_Years'           : 'Diabetes Duration (yrs)',
    'BMI'                      : 'BMI (kg/m2)',
    'Systolic_BP_mmHg'         : 'Systolic BP (mmHg)',
    'Diastolic_BP_mmHg'        : 'Diastolic BP (mmHg)',
    'Fasting_Glucose_mgdL'     : 'Fasting Glucose (mg/dL)',
    'Postprandial_Glucose_mgdL': 'Postprandial Glucose (mg/dL)',
    'HbA1c_Percent'            : 'HbA1c (%)',
    'Total_Cholesterol_mgdL'   : 'Total Cholesterol (mg/dL)',
    'LDL_mgdL'                 : 'LDL (mg/dL)',
    'HDL_mgdL'                 : 'HDL (mg/dL)',
    'Triglycerides_mgdL'       : 'Triglycerides (mg/dL)',
    'Creatinine_mgdL'          : 'Creatinine (mg/dL)',
    'eGFR_mLmin'               : 'eGFR (mL/min)',
    'Urine_Albumin_mgL'        : 'Urine Albumin (mg/L)',
}
for col, label in clin.items():
    m, s, med = df[col].mean(), df[col].std(), df[col].median()
    lo, hi = df[col].min(), df[col].max()
    print(f"  {label:<35s}: {m:6.1f} +/- {s:.1f}  |  {med:6.1f}  |  {lo:.1f}-{hi:.1f}")

# ── LIFESTYLE ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LIFESTYLE FACTORS")
print("=" * 60)
for col in ['Smoking_Status', 'Alcohol_Use', 'Physical_Activity', 'Diet_Adherence']:
    print(f"\n--- {col.replace('_', ' ')} ---")
    for v, c in df[col].value_counts().items():
        print(f"  {str(v):20s}: {c:4d} ({100*c/len(df):.1f}%)")

# ── MEDICATIONS ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEDICATION USE  (% on each medication)")
print("=" * 60)
meds = ['Insulin', 'Metformin', 'SGLT2_Inhibitor', 'GLP1_Agonist', 'Statin', 'Antihypertensive']
for m in meds:
    yes = (df[m] == 'Yes').sum()
    print(f"  {m:25s}: {yes:4d} ({100*yes/len(df):.1f}%)")

# ── COMPLICATIONS ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPLICATIONS")
print("=" * 60)
for col in ['Retinopathy', 'Neuropathy', 'Nephropathy', 'Cardiovascular_Complication']:
    has = (df[col] != 'None').sum()
    pct = 100 * has / len(df)
    print(f"\n--- {col.replace('_', ' ')} --- ({has}, {pct:.1f}%)")
    for v, c in df[col].value_counts().items():
        print(f"  {str(v):25s}: {c:4d} ({100*c/len(df):.1f}%)")

# ── HbA1c GLYCAEMIC CONTROL ───────────────────────────────────────
print("\n" + "=" * 60)
print("GLYCAEMIC CONTROL (HbA1c)")
print("=" * 60)
df['HbA1c_Category'] = pd.cut(df['HbA1c_Percent'],
    bins=[0, 7, 8, 9, 100], labels=['<7 (Target)', '7-8', '8-9', '>9 (Poor)'])
for v, c in df['HbA1c_Category'].value_counts().sort_index().items():
    print(f"  {str(v):20s}: {c:4d} ({100*c/len(df):.1f}%)")

# ── GENDER COMPARISON ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY METRICS BY GENDER  (independent t-test + Cohen's d)")
print("=" * 60)
key_cols = ['Age', 'BMI', 'HbA1c_Percent', 'Fasting_Glucose_mgdL',
            'Systolic_BP_mmHg', 'eGFR_mLmin']
males   = df[df['Gender'] == 'Male']
females = df[df['Gender'] == 'Female']
print(f"  {'Measure':<30s} {'Male (M+/-SD)':>18s} {'Female (M+/-SD)':>18s}  {'p':>7s}  {'d':>6s}")
print("  " + "-" * 82)
for col in key_cols:
    m1, s1 = males[col].mean(), males[col].std()
    m2, s2 = females[col].mean(), females[col].std()
    t, p = stats.ttest_ind(males[col], females[col])
    pooled = np.sqrt(((len(males)-1)*s1**2 + (len(females)-1)*s2**2) /
                     (len(males) + len(females) - 2))
    d = (m1 - m2) / pooled
    sig = ' *' if p < 0.05 else '  '
    print(f"  {col:<30s} {m1:6.1f} +/- {s1:4.1f}   {m2:6.1f} +/- {s2:4.1f}   {p:7.3f}{sig}  {d:+.2f}")

print()
print("  * p < .05  |  Cohen's d: 0.2 small, 0.5 medium, 0.8 large")
