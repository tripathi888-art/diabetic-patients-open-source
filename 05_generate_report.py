import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import base64, os, warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\tripa\OneDrive\Desktop\diabetic_patients_350.csv")

# ordinal encodings (needed for cluster section)
df['Retinopathy_Ord'] = df['Retinopathy'].map({'None':0,'Mild':1,'Moderate':2,'Severe':3})
df['Neuropathy_Ord']  = df['Neuropathy'].map({'None':0,'Peripheral':1,'Autonomic':2,'Both':3})
df['Nephropathy_Ord'] = df['Nephropathy'].map({'None':0,'Microalbuminuria':1,'Macroalbuminuria':2,'CKD':3})
df['CV_Ord']          = df['Cardiovascular_Complication'].map(
    {'None':0,'Hypertension':1,'CAD':2,'Heart Failure':3,'Stroke':3})
df['Burden'] = (df['Retinopathy_Ord'].fillna(0) + df['Neuropathy_Ord'].fillna(0) +
                df['Nephropathy_Ord'].fillna(0) + df['CV_Ord'].fillna(0))

# cluster memberships
cluster_cols = ['Age','Duration_Years','BMI','HbA1c_Percent','Fasting_Glucose_mgdL',
                'Systolic_BP_mmHg','eGFR_mLmin','Urine_Albumin_mgL','Triglycerides_mgdL',
                'Retinopathy_Ord','Neuropathy_Ord','Nephropathy_Ord','CV_Ord']
df_c = df[cluster_cols].copy()
for col in ['Retinopathy_Ord','Neuropathy_Ord','Nephropathy_Ord','CV_Ord']:
    df_c[col] = df_c[col].fillna(0)
df_c = df_c.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_c)
best_k = 8
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df.loc[df_c.index, 'Cluster'] = km.fit_predict(X_scaled) + 1
df['Cluster'] = df['Cluster'].astype('Int64')

DESKTOP = r"C:\Users\tripa\OneDrive\Desktop"

def img_tag(filename, alt='', width='100%'):
    path = os.path.join(DESKTOP, filename)
    if not os.path.exists(path):
        return f'<p style="color:red">Image not found: {filename}</p>'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15)">'

def stat_row(label, value):
    return f'<tr><td style="padding:5px 12px;color:#555">{label}</td><td style="padding:5px 12px;font-weight:600">{value}</td></tr>'

def section(title, anchor, content):
    return f'''
<section id="{anchor}" style="margin-bottom:48px">
  <h2 style="border-left:5px solid #2980b9;padding-left:12px;color:#2c3e50">{title}</h2>
  {content}
</section>'''

# ── build HTML ────────────────────────────────────────────────────
parts = []

# demographic stats
age_m, age_s = df['Age'].mean(), df['Age'].std()
n_female = (df['Gender']=='Female').sum()
n_male   = (df['Gender']=='Male').sum()

hba1c_target = (df['HbA1c_Percent'] < 7).sum()
hba1c_poor   = (df['HbA1c_Percent'] > 9).sum()

comp_rates = {
    'Retinopathy':   (df['Retinopathy'].notna()  & (df['Retinopathy']  != 'None')).mean()*100,
    'Neuropathy':    (df['Neuropathy'].notna()   & (df['Neuropathy']   != 'None')).mean()*100,
    'Nephropathy':   (df['Nephropathy'].notna()  & (df['Nephropathy']  != 'None')).mean()*100,
    'CV':            (df['Cardiovascular_Complication'].notna() &
                      (df['Cardiovascular_Complication'] != 'None')).mean()*100,
}

cluster_labels = {
    1: ('High BP + Nephropathy',   '#e74c3c'),
    2: ('Hyperglycaemic + Full Nephropathy', '#e67e22'),
    3: ('Lean, Poor HbA1c + High CV', '#f39c12'),
    4: ('Obese + Hypertensive, Low Burden', '#27ae60'),
    5: ('All CV Events',           '#2980b9'),
    6: ('Full Neuropathy',         '#8e44ad'),
    7: ('Moderate Multi-Complication', '#16a085'),
    8: ('Full Retinopathy, Well-Controlled HbA1c', '#c0392b'),
}

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diabetic Patients — Statistical Analysis Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",Arial,sans-serif;background:#f4f6f9;color:#2c3e50;line-height:1.6}}
  .page{{max-width:1100px;margin:0 auto;padding:32px 24px}}
  header{{background:linear-gradient(135deg,#1a5276,#2980b9);color:#fff;padding:36px 32px;border-radius:10px;margin-bottom:40px}}
  header h1{{font-size:2rem;margin-bottom:8px}}
  header p{{opacity:.85;font-size:1rem}}
  nav{{background:#fff;border-radius:8px;padding:18px 24px;margin-bottom:36px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
  nav a{{color:#2980b9;text-decoration:none;margin-right:20px;font-size:.92rem;font-weight:500}}
  nav a:hover{{text-decoration:underline}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px}}
  .kpi{{background:#fff;border-radius:8px;padding:18px 16px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
  .kpi .num{{font-size:2rem;font-weight:700;color:#2980b9}}
  .kpi .lbl{{font-size:.82rem;color:#7f8c8d;margin-top:4px}}
  .kpi.warn .num{{color:#e74c3c}}
  .kpi.ok   .num{{color:#27ae60}}
  table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px}}
  thead tr{{background:#2980b9;color:#fff}}
  th{{padding:10px 14px;text-align:left;font-size:.88rem;font-weight:600}}
  td{{padding:8px 14px;font-size:.88rem;border-bottom:1px solid #ecf0f1}}
  tr:last-child td{{border-bottom:none}}
  tr:nth-child(even) td{{background:#f8f9fa}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600;color:#fff}}
  .cluster-card{{display:inline-block;border-radius:8px;padding:12px 16px;margin:6px;font-size:.85rem;color:#fff;min-width:200px;vertical-align:top}}
  .note{{background:#fef9e7;border-left:4px solid #f39c12;padding:12px 16px;border-radius:4px;margin-bottom:20px;font-size:.88rem}}
  .fig-caption{{font-size:.8rem;color:#7f8c8d;text-align:center;margin-top:6px;margin-bottom:24px}}
  section h2{{font-size:1.3rem;margin-bottom:18px}}
  h3{{color:#2980b9;margin:20px 0 10px;font-size:1rem}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  @media(max-width:700px){{.kpi-grid{{grid-template-columns:repeat(2,1fr)}}.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="page">

<header>
  <h1>Diabetic Patients — Statistical Analysis Report</h1>
  <p>Dataset: <strong>diabetic_patients_350.csv</strong> &nbsp;|&nbsp; N = 350 patients &nbsp;|&nbsp; 34 variables &nbsp;|&nbsp; Generated 2026-05-19</p>
</header>

<nav>
  <strong>Jump to: </strong>
  <a href="#overview">Overview</a>
  <a href="#demographics">Demographics</a>
  <a href="#clinical">Clinical Measures</a>
  <a href="#glycaemic">Glycaemic Control</a>
  <a href="#complications">Complications</a>
  <a href="#severity">Severity Analysis</a>
  <a href="#correlations">Correlations</a>
  <a href="#subgroup">Subgroup by Type</a>
  <a href="#predictors">Complication Predictors</a>
  <a href="#clusters">Cluster Analysis</a>
  <a href="#notes">Data Quality Notes</a>
</nav>

<!-- KPIs -->
<div class="kpi-grid">
  <div class="kpi"><div class="num">350</div><div class="lbl">Total patients</div></div>
  <div class="kpi warn"><div class="num">{hba1c_poor} ({hba1c_poor/len(df)*100:.0f}%)</div><div class="lbl">HbA1c &gt; 9% (poor control)</div></div>
  <div class="kpi ok"><div class="num">{hba1c_target} ({hba1c_target/len(df)*100:.0f}%)</div><div class="lbl">HbA1c &lt; 7% (at target)</div></div>
  <div class="kpi warn"><div class="num">{comp_rates["CV"]:.0f}%</div><div class="lbl">CV complication prevalence</div></div>
</div>
'''

# ── SECTION 1: Overview ───────────────────────────────────────────
html += section('1. Dataset Overview', 'overview', f'''
<div class="two-col">
<div>
<table>
<thead><tr><th>Attribute</th><th>Value</th></tr></thead>
<tbody>
{stat_row("Total patients", "350")}
{stat_row("Total variables", "34")}
{stat_row("Missing values", "861 (spread across numeric & categorical columns)")}
{stat_row("Date range (Last Visit)", df["Last_Visit_Date"].min() + " – " + df["Last_Visit_Date"].max())}
</tbody>
</table>
</div>
<div>
<div class="note"><strong>Data Quality:</strong> 861 missing values detected. Analyses use pairwise-complete observations. The dataset shows characteristics of synthetic generation — clinical variables are statistically independent (see Correlations section).</div>
</div>
</div>
''')

# ── SECTION 2: Demographics ───────────────────────────────────────
eth_rows = ''.join(stat_row(v, f"{c} ({100*c/len(df):.1f}%)") for v, c in df['Ethnicity'].value_counts().items())
type_rows = ''.join(stat_row(v, f"{c} ({100*c/len(df):.1f}%)") for v, c in df['Diabetes_Type'].value_counts().items())

html += section('2. Demographics', 'demographics', f'''
<div class="two-col">
<div>
<h3>Age</h3>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
{stat_row("Mean ± SD", f"{age_m:.1f} ± {age_s:.1f} years")}
{stat_row("Median", f"{df['Age'].median():.1f} years")}
{stat_row("Range", f"{df['Age'].min()} – {df['Age'].max()} years")}
{stat_row("< 30 years", f"39 (11.1%)")}
{stat_row("30–44 years", f"75 (21.4%)")}
{stat_row("45–59 years", f"92 (26.3%)")}
{stat_row("60–74 years", f"83 (23.7%)")}
{stat_row("75+ years",   f"61 (17.4%)")}
</tbody></table>

<h3>Gender</h3>
<table><thead><tr><th>Gender</th><th>n (%)</th></tr></thead><tbody>
{stat_row("Female", f"{n_female} ({100*n_female/len(df):.1f}%)")}
{stat_row("Male",   f"{n_male} ({100*n_male/len(df):.1f}%)")}
</tbody></table>
</div>
<div>
<h3>Ethnicity</h3>
<table><thead><tr><th>Ethnicity</th><th>n (%)</th></tr></thead><tbody>{eth_rows}</tbody></table>

<h3>Diabetes Type</h3>
<table><thead><tr><th>Type</th><th>n (%)</th></tr></thead><tbody>{type_rows}</tbody></table>
</div>
</div>
{img_tag("fig4_overview_dashboard.png", "Overview dashboard")}
<p class="fig-caption">Figure 1. Demographic and clinical overview dashboard.</p>
''')

# ── SECTION 3: Clinical Measures ─────────────────────────────────
clin_data = [
    ('Diabetes Duration (yrs)', 'Duration_Years'),
    ('BMI (kg/m²)',             'BMI'),
    ('Systolic BP (mmHg)',      'Systolic_BP_mmHg'),
    ('Diastolic BP (mmHg)',     'Diastolic_BP_mmHg'),
    ('Fasting Glucose (mg/dL)', 'Fasting_Glucose_mgdL'),
    ('Postprandial Glucose',    'Postprandial_Glucose_mgdL'),
    ('HbA1c (%)',               'HbA1c_Percent'),
    ('Total Cholesterol',       'Total_Cholesterol_mgdL'),
    ('LDL (mg/dL)',             'LDL_mgdL'),
    ('HDL (mg/dL)',             'HDL_mgdL'),
    ('Triglycerides (mg/dL)',   'Triglycerides_mgdL'),
    ('Creatinine (mg/dL)',      'Creatinine_mgdL'),
    ('eGFR (mL/min)',           'eGFR_mLmin'),
    ('Urine Albumin (mg/L)',    'Urine_Albumin_mgL'),
]
clin_rows = ''
for label, col in clin_data:
    m, s, med = df[col].mean(), df[col].std(), df[col].median()
    lo, hi = df[col].min(), df[col].max()
    clin_rows += f'<tr><td>{label}</td><td>{m:.1f} ± {s:.1f}</td><td>{med:.1f}</td><td>{lo:.1f} – {hi:.1f}</td></tr>'

html += section('3. Clinical & Metabolic Measures', 'clinical', f'''
<table>
<thead><tr><th>Measure</th><th>Mean ± SD</th><th>Median</th><th>Range</th></tr></thead>
<tbody>{clin_rows}</tbody>
</table>
''')

# ── SECTION 4: Glycaemic Control ─────────────────────────────────
hba1c_cats = [('<7% (Target)', 97, 27.7), ('7–8%', 67, 19.1), ('8–9%', 80, 22.9), ('>9% (Poor)', 106, 30.3)]
gc_rows = ''.join(f'<tr><td>{c}</td><td>{n}</td><td>{p:.1f}%</td></tr>' for c,n,p in hba1c_cats)
html += section('4. Glycaemic Control', 'glycaemic', f'''
<p style="margin-bottom:12px"><strong>72.3% of patients are above the HbA1c target of 7%.</strong> Only 27.7% are at target.</p>
<table>
<thead><tr><th>HbA1c Category</th><th>n</th><th>%</th></tr></thead>
<tbody>{gc_rows}</tbody>
</table>
''')

# ── SECTION 5: Complications ──────────────────────────────────────
comp_detail = [
    ('Retinopathy',  41.7, [('Mild',73,20.9),('Moderate',46,13.1),('Severe',27,7.7)]),
    ('Neuropathy',   50.9, [('Peripheral',108,30.9),('Autonomic',29,8.3),('Both',41,11.7)]),
    ('Nephropathy',  40.6, [('Microalbuminuria',70,20.0),('Macroalbuminuria',45,12.9),('CKD',27,7.7)]),
    ('CV Complication', 55.4, [('Hypertension',110,31.4),('CAD',44,12.6),('Heart Failure',25,7.1),('Stroke',15,4.3)]),
]
comp_html = ''
for name, pct, cats in comp_detail:
    sub_rows = ''.join(f'<tr><td style="padding-left:24px">↳ {c}</td><td>{n}</td><td>{p:.1f}%</td></tr>' for c,n,p in cats)
    total_n = int(pct/100*350)
    comp_html += f'<tr style="background:#eaf3fb"><td><strong>{name}</strong></td><td><strong>{total_n}</strong></td><td><strong>{pct:.1f}%</strong></td></tr>{sub_rows}'

html += section('5. Complications', 'complications', f'''
<table>
<thead><tr><th>Complication / Subtype</th><th>n</th><th>%</th></tr></thead>
<tbody>{comp_html}</tbody>
</table>
''')

# ── SECTION 6: Severity Analysis ─────────────────────────────────
html += section('6. Severity & Burden Analysis', 'severity', f'''
<p style="margin-bottom:16px">A composite burden score (0–12) was computed by summing ordinal severity scores across all four complication domains.</p>
{img_tag("fig5_severity_analysis.png", "Severity distributions")}
<p class="fig-caption">Figure 2. Complication severity distributions (top) and key clinical markers by severity level (bottom). Kruskal–Wallis test results shown.</p>
{img_tag("fig6_burden_score.png", "Burden score")}
<p class="fig-caption">Figure 3. Composite complication burden score vs key clinical markers. Spearman ρ and significance shown.</p>
''')

# ── SECTION 7: Correlations ───────────────────────────────────────
html += section('7. Correlations', 'correlations', f'''
<div class="note"><strong>Notable finding:</strong> No clinically meaningful correlations (|r| &ge; 0.25) exist between any pair of numeric clinical variables — all pass p &gt; .05 or have near-zero effect sizes. This is consistent with synthetic data generation where variables are drawn independently.</div>
{img_tag("fig1_correlation_heatmap.png", "Correlation heatmap")}
<p class="fig-caption">Figure 4. Pearson correlation heatmap (lower triangle). Cells marked X are not statistically significant (p &gt; .05).</p>
''')

# ── SECTION 8: Subgroup by Type ───────────────────────────────────
subgroup_rows = [
    ('BMI (kg/m²)',        '26.3 ± 6.1', '28.5 ± 6.0', '29.9 ± 5.9', '31.1 ± 5.1', '0.025 *',  '0.027'),
    ('HbA1c (%)',          '7.9 ± 1.7',  '8.2 ± 1.7',  '8.0 ± 1.7',  '8.0 ± 1.8',  '0.774',     '0.003'),
    ('Fasting Glucose',    '137.7 ± 31', '145.6 ± 38', '146.3 ± 39', '148.7 ± 45', '0.715',     '0.004'),
    ('Systolic BP',        '129.2 ± 25', '137.0 ± 23', '133.0 ± 24', '133.7 ± 22', '0.314',     '0.010'),
    ('eGFR',               '66.7 ± 26',  '75.4 ± 23',  '74.1 ± 23',  '74.2 ± 25',  '0.321',     '0.010'),
]
sr_html = ''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td></tr>' for r in subgroup_rows)

html += section('8. Subgroup Comparison by Diabetes Type', 'subgroup', f'''
<p style="margin-bottom:12px">One-way ANOVA with η² effect size. Only BMI differed significantly across types.</p>
<table>
<thead><tr><th>Measure</th><th>Type 1 (n=29)</th><th>Type 2 (n=276)</th><th>LADA (n=22)</th><th>Gestational (n=23)</th><th>p</th><th>η²</th></tr></thead>
<tbody>{sr_html}</tbody>
</table>
<p style="margin-bottom:12px"><strong>Cardiovascular complications</strong> differed significantly by type (χ² p=.010): LADA 81.8%, Type 1 69.0%, Gestational 65.2%, Type 2 51.1%.</p>
{img_tag("fig2_boxplots_by_type.png", "Box plots by type")}
<p class="fig-caption">Figure 5. Key clinical metrics by diabetes type. Red dashed lines = clinical reference thresholds. * p &lt; .05, ANOVA.</p>
''')

# ── SECTION 9: Complication Predictors ───────────────────────────
pred_rows = [
    ('Retinopathy',    'BMI',  '0.79', '[0.63, 0.98]', '.029 *'),
    ('Neuropathy',     'Age',  '1.34', '[1.08, 1.66]', '.007 **'),
]
pr_html = ''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>' for r in pred_rows)

html += section('9. Complication Predictors', 'predictors', f'''
<p style="margin-bottom:12px">Single-predictor logistic regression; OR per 1-SD increase; Wald 95% CI. Only two significant predictors were identified.</p>
<table>
<thead><tr><th>Outcome</th><th>Predictor</th><th>OR</th><th>95% CI</th><th>p</th></tr></thead>
<tbody>{pr_html}</tbody>
</table>
<p style="margin-bottom:16px">All other predictors (HbA1c, glucose, duration, systolic BP, eGFR) were non-significant for all four outcomes, consistent with independent variable generation.</p>
{img_tag("fig3_roc_curves.png", "ROC curves")}
<p class="fig-caption">Figure 6. Single-predictor ROC curves. All AUCs cluster near 0.5 (chance), confirming absence of real predictive signal.</p>
''')

# ── SECTION 10: Cluster Analysis ─────────────────────────────────
cluster_cards = ''
cluster_summary = [
    (1, 21, 6.0,  '51.2', '26.5', '7.8', '156.7', '5.1', 'Retinopathy 90.5%, Nephropathy 95.2%'),
    (2, 36, 10.3, '55.9', '28.6', '8.2', '120.3', '3.8', 'Full Nephropathy 100%, High glucose'),
    (3, 55, 15.7, '59.8', '23.9', '9.0', '133.5', '1.9', 'Poor HbA1c, CV 58%, Lean BMI'),
    (4, 70, 20.0, '53.7', '33.2', '7.8', '145.8', '1.3', 'Obese + Hypertensive, Low burden'),
    (5, 52, 14.9, '50.6', '28.4', '8.3', '133.2', '4.1', 'All CV events 100%'),
    (6, 47, 13.4, '61.1', '28.3', '8.6', '137.1', '4.3', 'Full Neuropathy 100%, Older'),
    (7, 32, 9.1,  '52.2', '29.5', '8.4', '136.3', '2.0', 'Moderate multi-complication'),
    (8, 37, 10.6, '51.9', '28.2', '6.9', '125.8', '4.6', 'Full Retinopathy 100%, Best HbA1c'),
]
cs_rows = ''
for ci, n, pct, age, bmi, hba1c, sbp, burden, notes in cluster_summary:
    lbl, color = cluster_labels.get(ci, (f'Cluster {ci}', '#7f8c8d'))
    cs_rows += (f'<tr>'
                f'<td><span class="badge" style="background:{color}">C{ci}</span> {lbl}</td>'
                f'<td>{n} ({pct:.1f}%)</td><td>{age}</td><td>{bmi}</td>'
                f'<td>{hba1c}</td><td>{sbp}</td><td>{burden}</td><td style="font-size:.82rem">{notes}</td>'
                f'</tr>')

html += section('10. Cluster Analysis (K-Means, k=8)', 'clusters', f'''
<div class="note"><strong>Note:</strong> Silhouette score = 0.079 (low), expected for independently-generated synthetic data. Clusters are primarily driven by complication pattern combinations rather than coherent metabolic phenotypes.</div>
<table>
<thead><tr><th>Cluster / Label</th><th>n (%)</th><th>Age</th><th>BMI</th><th>HbA1c</th><th>Sys BP</th><th>Burden</th><th>Key characteristics</th></tr></thead>
<tbody>{cs_rows}</tbody>
</table>
{img_tag("fig7_cluster_selection.png", "Cluster selection")}
<p class="fig-caption">Figure 7. Elbow plot and silhouette scores used to select k=8.</p>
{img_tag("fig8_cluster_profiles.png", "Cluster profiles")}
<p class="fig-caption">Figure 8. PCA projection of patient clusters (left) and Z-score heatmap of cluster profiles (right).</p>
''')

# ── SECTION 11: Data Quality Notes ───────────────────────────────
html += section('11. Data Quality Notes', 'notes', f'''
<table>
<thead><tr><th>Observation</th><th>Implication</th></tr></thead>
<tbody>
{stat_row("861 missing values across the dataset", "Pairwise-complete observations used; may reduce effective sample size in some analyses")}
{stat_row("No pairwise correlations |r| ≥ 0.25", "Strongly indicates synthetic/independently-generated variables")}
{stat_row("HbA1c uncorrelated with fasting glucose (r=+0.04)", "In real data this correlation is typically r≈0.6–0.7")}
{stat_row("All ROC AUCs near 0.5 (chance)", "Confirms no predictive signal between standard risk factors and complications")}
{stat_row("Complication rates realistic but independent of clinical markers", "Dataset suitable for descriptive/demographic analysis; not for causal modelling")}
</tbody>
</table>
''')

html += '''
<footer style="text-align:center;padding:24px;color:#95a5a6;font-size:.8rem;margin-top:24px;border-top:1px solid #ddd">
  Analysis performed with Python (pandas, scipy, scikit-learn, matplotlib, seaborn) &nbsp;|&nbsp; Report auto-generated 2026-05-19
</footer>
</div>
</body>
</html>'''

out_path = os.path.join(DESKTOP, "diabetic_analysis_report.html")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Report saved to: {out_path}")
