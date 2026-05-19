# Diabetes Analysis Scripts

**Dataset:** `diabetic_patients_350.csv` (must stay on the Desktop)

## Scripts (run in order)

| File | What it does |
|------|--------------|
| `01_basic_stats.py` | Descriptive stats, demographics, medication use, glycaemic control, gender comparison |
| `02_correlations_subgroup_predictors.py` | Pearson/Spearman correlations, one-way ANOVA by diabetes type, logistic regression ORs |
| `03_visualisations.py` | Figures 1–4: correlation heatmap, box plots, ROC curves, overview dashboard |
| `04_severity_clusters.py` | Figures 5–8: severity analysis, burden score, elbow/silhouette, cluster profiles |
| `05_generate_report.py` | Self-contained HTML report embedding all figures |
| `run_all.py` | Runs all 5 scripts in sequence |

## Quick start

```
cd "C:\Users\tripa\OneDrive\Desktop\diabetes_analysis"
python run_all.py
```

## Requirements

```
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

## Outputs (saved to Desktop)

- `fig1_correlation_heatmap.png`
- `fig2_boxplots_by_type.png`
- `fig3_roc_curves.png`
- `fig4_overview_dashboard.png`
- `fig5_severity_analysis.png`
- `fig6_burden_score.png`
- `fig7_cluster_selection.png`
- `fig8_cluster_profiles.png`
- `diabetic_analysis_report.html`
