"""
Run all analysis scripts in order.
Outputs (PNGs + HTML report) are saved to the Desktop.
"""
import subprocess, sys, os

scripts = [
    "01_basic_stats.py",
    "02_correlations_subgroup_predictors.py",
    "03_visualisations.py",
    "04_severity_clusters.py",
    "05_generate_report.py",
]

base = os.path.dirname(os.path.abspath(__file__))
for script in scripts:
    path = os.path.join(base, script)
    print(f"\n{'='*55}\nRunning: {script}\n{'='*55}")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR in {script} — stopping.")
        sys.exit(1)

print("\nAll scripts completed. Check Desktop for outputs.")
