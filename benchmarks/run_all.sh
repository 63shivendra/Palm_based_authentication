#!/usr/bin/env bash
# benchmarks/run_all.sh - re-run v3 and v3c on every dataset present locally, keep a log per run,
# then write the CSV files and REPORT.md.        bash benchmarks/run_all.sh
set -u
cd "$(dirname "$0")/.."
L=benchmarks/results/logs
mkdir -p "$L"
E="python benchmarks/evaluate_palm_match.py"
G="data/GaborPalm/Palmprint Image Dataset with Gabor Filter Feature Enchancement/raw"

# Tongji is the long one: v3 and v3c run side by side
$E --dataset tongji --root data/Tongji --ids 600 --per-id 3 --methods v3 --enrol 3 --max-impostor 30000 --max-probes 100 > "$L/tongji_v3.log" 2>&1 &
T=$!
$E --dataset real --methods v3 v3c --enrol 2 > "$L/real.log" 2>&1
$E --dataset palmmatchdb --ids 400 --per-id 6 --methods v3 v3c --enrol 3 --max-impostor 20000 > "$L/palmmatchdb.log" 2>&1
$E --dataset folder --name gabor --root "$G" --no-detect --ids 100 --per-id 8 --methods v3 v3c --enrol 3 > "$L/gabor.log" 2>&1
python benchmarks/robustness_test.py --root data/Tongji --ids 200 --method v3 > "$L/robustness_v3.log" 2>&1
python benchmarks/robustness_test.py --root data/Tongji --ids 200 --method v3c > "$L/robustness_v3c.log" 2>&1
wait $T
$E --dataset tongji --root data/Tongji --ids 600 --per-id 3 --methods v3c --enrol 3 --max-impostor 30000 --max-probes 100 > "$L/tongji_v3c.log" 2>&1

python benchmarks/export_csv.py > "$L/export_csv.log" 2>&1
python benchmarks/make_report.py
python benchmarks/plot_results.py > "$L/plot_results.log" 2>&1
echo ALL DONE
