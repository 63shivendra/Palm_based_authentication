#!/usr/bin/env bash
# benchmarks/run_v5.sh - benchmark matcher v5 on every local dataset, plus the like-for-like Tongji protocol used
# in published papers (all 10 images per palm per session; session 1 enrolled, session 2 verified).
# Two chains run side by side on the GPU.  Watch progress:  python benchmarks/progress.py
set -u
cd "$(dirname "$0")/.."
L=benchmarks/results/logs
mkdir -p "$L"
E="python benchmarks/evaluate_palm_match.py"
G="data/GaborPalm/Palmprint Image Dataset with Gabor Filter Feature Enchancement/raw"
step() { echo "[step] $(date +%H:%M:%S) $*"; }

step "A: Tongji, published protocol (all 10 images per session), v3c and v5 - started"
( $E --dataset tongji --name tongji_full --root data/Tongji --ids 600 --per-id 10 --methods v3c v5 \
     --max-impostor 20000 --max-probes 0 > "$L/tongji_full.log" 2>&1; step "A: Tongji published protocol - finished" ) &
A=$!

step "B1/5 PalmMatchDB";   $E --dataset palmmatchdb --ids 400 --per-id 6 --methods v5 --enrol 3 --max-impostor 20000 > "$L/palmmatchdb_v5.log" 2>&1
step "B2/5 own photos";    $E --dataset real --methods v5 --enrol 2 > "$L/real_v5.log" 2>&1
step "B3/5 Gabor dataset"; $E --dataset folder --name gabor --root "$G" --no-detect --ids 100 --per-id 8 --methods v5 --enrol 3 > "$L/gabor_v5.log" 2>&1
step "B4/5 robustness";    python benchmarks/robustness_test.py --root data/Tongji --ids 200 --method v5 > "$L/robustness_v5.log" 2>&1
step "B5/5 Tongji, benchmark protocol (3 images per session)"
$E --dataset tongji --root data/Tongji --ids 600 --per-id 3 --methods v5 --enrol 3 --max-impostor 10000 --max-probes 10 > "$L/tongji_v5.log" 2>&1
step "B: finished, waiting for A"
wait $A
python benchmarks/export_csv.py > "$L/export_csv.log" 2>&1
python benchmarks/make_report.py > /dev/null
python benchmarks/plot_results.py > "$L/plot_results.log" 2>&1
step "ALL DONE"
