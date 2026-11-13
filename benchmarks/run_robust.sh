#!/usr/bin/env bash
# benchmarks/run_robust.sh - run the robust variant (v3cr, palm_match_robust.py) on the clean benchmarks,
# to check that its retries do not raise false accepts.        bash benchmarks/run_robust.sh
# Tongji uses 10,000 impostor comparisons (v3 and v3c use 30,000): every failed comparison is retried 3 times.
set -u
cd "$(dirname "$0")/.."
L=benchmarks/results/logs
mkdir -p "$L"
E="python benchmarks/evaluate_palm_match.py"
G="data/GaborPalm/Palmprint Image Dataset with Gabor Filter Feature Enchancement/raw"

$E --dataset tongji --root data/Tongji --ids 600 --per-id 3 --methods v3cr --enrol 3 --max-impostor 10000 --max-probes 10 > "$L/tongji_v3cr.log" 2>&1 &
T=$!
$E --dataset real --methods v3cr --enrol 2 > "$L/real_v3cr.log" 2>&1
$E --dataset palmmatchdb --ids 400 --per-id 6 --methods v3cr --enrol 3 --max-impostor 20000 > "$L/palmmatchdb_v3cr.log" 2>&1
$E --dataset folder --name gabor --root "$G" --no-detect --ids 100 --per-id 8 --methods v3cr --enrol 3 > "$L/gabor_v3cr.log" 2>&1
wait $T
echo ROBUST DONE
