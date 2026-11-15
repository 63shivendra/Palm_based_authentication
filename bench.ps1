# bench.ps1 - palm benchmark launcher (uses the Python 3.12 that has torch, kornia and mediapipe installed)
#
#   .\bench.ps1 fetch --list          datasets, licences, what is on disk
#   .\bench.ps1 fetch all             download every dataset that is missing
#   .\bench.ps1 run                   benchmark with the live dashboard (quick profile on CPU, full on GPU)
#   .\bench.ps1 run --profile full    the large run (meant for the GPU desktop)
#   .\bench.ps1 report <run folder>   rebuild CSV files, charts and REPORT.md of a run
#   .\bench.ps1 tune                  DEV-only tuning: preprocessing, score fusion, quality gate, lighting
#   .\bench.ps1 pad                   spoof-check metrics (ISO/IEC 30107-3), real attacks from data/pad/ if present
#   .\bench.ps1 embed --test          1:N search and template protection on TEST identities
#   .\bench.ps1 speed                 speed variants on DEV identities (run with nothing else busy)
#   .\bench.ps1 kiosk                 palm payment kiosk on http://127.0.0.1:5060
#   .\bench.ps1 test                  unit and end-to-end tests
$py = $null
Push-Location $env:TEMP                      # look up outside the repo: uv would pick the project's .venv first
try { $py = (& uv python find 3.12 2>$null | Select-Object -First 1) } catch { }
Pop-Location
if (-not $py) { $py = "python" }
$env:PYTHONIOENCODING = "utf-8"
$cmd, $rest = $args
switch ($cmd) {
    "fetch"  { & $py "$PSScriptRoot\scripts\fetch_datasets.py" @rest }
    "run"    { & $py "$PSScriptRoot\benchmarks\run_benchmark.py" @rest }
    "report" { & $py "$PSScriptRoot\benchmarks\bench_report.py" @rest }
    "tune"   { & $py "$PSScriptRoot\benchmarks\tune_dev.py" @rest }
    "pad"    { & $py "$PSScriptRoot\benchmarks\pad_eval.py" @rest }
    "embed"  { & $py "$PSScriptRoot\benchmarks\embed_eval.py" @rest }
    "speed"  { & $py "$PSScriptRoot\benchmarks\speed_dev.py" @rest }
    "kiosk"  { & $py "$PSScriptRoot\webapp\kiosk.py" @rest }
    "test"   { $env:PYTHONPATH = "$PSScriptRoot\.bench_env"; & $py -m pytest -q "$PSScriptRoot\tests\test_palm_pad.py" "$PSScriptRoot\tests\test_quality_embed_protect.py" "$PSScriptRoot\tests\test_kiosk.py" -p no:cacheprovider @rest }
    default  { Get-Content $PSCommandPath | Select-Object -Skip 1 -First 14 }
}
