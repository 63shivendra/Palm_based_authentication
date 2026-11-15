# DEV-only tuning

Generated 2026-11-15 02:20 by `benchmarks/tune_dev.py`. Only DEV identities of the benchmark's identity split are used (TEST identities are never touched). Up to 40 DEV identities per dataset, 150 genuine / 450 impostor pairs.

## Matcher variants

Shipped threshold: 12 for the v3c-style variants, 4 inliers for v2, 2.0 for fusion (each score divided by its own threshold, then added).

| Dataset | Variant | EER | FNMR at shipped threshold | FMR at shipped threshold | FNMR at FMR 1% | Impostor max | Genuine / impostor |
|---|---|---|---|---|---|---|---|
| bjtu2 | raw | 8.11% | 30.00% | 0.00% | 14.00% | 2.1 | 150 / 450 |
| bjtu2 | clahe | 7.22% | 26.67% | 0.00% | 16.67% | 4.5 | 150 / 450 |
| bjtu2 | v2 | 5.89% | 12.67% | 0.00% | 11.33% | 3.0 | 150 / 450 |
| bjtu2 | raw+v2 | 5.22% | 18.00% | 0.00% | 7.33% | 0.8 | 150 / 450 |
| bjtu2 | clahe+v2 | 4.67% | 17.33% | 0.00% | 7.33% | 1.0 | 150 / 450 |
| tongji | raw | 0.00% | 0.00% | 0.00% | 0.00% | 7.0 | 150 / 450 |
| tongji | clahe | 0.00% | 0.00% | 0.00% | 0.00% | 5.5 | 150 / 450 |
| tongji | v2 | 0.11% | 0.00% | 0.22% | 0.00% | 4.0 | 150 / 450 |
| tongji | raw+v2 | 0.00% | 0.00% | 0.00% | 0.00% | 1.0 | 150 / 450 |
| tongji | clahe+v2 | 0.00% | 0.00% | 0.00% | 0.00% | 1.0 | 150 / 450 |
| 11khands | raw | 0.00% | 0.67% | 0.00% | 0.00% | 3.9 | 150 / 450 |
| 11khands | clahe | 0.00% | 0.00% | 0.00% | 0.00% | 1.8 | 150 / 450 |
| 11khands | v2 | 0.00% | 0.00% | 0.44% | 0.00% | 4.0 | 150 / 450 |
| 11khands | raw+v2 | 0.00% | 0.00% | 0.00% | 0.00% | 1.1 | 150 / 450 |
| 11khands | clahe+v2 | 0.00% | 0.00% | 0.00% | 0.00% | 1.0 | 150 / 450 |

## Quality gate on DEV photos

A photo counts as a genuine failure when its weakest genuine comparison (raw variant) is below 12.

| Dataset | Photos | Rejected by the gate | Photos with a genuine failure | Of those, rejected by the gate | Rejection reasons |
|---|---|---|---|---|---|
| bjtu2 | 240 | 4 (1.67%) | 70 | 2 | {'faint_lines': 2, 'blurry': 2} |
| tongji | 240 | 0 (0.00%) | 0 | 0 | {} |
| 11khands | 200 | 5 (2.50%) | 2 | 0 | {'too_bright': 5} |
