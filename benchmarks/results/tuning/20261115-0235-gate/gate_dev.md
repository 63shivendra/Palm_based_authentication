# Quality gate limits (DEV identities only)

Generated 2026-11-15 02:35 by `benchmarks/gate_dev.py`. A photo 'would fail' when its v4 score against its own palm is below the v4 threshold (1.5).

## Matching failures per condition (no gate)

| Set | Condition | Photos | Would fail |
|---|---|---|---|
| natural | none | 562 | 5.69% |
| degraded | clean | 40 | 0.00% |
| degraded | brighter (x1.5) | 40 | 5.00% |
| degraded | overexposed (x2.0, clipped) | 40 | 50.00% |
| degraded | hard shadow (half palm x0.4) | 40 | 0.00% |
| degraded | gamma 2.0 (dim) | 40 | 0.00% |
| degraded | sensor noise (sigma 12) | 40 | 0.00% |
| degraded | darker (x0.5) | 40 | 0.00% |
| degraded | very dark (x0.25) | 40 | 0.00% |
| degraded | blur (sigma 4) | 40 | 0.00% |
| degraded | blur (sigma 8) | 40 | 17.50% |
| degraded | overexposed (x3.0, clipped) | 40 | 100.00% |

## Candidate limits

| Measure | Limit | Natural photos rejected | Degraded photos rejected | Rejected photos that would fail | Failing photos caught |
|---|---|---|---|---|---|
| clipped_white | > 0.05 | 0.00% | 21.82% | 64.58% | 61.39% |
| clipped_white | > 0.1 | 0.00% | 20.91% | 67.39% | 61.39% |
| clipped_white | > 0.2 | 0.00% | 20.00% | 70.45% | 61.39% |
| clipped_white | > 0.3 | 0.00% | 19.09% | 73.81% | 61.39% |
| clipped_white | > 0.4 | 0.00% | 18.41% | 76.54% | 61.39% |
| clipped_white | > 0.5 | 0.00% | 17.27% | 81.58% | 61.39% |
| clipped_black | > 0.05 | 0.00% | 0.00% | n/a | 0.00% |
| clipped_black | > 0.1 | 0.00% | 0.00% | n/a | 0.00% |
| clipped_black | > 0.2 | 0.00% | 0.00% | n/a | 0.00% |
| clipped_black | > 0.3 | 0.00% | 0.00% | n/a | 0.00% |
| brightness_high | > 215 | 3.56% | 20.23% | 56.88% | 61.39% |
| brightness_high | > 225 | 0.89% | 19.77% | 67.39% | 61.39% |
| brightness_high | > 235 | 0.00% | 17.27% | 81.58% | 61.39% |
| brightness_high | > 245 | 0.00% | 14.09% | 91.94% | 56.44% |
| brightness_low | < 25 | 0.00% | 0.68% | 0.00% | 0.00% |
| brightness_low | < 40 | 0.00% | 7.73% | 0.00% | 0.00% |
| brightness_low | < 55 | 0.00% | 11.82% | 0.00% | 0.00% |
| focus | < 0.08 | 0.00% | 2.50% | 100.00% | 10.89% |
| focus | < 0.1 | 0.00% | 2.50% | 100.00% | 10.89% |
| focus | < 0.12 | 0.00% | 6.59% | 48.28% | 13.86% |
| focus | < 0.15 | 0.00% | 11.59% | 35.29% | 17.82% |
| focus | < 0.2 | 0.36% | 17.05% | 24.68% | 18.81% |
| noise | > 4 | 0.00% | 9.09% | 0.00% | 0.00% |
| noise | > 6 | 0.00% | 9.09% | 0.00% | 0.00% |
| noise | > 8 | 0.00% | 9.09% | 0.00% | 0.00% |
| noise | > 10 | 0.00% | 9.09% | 0.00% | 0.00% |
| crease_contrast | < 0.005 | 0.00% | 8.18% | 100.00% | 35.64% |
| crease_contrast | < 0.008 | 0.00% | 10.68% | 100.00% | 46.53% |
| crease_contrast | < 0.01 | 0.36% | 12.27% | 92.86% | 51.49% |
| crease_contrast | < 0.014 | 9.79% | 17.50% | 51.52% | 67.33% |
