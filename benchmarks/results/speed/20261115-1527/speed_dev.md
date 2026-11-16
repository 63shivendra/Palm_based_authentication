# Speed variants (DEV identities only)

Generated 2026-11-15 15:27 by `benchmarks/speed_dev.py` on CPU (16 threads) (10 PyTorch threads). Times are medians on this machine.

| Variant | DISK input | Keypoints | LightGlue | Extract (ms) | Compare (ms) | Dataset | EER | FNMR at FMR 1% | Impostor max | Genuine median |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 512 px | 1024 | float32 | 1220 | 381 | tongji | 0.00% | 0.00% | 3.9 | 312 |
| kp512 | 512 px | 512 | float32 | 1335 | 208 | tongji | 0.00% | 0.00% | 2.7 | 114 |
| px384 | 384 px | 1024 | float32 | 696 | 417 | tongji | 0.00% | 0.00% | 7.9 | 356 |
| px384_kp512 | 384 px | 512 | float32 | 690 | 240 | tongji | 0.00% | 0.00% | 2.4 | 142 |
| base | 512 px | 1024 | float32 | 1220 | 381 | bjtu2 | 10.00% | 15.00% | 0.8 | 60 |
| kp512 | 512 px | 512 | float32 | 1335 | 208 | bjtu2 | 12.92% | 15.00% | 0.6 | 18 |
| px384 | 384 px | 1024 | float32 | 696 | 417 | bjtu2 | 11.67% | 13.33% | 2.0 | 110 |
| px384_kp512 | 384 px | 512 | float32 | 690 | 240 | bjtu2 | 10.00% | 10.00% | 1.2 | 41 |
