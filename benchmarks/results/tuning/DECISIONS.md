# v4 design decisions (DEV identities only)

Every choice below was made on DEV identities: the first 20% of each dataset's identities after the seeded shuffle of
the full-size sample (`global_split` in `benchmarks/run_benchmark.py`). TEST identities were never looked at before the
choices were fixed. Folders are under `benchmarks/results/`.

Protocol fix (15 Nov): the quick profile used to draw its own 40-identity sample and split it again, and for BJTU and
11k Hands its "TEST" identities were exactly the DEV identities used for tuning. The first v4 run was stopped and
deleted before any result was used. Since then every profile takes DEV and TEST from the one global split, and the
robustness test uses Tongji TEST identities only.

| Decision | Evidence (DEV) | Choice |
|---|---|---|
| Crop preprocessing before DISK | BJTU EER: raw 8.11%, CLAHE 7.22%, local contrast normalisation 8.80% (dropped after BJTU, the worst); Tongji and 11k Hands 0% for raw and CLAHE (`tuning/20261115-0219/dev_tuning.md`) | CLAHE |
| Score fusion with v2 (SIFT) | BJTU EER 4.67% (CLAHE + v2) vs 7.22% (CLAHE alone); FNMR at FMR 1%: 7.3% vs 16.7%; Tongji and 11k Hands stay at 0% | fused score = v3c score / 12 + v2 inliers / 4 |
| Decision threshold | Highest of 1,358 pooled DEV impostor fused scores: 1.05. At 1.5: 0 DEV false accepts, BJTU FNMR 12.3% (v3c at 12: 30.0%), Tongji and 11k Hands 0% | 1.5 (1.4 x the highest DEV impostor) |
| Noise | Without smoothing, 65-88% of genuine probes with sensor noise (sigma 12) fail (`dev_lighting_log.txt`); with v3cr's noise-adaptive smoothing added, 0% fail (`20261115-0235-gate/gate_dev.md`) | smoothing kept (only crops noisier than 4 grey levels) |
| Multi-scale retry for pre-cropped palms | Of 12 failing BJTU DEV genuine pairs, 2 had a scale outside the pose check; the others had too few matches because of lighting and contrast (diagnosis on 60 pairs) | not added |
| Quality gate limits | Each limit rejects 0% of 562 natural DEV photos. Over-exposure (> 30% clipped): 74% of rejected photos would fail, catches 61% of failures. Focus < 0.10 and faint creases < 0.008: every rejected photo would fail. Darkness and noise limits only catch extreme cases, because v4 matches dark and noisy photos (`20261115-0235-gate/gate_dev.md`) | `palm_quality.py` constants |
| Hand-pose limits of the gate | Not covered by gate_dev.py. The palm-size limit (160 px, an uncalibrated placeholder) rejected 96% of PalmMatchDB photos in the first TEST run, which was stopped. **This change was prompted by a TEST observation.** The new limit, 100 px, rests on DEV evidence: it gives crops at about the resolution of Tongji's 128 px palm images, which match at 0% EER on DEV. Tilt and finger limits reject 0 of 200 11k Hands DEV photos and were not changed | `MIN_PALM_PIXELS = 100` |
| Search vector | Rank-1 / recall@10: HOL 98.3% / 99.2% (Tongji), 86.7% / 97.5% (BJTU), 98.1% / 100% (11k Hands); DISK-VLAD 70.8%, 45.0%, 89.4% rank-1; combining both did not help (`embedding/20261115-0201/embed_report.md`) | HOL (no training at all) |
| Passive spoof threshold | 99th percentile of 366 genuine photos of 11k Hands DEV identities: 3.01 (`pad/`) | 3.01 |
| Peaks averaged in the spoof score | top-2/4/6/10 gave screen APCER 2.5-3.5%, print 11.5-14% at similar BPCER (simulated); no clear winner | kept 6 |
| Speed variants | 20 DEV identities each of Tongji and BJTU (`speed/20261115-1527/speed_dev.md`): DISK on a 384 px crop with 512 keypoints makes a template in 690 ms instead of 1,253 ms and compares in 241 ms instead of 377 ms, with the same EER (Tongji 0%, BJTU 10.0%; FNMR at FMR 1% on BJTU 10% vs 15%). int8 LightGlue gave no speed-up (dynamic-quantisation overhead) and was dropped | candidate, not adopted yet: v4's fused threshold must be re-checked on DEV and a TEST run repeated |
