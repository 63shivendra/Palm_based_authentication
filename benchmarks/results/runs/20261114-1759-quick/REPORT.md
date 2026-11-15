# Palm recognition benchmark (zero-shot)

Run `20261114-1759-quick`, finished 2026-11-14T19:26:31, on CPU (16 threads), profile **quick**.

No palm image was used to build any matcher: DISK and LightGlue are pretrained on outdoor scenes and MediaPipe only locates the hand. Every number below is measured on **held-out test identities**; for datasets with at least 40 identities, 20% of identities (DEV) were used only to pick the alternative thresholds.

## 1. Summary (1 enrolled photo, shipped threshold)

| Dataset | Method | Test identities | Comparisons (genuine / impostor) | Accuracy | EER | FAR | FRR | TAR @ FAR 0.1% | Rank-1 | FTE |
|---|---|---|---|---|---|---|---|---|---|---|
| Tongji contactless (ROI) | v3c compact | 32 | 128 / 500 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| Tongji contactless (ROI) | v3cr compact + robust | 32 | 128 / 125 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| Tongji contactless (ROI) | v2 SIFT baseline | 32 | 128 / 500 | 99.84% | **0.59%** | 0.00% | 0.78% | n/a | n/a | 0.00% |
| Tongji contactless (ROI) | v1 CompCode baseline | 32 | 128 / 500 | 81.37% | **3.16%** | 0.00% | 91.41% | n/a | n/a | 0.00% |
| PalmMatchDB (phone photos) | v3c compact | 39 | 102 / 500 | 99.50% | **0.99%** | 0.00% | 5.47% | n/a | n/a | 2.61% |
| PalmMatchDB (phone photos) | v3cr compact + robust | 39 | 102 / 125 | 98.68% | **0.89%** | 0.00% | 5.47% | n/a | n/a | 2.61% |
| PalmMatchDB (phone photos) | v2 SIFT baseline | 39 | 102 / 500 | 99.34% | **0.79%** | 0.58% | 3.56% | n/a | n/a | 2.61% |
| PalmMatchDB (phone photos) | v1 CompCode baseline | 39 | 74 / 500 | 95.82% | **2.75%** | 0.00% | 52.41% | n/a | n/a | 29.57% |
| 11k Hands (palmar photos) | v3c compact | 30 | 90 / 500 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| 11k Hands (palmar photos) | v3cr compact + robust | 30 | 90 / 125 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| 11k Hands (palmar photos) | v2 SIFT baseline | 30 | 90 / 500 | 99.49% | **0.00%** | 0.60% | 0.00% | n/a | n/a | 0.00% |
| 11k Hands (palmar photos) | v1 CompCode baseline | 30 | 90 / 500 | 95.42% | **2.21%** | 0.00% | 30.00% | n/a | n/a | 0.00% |
| BJTU palm v2 (ROI) | v3c compact | 32 | 128 / 500 | 93.63% | **13.84%** | 0.00% | 31.25% | n/a | n/a | 0.00% |
| BJTU palm v2 (ROI) | v3cr compact + robust | 32 | 128 / 125 | 84.58% | **12.65%** | 0.00% | 30.47% | n/a | n/a | 0.00% |
| BJTU palm v2 (ROI) | v2 SIFT baseline | 32 | 128 / 500 | 97.13% | **7.13%** | 0.00% | 14.06% | n/a | n/a | 0.00% |
| BJTU palm v2 (ROI) | v1 CompCode baseline | 32 | 128 / 500 | 80.25% | **7.81%** | 0.00% | 96.88% | n/a | n/a | 0.00% |
| XINHUA (ROI) | v3c compact | 30 | 90 / 500 | 99.83% | **0.10%** | 0.20% | 0.00% | n/a | n/a | 0.00% |
| XINHUA (ROI) | v3cr compact + robust | 30 | 90 / 125 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| XINHUA (ROI) | v2 SIFT baseline | 30 | 90 / 500 | 99.49% | **0.10%** | 0.60% | 0.00% | n/a | n/a | 0.00% |
| XINHUA (ROI) | v1 CompCode baseline | 30 | 90 / 500 | 85.93% | **1.16%** | 0.20% | 91.11% | n/a | n/a | 0.00% |
| Gabor palm dataset (raw ROI) | v3c compact | 15 | 90 / 500 | 99.15% | **1.06%** | 0.00% | 5.56% | n/a | n/a | 0.00% |
| Gabor palm dataset (raw ROI) | v3cr compact + robust | 15 | 90 / 125 | 98.14% | **0.00%** | 0.00% | 4.44% | n/a | n/a | 0.00% |
| Gabor palm dataset (raw ROI) | v2 SIFT baseline | 15 | 90 / 500 | 99.66% | **0.00%** | 0.40% | 0.00% | n/a | n/a | 0.00% |
| Gabor palm dataset (raw ROI) | v1 CompCode baseline | 15 | 90 / 500 | 86.44% | **17.99%** | 0.00% | 88.89% | n/a | n/a | 0.00% |
| Own phone photos | v3c compact | 2 | 3 / 3 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| Own phone photos | v3cr compact + robust | 2 | 3 / 3 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| Own phone photos | v2 SIFT baseline | 2 | 3 / 3 | 100.00% | **0.00%** | 0.00% | 0.00% | n/a | n/a | 0.00% |
| Own phone photos | v1 CompCode baseline | 2 | 3 / 3 | 50.00% | **33.33%** | 0.00% | 100.00% | n/a | n/a | 0.00% |

### With 3 enrolled photos (score = best match over the enrolled photos)

| Dataset | Method | Comparisons | Accuracy | EER | FAR | FRR |
|---|---|---|---|---|---|---|
| Gabor palm dataset (raw ROI) | v3c compact | 15 / 210 | 100.00% | **0.00%** | 0.00% | 0.00% |
| Gabor palm dataset (raw ROI) | v3cr compact + robust | 15 / 125 | 100.00% | **0.00%** | 0.00% | 0.00% |
| Gabor palm dataset (raw ROI) | v2 SIFT baseline | 15 / 210 | 98.67% | **0.00%** | 1.43% | 0.00% |
| Gabor palm dataset (raw ROI) | v1 CompCode baseline | 15 / 210 | 95.56% | **6.67%** | 0.00% | 66.67% |

### Thresholds chosen on DEV identities, applied to TEST identities

| Dataset | Method | shipped thr | FAR / FRR | DEV EER-point thr | FAR / FRR | DEV FAR-0.1% thr | FAR / FRR |
|---|---|---|---|---|---|---|---|
| Tongji contactless (ROI) | v3c compact | 12 | 0.00% / 0.00% | 31.5 | 0.00% / 0.00% | 58.7 | 0.00% / 0.00% |
| Tongji contactless (ROI) | v3cr compact + robust | 12 | 0.00% / 0.00% | 31.5 | 0.00% / 0.00% | 58.7 | 0.00% / 0.00% |
| Tongji contactless (ROI) | v2 SIFT baseline | 4 | 0.00% / 0.78% | 9.0 | 0.00% / 4.69% | 15.0 | 0.00% / 7.81% |
| Tongji contactless (ROI) | v1 CompCode baseline | -0.3 | 0.00% / 91.41% | -0.6 | 0.00% / 5.47% | -0.6 | 0.00% / 7.81% |
| BJTU palm v2 (ROI) | v3c compact | 12 | 0.00% / 31.25% | 0.5 | 6.20% / 17.19% | 7.5 | 0.00% / 30.47% |
| BJTU palm v2 (ROI) | v3cr compact + robust | 12 | 0.00% / 30.47% | 1.0 | 3.20% / 19.53% | 3.1 | 0.00% / 22.66% |
| BJTU palm v2 (ROI) | v2 SIFT baseline | 4 | 0.00% / 14.06% | 5.0 | 0.00% / 17.97% | 5.0 | 0.00% / 17.97% |
| BJTU palm v2 (ROI) | v1 CompCode baseline | -0.3 | 0.00% / 96.88% | -0.7 | 6.00% / 8.59% | -0.6 | 0.20% / 20.31% |

## 2. Datasets and licences

| Dataset | Licence | Use here | Images used | Identities | Protocol | Hand detection |
|---|---|---|---|---|---|---|
| Tongji contactless (ROI) | none stated on the official page | eval-only | 160 (160 usable) | 40 | cross-session | no (pre-cropped) |
| PalmMatchDB (phone photos) | Apache-2.0 (Hugging Face dataset tag) | permissive | 115 (112 usable) | 39 | all-vs-all (trusted-label pairs) | MediaPipe |
| 11k Hands (palmar photos) | free for reasonable academic fair use (project page) | eval-only | 90 (90 usable) | 30 | all-vs-all | MediaPipe |
| BJTU palm v2 (ROI) | CC BY 4.0 on the Zenodo record, but the images are re-hosted copies of other databases | eval-only | 160 (160 usable) | 40 | cross-session | no (pre-cropped) |
| XINHUA (ROI) | CC BY 4.0 on the Zenodo record, but the images are re-hosted copies of other databases | eval-only | 90 (90 usable) | 30 | all-vs-all | no (pre-cropped) |
| Gabor palm dataset (raw ROI) | CC BY 4.0 | permissive | 60 (60 usable) | 15 | all-vs-all | no (pre-cropped) |
| Own phone photos | own photos | own data | 4 (4 usable) | 2 | all-vs-all | MediaPipe |

`permissive` = licence allows research, development and redistribution. `eval-only` = no licence, research-only or unclear provenance: used only to measure a matcher that never trained on it, never redistributed. Notes per dataset are in `datasets.csv` (from `data/MANIFEST.json`).

## 3. Robustness to capture conditions (Tongji, simulated)

Clean enrolled photo, degraded probe photo; 20 genuine and 80 impostor comparisons per condition, shipped threshold. FRR / FAR:

| Condition | v3c compact | v3cr compact + robust |
|---|---|---|
| clean (baseline) | 0.0% / 0.0% | 0.0% / 0.0% |
| darker (x0.5) | 0.0% / 0.0% | 0.0% / 0.0% |
| brighter (x1.5) | 5.0% / 0.0% | 5.0% / 0.0% |
| uneven light (gradient) | 0.0% / 0.0% | 0.0% / 0.0% |
| overexposed (x2.0, clipped) | 50.0% / 0.0% | 50.0% / 0.0% |
| hard shadow (half palm x0.4) | 0.0% / 0.0% | 0.0% / 0.0% |
| blur (sigma 4) | 0.0% / 0.0% | 0.0% / 0.0% |
| rotation 20 deg | 5.0% / 0.0% | 0.0% / 0.0% |
| sensor noise (sigma 12) | 55.0% / 0.0% | 0.0% / 0.0% |
| JPEG quality 20 | 5.0% / 0.0% | 0.0% / 0.0% |

Degradations are applied to the cropped image, not captured with a real camera.

## 4. Latency per verification (median, ms)

| Stage | CPU (16 threads) |
|---|---|
| detect | 74 (p95 3,906) |
| align | 3 (p95 29) |
| extract_v3 | 1,344 (p95 1,375) |
| extract_v3c | 1,360 (p95 1,399) |
| extract_v3cr | 1,323 (p95 1,394) |
| match_v3 | 1,041 (p95 1,442) |
| match_v3c | 434 (p95 565) |
| match_v3cr | 437 (p95 558) |

Total for one verification = detect + align + extract + one match per enrolled photo. Measured on whole-hand phone photos. Not measured on a phone or payment terminal.

## 5. Template size (one enrolled photo, saved file)

| Method | Size |
|---|---|
| v3c compact | 97 KB |
| v3cr compact + robust | 97 KB |
| v2 SIFT baseline | 385 KB |
| v1 CompCode baseline | 82 KB |

## 6. Spoof detection

**Not measured.** The matchers have no liveness or presentation-attack detection yet, and no permissively licensed palm spoof dataset was found. A printed photo or a phone screen showing a palm may be accepted. Measuring APCER / BPCER (ISO/IEC 30107-3) needs our own spoof captures.

## 7. Comparison with published results

Published numbers are copied from the cited sources (not re-run) and use different protocols; most methods train on the dataset they are tested on. Ours is zero-shot.

| Dataset | Method | EER | Trained on palm images | Protocol | Source |
|---|---|---|---|---|---|
| Tongji contactless (ROI) | **ours: v3c compact** | 0.00% | no (zero-shot) | cross-session, test identities | this run |
| Tongji contactless (ROI) | **ours: v3cr compact + robust** | 0.00% | no (zero-shot) | cross-session, test identities | this run |
| Tongji contactless (ROI) | **ours: v2 SIFT baseline** | 0.59% | no (zero-shot) | cross-session, test identities | this run |
| Tongji contactless (ROI) | **ours: v1 CompCode baseline** | 3.16% | no (zero-shot) | cross-session, test identities | this run |
| PalmMatchDB (phone photos) | **ours: v3c compact** | 0.99% | no (zero-shot) | all-vs-all (trusted-label pairs), test identities | this run |
| PalmMatchDB (phone photos) | **ours: v3cr compact + robust** | 0.89% | no (zero-shot) | all-vs-all (trusted-label pairs), test identities | this run |
| PalmMatchDB (phone photos) | **ours: v2 SIFT baseline** | 0.79% | no (zero-shot) | all-vs-all (trusted-label pairs), test identities | this run |
| PalmMatchDB (phone photos) | **ours: v1 CompCode baseline** | 2.75% | no (zero-shot) | all-vs-all (trusted-label pairs), test identities | this run |
| Tongji | CCNet | 0.00004% | yes, on Tongji | closed-set, train and test split of Tongji | survey arXiv:2501.01166; arXiv:2504.11306 |
| Tongji | BOCV | 0.0056% | no (hand-crafted) | within Tongji | survey tables found in the literature search |
| Tongji | CompNet | 0.018 to 0.025% | yes, on Tongji | closed-set | arXiv:2501.01166 (0.018); arXiv:2504.11306 (0.025) |
| Tongji | PalmNet | 0.03 to 0.17% | partly (Gabor/PCA filters tuned on palms) | within Tongji | survey tables found in the literature search |
| Tongji | DHPN | 0.066% | yes | within Tongji | survey tables found in the literature search |
| Tongji | DHN | 0.088% | yes | within Tongji | survey tables found in the literature search |
| Tongji | CompCode | 0.10 to 0.11% | no (hand-crafted) | within Tongji | arXiv:2504.11306 and survey tables |
| Tongji | CCNet, trained on PolyU | 2.22% | yes, on another dataset | cross-dataset (PolyU to Tongji) | survey arXiv:2501.01166 |
| Tongji | PalmBridge, trained on PolyU | 2.25% | yes, on another dataset | cross-dataset open-set | arXiv:2601.20351 |
| Tongji | PalmBridge, trained on IITD | 2.88% | yes, on another dataset | cross-dataset open-set | arXiv:2601.20351 |
| PalmMatchDB | no published EER found | - |  | The dataset paper (Alausa et al., ICPECA 2023, IEEE Xplore 10076097) is paywalled; its results could not be read. No other paper reporting on this dataset was found. | web search, 2026-11-13 |
| Palmprint Image Dataset with Gabor Filter | no published result found | - |  | Mendeley Data record only; no paper reporting verification results on it was found. | web search, 2026-11-13 |

## 8. How far the numbers can be trusted

- Comparison counts are in the tables. With N impostor comparisons a FAR below about 3/N cannot be shown (rule of 3); the largest test here has 500.
- Fewer than 20 test identities (smoke test only, not an accuracy measurement): Gabor palm dataset (raw ROI), Own phone photos.
- Tongji, BJTU, XINHUA and Gabor are pre-cropped palms, so they do not test hand detection and alignment; PalmMatchDB, 11k Hands and the own photos test the whole chain.
- 11k Hands, XINHUA and Gabor photos of a palm were taken in one sitting (same session), an easier test than Tongji and BJTU, whose two sessions are weeks apart.
- Accuracy depends on the mix of genuine and impostor comparisons (v3cr uses fewer impostor comparisons because it retries every failed one); compare methods on EER, FAR and FRR.
- PalmMatchDB labels are unreliable; only pairs with trustworthy labels are kept (same palm: photos < 10 min apart; different palm: > 1 day apart).
- **Quick profile**: small samples sized for a laptop CPU. Run `--profile full` on the GPU machine for the numbers to quote.

## 9. Files

- `metrics.csv` every metric, one row per dataset x method x enrolment; `latency.csv`, `robustness.csv`, `datasets.csv`
- `scores/*.npz` the score of every comparison; `jobs/*.json` raw results; `run.log` the full log
- `plots/accuracy_enrol1.png`
- `plots/accuracy_enrol3.png`
- `plots/det_tongji.png`
- `plots/det_palmmatchdb.png`
- `plots/det_11khands.png`
- `plots/det_bjtu2.png`
- `plots/det_xinhua.png`
- `plots/det_gabor.png`
- `plots/det_real.png`
- `plots/latency.png`
- `plots/robustness.png`
- `plots/template_size.png`

![accuracy_enrol1.png](plots/accuracy_enrol1.png)

![accuracy_enrol3.png](plots/accuracy_enrol3.png)

![det_tongji.png](plots/det_tongji.png)

![det_palmmatchdb.png](plots/det_palmmatchdb.png)

![det_11khands.png](plots/det_11khands.png)

![det_bjtu2.png](plots/det_bjtu2.png)

![det_xinhua.png](plots/det_xinhua.png)

![det_gabor.png](plots/det_gabor.png)

![det_real.png](plots/det_real.png)

![latency.png](plots/latency.png)

![robustness.png](plots/robustness.png)

![template_size.png](plots/template_size.png)
