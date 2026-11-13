# Palm verification benchmark report

Generated 2026-11-13 by `benchmarks/make_report.py` from the JSON files in this folder.

Matcher: `palm_match.py` v3 (MediaPipe hand detection, own alignment and quality gates, DISK + LightGlue pretrained models used without any palm training, own pose-checked verification and coverage-weighted score). `v3c` is the compact-template variant in `palm_match_compact.py`. `v3cr` is v3c plus noise-adaptive smoothing and retry on failure, in `palm_match_robust.py`.

## 1. Test datasets

| Dataset | Source | Licence | Content used | Protocol |
|---|---|---|---|---|
| Tongji Contactless Palmprint | cslinzhang.github.io/ContactlessPalm (official ROI images) | No licence stated on the official page; treated as research-only. Used for evaluation only. | 600 palms, 2 sessions about 2 months apart, 12,000 pre-cropped 128x128 images; 3 images per palm per session used | enrol session 1, verify session 2 |
| PalmMatchDB | huggingface.co/datasets/aspmirlab/PalmMatchDB | Apache-2.0 (Hugging Face tag). Consent terms not verified. | phone photos of whole hands; 10,528 files but only 531 distinct images; labels unreliable | all pairs, restricted to pairs with trustworthy labels (same palm: photos < 10 min apart; different palm: > 1 day apart) |
| Palmprint Image Dataset with Gabor Filter (raw images) | data.mendeley.com/datasets/r8hxykxnk5 (version 2) | CC BY 4.0 | 15 people, 103 raw 128x128 palm crops | all pairs |
| Own phone photos | real palm images/ in this repository | own data | 4 photos: 3 of one right hand (2 sessions), 1 of the left hand | all pairs |

Not tested:

- **CASIA Palmprint:** needs registration and a signed release agreement; official portal unreachable.
- **SCUT-PV (palm vein):** needs a signed agreement sent to the lab; commercial use not allowed; near-infrared images.
- **GenPalm:** dataset announced in the paper, no public release found.
- **PCE-SynthPalm (Kaggle):** no licence stated; project policy prohibits downloading it.

## 2. Accuracy, EER, FAR, FRR

Accuracy = share of all comparisons decided correctly at the operating threshold. FAR and FRR include photos rejected by the quality gates (FTE). Rank-1 = identification accuracy against a gallery of all palms.

| Dataset | Method | Genuine / impostor comparisons | Accuracy | EER | FAR | FRR | TAR at FAR 0.1% | TAR at FAR 0.01% | Rank-1 | FTE |
|---|---|---|---|---|---|---|---|---|---|---|
| Tongji Contactless Palmprint | v3, 1 enrolled photo | 5,400 / 30,000 | 99.98% | 0.06% | 0.00% | 0.13% | 99.96% | 99.91% | 100.00% | 0.00% |
| Tongji Contactless Palmprint | v3, 3 enrolled photos | 1,800 / 30,000 | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% | n/a | 0.00% |
| Tongji Contactless Palmprint | v3c compact, 1 photo | 5,400 / 30,000 | 99.99% | 0.05% | 0.00% | 0.09% | 99.94% | 99.94% | 100.00% | 0.00% |
| Tongji Contactless Palmprint | v3c compact, 3 photos | 1,800 / 30,000 | 99.99% | 0.00% | 0.01% | 0.00% | 100.00% | 100.00% | n/a | 0.00% |
| Tongji Contactless Palmprint | v2 SIFT, 1 photo | 5,400 / 30,000 | 99.85% | 0.44% | 0.07% | 0.57% | 99.43% | 99.19% | 100.00% | 0.00% |
| PalmMatchDB | v3, 1 enrolled photo | 453 / 11,372 | 99.86% | 1.32% | 0.00% | 6.67% | 97.57% | n/a | n/a | 3.26% |
| PalmMatchDB | v3, 3 enrolled photos | 95 / 1,831 | 100.00% | 0.00% | 0.00% | 3.26% | n/a | n/a | n/a | 3.26% |
| PalmMatchDB | v3c compact, 1 photo | 453 / 11,372 | 99.87% | 1.11% | 0.00% | 6.46% | 98.01% | n/a | n/a | 3.26% |
| PalmMatchDB | v3c compact, 3 photos | 95 / 1,831 | 100.00% | 0.00% | 0.00% | 3.26% | n/a | n/a | n/a | 3.26% |
| PalmMatchDB | v2 SIFT, 1 photo | 453 / 11,372 | 98.85% | 0.46% | 1.14% | 3.68% | 98.45% | n/a | n/a | 3.26% |
| Palmprint Image Dataset with Gabor Filter (raw images) | v3, 1 enrolled photo | 311 / 4,942 | 99.83% | 0.29% | 0.00% | 2.89% | 99.36% | n/a | 100.00% | 0.00% |
| Palmprint Image Dataset with Gabor Filter (raw images) | v3, 3 enrolled photos | 58 / 812 | 99.89% | 0.00% | 0.00% | 1.72% | n/a | n/a | n/a | 0.00% |
| Palmprint Image Dataset with Gabor Filter (raw images) | v3c compact, 1 photo | 311 / 4,942 | 99.87% | 0.32% | 0.00% | 2.25% | 99.04% | n/a | 98.86% | 0.00% |
| Palmprint Image Dataset with Gabor Filter (raw images) | v3c compact, 3 photos | 58 / 812 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | n/a | 0.00% |
| Palmprint Image Dataset with Gabor Filter (raw images) | v2 SIFT, 1 photo | 311 / 4,942 | 99.90% | 0.00% | 0.10% | 0.00% | 100.00% | n/a | 100.00% | 0.00% |
| Own phone photos | v3, 1 enrolled photo | 3 / 3 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | 100.00% | 0.00% |
| Own phone photos | v3, 2 enrolled photos | 1 / 1 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | n/a | 0.00% |
| Own phone photos | v3c compact, 1 photo | 3 / 3 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | 100.00% | 0.00% |
| Own phone photos | v3c compact, 2 photos | 1 / 1 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | n/a | 0.00% |
| Own phone photos | v3cr robust, 1 photo | 3 / 3 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | 100.00% | 0.00% |
| Own phone photos | v3cr robust, 2 photos | 1 / 1 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | n/a | 0.00% |
| Own phone photos | v2 SIFT, 1 photo | 3 / 3 | 100.00% | 0.00% | 0.00% | 0.00% | n/a | n/a | 100.00% | 0.00% |
| Own phone photos | v1 CompCode, 1 photo | 3 / 3 | 50.00% | 33.33% | 0.00% | 100.00% | n/a | n/a | 100.00% | 0.00% |

Notes:

- The v3 threshold (26) was set using Tongji, so v3 FAR and FRR on Tongji are optimistic. EER does not depend on the threshold.
- The v3c threshold (12) was set from PalmMatchDB and the phone photos only. v3cr uses the same threshold; a retry must reach 1.5 times it.
- v3cr on Tongji uses 10,000 impostor comparisons, not 30,000, because every failed comparison is retried three times.
- `n/a` means the test has too few impostor comparisons to measure that false-accept level.
- The largest test has 30,000 impostor comparisons, which supports FAR claims down to about 0.01%, not the 0.001% project target.
- Tongji and the Gabor dataset are pre-cropped palms, so they do not exercise hand detection and alignment. Only PalmMatchDB and the phone photos test the whole chain.

## 3. Robustness to capture conditions

Tongji, 200 palms: clean enrolled photo, degraded probe photo, 200 genuine and 1,500 impostor comparisons per condition, v3 with 1 enrolled photo.

| Probe condition | FRR | FAR | EER | Median genuine score |
|---|---|---|---|---|
| clean (baseline) | 0.00% | 0.00% | 0.00% | 660 |
| darker (x0.5) | 0.00% | 0.00% | 0.00% | 685 |
| brighter (x1.5) | 4.00% | 0.00% | 2.10% | 615 |
| low contrast (x0.4) | 0.00% | 0.00% | 0.00% | 604 |
| uneven light (gradient) | 0.50% | 0.00% | 0.48% | 533 |
| blur (sigma 2) | 0.00% | 0.00% | 0.00% | 577 |
| blur (sigma 4) | 0.50% | 0.00% | 0.52% | 402 |
| rotation 10 deg | 0.00% | 0.00% | 0.00% | 658 |
| rotation 20 deg | 3.50% | 0.00% | 1.52% | 627 |
| closer (x1.15) | 0.00% | 0.00% | 0.00% | 529 |
| farther (x0.85) | 0.00% | 0.00% | 0.00% | 539 |
| sensor noise (sigma 12) | 82.00% | 0.00% | 24.95% | 0 |
| JPEG quality 20 | 7.00% | 0.00% | 1.00% | 519 |

Same test with the compact template (v3c):

| Probe condition | FRR | FAR | EER | Median genuine score |
|---|---|---|---|---|
| clean (baseline) | 0.00% | 0.00% | 0.00% | 305 |
| darker (x0.5) | 0.00% | 0.00% | 0.00% | 289 |
| brighter (x1.5) | 4.00% | 0.00% | 2.48% | 259 |
| low contrast (x0.4) | 0.00% | 0.00% | 0.00% | 279 |
| uneven light (gradient) | 0.50% | 0.00% | 0.00% | 257 |
| blur (sigma 2) | 0.00% | 0.00% | 0.00% | 298 |
| blur (sigma 4) | 2.00% | 0.00% | 0.52% | 198 |
| rotation 10 deg | 0.00% | 0.00% | 0.00% | 295 |
| rotation 20 deg | 3.50% | 0.00% | 1.52% | 272 |
| closer (x1.15) | 0.00% | 0.00% | 0.00% | 245 |
| farther (x0.85) | 0.00% | 0.00% | 0.00% | 236 |
| sensor noise (sigma 12) | 56.00% | 0.00% | 14.78% | 2 |
| JPEG quality 20 | 3.50% | 0.00% | 1.00% | 217 |

Same test with v3cr (`palm_match_robust.py`: v3c plus noise-adaptive smoothing and retry on failure). The additions were chosen on palms 300-499; this test uses palms 0-199:

| Probe condition | FRR | FAR | EER | Median genuine score |
|---|---|---|---|---|
| clean (baseline) | 0.00% | 0.00% | 0.00% | 305 |
| darker (x0.5) | 0.00% | 0.00% | 0.00% | 289 |
| brighter (x1.5) | 4.00% | 0.00% | 2.55% | 259 |
| low contrast (x0.4) | 0.00% | 0.00% | 0.00% | 279 |
| uneven light (gradient) | 0.50% | 0.00% | 0.00% | 257 |
| blur (sigma 2) | 0.00% | 0.00% | 0.00% | 298 |
| blur (sigma 4) | 2.00% | 0.00% | 0.52% | 198 |
| rotation 10 deg | 0.00% | 0.00% | 0.00% | 295 |
| rotation 20 deg | 0.00% | 0.00% | 0.00% | 272 |
| closer (x1.15) | 0.00% | 0.00% | 0.00% | 245 |
| farther (x0.85) | 0.00% | 0.00% | 0.00% | 236 |
| sensor noise (sigma 12) | 0.00% | 0.00% | 0.00% | 229 |
| JPEG quality 20 | 0.00% | 0.00% | 0.00% | 217 |

The degradations are simulated on the cropped image, not captured with a camera. Pixel noise added at this resolution breaks matching; real photos need to be tested for this.

## 4. Inference latency

Measured on desktop PC, NVIDIA RTX 5080 GPU. Median of repeated runs.

| Stage | GPU | CPU |
|---|---|---|
| Hand detection (MediaPipe, 12 MP photo) | - | 35 ms |
| Alignment, crop, quality gates | - | 1 ms |
| DISK feature extraction | 14 ms | 184 ms |
| Match + verify, v3 (2,048 keypoints) | 30 ms | 430 ms |
| Match + verify, v3c compact (1,024 keypoints) | 21 ms | 150 ms |
| Total for one verification, v3 | about 80 ms | about 650 ms |
| Total for one verification, v3c compact | about 71 ms | about 370 ms |

Totals = detection + alignment + extraction + one match. Each extra enrolled photo adds one match. Measured while another benchmark was using the GPU, so GPU figures are slightly high. Not measured on a phone or payment terminal.

## 5. Spoof detection

Not measured. The matcher has no liveness or anti-spoofing check; a printed photo of a palm may be accepted.

## 6. Comparison with published results

Published numbers are copied from the cited sources, not re-run here, and use different protocols (most train and test on Tongji itself). Ours is zero-shot: no palm image was used for training.

| Dataset | Method | EER | Trained on palm images | Protocol | Source |
|---|---|---|---|---|---|
| Tongji | **ours: v3, 1 enrolled photo** | 0.06% | no (zero-shot) | cross-session | this repository |
| Tongji | **ours: v3, 3 enrolled photos** | 0.00% | no (zero-shot) | cross-session, 3-photo enrolment (600 palms enrolled) | this repository |
| Tongji | **ours: v3c compact, 1 photo** | 0.05% | no (zero-shot) | cross-session | this repository |
| Tongji | **ours: v3c compact, 3 photos** | 0.00% | no (zero-shot) | cross-session, 3-photo enrolment (600 palms enrolled) | this repository |
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
| PalmMatchDB | **ours: v3, 1 enrolled photo** | 1.32% | no (zero-shot) | all-vs-all (trusted-label pairs) | this repository |
| PalmMatchDB | **ours: v3, 3 enrolled photos** | 0.00% | no (zero-shot) | all-vs-all (trusted-label pairs), 3-photo enrolment (36 palms enrolled) | this repository |
| PalmMatchDB | **ours: v3c compact, 1 photo** | 1.11% | no (zero-shot) | all-vs-all (trusted-label pairs) | this repository |
| PalmMatchDB | **ours: v3c compact, 3 photos** | 0.00% | no (zero-shot) | all-vs-all (trusted-label pairs), 3-photo enrolment (36 palms enrolled) | this repository |
| PalmMatchDB | no published EER found | - | - | The dataset paper (Alausa et al., ICPECA 2023, IEEE Xplore 10076097) is paywalled; its results could not be read. No other paper reporting on this dataset was found. | web search, 2026-11-13 |
| Palmprint Image Dataset with Gabor Filter | **ours: v3, 1 enrolled photo** | 0.29% | no (zero-shot) | all-vs-all | this repository |
| Palmprint Image Dataset with Gabor Filter | **ours: v3, 3 enrolled photos** | 0.00% | no (zero-shot) | all-vs-all, 3-photo enrolment (15 palms enrolled) | this repository |
| Palmprint Image Dataset with Gabor Filter | **ours: v3c compact, 1 photo** | 0.32% | no (zero-shot) | all-vs-all | this repository |
| Palmprint Image Dataset with Gabor Filter | **ours: v3c compact, 3 photos** | 0.00% | no (zero-shot) | all-vs-all, 3-photo enrolment (15 palms enrolled) | this repository |
| Palmprint Image Dataset with Gabor Filter | no published result found | - | - | Mendeley Data record only; no paper reporting verification results on it was found. | web search, 2026-11-13 |

## 7. Files

- `benchmark_results.csv`: every metric, one row per dataset and method
- `requested_metrics.csv`: accuracy, EER, FAR, FRR, latency, robustness and spoof detection, v3 next to v3c
- `robustness.csv`, `latency.csv`, `dataset_log.csv`, `literature_comparison.csv`
- `benchmark_scores.csv`: the score of every single comparison
- `plots/accuracy.png`, `plots/robustness.png`, `plots/size_speed.png`: summary charts (`benchmarks/plot_results.py`)
- `logs/`: console output of each run; `*.json`: raw results; `*.png`: score histograms and DET curves

Re-run v3 and v3c with `bash benchmarks/run_all.sh`, and v3cr with `bash benchmarks/run_robust.sh`.
