# IndiPalm: contactless palm verification for payments

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Contactless palm verification from an ordinary phone or webcam photo, built towards UPI-style payments. The matcher
needs **no palm training data**: it uses pretrained general-purpose models (MediaPipe hand landmarks, DISK keypoints,
LightGlue matching) with our own alignment, verification and scoring, so no personal palm data is needed to build it.

This README separates what is **measured** from what is **design or research work not yet benchmarked**.

---

## What works today (measured)

No matcher has been trained on palm images (zero-shot). v4's settings (preprocessing, score fusion, thresholds) were
chosen on DEV identities and are reported on separate TEST identities. Full tables, charts and caveats are in
[`benchmarks/results/REPORT.md`](benchmarks/results/REPORT.md) and in each run folder under `benchmarks/results/runs/`.

| Test | Result (v3c, compact template) |
|---|---|
| Tongji contactless, 600 palms, 2 sessions ~2 months apart, 5,400 genuine / 30,000 impostor comparisons | EER **0.05%** with 1 enrolled photo, **0.00%** with 3 |
| PalmMatchDB, whole-hand phone photos (detection + alignment + matching) | EER **1.11%** with 1 photo, **0.00%** with 3; 3.3% of photos rejected by the quality gates |
| 11k Hands, XINHUA, own phone photos (quick run, 30-40 people each) | EER 0.0-0.1% |
| Template size | 97 KB per enrolled photo |
| Latency, desktop CPU | hand detection 74 ms, feature extraction 1.35 s, one comparison 434 ms |

How this compares with published Tongji results (EER):

| Method | Trained on palms? | EER |
|---|---|---|
| CCNet, CompNet (trained and tested on Tongji) | yes, same dataset | 0.00004%, 0.018-0.025% |
| **This matcher (v3c), 1 enrolled photo** | **no** | **0.05%** |
| CompCode (hand-crafted) | no | 0.10% |
| CCNet / PalmBridge trained on another dataset, tested on Tongji | yes, other dataset | 2.2-2.9% |

Among matchers that have not seen the test dataset, this is the lowest Tongji error we found, about 40 times below
the published cross-dataset results. Models trained on Tongji itself are still more accurate on Tongji. Protocols differ
between papers; sources are in [`benchmarks/results/literature_comparison.csv`](benchmarks/results/literature_comparison.csv).

### v4: the production pipeline, on held-out TEST identities

v4 ([`palm_engine.py`](palm_engine.py)) adds to v3c: a capture quality gate, noise smoothing and contrast equalisation
(CLAHE), fusion with the SIFT score, liveness checks, a protected 1:N search and encrypted templates. Every v4 setting
was chosen on DEV identities only ([`benchmarks/results/tuning/DECISIONS.md`](benchmarks/results/tuning/DECISIONS.md));
the numbers below are TEST identities that no choice has seen. Quick profile on a CPU: 15-39 test identities per
dataset, up to 500 impostor comparisons, so 0% means "below about 0.6%" (rule of 3). Full report:
[`benchmarks/results/runs/20261115-0239-quick/REPORT.md`](benchmarks/results/runs/20261115-0239-quick/REPORT.md).

| Dataset (TEST identities) | v4 EER | v4 FRR (of which retake requests) | v3c EER | v3c FRR |
|---|---|---|---|---|
| Tongji (32) | 0.00% | 0.00% | 0.00% | 0.00% |
| BJTU (32), the hardest set | **10.18%** | **25.0%** (0%) | 15.90% | 41.4% |
| Gabor (15) | **0.00%** | **0.00%** | 1.06% | 5.6% |
| 11k Hands (24) | 0.00% | 2.2% (2.2%) | 0.00% | 0.00% |
| XINHUA (24) | 0.00% | 3.3% (3.3%) | 0.00% | 0.00% |
| PalmMatchDB (39) | 1.29% | 21.8% (20.9%) | 0.99% | 5.5% |
| Own photos (2) | 0.00% | 0.00% | 0.00% | 0.00% |

Side-by-side with the original benchmark (13 and 14 Nov), robustness, speed and new capabilities:
[`benchmarks/results/COMPARISON.md`](benchmarks/results/COMPARISON.md).

FAR was 0% for v4 on every dataset. v4 improves the two sets where v3c was weak (BJTU, Gabor). Its quality gate sends
back photos for a retake: on PalmMatchDB, 21% (palms narrower than 100 pixels, or tilted), which a live kiosk turns
into a "move closer" instruction but which counts as FRR here.

Robustness (Tongji TEST, 20 palms, degraded probe vs clean enrolment): v4 has 0% FRR under sensor noise (v3c 50%) and
JPEG 20 (v3c 5%). Under x1.5 brightness and x2 over-exposure v4 sends back 45% and 65% of photos for a retake (v3c
rejects 10% and 60% outright); of the photos v4 accepted, none was falsely rejected.

**1:N search and template protection** (Tongji TEST, 240 palms enrolled, searches with photos about 2 months later;
[`benchmarks/results/embedding/20261115-1440`](benchmarks/results/embedding/20261115-1440/embed_report.md)):

| | Result |
|---|---|
| Palm vector (HOL, no training), unprotected | rank-1 98.7%, recall@10 99.3% |
| Protected vector (IoM, 512 bytes, cancellable) | rank-1 95.3%, recall@10 98.0% |
| Two-stage search: protected top-5, then the v3c matcher decides | FNIR 2.67%, FPIR 0.00% (150 + 150 searches), 5 comparisons instead of 240 |
| Unlinkability between two keys (ISO/IEC 24745 D_sys) | 0.035 (0 = fully unlinkable) |
| Renewability | the same palm under a new key matches at chance level (0.066 equal indices) |

**Liveness and spoofing** (ISO/IEC 30107-3 terms;
[`benchmarks/results/pad/20261115-1441`](benchmarks/results/pad/20261115-1441/pad_report.md)):

| Check | What it does | Result |
|---|---|---|
| Active: random finger counts | the user shows e.g. 2, then 4, then 1 fingers, then the open hand, each prompt after a random delay; checked on the server | finger counting correct on 98.1% of 2,039 real open-hand photos; a printed photo fails every simulated attempt; replayed videos pass 0-2.7% of attempts (best possible guess 3.7%) |
| Passive: screen / print pattern | narrow moire peaks in the palm's spectrum (no training) | genuine photos wrongly flagged (BPCER) 2.9% of 1,673; simulated screen replays accepted (APCER) 5.0%, simulated prints 9.3% |

The simulated attacks only show that the checks react to the patterns. **Real attack photos (prints, phone screens)
have not been collected yet**; `scripts/capture_pad_set.py` records them with the webcam and `.\bench.ps1 pad` then
reports real APCER / BPCER.

## Known gaps (not solved yet)

| Gap | Status |
|---|---|
| Spoof detection on real attacks | the checks exist and are measured on genuine photos and simulated attacks; real printed / on-screen attack photos still have to be collected (`scripts/capture_pad_set.py`) |
| Speed | v4 takes about 1.4 s to make a template and 0.6 s per comparison on a desktop CPU. A faster variant (DISK on a 384 px crop, 512 keypoints) halves template time and cuts comparison time by a third with the same DEV accuracy ([`benchmarks/results/speed/`](benchmarks/results/speed/20261115-1527/speed_dev.md)); it is not adopted until the TEST run is repeated. Not yet suitable for a low-power terminal |
| BJTU (pre-cropped, inconsistent crops and lighting) | v4 EER 10.2%, better than v3c (15.9%) but far from the other datasets |
| Strong over-exposure | v4 asks for a retake instead of failing, but cannot match a palm that is half white |
| Template size | v4 stores about 430 KB per enrolled photo (v3c part 97 KB + SIFT part); v3c alone is 97 KB |
| Data | only PalmMatchDB (Apache-2.0) and the Gabor set (CC BY 4.0) are permissive; Tongji, 11k Hands, BJTU and XINHUA are evaluation-only. A consented dataset of our own is still needed |
| Scale of evidence | the v4 numbers come from the quick profile; run `.\bench.ps1 run --profile full` on a GPU for 20,000 impostor comparisons per dataset |

---

## Palm payment kiosk (register, then pay with your palm)

```
.\bench.ps1 kiosk                       # http://127.0.0.1:5060   (or: python webapp/kiosk.py)
.\bench.ps1 kiosk --pad enforce         # also reject photos the screen/print check flags
```

1. **Register**: show the open palm; the screen gives live instructions (closer, too bright, open your hand, ...) until
   the photo is usable. Then a liveness check asks for random finger counts at random times, ending with the open hand.
   Three open-hand frames are captured. A duplicate search runs over everyone registered; a new palm gets the next free
   ID (`PALM-0001`, `PALM-0002`, ...), an optional name is for the admin only.
2. **Pay**: the same capture and liveness check, then a 1:N search: the protected palm vector picks the 3 most similar
   registered palms and the matcher decides. The result shows the palm ID, the score and the time of every stage.
3. **Admin**: registered palm IDs (no images), template sizes, delete (erases everything stored for that person), audit
   log with CSV export.

Every check runs on the server, not in the browser. Photos are never written to disk. Stored per person in
`webapp/data/kiosk/` (git-ignored): the keypoint templates encrypted with Windows DPAPI, and 3 x 512 bytes of cancellable
IoM search hashes. Three failed liveness attempts within 2 minutes lock the kiosk for 60 seconds.

The passive screen/print check defaults to warn-only: its threshold was set on photos from other cameras, so check it on
the kiosk camera first (`python scripts/capture_pad_set.py`, then `.\bench.ps1 pad`).

---

## Training-free palm matcher and live test web app

The matcher is in three files: `palm_match.py` (v3), `palm_match_compact.py` (v3c, compact template) and
`palm_match_robust.py` (v3cr, v3c plus noise smoothing and retry). Baselines in `palm_match.py`: v2 (SIFT + RANSAC)
and v1 (Competitive Code).

Pipeline: MediaPipe hand landmarks -> quality gates (hand found, distance, blur) -> 7-landmark alignment to a
512x512 right-hand palm crop -> DISK keypoints + LightGlue matches -> RANSAC with a pose check -> coverage-weighted
score (inliers x palm cells covered / 64) -> best score over 1-3 enrolled photos.

To compare v3, v3c, v3cr and v4 side by side with a webcam or uploaded photos (people enrolled before v4 existed
need to be enrolled again for the v4 column):

```
pip install flask torch kornia mediapipe opencv-python
python webapp/app.py          # then open http://127.0.0.1:5050
```

Enrol a person with 1-3 palm photos, then verify new photos. The dashboard shows each version's decision,
score and timing. Templates are stored only in `webapp/data/` on the local machine (not in git); photos are
not saved.

---

## Zero-shot benchmark on public palm datasets (live in the terminal)

```
.\bench.ps1 fetch all          # download the datasets into data/ (git-ignored); data/MANIFEST.json records licence + SHA-256
.\bench.ps1 run                # live dashboard: progress, running EER/FAR/FRR, per-job results, event log
.\bench.ps1 run --profile full # the large run, sized for a GPU
```

| Dataset | Licence | Use | What it tests |
|---|---|---|---|
| PalmMatchDB (Hugging Face) | Apache-2.0 | permissive | whole-hand phone photos: detection + alignment + matching |
| Palmprint Image Dataset with Gabor Filter (Mendeley) | CC BY 4.0 | permissive | small set of palm crops (smoke test) |
| Tongji contactless (ROI) | none stated | evaluation only | standard benchmark, 2 sessions ~61 days apart |
| 11k Hands, palmar photos | academic fair use | evaluation only | whole-hand photos, 190 people |
| BJTU palm v2, XINHUA (Zenodo bundle) | CC BY on a re-upload, provenance unclear | evaluation only | phone palm crops; BJTU has 2 sessions |
| Own phone photos (`real palm images/`) | own data | | whole chain on our own photos |

No matcher has been trained on palm images (zero-shot). Datasets with at least 40 identities are split by identity once,
on the full-size sample: 20% DEV and 80% TEST, the same split for every profile. DEV identities are the only ones the
tuning scripts (`.\bench.ps1 tune`, `benchmarks/gate_dev.py`, `embed`, `speed`) and the passive spoof threshold have
used; every reported number is measured on TEST identities. Reported per dataset and method: accuracy, EER, FAR, FRR
(including photos rejected by the quality gates), TAR at FAR 1% / 0.1% / 0.01%, AUC, d', rank-1, failure-to-enrol,
3-photo enrolment, latency per stage, template size and robustness under simulated lighting / blur / noise / pose.
Spoof detection is measured separately (`.\bench.ps1 pad`), and so are 1:N search and template protection
(`.\bench.ps1 embed --test`). Baselines run on the same protocol: v2 (SIFT + RANSAC) and v1 (Competitive Code).

Each run writes `benchmarks/results/runs/<date-time>-<profile>/`: `run.log`, `REPORT.md`, `metrics.csv`, `latency.csv`,
`robustness.csv`, `datasets.csv`, `scores/*.npz` and `plots/*.png`. An interrupted run continues with
`.\bench.ps1 run --resume <run folder>`; templates are cached in `data/_cache/`, so a second run skips feature extraction.

---

## Design and research work (not benchmarked)

These folders and documents describe the longer-term system: a dedicated near-infrared palm-vein terminal, a model
trained on synthetic palms, cancellable templates and a two-tier payment switch. They contain design documents and
research code with unit tests. **None of their performance figures (accuracy, latency on an NPU, sunlight tolerance,
1:N search time) have been measured on real palms**, and they are not used by the matcher or the benchmark above.

| Path | Content | Status |
|---|---|---|
| [`solution_v0.md`](solution_v0.md), [`approch.md`](approch.md), [`data_and_diffusion_strategy.md`](data_and_diffusion_strategy.md), [`master_implementation_plan.md`](master_implementation_plan.md) | target architecture, hardware concept (850 nm illumination, NXP i.MX8M Plus), data strategy, roadmap | design documents; figures in them are targets, not measurements |
| `common/crypto/` | IoM-URP cancellable hashing, Bloom filters, ISO/IEC 24745 metrics | IoM hashing and the unlinkability metric are used by `palm_protect.py` and measured (see above); Bloom filters unused |
| `common/generators/`, `indidiff_palm/` | procedural and diffusion-based synthetic palm generation | research code; no model trained on it is benchmarked |
| `indipalm_edge/` | edge model components (flow-field unwarping, cross-attention fusion, active PAD) | research code; untrained |
| `npci_switch/` | 1:1 and 1:N matching-switch prototypes, UPI token service mock-up | prototype; not connected to any payment system |
| `android/` | native Android app running the matcher on a phone | demo |

---

## Directory structure

```
.
|-- palm_match.py, palm_match_compact.py, palm_match_robust.py   # the matcher (v3, v3c, v3cr; baselines v2, v1)
|-- palm_engine.py           # v4: the full pipeline used by the kiosk (quality, liveness, template, search)
|-- palm_quality.py          # capture quality gate with a retake instruction per problem
|-- palm_pad.py              # spoof checks: passive screen/print check, active finger-count liveness
|-- palm_embed.py            # fixed-length palm vector (HOL) for 1:N search
|-- palm_protect.py          # cancellable IoM search template, encrypted keypoint templates
|-- webapp/                  # kiosk.py (register / pay / admin) and app.py (side-by-side matcher dashboard)
|-- benchmarks/              # benchmark runner and report; DEV tuning, spoof, search and speed evaluations; results
|-- scripts/                 # dataset download, spoof test-set capture, debugging scripts
|-- bench.ps1                # launcher: fetch, run, report, tune, pad, embed, speed, kiosk, test
|-- common/, indidiff_palm/, indipalm_edge/, npci_switch/   # design and research code (see above)
|-- android/                 # Android demo app
|-- tests/                   # unit and end-to-end tests (.\bench.ps1 test runs the ones for the matcher and kiosk)
```

---

## License

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for details.
MediaPipe, DISK and LightGlue are used under their own licences. Datasets keep their own licences; see
`data/MANIFEST.json` after downloading.
