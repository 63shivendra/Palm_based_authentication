# Spoof detection (passive recapture check)

Generated 2026-11-15 14:50 by `benchmarks/pad_eval.py`. Metrics follow ISO/IEC 30107-3.

Threshold **3.01**: the 99th percentile of the scores of 366 genuine photos of 11k Hands DEV identities (BPCER 1% on DEV). Everything below is measured on other photos.

## Genuine photos (BPCER)

| Set | Photos | BPCER |
|---|---|---|
| 11khands (TEST) | 1461 | 3.22% |
| palmmatchdb (TEST) | 208 | 0.48% |
| real (TEST) | 4 | 0.00% |
| all TEST genuine photos | 1673 | 2.87% |

## Simulated attacks (APCER)

Rendered from TEST genuine photos. They show that the check reacts to screen and print patterns. **They do not replace real attack photos**, and real attacks may be harder (higher resolution screens, photo paper, blur hiding the pattern).

| Attack | Presentations | APCER (accepted as live) |
|---|---|---|
| screen (simulated) | 300 | 5.00% |
| print (simulated) | 300 | 9.33% |

D-EER 6.16%, BPCER20 18.41%, BPCER10 0.00% (simulated attacks).

## Real attack photos

Not collected yet. Capture them with `python scripts/capture_pad_set.py` (printed palm photos and palm photos shown on a phone or monitor, plus live photos of the same people), then run this script again.


## Active check (random finger counts)

**Finger counting on real hands.** The active check counts straight fingers from the MediaPipe landmarks. On 2039 genuine open-hand photos (11k Hands, PalmMatchDB, own photos) it counted 5 fingers for 98.14% and 4 for 1.86% (thumb held close to the hand). Finger spread on these photos: median 6.5 deg; the final step asks for >= 20 deg.

**Replayed video (simulated landmark sequences, 300 attempts each).** A recording that shows finger counts in a fixed or random order, at 15 frames per second:

| Replayed video | Accepted (APCER) |
|---|---|
| cycles 1-2-3-4-5, 0.8 s per pose | 0.00% |
| cycles 1-2-3-4-5, 1.5 s per pose | 1.67% |
| cycles 1-2-3-4-5, 2.5 s per pose | 2.67% |
| cycles 1-2-3-4-5, 3.5 s per pose | 2.67% |
| random order, 1.5 s per pose | 0.00% |

The best possible replay has to guess each random step (1 in 3 per step, 1 in 27 for three steps, 3.7%). Lock the account for a while after 3 failed attempts. A printed photo never changes its finger count and failed every simulated attempt (tests/test_palm_pad.py).
