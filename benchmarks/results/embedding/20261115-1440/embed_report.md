# 1:N search and template protection (TEST identities)

Generated 2026-11-15 14:40 by `benchmarks/embed_eval.py --test`. Tongji TEST identities (never used for any choice). Vector: **hol**. 240 palms enrolled with 1 session-1 photo(s); searches use session-2 photos about 2 months later. Half of the TEST identities never enrol and give the non-mated searches.

## Identification

| Search | Rank-1 | Recall@10 | Verification EER |
|---|---|---|---|
| vector only, unprotected | 98.67% | 99.33% | 0.74% |
| vector only, protected (IoM, 512 x 16) | 95.33% | 98.00% | 3.34% |

**Two-stage search** (protected vector picks the top 5, the v3c matcher decides at threshold 12): FNIR **2.67%** (150 mated searches; the true palm was in the shortlist for 97.33%), FPIR **0.00%** (150 non-mated searches). Cost per search: 5 LightGlue comparisons instead of 240.

## Template protection (ISO/IEC 24745)

- Protected search template: 512 IoM indices = 512 bytes; the raw vector is not stored.
- Same palm, same key: median 0.620 equal indices (chance level 0.062).
- **Renewability**: same palm under a new key: median 0.066, the level of a different palm (median 0.066). A revoked template stops matching.
- **Unlinkability**: D_sys = **0.035** between key A and key B templates (0 = fully unlinkable, 1 = fully linkable; 480 mated / 480 non-mated pairs).
- **Irreversibility** is not proven: IoM discards magnitudes, but published attacks can approximate a vector from hashes when the key is stolen too. Keys belong in a hardware keystore.
- The keypoint template used by the matcher is encrypted at rest (palm_protect.py: Windows DPAPI or AES-256-GCM) and decrypted only in memory at match time.
