# Palm vector selection (DEV identities only)

Generated 2026-11-15 02:01 by `benchmarks/embed_eval.py --select`. Gallery = one photo per identity (session 1 where there are sessions); every other photo (session 2) is searched against it. DISK-VLAD is fitted on the other datasets' DEV photos (no identity labels).

| Dataset | Vector | Gallery | Searches | Rank-1 | Recall@5 | Recall@10 | Recall@20 | EER |
|---|---|---|---|---|---|---|---|---|
| tongji | hol | 40 | 120 | 98.33% | 98.33% | 99.17% | 100.00% | 2.50% |
| tongji | disk | 40 | 120 | 70.83% | 90.00% | 95.83% | 99.17% | 13.49% |
| tongji | hol+disk | 40 | 120 | 95.00% | 97.50% | 97.50% | 99.17% | 5.83% |
| bjtu2 | hol | 40 | 120 | 86.67% | 94.17% | 97.50% | 100.00% | 12.50% |
| bjtu2 | disk | 40 | 120 | 45.00% | 62.50% | 78.33% | 90.00% | 23.45% |
| bjtu2 | hol+disk | 40 | 120 | 56.67% | 78.33% | 85.83% | 94.17% | 18.33% |
| 11khands | hol | 40 | 160 | 98.12% | 100.00% | 100.00% | 100.00% | 5.00% |
| 11khands | disk | 40 | 160 | 89.38% | 95.62% | 97.50% | 98.75% | 7.50% |
| 11khands | hol+disk | 40 | 160 | 95.62% | 98.12% | 98.12% | 98.75% | 4.37% |
