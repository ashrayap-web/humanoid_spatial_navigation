# Evaluation summary

| pair | run | GT changes | predicted (confirmed / unverified) | P / R / F1 (all claims) | P / R / F1 (confirmed only) |
|---|---|---|---|---|---|
| main | `demo` | 1 | 1 / 2 | 0.33 / 1.00 / 0.50 | 1.00 / 1.00 / 1.00 |

## Synthetic suite (C5 only): 200 scenes, 414 ground-truth changes

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| overall | 408 | 2 | 6 | 1.00 | 0.99 | 0.99 |
| removed | 84 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| added | 83 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| moved | 157 | 2 | 6 | 0.99 | 0.96 | 0.98 |
| replaced | 84 | 0 | 0 | 1.00 | 1.00 | 1.00 |

Moved / rotated objects: mean distance error 0.093 m, median yaw error 0.7°.

---

# Evaluation — pair `main` (run `demo`)

Based on **1 ground-truth change(s)** and 3 prediction(s) (1 confirmed, 2 unverified; rejected candidates excluded).

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| all claims (confirmed + unverified) | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| confirmed only | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |

Per type (all claims):

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| removed | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |

Matches: removed bag ← chg_000
False positives: chg_001 removed jacket (unverified); chg_003 removed monitor (unverified)
False negatives: none

Unverified predictions: 2 — expected: 0, found as unverified: 0.
