# Experiment Data Preparation Report

**Seed**: 42
**Pool**: candidates_merged.jsonl (11303 samples)
**Guidance levels**: [0, 0, 0.25, 0.5, 0.75, 1.0]

---

## 1. Pool Overview

=== Full Pool (11k) ===
Total samples: 11303
Mean pass_rate: 0.5425
Median pass_rate: 0.5938

Difficulty distribution:
       Trivial:  4167 ( 36.9%)
          Easy:  3324 ( 29.4%)
        Medium:  2031 ( 18.0%)
          Hard:   509 (  4.5%)
    Unsolvable:  1272 ( 11.3%)

Steps count distribution:
   6 steps: 1776
   7 steps: 2051
   8 steps: 2108
   9 steps: 1728
  10 steps: 1469
  11 steps:  890
  12 steps:  641
  13 steps:  387
  14 steps:  253

## 2. Standard Training Set (3k)

=== Standard Training Set ===
Total samples: 3000
Mean pass_rate: 0.5400
Median pass_rate: 0.5938

Difficulty distribution:
       Trivial:  1088 ( 36.3%)
          Easy:   895 ( 29.8%)
        Medium:   550 ( 18.3%)
          Hard:   137 (  4.6%)
    Unsolvable:   330 ( 11.0%)

Steps count distribution:
   6 steps:  492
   7 steps:  532
   8 steps:  555
   9 steps:  439
  10 steps:  417
  11 steps:  224
  12 steps:  179
  13 steps:  106
  14 steps:   56

## 3. Unsolvable Subset

Total unsolvable in standard set: 330
Fraction of standard set: 11.0%

Steps count distribution (unsolvable):
   6 steps:   47
   7 steps:   54
   8 steps:   64
   9 steps:   33
  10 steps:   52
  11 steps:   32
  12 steps:   21
  13 steps:   18
  14 steps:    9
  Mean: 8.95

Question length: mean=280, min=50, max=496

## 4. Expanded Unsolvable Set

Guidance levels: [0, 0, 0.25, 0.5, 0.75, 1.0]
Instances per original problem: 6
Total expanded instances: 1980

By guidance level:
  g=0.00:  660 instances, avg guidance_steps=0.0, avg student_steps=8.9
  g=0.25:  330 instances, avg guidance_steps=2.3, avg student_steps=6.7
  g=0.50:  330 instances, avg guidance_steps=4.5, avg student_steps=4.4
  g=0.75:  330 instances, avg guidance_steps=6.7, avg student_steps=2.3
  g=1.00:  330 instances, avg guidance_steps=7.9, avg student_steps=1.0

## 5. Output Files

| File | Description | Count |
|------|-------------|-------|
| `standard_train_3k.jsonl` | Randomly sampled standard training set | 3000 |
| `unsolvable_subset.jsonl` | Unsolvable problems (pass_rate=0) from standard set | 330 |
| `unsolvable_expanded.jsonl` | Guidance-expanded unsolvable set | 1980 |
