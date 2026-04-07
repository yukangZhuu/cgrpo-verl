# TTN-2K Data Report

**Seed**: 42
**Pool**: 11303 problems
**Sampled**: 2200 → 2000 train + 200 test

---

## 1. Pool Distribution
### Full Pool (11k) (n=11303)

| Category | Count | % |
|----------|-------|---|
| Trivial | 4167 | 36.9% |
| Easy | 3324 | 29.4% |
| Medium | 2031 | 18.0% |
| Hard | 509 | 4.5% |
| Unsolvable | 1272 | 11.3% |

Mean pass_rate: 0.5425

Steps distribution:
   6 steps: 1776
   7 steps: 2051
   8 steps: 2108
   9 steps: 1728
  10 steps: 1469
  11 steps:  890
  12 steps:  641
  13 steps:  387
  14 steps:  253
  Mean: 8.69

## 2. Train Set Distribution
### Train (2k) (n=2000)

| Category | Count | % |
|----------|-------|---|
| Trivial | 712 | 35.6% |
| Easy | 604 | 30.2% |
| Medium | 375 | 18.8% |
| Hard | 91 | 4.5% |
| Unsolvable | 218 | 10.9% |

Mean pass_rate: 0.5345

Steps distribution:
   6 steps:  334
   7 steps:  348
   8 steps:  362
   9 steps:  298
  10 steps:  276
  11 steps:  137
  12 steps:  131
  13 steps:   72
  14 steps:   42
  Mean: 8.69

## 3. Test Set Distribution
### Test (200) (n=200)

| Category | Count | % |
|----------|-------|---|
| Trivial | 72 | 36.0% |
| Easy | 60 | 30.0% |
| Medium | 37 | 18.5% |
| Hard | 9 | 4.5% |
| Unsolvable | 22 | 11.0% |

Mean pass_rate: 0.5387

Steps distribution:
   6 steps:   35
   7 steps:   41
   8 steps:   41
   9 steps:   36
  10 steps:   26
  11 steps:    9
  12 steps:    5
  13 steps:    5
  14 steps:    2
  Mean: 8.30

## 4. Unsolvable Subset (pass@32=0, from train)

Total: 218 (10.9% of train)

Steps distribution:
   6 steps:   30
   7 steps:   40
   8 steps:   39
   9 steps:   18
  10 steps:   33
  11 steps:   22
  12 steps:   18
  13 steps:   12
  14 steps:    6
  Mean: 9.00

Question length: mean=282, min=50, max=496

## 5. Distribution Comparison (Train vs Pool)

| Category | Pool % | Train % | Test % |
|----------|--------|---------|--------|
| Trivial | 36.9% | 35.6% | 36.0% |
| Easy | 29.4% | 30.2% | 30.0% |
| Medium | 18.0% | 18.8% | 18.5% |
| Hard | 4.5% | 4.5% | 4.5% |
| Unsolvable | 11.3% | 10.9% | 11.0% |

## 6. Output Files

| File | Description | Count |
|------|-------------|-------|
| `train_2k.jsonl` | Training set | 2000 |
| `test_200.jsonl` | Test set | 200 |
| `unsolvable_pass32.jsonl` | pass@32=0 from train (for re-scoring) | 218 |
