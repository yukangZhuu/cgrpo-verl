# MFC — Monotone Frontier Curriculum  
## Algorithm Design Specification

> Target regime: **unsolvable-only RLVR training data** (all training samples have student-model pass@K = 0 under zero guidance).  
> Relation to paper: replaces / upgrades the per-sample adaptive curriculum slot; AdaBack remains as a primary baseline.

---

## 1. Logic Chain (problem → method)

### 1.1 Three-level objective hierarchy

| Level | Objective | Notes |
|-------|-----------|-------|
| Terminal | Test-time zero-guidance pass rate on originally-unsolvable problems | What the paper ultimately measures |
| **Surrogate** | **Maximize `frac_at_zero` = fraction of training samples whose frontier `ρ★ = 0`** | Directly measurable during training |
| Tool | Maximize per-step RL gradient signal (signal quantity) | Proxy used by AdaBack-style methods |

### 1.2 Distribution-shift argument

- Training with `ρ > 0` exposes the policy to a guided input distribution `p_train(s | ρ)` that is **not** the test distribution `p_test(s)` (no hint).
- GRPO's gradient direction under `ρ > 0` is
  `∇_θ E_{s ~ p_train(· | ρ)} [log π_θ(a | s) · A(s, a)]`,
  which optimizes `π_θ(a | s_guided)` directly, and `π_θ(a | s_unguided)` only indirectly through parameter sharing.
- At `ρ = 0`, `p_train = p_test` and the gradient is "on-target".
- The gap between guided and unguided distributions (`KL(p_train(·|ρ) ‖ p_test)`) is monotone in `ρ`.

**Implication.** Signal quantity under `ρ > 0` is partially "off-target"; getting a sample to `ρ = 0` is a *precondition* for purely on-target training. Hence `frac_at_zero` is a better surrogate of the terminal objective than per-step signal magnitude.

### 1.3 Why AdaBack is sub-optimal on unsolvable-only

AdaBack's updates drive each sample toward `avg_reward(ρᵢ) ≈ τ`, i.e. the region where reward variance (and hence GRPO signal) is maximal under `p_train(·|ρᵢ)`. This is equivalent to  
`AdaBack ≈ argmax_ρᵢ Var_{s ∼ p_train(·|ρᵢ)}[r]`.

When `p_train = p_test` (general data), this is close to the surrogate objective.  
When `p_train ≠ p_test` (unsolvable-only), it is not.

Empirical evidence from our AdaBack run on 100-problem TTN-unsolvable (hint mode, 8×PRO6000, 1200 steps):

- Slowest convergence among `baseline GRPO / mixture (R3) / AdaBack`.
- `mean_rho ≈ 0.15`, `frac_at_zero ≈ 0.45`. About **55%** of samples remain trapped at non-zero `ρ`, occupying compute while producing only shifted-distribution signal.

Mechanism of the trap: on failure (`avg_reward < τ`), AdaBack raises `ρ_min` upward, forgetting that lower `ρ` was ever achievable. On the next success it allows `ρ_min = 0` again — the lower bound bounces, creating oscillation rather than monotone descent.

### 1.4 MFC in one sentence

> **MFC directly optimizes `frac_at_zero` on unsolvable-only data by (i) representing per-sample state as a monotonically-descending frontier instead of an interval, and (ii) biasing rollout compute toward the current frontier with periodic probes below it.**

---

## 2. Related-work framing (for paper)

Three-way taxonomy of backward curriculum under RL for reasoning:

1. **Staged curriculum** — global, manually scheduled difficulty progression. Classic in pre-RLVR reasoning work, less prevalent in recent RLVR literature because the scheduler design is ad-hoc.
2. **Mixture curriculum** (R3-style, 2024). Samples of multiple difficulty / guidance levels are mixed in a fixed ratio. Simple, reproducible; no per-sample state. Used as one of our baselines.
3. **Per-sample curriculum** (AdaBack, 2025). Per-sample `[ρ_min, ρ_max]` interval; bisection-style updates to drive each sample toward the τ-success region. Current SOTA for per-sample backward curriculum.

All three were designed against **generic** datasets. They implicitly assume `p_train ≈ p_test`. Our unsolvable-only regime violates that assumption — **this is the opening through which MFC enters the paper**.

---

## 3. Algorithm

### 3.1 Per-sample state

For each sample `i`:

| Field | Type | Initial | Meaning |
|-------|------|---------|---------|
| `ρ★_i` | `float ∈ [0,1]` | `default_rho_star = 1.0` | Frontier: lowest `ρ` empirically verified as solvable |
| `safety_counter_i` | `int` | `0` | Consecutive exploit-mode failures at `ρ★_i` |
| `visits_i` | `int` | `0` | Total number of visits (for logging) |
| `mode_i` | `"exploit" | "probe"` | `"exploit"` | Mode of last visit |
| `rho_used_i` | `float` | same as `ρ★_i` | ρ used in the last visit (for logging / update) |
| `safety_triggered_total_i` | `int` | `0` | Count of safety-valve activations (for logging) |

Contrast with AdaBack: `(ρ_min, ρ_max, ρ)` triple (interval) → MFC: single `ρ★` (point).

### 3.2 Success criterion

`ε = 1 / n_rollouts`. A visit is *successful* iff `avg_reward ≥ ε` (≥ 1 of `n` rollouts correct). `ε` is **derived**, not a free hyper-parameter.

### 3.3 Visit-time ρ sampling (Mechanism 2 — Frontier-Biased Visit Sampling)

Since the actual supervision a rollout sees is determined by the **discrete teacher-step count** `g = round(ρ * num_steps_i)` (clipped to `[0, num_steps_i - 1]`), the probe branch must sample in that discrete space. Otherwise, tiny continuous changes in ρ produce no change in the hint the model actually reads and waste compute.

At each visit of sample `i`:

```
g_curr_i = clip(round(ρ★_i * num_steps_i), 0, num_steps_i - 1)

u = Uniform(0, 1)
if u < (1 - p_probe):
    mode = "exploit"
    ρ_used = ρ★_i
else:
    mode = "probe"
    if g_curr_i == 0:
        # Discrete floor reached → no lower step to probe.  Degenerate to
        # exploit at the current (now ≡ 0) frontier.
        ρ_used = ρ★_i        # functionally 0
        mode = "exploit"
    else:
        g_probe = Uniform({0, 1, ..., g_curr_i - 1})
        ρ_used = g_probe / num_steps_i   # strictly fewer teacher steps
```

Two free consequences of the discrete sampling:

1. **Minimum-advance guarantee.** Every probe strictly reduces the revealed teacher-step count relative to exploit — no wasted probes that happen to map to the same `g` after `round()`.
2. **`ρ★ = 0` is reachable in finite time.** `g_probe = 0` is a positive-probability outcome whenever `g_curr > 0`; on probe success the frontier snaps exactly to `0.0` (not to a tiny residual like `1/n_steps²`). This subsumes an earlier "snap-to-zero" proposal — no dedicated rule needed.

**All `n` rollouts of a group use the single `ρ_used`.** GRPO's intra-group identical-prompt assumption is preserved — the only change vs. AdaBack is the ρ distribution.

### 3.3.1 Lattice invariant

After any probe success or safety-valve event, `ρ★_i` is exactly on the per-sample lattice `{g / num_steps_i : g ∈ [0, num_steps_i - 1]}`. This is maintained by:

- **Initialization**: on the first `get_rho` call for sample `i`, `ρ★_i` is snapped from `default_rho_star = 1.0` to `(num_steps_i - 1) / num_steps_i`, which is the top valid lattice point.
- **Probe success**: `ρ★_i ← g_probe / num_steps_i`, which is on-lattice by construction.
- **Safety bump**: handled in §3.4 below.
- **FP precision**: the `g_from_rho` helper uses a fixed `+1e-9` epsilon before rounding to avoid banker's-rounding flips when values like `3/7 * 7` lie just below an integer boundary.

### 3.4 Post-rollout update (Mechanism 1 — Monotone Frontier / Ratchet)

After the `n` rollouts complete and `avg_reward` is known:

```
Case A: avg_reward ≥ ε  AND  ρ_used ≤ ρ★_i
    ρ★_i ← min(ρ★_i, ρ_used)             # advance frontier (stays on-lattice)
    safety_counter_i ← 0

Case B: avg_reward < ε   AND  mode == "exploit"   (failure at frontier)
    safety_counter_i += 1
    if safety_counter_i ≥ safety_K:
        # Continuous bump, then snap to lattice, then enforce +1-step advance:
        g_curr = clip(round(ρ★_i * num_steps_i), 0, num_steps_i - 1)
        g_bump = clip(round((ρ★_i + delta_safe) * num_steps_i), 0, num_steps_i - 1)
        g_new  = min(num_steps_i - 1, max(g_curr + 1, g_bump))
        ρ★_i   ← g_new / num_steps_i
        safety_counter_i ← 0
        safety_triggered_total_i += 1

Case C: avg_reward < ε   AND  mode == "probe"     (probe failure)
    # No change — core ratchet property: probe failures don't regress.
```

Two properties of the Case B formulation:

- **Lattice-preserving** — `ρ★_i` remains on the per-sample lattice after the valve fires.
- **Minimum-advance guarantee** — even if `delta_safe` is smaller than `1/num_steps_i`, the `max(g_curr + 1, …)` guard ensures the valve actually advances by one teacher-step. This removes the silent "bump by 0.01 but `round()` returns the same `g`" failure mode.

Contrast with AdaBack: AdaBack in Case C does `ρ_min ← ρ_used`, collapsing the usable interval. MFC retains the information.

### 3.5 Hyperparameters (complete list)

| Hyper-param | Default | Range | Rationale |
|-------------|---------|-------|-----------|
| `p_probe` | 0.25 | 0.1 – 0.4 | ≈ 1 probe every 4 visits; balances exploit-signal stability vs. probe-descent speed |
| `safety_K` | 3 | 2 – 5 | Tolerate occasional noise at the frontier before regressing |
| `delta_safe` | 0.1 | 0.05 – 0.2 | Regression step; same order of magnitude as a single teacher-step fraction for typical `num_steps ≈ 8` |
| `default_rho_star` | 1.0 | — | Cold start with full guidance; first successful exploit at ρ=1 will immediately drop the frontier once a probe lands |
| `ε = 1 / n_rollouts` | derived | — | Success = at least one correct rollout in the group |

**Three free knobs** (`p_probe`, `safety_K`, `delta_safe`), one fewer than AdaBack's 4.

### 3.6 Checkpointing

Serialize `{sample_id → state dict}` + all scalar hyperparameters to `mfc_curriculum_state.json` alongside the trainer checkpoint. Disjoint path from `adaptive_curriculum_state.json`.

---

## 4. Comparison with AdaBack

| Dimension | AdaBack | MFC |
|-----------|---------|-----|
| Implicit objective | `argmax_ρ Var_{p_train(·|ρ)}[r]` (tool-level signal) | `max frac_at_zero` (surrogate of terminal) |
| Per-sample state | `(ρ_min, ρ_max, ρ)` interval | `ρ★` point + safety counter |
| Failure at lower ρ | `ρ_min ← ρ` (regression) | No change (ratchet) |
| Success behavior | `ρ_max ← ρ`, `ρ_min ← 0` | `ρ★ ← min(ρ★, ρ)` |
| ρ sampling | `Uniform(ρ_min, ρ_max)` + forced-zero with prob `p_zero` | `Delta(ρ★)·(1-p_probe) + Uniform(0, ρ★)·p_probe` |
| Compute allocation | Bidirectional interior of the interval | Single direction: at / below frontier |
| Distribution-shift awareness | None | Objective directly favors shift-free training |
| Free hyper-parameters | 4: `τ`, `p_zero`, `default_rho`, `min_step_delta` | 3: `p_probe`, `safety_K`, `delta_safe` |
| Designed for | Generic curriculum data | **Unsolvable-only** regime |

---

## 5. Expected experimental signatures

If MFC's design is correct, at matched compute (step count × batch size × rollouts) we expect:

1. **`mfc/frac_at_zero > adaptive/frac_at_zero`** (primary surrogate metric).
2. **Per-sample `ρ★` trajectory is (nearly) monotone**, whereas AdaBack's `ρ_min / ρ_max` oscillate.
3. **Training reward curve converges faster** — fewer wasted visits re-exploring already-known territory.
4. **Zero-guidance evaluation lifts** on held-out benchmarks (MATH-500, AIME 24/25/26, AMC23, SciBench, GPQA), especially on boundary-expansion metrics (pass@16 / pass@64 on AIME).

Failure signatures and their interpretation:

- High `frac_at_zero` but no benchmark lift → surrogate ≠ terminal on this dataset; discussion section addresses the gap.
- Low `frac_at_zero` growth → probe rate too low or safety valve too loose; ablation candidate.
- Rapid descent followed by safety-valve-triggered regression → `p_probe` too aggressive at early training; consider cooldown schedule (left as future work).

---

## 6. Scope and limitations

- **Scope claim**: MFC is designed for, and evaluated on, the unsolvable-only regime. We do **not** claim MFC dominates AdaBack on generic curriculum datasets; in that regime `p_train ≈ p_test` and AdaBack's objective aligns with ours.
- **Dependency on teacher trace quality**: MFC inherits all unlock-curve assumptions from the teacher-guidance baseline.
- **Safety valve is a heuristic**: a fully Bayesian treatment is left to future work.
- **No cross-sample reweighting**: because our dataset size (≈128) equals the training batch, every sample is visited once per step; cross-sample compute allocation (LILO-style) is not applicable here and is omitted.

---

## 7. Integration into the paper

- **Related-work**: introduce the three-category taxonomy; AdaBack is the per-sample SOTA baseline.
- **Methods**: one section each for (a) distribution-shift motivation, (b) MFC mechanism (two sub-sections per mechanism), (c) comparison table with AdaBack.
- **Experiments**: MFC run replaces the "per-sample adaptive" slot; AdaBack run remains as the direct baseline. Ablations deferred (all three hyper-parameters held at defaults).
- **Analysis**: per-sample `ρ★` trajectory plot (AdaBack vs MFC), `frac_at_zero` vs step, benchmark tables.

---

## 8. Implementation mapping (for developers)

- Core class: `MonotoneFrontierCurriculumState` in [verl/utils/curriculum.py](../verl/utils/curriculum.py). Public API mirrors `PerSampleCurriculumState` (`get_rho`, `update`, `state_dict`, `load_state_dict`, `get_metrics`).
- Config: `mfc_curriculum.*` section in [verl/trainer/config/cgrpo_trainer.yaml](../verl/trainer/config/cgrpo_trainer.yaml).
- Trainer: `curriculum_method = "mfc"` in [verl/trainer/cgrpo_trainer.py](../verl/trainer/cgrpo_trainer.py), reusing the existing `_apply_adaptive_guidance` / `_update_adaptive_state` hooks via a polymorphic `self.curriculum_state` pointer.
- Launcher: [examples/cgrpo_trainer/run_ttn2k_unsolvable_mfc_hint_8xpro6000.sh](../examples/cgrpo_trainer/run_ttn2k_unsolvable_mfc_hint_8xpro6000.sh).

## 9. Manual verification checklist

Before launching the real MFC training run, verify each dispatch branch:

- `data.curriculum_method=none` → no `mfc/*` and no `adaptive/*` metrics; DAPO baseline path unchanged.
- `data.curriculum_method=mixture` → dataset-side `g_level` respected; no adaptive / MFC state constructed.
- `data.curriculum_method=adaptive` → only `adaptive/*` metrics; `adaptive_curriculum_state.json` saved.
- `data.curriculum_method=mfc` → only `mfc/*` metrics; `mfc_curriculum_state.json` saved.
- `guidance_mode=hint` and `guidance_mode=prefix` both function identically under all four curriculum methods.

---

## 10. Metric semantics — faithful convergence indicators

The naive `frac_at_zero` metric as originally logged in AdaBack is a **stochastic / polluted signal**: it counts the fraction of samples whose *last sampled* `rho` landed below `0.05`, which is inflated by (a) `p_zero`-forced zeros and (b) `Uniform(rho_min, rho_max)` draws that happen to fall in the sub-`0.05` tail while `rho_max` is still far above 0. This over-counts convergence.

MFC and AdaBack therefore export additional **faithful** metrics (based on the state itself, not on a single realisation):

| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| `adaptive/frac_rho_max_below_0_05` | `|{i : rho_max_i < 0.05}| / n` | Fraction whose AdaBack interval has collapsed near zero (unbiased). |
| `adaptive/frac_rho_max_below_0_1` | same with `< 0.1` | Looser threshold for the same concept. |
| `mfc/frac_effective_zero` | `|{i : round(rho_star_i * num_steps_i) == 0}| / n` | Fraction whose discrete hint is actually empty — strongest surrogate of "entered no-shift training". |
| `mfc/frac_effective_below_2_steps` | `|{i : g_star_i < 2}| / n` | Samples within one teacher step of independence. |
| `mfc/frac_rho_star_below_0_05` | `|{i : rho_star_i < 0.05}| / n` | Continuous analogue of `adaptive/frac_rho_max_below_0_05` — directly comparable across methods. |
| `mfc/frac_rho_star_at_zero_exact` | `|{i : rho_star_i == 0}| / n` | Strict form; meaningful because discrete probing lets the frontier reach 0 exactly. |
| `mfc/frac_at_zero` | alias of `mfc/frac_rho_star_below_0_05` | Kept for AdaBack cross-comparison. |

The paper's "data entered no-shift training" surrogate should be reported as `mfc/frac_effective_zero` (or `adaptive/frac_rho_max_below_0_05` for AdaBack), **not** the originally-misleading `frac_at_zero` as computed in the first-generation AdaBack code.

A back-of-the-envelope calibration on our production AdaBack run (`mean_rho ≈ 0.15, frac_at_zero ≈ 0.45, p_zero = 0.05`) suggests the "true" converged fraction was closer to `35–40%`, not `45%`. This only strengthens MFC's motivation.

---

## 11. Errata / design refinements relative to the initial spec

Two refinements were adopted after the first smoke-test run surfaced edge cases that the continuous-ρ formulation could not address:

### 11.1 Discrete probe (supersedes continuous `Uniform(0, ρ★)` and the earlier "snap-to-zero" idea)

**Problem.** A probe at `ρ = 0.28` with `num_steps = 10` maps to `round(2.8) = 3` revealed steps, identical to exploit at `ρ★ = 0.30` (also 3 steps). The probe wastes a visit.

**Fix.** Probe in discrete step space: `g_probe ~ Uniform{0, …, g_curr - 1}`, `ρ_used = g_probe / num_steps`. This is the formulation now in §3.3.

**Bonus.** Since `g_probe = 0` is a positive-probability outcome, `ρ★ = 0` is reachable in finite time — eliminating the need for a separate snap-to-zero rule that would have been required under continuous probing.

### 11.2 Safety valve is lattice-snapping with a minimum-advance guard

**Problem.** A continuous bump `ρ★ ← ρ★ + delta_safe` with `delta_safe = 0.05` and `num_steps = 10` goes from `ρ★ = 0.3 (g = 3)` to `0.35 (round → 4)`. But with `delta_safe = 0.01` the bump goes `0.3 → 0.31 (round → 3)` — the same discrete step as before, so the valve silently no-ops.

**Fix.** The post-bump `ρ★` is snapped to the lattice `g / num_steps` and a `max(g_curr + 1, …)` guard ensures at least one discrete step of advance. See §3.4, Case B.

### 11.3 FP precision

Every `ρ → g` conversion uses `round(ρ * num_steps + 1e-9)` to avoid banker's-rounding flips at half-step boundaries (e.g. `3/7 * 7 = 2.9999…95` would otherwise round down to 2). A dedicated unit test exhaustively verifies the lattice round-trip for all `(g, num_steps)` with `num_steps ∈ [2, 30)`.
