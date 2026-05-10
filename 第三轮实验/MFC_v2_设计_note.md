# MFC v2 — Design Note

> Companion to [MFC_算法设计_spec.md](MFC_算法设计_spec.md). Read that first for v1 and the
> distribution-shift motivation. This note documents only the v2 redesign and
> the empirical findings on v1 that drove it.

## 1. Why v2

The v1 1200-step run on `ttn_unsolvable_pass64_n128` produced these signals:

| Symptom | Reading |
|---------|---------|
| `safety_trigger_total` grew **linearly** to ~3000 | Persistent oscillation, not a transient. ~30-40 "ceiling" samples each triggering ~75-100 safety bumps over the run. |
| `mean_rho_used` plateaued at ~0.16 after step 500 | Frontier descent stalled; further compute fed the cycle, not progress. |
| `probe_fraction` dropped from 0.30 to 0.14, while `frac_at_zero ≈ 0.55` | Confirms most "rho_star=0" samples no longer probe. The frac_at_zero metric mixes (i) genuinely converged samples with (ii) cycle samples passing through ρ★=0 in their oscillation. |
| pass@1 vs AdaBack tied; pass@k boundary +3pp | v1's surrogate gain (frac_at_zero +10pp) only weakly transferred to terminal metrics. |

Mechanism: v1's `epsilon = 1/n` ratchet trips on a single lucky rollout (~5-30% noise floor for unsolvables). For ceiling samples (those whose true success rate at low ρ is below threshold), this produces **false convergence to ρ★=0**, where exploit fails repeatedly → safety valve fires → ρ★ bumps up by `delta_safe` → next probe gets lucky again → ratchets back down. The resulting limit cycle wastes compute and pollutes the surrogate metric.

## 2. The v2 redesign in one line

> **MFC v2 = AdaBack with the `ρ_min` state removed, projected onto the per-sample step lattice.**

Concretely:

| Aspect | AdaBack | MFC v1 | MFC v2 |
|--------|---------|--------|--------|
| State | `(ρ_min, ρ_max)` interval + p_zero | `ρ★` + safety_counter + mode + 4 more | `ρ_max` |
| Sampling space | continuous | discrete step lattice | discrete step lattice |
| Visit | `ρ ~ U(ρ_min, ρ_max)` | 1−p_probe at `ρ★`; p_probe over `{0,…,g_curr−1}/n` | **`g_used ~ U{0, 1, …, g_curr}`, `ρ_used = g_used/n`** |
| Update on success | `ρ_max ← ρ`, `ρ_min ← 0` | `ρ★ ← ρ` (if probe), exploit success keeps `ρ★` | `ρ_max ← ρ_used` (if `ρ_used < ρ_max`) |
| Update on failure | `ρ_min ← ρ` | exploit: counter++, safety valve; probe: noop | nothing |
| Threshold | `τ` (default 0.4) | `ε = 1/n` (≈ 0.125 for n=8) | `τ` (default 0.5) |
| Hyperparameters | 4 | 3 | **1** |

The visit distribution deserves its own paragraph. v2 samples on the **per-sample discrete lattice** `{k / num_steps : k ∈ {0, 1, …, g_curr}}` where `g_curr = round(ρ_max · num_steps)`. Both endpoints are included and every level gets equal probability `1 / (g_curr + 1)`.

### Why discrete (and why include both endpoints)

A continuous `ρ ~ Uniform(0, ρ_max)` looks simpler at first glance, but `compute_guidance_steps` rounds `ρ · num_steps` to the nearest integer. Under that rounding, `g = 0` and `g = g_curr` each get only a half-width region of probability, while every interior `g` gets a full-width region. For `ρ_max = 0.5`, `num_steps = 10` (so `g_curr = 5`):

| `g`      | `ρ` interval mapped here | P(g) |
|----------|--------------------------|------|
| 0        | [0, 0.05)                | 0.10 |
| 1, 2, 3, 4 | full lattice cell each  | 0.20 each |
| 5 (= ρ_max) | [0.45, 0.5]            | 0.10 |

In other words, **the two points the algorithm cares about most are the two it under-samples**:
* `g = 0` is the only no-shift training position (the surrogate objective).
* `g = g_curr` is the implicit "exploit at frontier" — the most stable-gradient anchor.

Discrete-uniform sampling fixes this. Every reachable level — including both endpoints — gets `1 / (g_curr + 1)` probability. As the frontier descends, the relative weight of `g = 0` automatically rises (at `g_curr = 1` the split is 50/50 between exploit and on-target training; at `g_curr = 0` it collapses to deterministic on-target).

This also makes `ρ_max = 0` exactly reachable from any `g_curr ≥ 1` in finite expected time without relying on the snap-to-zero rule, which is now purely defensive (it only fires for state mutated externally — e.g., loaded from an old continuous-mode checkpoint).

## 3. Why this addresses v1's failure modes

| v1 problem | v2 fix |
|------------|--------|
| ε = 1/n trips on noise → false convergence | τ = 0.5: noise floor drops to ~5%, only real learning ratchets |
| Safety valve creates limit cycle | Removed entirely. Without false ratchets, no need to compensate. |
| `frac_at_zero` is contaminated by cycles | Strict monotone non-increasing ρ_max → samples at `ρ_max=0` are permanently there |
| 75% exploit at ρ★ when stuck = shifted-distribution gradient pollution | Uniform sampling: stuck samples get half their visits at very low ρ where they fail silently (zero gradient instead of off-target gradient) |

## 4. Why v2 ≠ "AdaBack with smaller τ"

A reasonable challenge: doesn't lowering AdaBack's τ achieve the same effect (more aggressive descent)?

No, because **τ is symmetric**: it controls both the success ratchet and the failure ratchet. Lowering τ in AdaBack:

- Makes descent easier on success ✓
- **Also makes `ρ_min` raise faster on failure** ✗ — stronger lockout, worse for unsolvable-only data

The structural fix is removing `ρ_min` itself, not tuning τ. v2 keeps the descent threshold strict (τ=0.5) precisely because there's no asymmetric coupling to worry about.

## 5. The epistemic argument

This is the framing for the paper:

> **AdaBack treats failure as strong evidence about ρ-solvability.** That is appropriate when the model is fixed and ρ-difficulty is the only variable. **MFC v2 treats failure as weak evidence**, because in unsolvable-only training the model itself is changing — yesterday's "this ρ doesn't work" may be today's "this ρ works". Only success is committed; failure is forgotten.

This asymmetric epistemic stance is what justifies removing `ρ_min`. v2 is less an algorithmic change than a different prior on what failure means.

## 6. Predicted v2 vs v1 metric signatures

Based on the reasoning above:

| Metric | v1 (observed) | v2 (predicted) | Mechanism |
|--------|---------------|----------------|-----------|
| `safety_trigger_total` | linear → 3000 | **N/A (always 0)** | by construction |
| `frac_at_zero` (displayed) | 0.55 | **0.40-0.55** | false positives gone, but uniform pushes more compute below ρ_max so genuine converged count similar |
| `frac_at_zero` (filtered: ρ_max=0 AND avg_reward ≥ τ) | ~0.30-0.40 | **higher** | v2's converged samples are real |
| `mean_rho_used` | 0.16 plateau | **0.10-0.13 plateau** | discrete uniform (mean ≈ ρ_max / 2) vs 75% exploit at ρ_max |
| pass@1 | unchanged from AdaBack | **= or slight ↑** | shifted-gradient pollution removed |
| pass@k boundary | +3pp over AdaBack | **+3 to +6pp** | cleaner gradient → better generalization |

If v2 lifts pass@1 noticeably → the v1 limit-cycle hypothesis is confirmed and the simpler algorithm wins.

If v2 ties on pass@1 → ceiling effect dominates; v2 still wins on simplicity / hyperparameter count, and the cleaner `frac_at_zero` is itself a paper-grade contribution (a true measurement of frontier-pushing capacity).

## 7. Implementation summary

- New class: [MonotoneFrontierCurriculumStateV2](../verl/utils/curriculum.py) — added alongside v1, no v1 logic changed.
- Config switch: `mfc_curriculum.variant ∈ {v1, v2}`, default `v1` so existing launchers byte-identical.
- Trainer dispatches on `variant`; checkpoint files are disjoint (`mfc_curriculum_state.json` for v1, `mfc_curriculum_state_v2.json` for v2) so cross-variant resume cannot corrupt state.
- New launchers: [run_ttn2k_unsolvable_mfc_v2_hint_8xpro6000.sh](../examples/cgrpo_trainer/run_ttn2k_unsolvable_mfc_v2_hint_8xpro6000.sh) and [run_ttn2k_mfc_v2_trace_smoke.sh](../examples/cgrpo_trainer/run_ttn2k_mfc_v2_trace_smoke.sh). All training knobs identical to the v1 / AdaBack launchers — only `mfc_curriculum.*`, `experiment_name`, `default_local_dir` differ.

## 8. Cross-method wandb metrics

Paper figures need apples-to-apples overlays of AdaBack / v1 / v2. To support this without renaming any existing keys, all three methods now emit a unified `curriculum/*` namespace **in addition to** their method-private keys:

| Key | Definition (per-method "frontier" interpretation) |
|-----|---------------------------------------------------|
| `curriculum/method` | `"adaptive"` / `"mfc_v1"` / `"mfc_v2"` |
| `curriculum/mean_frontier` | mean of frontier values: `ρ_max` (AdaBack), `ρ★` (v1), `ρ_max` (v2) |
| `curriculum/median_frontier` | median of the same |
| `curriculum/frac_at_zero_strict` | fraction with frontier `≤ 0` |
| `curriculum/frac_at_zero_loose` | fraction with frontier `< 0.05` |
| `curriculum/frac_below_0_1` | fraction with frontier `< 0.1` |
| `curriculum/frac_effective_zero` | fraction whose `round(ρ * num_steps) == 0` (most faithful "no-shift training" indicator) |
| `curriculum/mean_rho_used` | mean of last-visit ρ across samples |
| `curriculum/mean_visits` | mean visit count per sample |
| `curriculum/num_tracked` | number of tracked samples |

These keys are **bit-identical in name** across all three methods (a unit test in `test_mfc_v2_curriculum.py::test_unified_metric_keys_are_identical_across_methods` enforces this).

For paper figures: plot any `curriculum/*` key with three lines from three runs and the comparison is fair by construction.

## 9. Reading list for the next iteration

If v2 still doesn't deliver pass@1 lift, the path is no longer algorithmic — it's data-side. The follow-up to consider in priority order:

1. **Multi-seed v2** — confirm the result is stable, not a single-seed artifact.
2. **Larger unsolvable pool** (256 / 512 samples) — test whether the bottleneck was 128-sample memorization, not algorithm efficiency.
3. **Smaller / larger base model** — verify the frontier-pushing advantage scales with the unsolvable fraction.

These are scope expansions, not algorithm rewrites; v2 should be the final word on the per-sample curriculum mechanism for this paper.
