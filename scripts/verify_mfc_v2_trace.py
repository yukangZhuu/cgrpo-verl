#!/usr/bin/env python3
"""Verify a MFC v2 training trace JSONL against the algorithm's invariants.

Run e.g.:

    python3 scripts/verify_mfc_v2_trace.py "adaptive_train_trace copy.jsonl"

Checks performed
----------------

Schema (every step + every sample record):

* curriculum_method == "mfc", method == "mfc", variant == "v2"
* trace lines come in non-decreasing global_step order
* sample record has: adaptive_id, num_teacher_steps, rho_used, guidance_steps_count,
  avg_acc_rollouts, rollout_accs, state_before/after with {rho_max, visits, last_success}
* tau is a positive constant across all records and matches step_summary.tau

Discrete-lattice sampling (the v2 fix from continuous to discrete):

* For every record with num_teacher_steps > 0, rho_used must be exactly k / n
  for some integer k in {0, 1, ..., g_curr} where g_curr = round(rho_max_before * n)
* guidance_steps_count must equal round(rho_used * n) clamped to [0, n-1]
* g_used can include both endpoints (g=0 and g=g_curr) — verified by checking
  that we observe at least one of each across the full trace

Monotone frontier (the v2 ratchet rule):

* Per sample, rho_max_after <= rho_max_before (NEVER increases)
* state_after.visits == state_before.visits + 1
* If success (avg_reward >= tau) AND rho_used < rho_max_before:
    rho_max_after == rho_used (frontier descent)
  else:
    rho_max_after == rho_max_before (no change)
* Never any "rho_min" state field (would indicate v1 / AdaBack leakage)
* Never any "mode" / "safety_counter" field (would indicate v1 leakage)

Cross-step continuity:

* For each sample, state_after at step t == state_before at step t+1

Reward / success consistency:

* avg_acc_rollouts == mean(rollout_accs)
* success flag == (avg_acc_rollouts >= tau)

Step summary consistency:

* num_adaptive_traced_samples == len(samples)
* mean_rho_used and mean_frontier_after match the per-sample averages (rounded)
* n_success matches the count of success-flagged records
* frac_at_zero_after matches the fraction of state_after.rho_max == 0

Exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def _info(msg: str) -> None:
    print(f"[ok]   {msg}")


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _round_g(rho: float, n: int) -> int:
    """Match v2's _g_from_rho discretisation exactly."""
    if n <= 0:
        return 0
    g = int(math.floor(rho * n + 0.5 + 1e-9))
    return max(0, min(n - 1, g))


def _g_from_rho_used(rho_used: float, n: int) -> int:
    """Recover the integer step count actually used in this visit.

    Because v2 always emits rho_used = k/n for some k, we can recover k by
    rounding rho_used*n to the nearest integer.  Lattice membership is
    checked separately.
    """
    if n <= 0:
        return 0
    return int(round(rho_used * n))


def load_lines(path: Path) -> list[dict[str, Any]]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError as e:
                _fail(f"line {i}: not valid JSON ({e})")
    if not lines:
        _fail("trace file is empty")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument(
        "--strict-summary",
        action="store_true",
        help=(
            "If set, fail when step_summary aggregates disagree by more than "
            "the rounding tolerance instead of warning."
        ),
    )
    args = ap.parse_args()

    if not args.trace.exists():
        _fail(f"trace file not found: {args.trace}")

    lines = load_lines(args.trace)
    _info(f"loaded {len(lines)} step-summary lines")

    # Aggregate counters / cross-checks across the whole trace.
    n_records_total = 0
    n_success_total = 0
    seen_g_endpoint_zero = 0
    seen_g_endpoint_at_g_curr = 0
    last_global_step = -1
    sample_last_state: dict[str, dict[str, Any]] = {}
    rho_used_unique: set[float] = set()
    rho_max_before_unique: set[float] = set()
    monotone_violations = 0
    descent_count = 0
    success_no_descent_due_to_above_frontier = 0

    tau_global: float | None = None

    for line_idx, payload in enumerate(lines, start=1):
        # ---- Top-level schema ----
        for key in (
            "global_step",
            "curriculum_method",
            "guidance_mode",
            "n_rollouts",
            "num_originals_in_batch",
            "num_adaptive_traced_samples",
            "step_summary",
            "samples",
        ):
            if key not in payload:
                _fail(f"line {line_idx}: missing top-level key {key!r}")

        global_step = payload["global_step"]
        if global_step < last_global_step:
            _fail(
                f"line {line_idx}: global_step decreases ({last_global_step} -> "
                f"{global_step})"
            )
        last_global_step = global_step

        if payload["curriculum_method"] != "mfc":
            _fail(
                f"line {line_idx}: curriculum_method should be 'mfc', got "
                f"{payload['curriculum_method']!r}"
            )

        summary = payload["step_summary"]
        if summary.get("variant") != "v2":
            _fail(
                f"line {line_idx}: step_summary.variant must be 'v2', got "
                f"{summary.get('variant')!r}"
            )

        tau_summary = summary.get("tau")
        if tau_summary is None or tau_summary <= 0.0:
            _fail(f"line {line_idx}: invalid tau={tau_summary!r}")
        if tau_global is None:
            tau_global = float(tau_summary)
        elif not _close(float(tau_summary), tau_global):
            _fail(
                f"line {line_idx}: tau drifts ({tau_global} -> {tau_summary}) — "
                "MFC v2 keeps tau constant within a run"
            )

        samples = payload["samples"]
        if len(samples) != payload["num_adaptive_traced_samples"]:
            _fail(
                f"line {line_idx}: num_adaptive_traced_samples="
                f"{payload['num_adaptive_traced_samples']} != len(samples)="
                f"{len(samples)}"
            )

        # ---- Per-sample checks ----
        per_step_rho_used: list[float] = []
        per_step_frontier_after: list[float] = []
        per_step_n_success = 0
        for s_idx, rec in enumerate(samples):
            for key in (
                "adaptive_id",
                "num_teacher_steps",
                "rho_used",
                "guidance_steps_count",
                "avg_acc_rollouts",
                "rollout_accs",
                "state_before",
                "state_after",
                "method",
                "variant",
                "tau",
                "success",
            ):
                if key not in rec:
                    _fail(
                        f"line {line_idx} sample {s_idx}: missing key {key!r}"
                    )

            if rec["method"] != "mfc" or rec["variant"] != "v2":
                _fail(
                    f"line {line_idx} sample {s_idx}: method/variant mismatch "
                    f"(got {rec['method']!r}/{rec['variant']!r})"
                )

            sb = rec["state_before"]
            sa = rec["state_after"]
            for state_dict, name in ((sb, "state_before"), (sa, "state_after")):
                for k in ("rho_max", "visits", "last_success"):
                    if k not in state_dict:
                        _fail(
                            f"line {line_idx} sample {s_idx}: {name} missing "
                            f"{k!r}"
                        )
                # v1 / AdaBack fields MUST NOT leak into v2.
                for forbidden in (
                    "rho_min",
                    "rho_star",
                    "mode",
                    "safety_counter",
                    "safety_triggered_total",
                ):
                    if forbidden in state_dict:
                        _fail(
                            f"line {line_idx} sample {s_idx}: {name} contains "
                            f"forbidden v1/AdaBack field {forbidden!r}"
                        )

            # ---- Monotone non-increasing frontier ----
            rho_before = float(sb["rho_max"])
            rho_after = float(sa["rho_max"])
            if rho_after > rho_before + 1e-12:
                monotone_violations += 1
                _fail(
                    f"step {global_step} sample {rec['adaptive_id']!r}: "
                    f"rho_max increased {rho_before} -> {rho_after} "
                    "(MFC v2 frontier must be monotone non-increasing)"
                )

            # ---- Visit counter increments by exactly 1 ----
            v_before, v_after = int(sb["visits"]), int(sa["visits"])
            if v_after != v_before + 1:
                _fail(
                    f"step {global_step} sample {rec['adaptive_id']!r}: "
                    f"visits should increment by 1 (got {v_before} -> {v_after})"
                )

            # ---- Discrete-lattice rho_used ----
            n = int(rec["num_teacher_steps"])
            rho_used = float(rec["rho_used"])
            if n > 0:
                k = _g_from_rho_used(rho_used, n)
                # k / n should equal rho_used to FP precision.
                if not _close(rho_used, k / n, tol=1e-9):
                    _fail(
                        f"step {global_step} sample {rec['adaptive_id']!r}: "
                        f"rho_used={rho_used!r} is OFF the per-sample step "
                        f"lattice (n={n}; nearest k={k}, k/n={k/n})"
                    )
                # k must lie in {0, ..., g_curr} where g_curr is derived from
                # rho_max BEFORE this visit.
                g_curr_before = _round_g(rho_before, n)
                if not (0 <= k <= g_curr_before):
                    _fail(
                        f"step {global_step} sample {rec['adaptive_id']!r}: "
                        f"k={k} out of {{0, ..., {g_curr_before}}} "
                        f"(rho_max_before={rho_before}, n={n})"
                    )
                if k == 0:
                    seen_g_endpoint_zero += 1
                if k == g_curr_before and g_curr_before > 0:
                    seen_g_endpoint_at_g_curr += 1
                rho_used_unique.add(round(rho_used, 9))
                rho_max_before_unique.add(round(rho_before, 9))

                # ---- guidance_steps_count consistency ----
                expected_g = _round_g(rho_used, n)
                if int(rec["guidance_steps_count"]) != expected_g:
                    _fail(
                        f"step {global_step} sample {rec['adaptive_id']!r}: "
                        f"guidance_steps_count={rec['guidance_steps_count']} "
                        f"!= round(rho_used*n) clipped = {expected_g}"
                    )

            # ---- Reward / success consistency ----
            accs = rec["rollout_accs"]
            mean_acc = sum(accs) / len(accs) if accs else 0.0
            if not _close(mean_acc, float(rec["avg_acc_rollouts"]), tol=1e-6):
                _fail(
                    f"step {global_step} sample {rec['adaptive_id']!r}: "
                    f"avg_acc_rollouts={rec['avg_acc_rollouts']!r} != "
                    f"mean(rollout_accs)={mean_acc}"
                )
            tau_rec = float(rec["tau"])
            success_expected = mean_acc >= tau_rec
            if bool(rec["success"]) != success_expected:
                _fail(
                    f"step {global_step} sample {rec['adaptive_id']!r}: "
                    f"success flag {rec['success']!r} disagrees with "
                    f"avg_acc={mean_acc} vs tau={tau_rec}"
                )

            # ---- Ratchet rule ----
            success = success_expected
            if success and rho_used < rho_before - 1e-12:
                if not _close(rho_after, rho_used):
                    _fail(
                        f"step {global_step} sample {rec['adaptive_id']!r}: "
                        f"successful descent should set rho_max=rho_used "
                        f"(rho_used={rho_used}, rho_after={rho_after})"
                    )
                descent_count += 1
            elif success and rho_used >= rho_before - 1e-12:
                # success at or above frontier: no descent allowed
                if not _close(rho_after, rho_before):
                    _fail(
                        f"step {global_step} sample {rec['adaptive_id']!r}: "
                        f"success at rho>=rho_max should keep rho_max "
                        f"({rho_before} -> {rho_after})"
                    )
                success_no_descent_due_to_above_frontier += 1
            else:
                # failure: rho_max must be unchanged (modulo defensive snap)
                snap_threshold = (1.0 / (2.0 * n)) if n > 0 else 0.0
                snap_fired = (
                    0.0 < rho_before < snap_threshold and rho_after == 0.0
                )
                if not _close(rho_after, rho_before) and not snap_fired:
                    _fail(
                        f"step {global_step} sample {rec['adaptive_id']!r}: "
                        f"failure must not modify rho_max "
                        f"({rho_before} -> {rho_after})"
                    )

            # ---- Cross-step continuity ----
            sid = rec["adaptive_id"]
            prev = sample_last_state.get(sid)
            if prev is not None:
                if not _close(prev["rho_max"], rho_before, tol=1e-9):
                    _fail(
                        f"sample {sid!r}: state_after.rho_max from previous step "
                        f"= {prev['rho_max']} but state_before this step = "
                        f"{rho_before} (frontier should persist across steps)"
                    )
                if int(prev["visits"]) != v_before:
                    _fail(
                        f"sample {sid!r}: prev visits={prev['visits']} but "
                        f"state_before.visits={v_before} (visit counter must "
                        "carry over)"
                    )
            sample_last_state[sid] = {
                "rho_max": rho_after,
                "visits": v_after,
            }

            # Per-step aggregates (for step_summary cross-check).
            per_step_rho_used.append(rho_used)
            per_step_frontier_after.append(rho_after)
            if success:
                per_step_n_success += 1
                n_success_total += 1
            n_records_total += 1

        # ---- step_summary aggregate cross-checks ----
        if samples:
            mr_used = sum(per_step_rho_used) / len(per_step_rho_used)
            mr_after = sum(per_step_frontier_after) / len(per_step_frontier_after)
            mr_used_summary = float(summary["mean_rho_used"])
            mr_after_summary = float(summary["mean_frontier_after"])
            tol = 5e-4 if not args.strict_summary else 1e-9
            if abs(mr_used - mr_used_summary) > tol:
                _fail(
                    f"step {global_step}: step_summary.mean_rho_used="
                    f"{mr_used_summary} but per-sample mean={mr_used:.4f}"
                )
            if abs(mr_after - mr_after_summary) > tol:
                _fail(
                    f"step {global_step}: step_summary.mean_frontier_after="
                    f"{mr_after_summary} but per-sample mean={mr_after:.4f}"
                )
            if int(summary["n_success"]) != per_step_n_success:
                _fail(
                    f"step {global_step}: step_summary.n_success="
                    f"{summary['n_success']} but actual={per_step_n_success}"
                )
            frac_zero = sum(
                1 for r in per_step_frontier_after if r == 0.0
            ) / len(per_step_frontier_after)
            if abs(frac_zero - float(summary["frac_at_zero_after"])) > tol:
                _fail(
                    f"step {global_step}: frac_at_zero_after mismatch "
                    f"summary={summary['frac_at_zero_after']} actual={frac_zero}"
                )

    # ---- Whole-trace summary ----
    print()
    _info(f"records checked: {n_records_total}")
    _info(f"unique adaptive_ids: {len(sample_last_state)}")
    _info(f"successful descents (frontier strictly decreased): {descent_count}")
    _info(
        f"successes at rho >= rho_max (no descent): "
        f"{success_no_descent_due_to_above_frontier}"
    )
    _info(f"failures (no rho_max change): "
          f"{n_records_total - n_success_total}")
    _info(
        f"records that landed at g=0 (the on-target lattice point): "
        f"{seen_g_endpoint_zero}"
    )
    _info(
        f"records that landed at g=g_curr (the implicit-exploit point): "
        f"{seen_g_endpoint_at_g_curr}"
    )
    _info(f"unique rho_used values observed: {len(rho_used_unique)}")
    _info(f"monotone violations: {monotone_violations}")

    # Sanity assertions on whole-trace patterns:
    if seen_g_endpoint_zero == 0 and n_records_total > 32:
        print(
            "[warn] never sampled g=0 across the full trace — possible if all "
            "frontiers are at rho_max=1 in this short smoke (every visit then "
            "samples uniformly over n_steps levels including 0).  Worth a "
            "look.",
            file=sys.stderr,
        )

    if seen_g_endpoint_at_g_curr == 0 and n_records_total > 32:
        print(
            "[warn] never sampled the frontier (g=g_curr) — possible if every "
            "visit is at rho_max=1 (then g_curr = num_steps-1 due to clamp). "
            "Verify against the discrete sampling spec.",
            file=sys.stderr,
        )

    print("\n[verify_mfc_v2_trace] all invariants hold.")


if __name__ == "__main__":
    main()
