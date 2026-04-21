#!/usr/bin/env python3
"""Build a pure-128 unsolvable training set (no frozen anchors).

Motivation
----------
The original adaptive training dataset under
``data/ttn2k/final/ttn_unsolvable_pass64_max2600_n100/dataset_adaptive.jsonl``
is composed of 100 unsolvable problems plus 28 frozen anchors (14 problems
anchored at g_level=0 and g_level=1) so that the batch size of 128 was
exactly filled.

For MFC (and for a fairer AdaBack re-run), the anchors are actively
undesirable:

* ``frozen_g_level=0`` anchors are unsolvable at rho=0, so all rollouts fail
  and GRPO advantage is identically zero — wasted compute.
* ``frozen_g_level=1`` anchors are almost always solved when all teacher
  steps are revealed, so reward variance is ~0 — also wasted, and the
  heavily shifted input distribution leaks shift-only gradient.
* Under MFC's ``frac_at_zero`` surrogate, the frozen g=0 anchors would
  permanently inflate the metric by 14/128 ≈ 11%.

This script regenerates the training set as a pure 128 unsolvable pool:

1. Loads all rows from ``ttn_unsolvable_pass64.jsonl`` (source of truth).
2. Sorts by ``question_chars + steps_chars`` ascending.
3. Keeps the 128 shortest rows (deterministic; no RNG).
4. Writes each row through unchanged except for ``adaptive_id = str(index)``.
5. Emits a ``report.json`` describing provenance + selection.

The "128 shortest" rule is chosen because (i) the source pool has only 136
unsolvable rows total, so we cannot afford a strong length filter, and
(ii) shorter rows minimise the chance of hitting
``data.filter_overlong_prompts`` at training time.  The trainer still enforces
``max_prompt_length`` at load time, so any edge-case truncation is handled
there.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/ttn2k/final/ttn_unsolvable_pass64.jsonl"
DEFAULT_OUT_DIR = ROOT / "data/ttn2k/final/ttn_unsolvable_pass64_n128"


def load_rows(source: Path) -> list[dict]:
    rows: list[dict] = []
    with open(source, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Failed to parse line {i} of {source}: {e}")
    return rows


def total_length(row: dict) -> int:
    q = row.get("question", "") or ""
    steps = row.get("steps", []) or []
    return len(q) + sum(len(s) for s in steps)


def build(
    source: Path,
    out_dir: Path,
    target_count: int,
    verify_unsolvable: bool = True,
) -> None:
    if not source.exists():
        raise SystemExit(f"Source file not found: {source}")

    rows = load_rows(source)
    n_raw = len(rows)
    if n_raw < target_count:
        raise SystemExit(
            f"Source has only {n_raw} rows, need >= {target_count}"
        )

    if verify_unsolvable:
        bad = [r for r in rows if r.get("pass_rate", 0.0) != 0.0]
        if bad:
            raise SystemExit(
                f"Source contains {len(bad)} rows with pass_rate != 0; "
                "refusing to build an 'unsolvable' dataset from mixed data."
            )

    # Stable sort: primary key is total_length, secondary is index so output
    # is deterministic across machines / Python versions.
    rows_sorted = sorted(rows, key=lambda r: (total_length(r), r.get("index", 0)))

    kept = rows_sorted[:target_count]
    dropped = rows_sorted[target_count:]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            payload = dict(r)  # shallow copy; we do not mutate the original
            payload["adaptive_id"] = str(r.get("index"))
            # Pure unsolvable rows MUST NOT carry frozen_g_level / g_level /
            # guidance_steps fields — the dataset loader treats their absence
            # as "let the trainer sample rho via MFC/AdaBack dynamic logic".
            for k in ("frozen_g_level", "g_level", "guidance_steps",
                      "guidance_steps_count", "student_steps_count"):
                payload.pop(k, None)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    report_path = out_dir / "report.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "source_file": str(source.relative_to(ROOT)),
        "output_file": str(out_path.relative_to(ROOT)),
        "selection_strategy": "sort by (question_chars + steps_chars, index) asc; take shortest N",
        "target_count": target_count,
        "counts": {
            "raw_source_rows": n_raw,
            "kept": len(kept),
            "dropped": len(dropped),
        },
        "length_summary_kept": {
            "min": total_length(kept[0]) if kept else 0,
            "max": total_length(kept[-1]) if kept else 0,
            "median": total_length(kept[len(kept) // 2]) if kept else 0,
        },
        "length_summary_dropped": [
            {
                "dataset_index": r.get("index"),
                "question_plus_steps_chars": total_length(r),
            }
            for r in dropped
        ],
        "has_anchors": False,
        "notes": (
            "Pure unsolvable pool: every row has pass_rate=0 under the scorer "
            "model. No frozen_g_level anchors. adaptive_id = str(index)."
        ),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {len(kept)} rows -> {out_path}")
    print(f"[ok] wrote report -> {report_path}")
    print(
        f"     kept length range: "
        f"[{report['length_summary_kept']['min']}, "
        f"{report['length_summary_kept']['max']}] chars "
        f"(median {report['length_summary_kept']['median']})"
    )
    print(f"     dropped {len(dropped)} row(s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--target-count", type=int, default=128)
    p.add_argument(
        "--skip-unsolvable-check",
        action="store_true",
        help="do not require pass_rate=0 on every source row",
    )
    args = p.parse_args()
    build(
        source=args.source,
        out_dir=args.out_dir,
        target_count=args.target_count,
        verify_unsolvable=not args.skip_unsolvable_check,
    )


if __name__ == "__main__":
    main()
