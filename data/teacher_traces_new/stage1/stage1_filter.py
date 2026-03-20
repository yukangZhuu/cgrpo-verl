"""
Stage 1: Static Quality & Length Filtering for Teacher Traces Dataset.

Filters the raw teacher_traces.jsonl into a clean candidate pool based on:
1. Basic quality (non-empty fields, no duplicates, correct teacher traces)
2. Steps count range [6, 14]
3. Question character length <= 500
4. Steps total character length <= 2500
5. Step quality (no degenerate step lengths)
6. Answer verification (teacher_answer must be verified equivalent to ground_truth)
7. Multi-part answer exclusion

Input:  teacher_traces.jsonl (16,559 samples)
Output: candidates_stage1.jsonl (filtered candidates)
        stage1_stats.json (filtering statistics)

Usage:
    conda activate cg
    python stage1_filter.py
"""

import json
import os
import re
import sys
from collections import Counter, OrderedDict

# ---- Configuration ----
INPUT_PATH = os.path.join(os.path.dirname(__file__), "teacher_traces.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "candidates_stage1.jsonl")
STATS_PATH = os.path.join(os.path.dirname(__file__), "stage1_stats.json")
REMOVED_PATH = os.path.join(os.path.dirname(__file__), "stage1_removed.jsonl")

MIN_STEPS = 6
MAX_STEPS = 14
MAX_QUESTION_CHARS = 500
MAX_STEPS_TOTAL_CHARS = 2800
MIN_STEP_CHARS = 20
MAX_STEP_RATIO = 5.0  # max step length / median step length


# ---- Answer Verification (replicating reward manager logic) ----
try:
    from math_verify import (
        parse, verify,
        LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig,
    )
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("WARNING: math_verify not available. Using fallback verification only.")


def is_empty(text: str) -> bool:
    return not text or text.strip() == ""


def is_number(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def compare_numbers(num1: str, num2: str) -> bool:
    try:
        return abs(float(num1.strip()) - float(num2.strip())) < 1e-6
    except ValueError:
        return False


def is_latex_wrapped(text: str) -> bool:
    text = text.strip()
    for pattern in [
        r'^\\\[.*\\\]$', r'^\$\$.*\$\$$', r'^\\boxed\{.*\}$',
        r'^\$.*\$', r'^\\\(.*\\\)$', r'^\[.*\]$',
    ]:
        if re.match(pattern, text, re.DOTALL):
            return True
    return False


def wrap_latex(text: str) -> str:
    text = text.strip()
    return text if is_latex_wrapped(text) else f'\\boxed{{{text}}}'


def verify_answer(ground_truth: str, teacher_answer: str) -> tuple[bool, str]:
    """Verify teacher_answer matches ground_truth. Returns (is_match, method)."""
    if is_empty(ground_truth) or is_empty(teacher_answer):
        return False, "empty"

    gt_clean = re.sub(r'\s+', '', ground_truth.strip())
    ta_clean = re.sub(r'\s+', '', teacher_answer.strip())
    if gt_clean == ta_clean:
        return True, "exact_match"

    if is_number(ground_truth) and is_number(teacher_answer):
        if compare_numbers(ground_truth, teacher_answer):
            return True, "numeric_match"

    if MATH_VERIFY_AVAILABLE:
        try:
            parsed_gt = parse(ground_truth, extraction_config=[
                LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()
            ])
            parsed_ta = parse(teacher_answer, extraction_config=[
                LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()
            ])
            if parsed_gt and parsed_ta and verify(parsed_gt, parsed_ta):
                return True, "math_verify_full"
        except Exception:
            pass

        try:
            parsed_gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
            parsed_ta = parse(teacher_answer, extraction_config=[ExprExtractionConfig()])
            if parsed_gt and parsed_ta and verify(parsed_gt, parsed_ta):
                return True, "expr_match"
        except Exception:
            pass

        try:
            wrapped_gt = wrap_latex(ground_truth)
            wrapped_ta = wrap_latex(teacher_answer)
            parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
            parsed_ta = parse(wrapped_ta, extraction_config=[LatexExtractionConfig()])
            if parsed_gt and parsed_ta and verify(parsed_gt, parsed_ta):
                return True, "latex_wrapped_match"
        except Exception:
            pass

    # regex_number_match is intentionally EXCLUDED here as it is too permissive
    # and can produce false positives. Samples that only match via regex will be
    # flagged for review rather than automatically accepted.
    return False, "no_match"


def is_multi_part_answer(teacher_answer: str) -> bool:
    """Detect multi-part answers like '(1) ...; (2) ...'."""
    return bool(re.search(r'\(1\)|\(2\)|\(i\)|\(ii\)|^\s*\(a\)', teacher_answer))


def check_step_quality(steps: list[str]) -> tuple[bool, str]:
    """Check if steps have reasonable quality."""
    if not steps:
        return False, "empty_steps"

    step_lens = [len(s.strip()) for s in steps]

    if any(l < MIN_STEP_CHARS for l in step_lens):
        return False, "step_too_short"

    if len(steps) >= 3:
        sorted_lens = sorted(step_lens)
        median_len = sorted_lens[len(sorted_lens) // 2]
        if median_len > 0 and max(step_lens) > MAX_STEP_RATIO * median_len:
            return False, "step_imbalanced"

    return True, "ok"


# ---- Main Filtering Pipeline ----
def main():
    print(f"Stage 1: Static Quality & Length Filtering")
    print(f"Input: {INPUT_PATH}")
    print(f"Filters: steps[{MIN_STEPS},{MAX_STEPS}], q<={MAX_QUESTION_CHARS}chars, "
          f"steps_total<={MAX_STEPS_TOTAL_CHARS}chars")
    print()

    # Tracking
    total = 0
    questions_seen = {}
    removal_reasons = Counter()
    kept = []
    removed = []
    verify_methods = Counter()

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            d = json.loads(line)
            total += 1
            idx = d.get('index', line_num)
            question = d.get('question', '').strip()
            steps = d.get('steps', [])
            ground_truth = str(d.get('ground_truth', '')).strip()
            teacher_answer = str(d.get('teacher_answer', '')).strip()
            n_steps = len(steps)

            # ---- Filter 1: Basic field presence ----
            if not question or not steps or not ground_truth:
                removal_reasons["missing_fields"] += 1
                removed.append({"index": idx, "reason": "missing_fields"})
                continue

            # ---- Filter 2: Duplicate question ----
            if question in questions_seen:
                removal_reasons["duplicate"] += 1
                removed.append({"index": idx, "reason": "duplicate"})
                continue
            questions_seen[question] = idx

            # ---- Filter 3: Steps count range [6, 14] ----
            if not (MIN_STEPS <= n_steps <= MAX_STEPS):
                removal_reasons["steps_out_of_range"] += 1
                removed.append({"index": idx, "reason": f"steps={n_steps}"})
                continue

            # ---- Filter 4: Question length ----
            if len(question) > MAX_QUESTION_CHARS:
                removal_reasons["question_too_long"] += 1
                removed.append({"index": idx, "reason": f"q_len={len(question)}"})
                continue

            # ---- Filter 5: Steps total length ----
            steps_text = '\n'.join(steps)
            if len(steps_text) > MAX_STEPS_TOTAL_CHARS:
                removal_reasons["steps_too_long"] += 1
                removed.append({"index": idx, "reason": f"steps_len={len(steps_text)}"})
                continue

            # ---- Filter 6: Step quality ----
            quality_ok, quality_reason = check_step_quality(steps)
            if not quality_ok:
                removal_reasons[f"quality:{quality_reason}"] += 1
                removed.append({"index": idx, "reason": quality_reason})
                continue

            # ---- Filter 7: Multi-part answer exclusion ----
            if is_multi_part_answer(teacher_answer):
                removal_reasons["multi_part_answer"] += 1
                removed.append({"index": idx, "reason": "multi_part_answer"})
                continue

            # ---- Filter 8: Answer verification ----
            is_match, method = verify_answer(ground_truth, teacher_answer)
            verify_methods[method] += 1

            if not is_match:
                removal_reasons["answer_not_verified"] += 1
                removed.append({
                    "index": idx,
                    "reason": "answer_not_verified",
                    "teacher_answer": teacher_answer[:100],
                    "ground_truth": ground_truth[:100],
                })
                continue

            # ---- Passed all filters ----
            d["_verify_method"] = method
            d["_num_steps"] = n_steps
            d["_q_char_len"] = len(question)
            d["_steps_total_char_len"] = len(steps_text)
            kept.append(d)

            if (line_num + 1) % 2000 == 0:
                print(f"  Processed {line_num + 1}... (kept {len(kept)} so far)")

    # ---- Write outputs ----
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open(REMOVED_PATH, 'w', encoding='utf-8') as f:
        for d in removed:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # Build steps distribution for kept samples
    steps_dist = Counter(d["_num_steps"] for d in kept)

    stats = OrderedDict([
        ("input_file", INPUT_PATH),
        ("output_file", OUTPUT_PATH),
        ("total_input", total),
        ("total_kept", len(kept)),
        ("total_removed", len(removed)),
        ("keep_rate", f"{len(kept)/total*100:.1f}%"),
        ("filter_config", {
            "min_steps": MIN_STEPS,
            "max_steps": MAX_STEPS,
            "max_question_chars": MAX_QUESTION_CHARS,
            "max_steps_total_chars": MAX_STEPS_TOTAL_CHARS,
            "min_step_chars": MIN_STEP_CHARS,
            "max_step_ratio": MAX_STEP_RATIO,
        }),
        ("removal_breakdown", dict(removal_reasons.most_common())),
        ("verify_method_breakdown", dict(verify_methods.most_common())),
        ("kept_steps_distribution", dict(sorted(steps_dist.items()))),
    ])

    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ---- Print Summary ----
    print()
    print("=" * 60)
    print("STAGE 1 FILTERING SUMMARY")
    print("=" * 60)
    print(f"Input:   {total} samples")
    print(f"Kept:    {len(kept)} samples ({len(kept)/total*100:.1f}%)")
    print(f"Removed: {len(removed)} samples")
    print()
    print("Removal breakdown:")
    for reason, count in removal_reasons.most_common():
        print(f"  {reason}: {count}")
    print()
    print("Verification method for kept samples:")
    for method, count in verify_methods.most_common():
        print(f"  {method}: {count}")
    print()
    print("Steps distribution (kept):")
    for s in range(MIN_STEPS, MAX_STEPS + 1):
        count = steps_dist.get(s, 0)
        bar = '#' * (count // 20)
        print(f"  steps={s:2d}: {count:5d} {bar}")
    print()
    print(f"Output written to: {OUTPUT_PATH}")
    print(f"Stats written to:  {STATS_PATH}")
    print(f"Removed log:       {REMOVED_PATH}")


if __name__ == "__main__":
    main()
