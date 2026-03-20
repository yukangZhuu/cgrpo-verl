"""
Answer Verification Script for Teacher Traces Dataset.

Verifies that teacher_answer and ground_truth are mathematically equivalent
using the same verification pipeline as the training reward manager.
Identifies potential false positives and problematic samples.

Usage:
    conda activate cg
    python verify_answers.py
"""

import json
import re
import sys
import os
from collections import Counter

from math_verify import (
    parse, verify,
    LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "teacher_traces.jsonl")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "verify_report.jsonl")


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


def verify_answer_full(ground_truth: str, model_answer: str) -> tuple[bool, str]:
    """
    Replicate the exact verification pipeline from CurriculumGRPORewardManager.
    Returns (is_match, method_used).
    """
    if is_empty(ground_truth) or is_empty(model_answer):
        return False, "empty"

    gt_clean = re.sub(r'\s+', '', ground_truth.strip())
    ma_clean = re.sub(r'\s+', '', model_answer.strip())
    if gt_clean == ma_clean:
        return True, "exact_match"

    if is_number(ground_truth) and is_number(model_answer):
        if compare_numbers(ground_truth, model_answer):
            return True, "numeric_match"

    try:
        parsed_gt = parse(ground_truth, extraction_config=[
            LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()
        ])
        parsed_ma = parse(model_answer, extraction_config=[
            LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()
        ])
        if parsed_gt and parsed_ma:
            if verify(parsed_gt, parsed_ma):
                return True, "math_verify_full"
    except Exception:
        pass

    try:
        mcq_answers = ['A', 'B', 'C', 'D', 'E']
        config = StringExtractionConfig(strings=tuple(mcq_answers))
        parsed_gt = parse(ground_truth, extraction_config=[config])
        parsed_ma = parse(model_answer, extraction_config=[config])
        if parsed_gt and parsed_ma:
            if verify(parsed_gt, parsed_ma):
                return True, "mcq_match"
    except Exception:
        pass

    try:
        parsed_gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
        parsed_ma = parse(model_answer, extraction_config=[ExprExtractionConfig()])
        if parsed_gt and parsed_ma:
            if verify(parsed_gt, parsed_ma):
                return True, "expr_match"
    except Exception:
        pass

    try:
        wrapped_gt = wrap_latex(ground_truth)
        wrapped_ma = wrap_latex(model_answer)
        parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
        parsed_ma = parse(wrapped_ma, extraction_config=[LatexExtractionConfig()])
        if parsed_gt and parsed_ma:
            if verify(parsed_gt, parsed_ma):
                return True, "latex_wrapped_match"
    except Exception:
        pass

    try:
        numbers_gt = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
        numbers_ma = re.findall(r'[-+]?\d*\.?\d+', model_answer)
        if numbers_gt and numbers_ma:
            if numbers_gt == numbers_ma:
                return True, "regex_number_match"
    except Exception:
        pass

    return False, "no_match"


def main():
    print(f"Loading data from {DATA_PATH}")

    results = {
        "total": 0,
        "format_match": 0,
        "format_mismatch": 0,
        "verified_match": 0,
        "verified_no_match": 0,
        "method_counts": Counter(),
    }

    problems = []
    mismatched_not_verified = []
    potential_false_positives = []

    with open(DATA_PATH, 'r') as f:
        for line_num, line in enumerate(f):
            d = json.loads(line)
            ta = str(d.get('teacher_answer', '')).strip()
            gt = str(d.get('ground_truth', '')).strip()
            idx = d.get('index', line_num)
            results["total"] += 1

            format_match = (ta == gt)
            if format_match:
                results["format_match"] += 1
            else:
                results["format_mismatch"] += 1

            is_match, method = verify_answer_full(gt, ta)
            results["method_counts"][method] += 1

            if is_match:
                results["verified_match"] += 1
            else:
                results["verified_no_match"] += 1
                mismatched_not_verified.append({
                    "index": idx,
                    "teacher_answer": ta,
                    "ground_truth": gt,
                    "method": method,
                    "question_preview": d.get('question', '')[:120],
                })

            # Detect potential false positives from the regex_number_match fallback:
            # this is the most dangerous match method as it could match irrelevant numbers
            if is_match and method == "regex_number_match" and not format_match:
                potential_false_positives.append({
                    "index": idx,
                    "teacher_answer": ta,
                    "ground_truth": gt,
                    "method": method,
                })

            if line_num % 2000 == 0:
                print(f"  Processed {line_num + 1}...")

    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"Total samples: {results['total']}")
    print(f"Format exact match (ta == gt): {results['format_match']} ({results['format_match']/results['total']*100:.1f}%)")
    print(f"Format mismatch: {results['format_mismatch']} ({results['format_mismatch']/results['total']*100:.1f}%)")
    print(f"Verified equivalent: {results['verified_match']} ({results['verified_match']/results['total']*100:.1f}%)")
    print(f"NOT verified equivalent: {results['verified_no_match']} ({results['verified_no_match']/results['total']*100:.1f}%)")
    print()
    print("Verification method breakdown:")
    for method, count in results["method_counts"].most_common():
        print(f"  {method}: {count} ({count/results['total']*100:.1f}%)")
    print()
    print(f"Potential false positives (regex_number_match on format-mismatched): {len(potential_false_positives)}")
    if potential_false_positives:
        for item in potential_false_positives[:10]:
            print(f"  idx={item['index']}: ta='{item['teacher_answer'][:60]}' vs gt='{item['ground_truth'][:60]}'")
        if len(potential_false_positives) > 10:
            print(f"  ... and {len(potential_false_positives) - 10} more")
    print()
    print(f"Samples where teacher_answer != ground_truth AND verification FAILED: {len(mismatched_not_verified)}")
    if mismatched_not_verified:
        print("  These are potentially WRONG teacher traces:")
        for item in mismatched_not_verified[:20]:
            print(f"  idx={item['index']}: ta='{item['teacher_answer'][:50]}' vs gt='{item['ground_truth'][:50]}' | q: {item['question_preview'][:60]}...")
        if len(mismatched_not_verified) > 20:
            print(f"  ... and {len(mismatched_not_verified) - 20} more")

    # Save detailed report
    report = {
        "summary": {
            "total": results["total"],
            "format_match": results["format_match"],
            "format_mismatch": results["format_mismatch"],
            "verified_match": results["verified_match"],
            "verified_no_match": results["verified_no_match"],
            "method_counts": dict(results["method_counts"]),
            "potential_false_positive_count": len(potential_false_positives),
        },
        "mismatched_not_verified": mismatched_not_verified,
        "potential_false_positives": potential_false_positives,
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
