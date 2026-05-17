#!/usr/bin/env python3
"""
Build the SFT parquet dataset for the Uns-128 teacher-trace baseline.

Input:
  - data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset.jsonl  (128 rows, no teacher_answer)
  - data/ttn2k/final/train_2k.jsonl                            (2000 rows, has teacher_answer)

Output:
  - data/ttn2k/final/ttn_unsolvable_pass64_n128/sft_dataset.parquet

Each output row has a single column `messages` containing a 3-turn conversation:
  [system, user, assistant]

The **user** message uses the exact same ρ=0 prompt template as the RL
curriculum pipeline (CurriculumGRPODataset._build_standard_user_content).

The **assistant** message contains the full teacher reasoning + boxed answer
in the same <think>...</think> format the model is trained to produce:
  <think>
  {step_1}
  {step_2}
  ...
  {step_N}
  </think>
  \\boxed{teacher_answer}

enable_thinking is set to False per row so that the Qwen3 chat template
does NOT inject its own <think> tag (we control it explicitly).
"""

import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

N128_PATH = os.path.join(
    PROJECT_ROOT,
    "data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset.jsonl",
)
TRAIN2K_PATH = os.path.join(
    PROJECT_ROOT,
    "data/ttn2k/final/train_2k.jsonl",
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data/ttn2k/final/ttn_unsolvable_pass64_n128/sft_dataset.parquet",
)

SYSTEM_PROMPT = (
    "You are an expert mathematician with strong problem-solving skills. "
    "Think step by step."
)

FORMAT_BLOCK = (
    "Use this format:\n"
    "<think>\n"
    "[Your reasoning process here, showing how YOU would reach the solution]\n"
    "</think>\n"
    "\\boxed{answer}"
)


def build_standard_user_content(question: str) -> str:
    return (
        f"{question}\n"
        f"Please reason step by step to solve this problem.\n"
        f"{FORMAT_BLOCK}"
    )


def build_assistant_content(steps: list[str], teacher_answer: str) -> str:
    steps_text = "\n".join(steps)
    return f"<think>\n{steps_text}\n</think>\n\\boxed{{{teacher_answer}}}"


def main():
    # 1. Load n128 dataset
    n128_rows = []
    with open(N128_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                n128_rows.append(json.loads(line))
    print(f"Loaded {len(n128_rows)} rows from n128 dataset")

    n128_indices = {r["index"] for r in n128_rows}

    # 2. Recover teacher_answer from train_2k
    idx_to_teacher_answer = {}
    with open(TRAIN2K_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["index"] in n128_indices:
                idx_to_teacher_answer[row["index"]] = row["teacher_answer"]

    missing = n128_indices - set(idx_to_teacher_answer.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} n128 indices not found in train_2k: {missing}"
        )
    print(f"Recovered teacher_answer for all {len(idx_to_teacher_answer)} rows")

    # 3. Build messages
    records = []
    for row in n128_rows:
        question = row["question"]
        steps = row["steps"]
        teacher_answer = idx_to_teacher_answer[row["index"]]

        user_content = build_standard_user_content(question)
        assistant_content = build_assistant_content(steps, teacher_answer)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        records.append({
            "messages": messages,
            "enable_thinking": False,
        })

    # 4. Write parquet
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")

    # 5. Sanity check: print first sample
    sample = records[0]
    print("\n=== Sample 0 ===")
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"[{role}] {preview}")
    print(f"enable_thinking: {sample['enable_thinking']}")


if __name__ == "__main__":
    main()
