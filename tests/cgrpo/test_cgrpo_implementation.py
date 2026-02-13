#!/usr/bin/env python3
"""
Test script for Curriculum-GRPO implementation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from transformers import AutoTokenizer


def test_curriculum_manager():
    """Test CurriculumManager."""
    print("=" * 50)
    print("Testing CurriculumManager")
    print("=" * 50)
    
    from verl.utils.curriculum import CurriculumManager, CurriculumConfig
    
    config = CurriculumConfig(
        initial_k=1,
        max_k=5,
        ema_alpha=0.1,
        base_threshold=0.9,
        threshold_decay=0.95,
        patience=10,
        min_steps_per_k=5,
    )
    
    manager = CurriculumManager(config)
    
    assert manager.get_current_k() == 1, f"Expected k=1, got {manager.get_current_k()}"
    print(f"✓ Initial k: {manager.get_current_k()}")
    
    for step in range(20):
        success_rate = 0.95 if step < 10 else 0.85
        metrics = manager.update(success_rate)
        print(f"Step {step}: k={metrics['curriculum/k']}, sr_ema={metrics['curriculum/sr_ema']:.4f}")
    
    print(f"✓ Final k: {manager.get_current_k()}")
    print(f"✓ Curriculum summary: {manager.get_summary()}")
    
    state = manager.state_dict()
    manager2 = CurriculumManager(config)
    manager2.load_state_dict(state)
    assert manager2.get_current_k() == manager.get_current_k()
    print("✓ State save/load works")
    
    print()


def test_curriculum_dataset():
    """Test CurriculumGRPODataset."""
    print("=" * 50)
    print("Testing CurriculumGRPODataset")
    print("=" * 50)
    
    from verl.utils.dataset.curriculum_dataset import CurriculumGRPODataset
    from omegaconf import DictConfig
    
    test_data = [
        {
            "question": "What is 2 + 2?",
            "steps": ["First, we have 2.", "Then, we add 2 more.", "The result is 4."],
            "teacher_answer": "4",
            "ground_truth": "4",
            "index": 0,
        },
        {
            "question": "What is 3 * 3?",
            "steps": ["We need to multiply 3 by 3.", "3 times 3 equals 9."],
            "teacher_answer": "9",
            "ground_truth": "9",
            "index": 1,
        },
    ]
    
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        temp_file = f.name
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
        
        config = DictConfig({
            "max_prompt_length": 512,
            "max_response_length": 256,
            "thinking_start": "<think",
            "thinking_end": "</think",
            "answer_start": "<answer>",
            "answer_end": "</answer>",
            "use_chat_template": True,
        })
        
        dataset = CurriculumGRPODataset(
            data_files=temp_file,
            tokenizer=tokenizer,
            config=config,
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        item = dataset[0]
        print(f"✓ Sample keys: {item.keys()}")
        print(f"  - Question: {item['raw_prompt'][0]['content']}")
        print(f"  - Steps: {item['steps']}")
        print(f"  - Ground truth: {item['ground_truth']}")
        
        for k in [1, 2, 3]:
            prompt, prefix, cut_idx = dataset.build_curriculum_prompt(item, current_k=k)
            print(f"\n✓ Curriculum prompt for k={k}:")
            print(f"  - Cut index: {cut_idx}")
            print(f"  - Teacher prefix: {prefix[:100]}..." if len(prefix) > 100 else f"  - Teacher prefix: {prefix}")
        
        print()
        
    finally:
        os.unlink(temp_file)


def test_reward_manager():
    """Test CurriculumGRPORewardManager."""
    print("=" * 50)
    print("Testing CurriculumGRPORewardManager")
    print("=" * 50)
    
    from verl.workers.reward_manager.cgrpo import CurriculumGRPORewardManager
    from verl import DataProto
    from tensordict import TensorDict
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    
    manager = CurriculumGRPORewardManager(
        tokenizer=tokenizer,
        correct_score=1.0,
        format_score=0.0,
    )
    
    responses = [
        "<think\nStep 1\n</think\n<answer>42</answer>",
        "<think\nStep 1\n</think\n<answer>24</answer>",
        "The answer is 42.",
    ]
    
    response_ids = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    max_len = max(len(ids) for ids in response_ids)
    padded_responses = torch.zeros(len(responses), max_len, dtype=torch.long)
    for i, ids in enumerate(response_ids):
        padded_responses[i, :len(ids)] = torch.tensor(ids)
    
    batch = TensorDict({
        "responses": padded_responses,
    })
    
    non_tensor_batch = {
        "reward_model": np.array([
            {"ground_truth": "42"},
            {"ground_truth": "42"},
            {"ground_truth": "42"},
        ], dtype=object),
    }
    
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    reward_tensor, extra_info = manager(data, return_dict=True)
    
    print(f"✓ Reward shape: {reward_tensor.shape}")
    print(f"✓ Extracted answers: {extra_info['extracted_answers']}")
    print(f"✓ Ground truths: {extra_info['ground_truths']}")
    print(f"✓ Is correct: {extra_info['is_correct']}")
    print(f"✓ Rewards: {reward_tensor.sum(dim=-1).tolist()}")
    
    print()


def test_prompt_building():
    """Test curriculum prompt building."""
    print("=" * 50)
    print("Testing Prompt Building")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    
    question = "What is 5 + 3?"
    steps = [
        "We need to add 5 and 3.",
        "5 plus 3 equals 8.",
    ]
    
    for k in [1, 2]:
        num_steps = len(steps)
        cut_index = max(0, num_steps - k)
        teacher_prefix = "\n".join(steps[:cut_index]) if cut_index > 0 else ""
        
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        if teacher_prefix:
            prompt += f"<think\n{teacher_prefix}"
        
        print(f"\n✓ Prompt for k={k}:")
        print(f"  Cut index: {cut_index}")
        print(f"  Teacher prefix: '{teacher_prefix}'")
        print(f"  Full prompt:\n{prompt}")
        print(f"  Student needs to generate: {steps[cut_index:]}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Curriculum-GRPO Implementation Tests")
    print("=" * 50 + "\n")
    
    try:
        test_curriculum_manager()
        test_curriculum_dataset()
        test_reward_manager()
        test_prompt_building()
        
        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
