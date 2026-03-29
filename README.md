# GRPO on Apple Silicon (MLX)

**First open-source implementation of Group Relative Policy Optimization on MLX.**

GRPO is the RL (reinforcement learning) step that makes fine-tuned LLMs produce more natural, flexible responses. Until now, it required CUDA/PyTorch via HuggingFace's `trl` library — leaving Mac users unable to do RL-based fine-tuning.

This script implements the complete GRPO training loop **natively on MLX**, with actual weight updates. No trl, no PyTorch, no CUDA. Just MLX on Apple Silicon.

**Tested:** Mac Mini M4 Pro 64GB, Qwen3.5-9B-MLX-4bit — reward score **0.922**.

Based on: *"Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO"* ([arXiv:2509.13081](https://arxiv.org/abs/2509.13081))

## How It Works

```
For each training iteration:
  1. Sample prompts from your dataset
  2. Generate N candidate responses per prompt
  3. Score each candidate with semantic similarity (sentence-transformers)
  4. Compute GRPO loss: reward-weighted cross-entropy
     - Good candidates (above group average) → reinforce
     - Bad candidates (below group average) → penalize
  5. Backpropagate through MLX and update LoRA weights
```

The key insight: `mlx.nn.value_and_grad` computes both loss and gradients in a single call, making GRPO feasible on MLX without any external RL framework.

## Requirements

- Mac with Apple Silicon (M1/M2/M3/M4), 16GB+ RAM recommended
- Python 3.10+ (3.12 recommended)

```bash
pip install mlx-lm sentence-transformers numpy
```

## Quick Start

### 1. Prepare your data

JSONL file with the same format as SFT training data:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a branch of AI that enables systems to learn from data..."}]}
```

### 2. Run GRPO

```bash
# With SFT adapter (recommended — start from a fine-tuned model)
python grpo_mlx.py \
    --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --adapter path/to/sft/adapters \
    --data train.jsonl

# Without adapter (GRPO on base model)
python grpo_mlx.py \
    --model mlx-community/Qwen3.5-4B-MLX-4bit \
    --data train.jsonl

# Custom parameters
python grpo_mlx.py \
    --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --data train.jsonl \
    --n-candidates 6 \
    --n-iters 50 \
    --lr 5e-7 \
    --lora-layers 16 \
    --log grpo_train.log
```

### 3. Expected output

```
[14:23:01] === GRPO Full Training on MLX ===
[14:23:01] Model: mlx-community/Qwen3.5-9B-MLX-4bit
[14:23:15] Reward encoder loaded (all-MiniLM-L6-v2)
[14:23:22] Model loaded
[14:23:22] LoRA applied (16 layers, rank=8)
[14:23:22] Reference data: 847 entries
[14:24:10] Iter 1/30: reward=0.412 (max=0.623), loss=0.0821, 48.3s
[14:25:02] Iter 2/30: reward=0.534 (max=0.712), loss=0.0654, 51.7s
...
[14:48:33] Iter 30/30: reward=0.891 (max=0.922), loss=0.0123, 47.1s
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Path to MLX model or HuggingFace repo |
| `--adapter` | None | Path to SFT LoRA adapters |
| `--data` | (required) | Path to JSONL training data |
| `--n-candidates` | 4 | Candidates per prompt (4-8) |
| `--n-iters` | 30 | Training iterations |
| `--batch-size` | 3 | Prompts per iteration |
| `--max-tokens` | 256 | Max response tokens |
| `--lr` | 1e-6 | Learning rate (keep low for RL stability) |
| `--lora-layers` | 16 | Number of LoRA layers |
| `--temperature` | 0.7 | Generation temperature |
| `--min-response-length` | 80 | Min chars in reference response |
| `--log` | None | Log file path |

## Hardware Guidelines (4-bit quantized models)

| Model | RAM needed | Recommended settings |
|-------|-----------|---------------------|
| 4B | ~8 GB | batch 3, 12 layers, candidates 6 |
| 9B | ~20 GB | batch 3, 16 layers, candidates 4 |
| 14B | ~35 GB | batch 2, 16 layers, candidates 4 |

For long runs, use `nohup`:
```bash
nohup python grpo_mlx.py --model ... --data ... --log grpo.log > /dev/null 2>&1 &
tail -f grpo.log
```

## Pipeline: CPT → SFT → GRPO

GRPO is the third and final stage of the fine-tuning pipeline:

1. **CPT** (Continued Pre-Training) — teach the model new vocabulary/domain knowledge with raw text
2. **SFT** (Supervised Fine-Tuning) — teach the model to respond in the desired format with Q&A pairs
3. **GRPO** (this script) — refine response quality using reward-weighted RL

Each stage builds on the previous one. GRPO works best when starting from an SFT adapter (`--adapter`).

For CPT and SFT on MLX, use [`mlx-lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm):
```bash
python -m mlx_lm lora -c config_cpt.yaml   # Stage 1
python -m mlx_lm lora -c config_sft.yaml   # Stage 2
python grpo_mlx.py --adapter sft_adapters  # Stage 3
```

## Important Notes

### Qwen3.5 Thinking Mode

If using Qwen3.5 models, **always disable thinking mode**. The script handles this automatically via `enable_thinking=False` in `apply_chat_template`. Without this, Qwen3.5 prepends "Thinking Process:" to every response, destroying semantic similarity scores (reward drops from 0.92 to 0.27).

### Semantic Reward

The reward signal comes from cosine similarity between generated and reference responses, computed by a lightweight sentence-transformer encoder (~80MB). This is more flexible than keyword matching and much cheaper than using an LLM judge.

### Early Stopping

Training stops automatically if average reward doesn't improve for 5 consecutive iterations (after at least 10 iterations). This prevents overfitting.

## Citation

If you use this in your work:

```bibtex
@software{grpo_mlx,
    title={GRPO on Apple Silicon (MLX)},
    author={Roberto Marras and Claude Opus 4.6},
    year={2025},
    url={https://github.com/onepixac/grpo-mlx}
}
```

Based on:
```bibtex
@article{grpo_semantic_reward,
    title={Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO},
    year={2025},
    eprint={2509.13081},
    archivePrefix={arXiv}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
