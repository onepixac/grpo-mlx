# GRPO on Apple Silicon (MLX)

**First open-source implementation of Group Relative Policy Optimization on MLX.**

## The Problem

Fine-tuning an LLM to its full potential requires three stages: **CPT** (domain knowledge) → **SFT** (instruction tuning) → **GRPO** (reinforcement learning). On Apple Silicon, `mlx-lm` handles CPT and SFT beautifully — but the pipeline stops there. The third stage, GRPO, only exists in HuggingFace's `trl` library, which requires CUDA and PyTorch. **It does not run on MLX.**

This means every Mac user doing local fine-tuning is stuck at stage 2. They can teach a model *what* to say (SFT), but they can't optimize *how well* it says it (GRPO). The result: responses that are correct but stiff, repetitive, or unnatural — exactly what RL is designed to fix.

## The Solution

This script implements the complete GRPO training loop **natively on MLX**, with actual LoRA weight updates via backpropagation. No trl, no PyTorch, no CUDA.

Works with **any open-source model** that runs on `mlx-lm` — Qwen, Llama, Mistral, Gemma, Phi, or any other architecture, at any size and quantization.

Now Mac users can run the full fine-tuning pipeline locally:
- `mlx-lm` for CPT and SFT (stages 1-2)
- **this script** for GRPO (stage 3)

**Tested:** Mac Mini M4 Pro 64GB — reward score **0.922**.

Based on: *"Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO"* ([arXiv:2509.13081](https://arxiv.org/abs/2509.13081))

## How It Works

```
For each training iteration:
  1. Sample prompts from your dataset
  2. Generate N candidate responses per prompt (with sampling for diversity)
  3. Score each candidate with semantic similarity (sentence-transformers)
     → cosine similarity between candidate and reference response
  4. Compute GRPO loss: reward-weighted cross-entropy
     - Candidates above group average → reinforce (make more likely)
     - Candidates below group average → penalize (make less likely)
  5. Backpropagate through MLX and update LoRA weights
```

The key insight: `mlx.nn.value_and_grad` computes both loss and gradients in a single call, making GRPO feasible on MLX without any external RL framework.

## Requirements

- Mac with Apple Silicon (M1/M2/M3/M4), 16GB+ RAM recommended
- Python 3.10+ (3.12 recommended)
- Any model compatible with `mlx-lm`

```bash
pip install mlx-lm sentence-transformers numpy
```

## Quick Start

### 1. Prepare your data

JSONL file with the same format as SFT training data. The `assistant` response is used as the reference ("gold standard") that GRPO optimizes toward:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a branch of AI that enables systems to learn from data..."}]}
```

### 2. Run GRPO

Use any MLX-compatible model — local path or HuggingFace repo:

```bash
# With SFT adapter (recommended — start from a fine-tuned model)
python grpo_mlx.py \
    --model <your-mlx-model> \
    --adapter path/to/sft/adapters \
    --data train.jsonl

# Without adapter (GRPO on base/instruct model)
python grpo_mlx.py \
    --model <your-mlx-model> \
    --data train.jsonl

# Custom parameters
python grpo_mlx.py \
    --model <your-mlx-model> \
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
[14:23:15] Reward encoder loaded (all-MiniLM-L6-v2)
[14:23:22] Model loaded
[14:23:22] LoRA applied (16 layers, rank=8)
[14:23:22] Reference data: 847 entries
[14:24:10] Iter 1/30: reward=0.412 (max=0.623), loss=0.0821, 48.3s
[14:25:02] Iter 2/30: reward=0.534 (max=0.712), loss=0.0654, 51.7s
...
[14:48:33] Iter 30/30: reward=0.891 (max=0.922), loss=0.0123, 47.1s

SUCCESS: final reward 0.891
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Path to any MLX model or HuggingFace repo |
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

| Model size | RAM needed | Recommended settings |
|------------|-----------|---------------------|
| ~4B | ~8 GB | batch 3, 12 layers, candidates 6 |
| ~9B | ~20 GB | batch 3, 16 layers, candidates 4 |
| ~14B | ~35 GB | batch 2, 16 layers, candidates 4 |

For long runs, use `nohup`:
```bash
nohup python grpo_mlx.py --model ... --data ... --log grpo.log > /dev/null 2>&1 &
tail -f grpo.log
```

## The Full Pipeline: CPT → SFT → GRPO

GRPO is the third and final stage of the fine-tuning pipeline:

| Stage | What it does | Tool |
|-------|-------------|------|
| **CPT** | Teach domain vocabulary and patterns with raw text | `mlx-lm` |
| **SFT** | Teach instruction-following with Q&A pairs | `mlx-lm` |
| **GRPO** | Optimize response quality with reward-weighted RL | **this script** |

Each stage builds on the previous one. GRPO works best starting from SFT adapters (`--adapter`).

```bash
python -m mlx_lm lora -c config_cpt.yaml                    # Stage 1: CPT
python -m mlx_lm lora -c config_sft.yaml                    # Stage 2: SFT
python grpo_mlx.py --model ... --adapter sft_adapters ...   # Stage 3: GRPO
```

## Important Notes

### Thinking Models

If your model has a "thinking mode" (e.g. Qwen3.5, QwQ, or similar reasoning models), **you must disable it**. The script handles this via `enable_thinking=False` in `apply_chat_template`. Thinking models prepend internal reasoning to every response, which destroys semantic similarity scores. In our tests: reward 0.92 without thinking, 0.27 with thinking.

### Semantic Reward

The reward signal comes from cosine similarity between generated and reference responses, computed by a lightweight sentence-transformer encoder (~80MB, runs on CPU). This is more flexible than keyword matching and orders of magnitude cheaper than using an LLM judge.

### Early Stopping

Training stops automatically if average reward doesn't improve for 5 consecutive iterations (after at least 10 iterations). This prevents overfitting.

### Model Compatibility

Any model that works with `mlx-lm` works here. This includes MLX-converted models from:
- [mlx-community](https://huggingface.co/mlx-community) on HuggingFace (thousands of models)
- Models you convert yourself with `mlx_lm.convert`

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
