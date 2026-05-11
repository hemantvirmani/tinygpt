# FemtoGPT & TinyGPT

This repo is a learning-first GPT project. My goal was to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation — understanding how GPT learns, not just how to run it.

We start with a minimal, readable model in `femtogpt.py` (~10M parameter model) based on Andrej Karpathy's minimal GPT, using a character-level tokenizer trained on the Shakespeare dataset. Reference: [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY). It includes all core features including multi-head attention.

`tinygpt.py` is a GPT-2/nanoGPT-class decoder-only Transformer with **163.04M parameters**. It uses tiktoken's GPT-2 tokenizer and was trained on FineWeb-Edu (sample-100BT).

## Checkpoints

TinyGPT saves a checkpoint every 100 steps.

## Reproducibility

A fixed random seed (`G_SEED`) seeds both CPU and CUDA so experiments are easier to compare.

## Performance Optimizations

### Hardware Performance Optimizations
- TensorFloat-32 (TF32) precision: `torch.set_float32_matmul_precision('high')` for 19-bit internal matmul precision. [DONE]
- bfloat16 mixed precision: `torch.autocast` for faster, lower-precision training without float16 overflow issues. [DONE]
- Kernel fusion via `torch.compile`: fuses ops into a single CUDA kernel to reduce overhead. [DONE]
- FlashAttention: `torch.nn.functional.scaled_dot_product_attention` to avoid materializing the NxN attention matrix. [DONE]

### Algorithmic Optimizations
- AdamW optimizer with GPT-3 hyperparameters: beta1=0.9, beta2=0.95, epsilon=1e-8. [DONE]
- Weight decay 0.1 on 2D parameters only (exclude biases and LayerNorm scales). [DONE]
- Cosine LR schedule: linear warmup to max LR, then decay to 10% of max. [DONE]
- Global grad clip 1.0 to prevent loss spikes. [DONE]
- Fused AdamW (`fused=True`) for faster updates. [DONE]
- Gradient accumulation to reach large effective batch sizes without OOM. [DONE]

## TODO (Living List, not in order of implementation)

- Temperature sampling, Top-k / Top-p sampling
- Fine Tuning
- RLHF

## Dataset

**FemtoGPT** — Uses Shakespeare (Karpathy's dataset).

**TinyGPT** — Streams directly from [FineWeb-Edu sample-100BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~100B tokens of high-quality educational web text) during training. No offline dataset.bin required. Earlier attempts used a smaller offline dataset (~2.5 GB from FineWeb-Edu sample-10BT).

### prepare_dataset.py

`prepare_dataset.py` builds an offline mixed dataset (WikiText-103 + OpenWebText sample + FineWeb-Edu sample), cleans it, and saves `dataset.txt` and `dataset.bin`. Kept as a fallback; set `G_USE_STREAMING = False` in `tinygpt.py` to use it.

Tokenizer: `tiktoken` with GPT-2 encoding.

## Results

### FemtoGPT (~10M params, 6 layers, 6 heads, embd=384, ctx=256, char-level tokenizer)

| Config | Train Loss | Val Loss |
|--------|-----------|---------|
| batch=64, iters=2000, lr=3e-4 | 1.3349 | 1.6529 |

### TinyGPT (163.04M params, 12 layers, 12 heads, embd=768, ctx=1024)

| Attempt | Steps | Dataset | Eff. Batch | LR | Best Val Loss | Notes |
|---------|-------|---------|-----------|-----|--------------|-------|
| 1 | ~60K | Offline curated corpus | 16 | 1e-5 | ~3.64 | Resumed from prior run; very low LR; dataset exhausted |
| 2 | 100K | FineWeb-Edu 10BT (~2.5 GB), offline | 32 | 3e-4 | **3.34** (step 99.4K) | Still declining at cutoff; run stopped at 100K step limit |
| 3 | 379,400 | FineWeb-Edu 100BT, streaming | 32 | 3e-4 | **3.1878** (step 376K) | Slow crawl due to small effective batch; checkpoint saved for attempt 4 |
| 4 | 59,600 | FineWeb-Edu 100BT, streaming | 512 | 6e-4 | **2.8368** (step 436.5K) | 16× batch increase; matched nanoGPT GPT-2 small benchmark |

**Key lesson across attempts:** Dataset size and effective batch size are the two most important variables. Attempts 1 and 2 were limited by dataset size. Attempt 3 was limited by gradient noise from a small effective batch (32). Scaling the batch 16× in attempt 4 delivered more improvement in 60K steps than attempt 3 did in its final 300K steps.

**nanoGPT GPT-2 small reference** (124M params, OpenWebText): val loss ~2.85. TinyGPT at 163M params on FineWeb-Edu reached **2.8368** — effectively matching the benchmark.

## Training Story & Learning Journey

- [Training Story](gpt2-training-artifacts/tinygpt-training-story.md) — detailed account of all four attempts, loss curve analysis, and final results.
- [Learning Journey](gpt2-training-artifacts/tinygpt-learning-journey.md) — how the codebase evolved commit-by-commit, what each phase taught the model, and open questions about Transformer architecture.

## Learning Roadmap

Run → Understand → Control → Scale → Customize

## Credits

* Inspired by Andrej Karpathy and GPT architectures.
* For TinyGPT, I used ChatGPT for helping with the starting code.
* Both ChatGPT and Gemini answered a lot of questions along the way.
* Thanks to Kaggle and RunPod for GPU access.
