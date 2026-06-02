# FemtoGPT & TinyGPT

This repo is meant to build hands-on intuition for attention mechanisms, embeddings, fine-tuning, etc. By the end of it, we will understand working of GPTs by building a toy model and an evolved model.

We start with a minimal, readable model in `femtogpt.py` (~10M parameter model) based on Andrej Karpathy's minimal GPT, using a character-level tokenizer trained on the Shakespeare dataset. Reference: [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY). It includes all core features including multi-head attention.

`tinygpt.py` is a GPT-2/nanoGPT-class decoder-only Transformer. Current version: **124.44M parameters** (weight-tied lm_head, matching nanoGPT GPT-2 small architecture). Uses tiktoken's GPT-2 tokenizer.

## Checkpoints

TinyGPT saves a checkpoint every 1000 steps.

## Inference and Hosted Model

### Pretrained TinyGPT

Both formats are hosted on Hugging Face at [hemantvirmani/tinyGPT](https://huggingface.co/hemantvirmani/tinyGPT/):

- **PyTorch-native weights** (`tinygpt_pretrained_weights.pt`) — load with `tinygpt.load_model_for_inference()`
- **HuggingFace format** (`model.safetensors` + `config.json`) — load with `GPT2LMHeadModel.from_pretrained("hemantvirmani/tinyGPT")`

**Inference scripts:**

- [infer_pytorch.py](infer_pytorch.py) — loads the PyTorch-native model and runs the prompt suite used for the sample generations.
- [infer_hf.py](infer_hf.py) — loads the HuggingFace-format model from `tinygpt_pretrained_model_hf/` and runs the same prompt suite.

**Export scripts:**

- [export_to_hf.py](export_to_hf.py) — converts `tinygpt_pretrained_weights.pt` → `tinygpt_pretrained_model_hf/` (HuggingFace format).

- [pretraining_output_1.0.txt](gpt2-training-artifacts/pretraining_output_1.0.txt) contains sample outputs from PyTorch-native inference with `temperature=1.0` and `max_tokens=500`.
- [pretraining_output_0.7.txt](gpt2-training-artifacts/pretraining_output_0.7.txt) contains the better sample outputs from PyTorch-native inference with `temperature=0.7` and `max_tokens=500`.

Temperature controls sampling randomness: lower values usually make generations more conservative and coherent, while higher values add variety but can drift more.

### Instruction-Tuned TinyGPT (Alpaca)

Fine-tuned on Alpaca Cleaned 52K. Upload to Hugging Face pending.

- **PyTorch-native weights** (`tinygpt_finetuned_checkpoint_alpaca.pt`) — _not yet uploaded_
- **HuggingFace format** (`tinygpt_pretrained_model_hf_alpaca/`) — _not yet exported or uploaded_

**Export script:** [export_to_hf_alpaca.py](export_to_hf_alpaca.py) — converts `tinygpt_finetuned_checkpoint_alpaca.pt` → `tinygpt_pretrained_model_hf_alpaca/` (HuggingFace format).

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

- Fine Tuning [DONE]
- RLHF

## Attempt 6 Plan — OpenWebText (Direct nanoGPT Comparison)

**Goal:** Validate the 124M architecture against nanoGPT's 2.85 benchmark on the same dataset (OpenWebText). If TinyGPT hits ~2.85 on OpenWebText, the architecture and training setup are confirmed correct, and the higher attempt 5 number is purely the harder FineWeb base dataset.

### Step budget

OpenWebText has ~9.4B tokens. At `effective_batch_size=512`, `block_size=1024`: one full pass ≈ 17.9K steps.

- `max_iters=30_000` — 1.67 passes through 9.4B tokens (within Muennighoff's ≤2-pass safe zone; same schedule as attempt 5)
- `warmup_iters=1_800` — 6% warmup, identical to attempt 5
- Early stopping (`patience=6_000`) will halt if loss plateaus before 25K

The LR schedule is calibrated to the data: the cosine decay hits its minimum right as the data is exhausted. Setting `max_iters` larger than the data would waste compute at minimal LR — this budget avoids that.

### Expected outcome

| Model | Params | Dataset | Expected val loss |
| --- | --- | --- | --- |
| nanoGPT GPT-2 small | 124M | OpenWebText | ~2.85 |
| TinyGPT attempt 6 | 124M | OpenWebText | ~2.85 (target) |

Both models are 124M params (weight-tied) on the same data — the fairest possible comparison.

### GPU options

| GPU | `batch_size` | Est. hours |
| --- | --- | --- |
| RTX 4090 | 16 | ~62 hrs |
| RTX 5090 | 16 | ~26 hrs |

## Dataset

**FemtoGPT** — Uses Shakespeare (Karpathy's dataset).

**TinyGPT** — Streams directly from [FineWeb-Edu sample-100BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~100B tokens of high-quality educational web text) during training. No offline dataset.bin required. Earlier attempts used a smaller offline dataset (~2.5 GB from FineWeb-Edu sample-10BT).

### prepare_dataset.py

`prepare_dataset.py` builds an offline mixed dataset (WikiText-103 + OpenWebText sample + FineWeb-Edu sample), cleans it, and saves `dataset.txt` and `dataset.bin`. Kept as a fallback; set `G_USE_STREAMING = False` in `tinygpt.py` to use it.

Tokenizer: `tiktoken` with GPT-2 encoding.

## Results

> All results below were produced with code at tag **[v1.0.0](https://github.com/hemantvirmani/tinygpt/tree/v1.0.0)**. The current `main` branch reflects the next planned pretraining configuration (FineWeb base 10BT, 30K steps) — see [Attempt 6 Plan](#attempt-6-plan--openwebtext-direct-nanogpt-comparison).

### FemtoGPT (~10M params, 6 layers, 6 heads, embd=384, ctx=256, char-level tokenizer)

| Config                          | Train Loss | Val Loss |
|---------------------------------|------------|----------|
| batch=64, iters=2000, lr=3e-4   | 1.3349     | 1.6529   |

### TinyGPT (12 layers, 12 heads, embd=768, ctx=1024)

Attempts 1–4: **163.04M params** (no weight tying). Attempt 5+: **124.44M params** (weight-tied lm_head = token embedding, matching nanoGPT GPT-2 small).

| Attempt | Steps | Dataset | Params | Eff. Batch | LR | Best Val Loss | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ~60K | Offline curated corpus | 163M | 16 | 1e-5 | ~3.64 | Resumed from prior run; very low LR; dataset exhausted |
| 2 | 100K | FineWeb-Edu 10BT (~2.5 GB), offline | 163M | 32 | 3e-4 | **3.34** (step 99.4K) | Still declining at cutoff; run stopped at 100K step limit |
| 3 | 379,400 | FineWeb-Edu 100BT, streaming | 163M | 32 | 3e-4 | **3.1878** (step 376K) | Slow crawl due to small effective batch; checkpoint saved for attempt 4 |
| 4 | 59,600 | FineWeb-Edu 100BT, streaming | 163M | 512 | 6e-4 | **2.8368** (step 436.5K) | 16× batch increase; matched nanoGPT GPT-2 small benchmark |
| 5 | 24,700 (best step 23,600) | FineWeb base 10BT, streaming | 124M | 512 | 6e-4 | **3.2457** (step 23,600) | First run with weight tying (163M → 124M). Higher val loss vs attempt 4 is expected — see note below. |

**Key lesson across attempts:** Dataset size and effective batch size are the two most important variables. Attempts 1 and 2 were limited by dataset size. Attempt 3 was limited by gradient noise from a small effective batch (32). Scaling the batch 16× in attempt 4 delivered more improvement in 60K steps than attempt 3 did in its final 300K steps.

**Attempt 5 note — why val loss is higher than attempt 4:** The 3.2457 vs 2.8368 gap reflects three compounding changes, none of which indicate model degradation:

1. **Harder dataset** — FineWeb base (diverse web text) has a higher entropy floor than FineWeb-Edu (filtered educational text). A model trained on base text is harder to evaluate with cross-entropy.
2. **Fewer parameters** — weight tying saved 38.6M params (163M → 124M). Fewer free parameters = less expressive capacity = higher irreducible loss.
3. **Data ceiling** — 10BT tokens is not enough for the model to converge strongly on FineWeb base. The loss plateaued around step 23K and stopped improving meaningfully.

**nanoGPT GPT-2 small reference** (124M params, OpenWebText): val loss ~2.85. TinyGPT attempt 4 at 163M params on FineWeb-Edu reached **2.8368** — effectively matching the benchmark. Attempt 5 at 124M params on FineWeb base is the next step toward a direct apples-to-apples comparison (see Attempt 6 plan below).

### Instruction Fine-Tuning — Alpaca Cleaned (52K examples)

Full fine-tune of the pretrained TinyGPT weights on the [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset using a custom PyTorch training loop. Prompt template: `### Instruction / ### Input (optional) / ### Response`. Loss masked on padding tokens.

| Setting | Value |
| --- | --- |
| Base model | TinyGPT pretrained (val loss 2.8368) |
| Dataset | Alpaca Cleaned (52K examples) |
| Dropout | 0.1 |
| Learning rate | 1e-4 (warmup + cosine decay) |
| Effective batch size | 64 (4 micro-batch × 16 accumulation steps) |
| Best val loss | **1.8405** (step 3,600 of 5,000) |
| Final weights | `tinygpt_finetuned_checkpoint_alpaca.pt` |

**Key lesson:** Dropout = 0.1 is critical at fine-tuning time — without it, the train/val gap exceeded 0.80 within 2,000 steps. With it, the gap stayed at 0.40–0.50. Format acquisition is fast (first 100 steps); factual accuracy is limited by the 163M parameter capacity.

The notebook and raw fine-tuning outputs are in [`gpt2-training-artifacts/finetuning_outputs_alpaca_dataset.md`](gpt2-training-artifacts/finetuning_outputs_alpaca_dataset.md).

## Training Story & Learning Journey

- [Training Story](gpt2-training-artifacts/tinygpt-training-story.md) — detailed account of all four pretraining attempts, loss curve analysis, and final results.
- [Learning Journey](gpt2-training-artifacts/tinygpt-learning-journey.md) — how the codebase evolved commit-by-commit, what each phase taught the model, instruction fine-tuning on Alpaca Cleaned, and open questions about Transformer architecture.

## Learning Roadmap

Run → Understand → Control → Scale → Customize

## Credits

- Inspired by Andrej Karpathy and GPT architectures.
- For TinyGPT, I used ChatGPT for helping with the starting code.
- Both ChatGPT and Gemini answered a lot of questions along the way.
- Thanks to Kaggle and RunPod for GPU access.
