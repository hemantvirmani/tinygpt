# Future Enhancements — TinyGPT Pretraining

If pretraining from scratch again with free GPU access, apply these changes.

---

## 1. Switch Dataset: FineWeb-Edu → FineWeb [DONE]

Switched to `HuggingFaceFW/fineweb` `sample-10BT` in `tinygpt.py`. FineWeb-Edu
biased the model toward formal educational text; FineWeb base is more diverse
and produces a stronger general-purpose base for instruction fine-tuning.

Training target: 1.57 passes × 10B tokens = 15.7B tokens = `max_iters=30_000`,
`warmup_iters=1_800` (6% warmup). Kept to ≤2 passes per Muennighoff et al. 2023.

If running on RTX 5090: set `batch_size=32` (one-line change). Accumulation
steps drop 32 → 16 automatically. Same training dynamics, ~26 hrs vs ~62 hrs
on 4090, and cheaper ($26 vs $43 at RunPod community pricing).

---

## 1b. Try OpenWebText for a Direct nanoGPT Comparison

**File:** `tinygpt.py`

```python
STREAMING_HF_DATASET = "Skylion007/openwebtext"
STREAMING_HF_SUBSET  = None   # no subset — full dataset is ~9B tokens
```

nanoGPT GPT-2 small (124M params) trained on OpenWebText and reached val loss
~2.85. TinyGPT reached 2.84 on FineWeb-Edu, but those are different datasets
so the comparison isn't apples-to-apples. Training on the same OpenWebText
would isolate the architecture difference (163M vs 124M, lm_head bias, etc.)
from the dataset difference. Expected outcome: TinyGPT should beat 2.85 on
the same data given its larger size, if it doesn't, that's a signal something
in the architecture or training setup needs revisiting.

At `effective_batch_size=512`, 9B tokens ≈ 18K steps — a very short run.

---

## 2. lm_head bias = False

**File:** `tinygpt.py`, `TinyGPT.__init__`

```python
# Before
self.head = nn.Linear(G_N_EMBD, state.vocab_size)

# After
self.head = nn.Linear(G_N_EMBD, state.vocab_size, bias=False)
```

The trained bias (norm ~1459) caused a silent bug during HuggingFace export —
the bias was dropped, reducing logit correlation to 0.8. `bias=False` is the
standard (GPT-2, GPT-3, Llama, Mistral all use no lm_head bias), makes HF
export trivial, and unlocks weight tying.

---

## 3. Weight Tying: lm_head.weight = wte.weight

**File:** `tinygpt.py`, `TinyGPT.__init__` (after removing lm_head bias above)

```python
self.head.weight = self.token_embedding_table.weight
```

Both `token_embedding_table` (vocab→embd) and `lm_head` (embd→vocab) are
(50257 × 768) matrices doing inverse jobs. Tying them means one matrix does
both — the intuition being that the vector representing "cat" as input should
point in the same direction the model votes for when predicting "cat" as output.

Karpathy uses this in nanoGPT. The original GPT-2 paper used it. It was shown
in "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)
to improve perplexity across benchmarks.

Parameter impact: saves 38.6M params (768 × 50257), dropping the model from
163M → **124M**. Also acts as regularization — fewer free parameters means
less overfitting risk during fine-tuning. Requires `bias=False` on lm_head first.

GPT-3 and modern LLMs (Llama, Mistral) dropped weight tying for more
expressiveness at scale. At 163M params it is a net positive.

---

## 4. Dropout = 0.0 During Pretraining

**File:** `tinygpt.py`

```python
G_DROPOUT_PROB = 0.0   # keep at 0 for pretraining
```

Modern LLMs (GPT-3, Llama, Mistral) use zero dropout during pretraining.
When training on billions of tokens, the model won't overfit the data —
dropout only wastes compute by zeroing activations that could be learning.
Add dropout (0.1) only at fine-tuning time via the fine-tuning notebook.

---

## 5. Fix G_EVAL_ITERATIONS Hardcoding Bug [DONE]

The variable `G_EVAL_ITERATIONS` was not being used — the function had `10`
hardcoded instead. Fixed in a prior commit; `eval_iterations` is now a field
in the `Hyperparameters` dataclass (default: 50 batches).

---

## 6. Stop Training When Val Loss Plateaus

Rather than a fixed `max_iters` (now a `Hyperparameters` field, default
`100_000`), stop when val loss has not improved for ~10K steps after the LR
has decayed to its minimum. This is more principled than an arbitrary step
count — you may be leaving performance on the table or wasting compute past
the plateau.

The 100K default was chosen because at `effective_batch_size=512` it covers
~52B tokens — already well beyond what nanoGPT used to reach val loss 2.85.
For a principled stopping rule: once the cosine LR reaches `eta_min` and val
loss is flat for 10K steps, stop and save.
