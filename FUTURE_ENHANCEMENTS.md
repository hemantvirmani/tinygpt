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

## 2. lm_head bias = False [DONE]

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

## 3. Weight Tying: lm_head.weight = wte.weight [DONE]

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

## 4. Fix G_EVAL_ITERATIONS Hardcoding Bug [DONE]

The variable `G_EVAL_ITERATIONS` was not being used — the function had `10`
hardcoded instead. Fixed in a prior commit; `eval_iterations` is now a field
in the `Hyperparameters` dataclass (default: 50 batches).

---

## 5. Stop Training When Val Loss Plateaus [DONE]

Rather than a fixed `max_iters` (which was fine for initial runs), make
stopping dynamic: track the best val loss seen; if it does not improve by
at least `min_delta=0.01` for `patience=6_000` steps (12 consecutive evals
at `eval_interval=500`), save the best checkpoint and exit early.

`patience=6_000` is ~20% of `max_iters=30_000` — enough to confirm a real
plateau, not just noise. This avoids wasting compute past the plateau and
removes the need to pick a magic step count upfront.
