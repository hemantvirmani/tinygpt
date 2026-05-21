# Future Enhancements — TinyGPT Pretraining

If pretraining from scratch again with free GPU access, apply these changes.

---

## 1. Switch Dataset: FineWeb-Edu → FineWeb

**File:** `tinygpt.py`

```python
STREAMING_HF_DATASET = "HuggingFaceFW/fineweb"
STREAMING_HF_SUBSET  = "sample-10BT"   # 10B tokens — good starting point
```

FineWeb-Edu biases the model toward formal educational text. Full FineWeb is
more diverse, better filtered, and produces a stronger general-purpose base
for instruction fine-tuning. Start with `sample-10BT` (~what GPT-2 saw);
only go to `sample-100BT` if GPU time is abundant.

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

## 5. Fix G_EVAL_ITERATIONS Hardcoding Bug

**File:** `tinygpt.py`, `_evaluate_loss` (and anywhere else it's hardcoded)

The variable `G_EVAL_ITERATIONS` was not being used — the function had `10`
hardcoded instead. This made val loss readings during pretraining extremely
noisy (10 batches = ~160 tokens evaluated). Fix it and set a proper value:

```python
G_EVAL_ITERATIONS = 200   # 200 batches × 16 = 3200 tokens per eval
```

---

## 6. Stop Training When Val Loss Plateaus

Rather than a fixed `G_MAX_ITERS = 600000`, stop when val loss has not
improved for ~10K steps after the LR has decayed to its minimum. This is more
principled than an arbitrary step count — you may be leaving performance on
the table or wasting compute past the plateau.

Practical rule: once the cosine LR reaches `eta_min` and val loss is flat for
10K steps, stop and save.
