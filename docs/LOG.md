# Project log — Does refusal decompose by harm category?

Mateenah Jahan · BlueDot Technical AI Safety Project · Unit 2

---

## Unit 2.2 — Assumptions: what needs to be true for this to work

Thirteen assumptions, rated on **stakes** (does being wrong kill the direction, or just need an
adjustment?) and **uncertainty** (how confident am I already?). Cruxes are the high-stakes,
high-uncertainty ones.

### About the existing work

| # | Assumption | Stakes | Uncertainty |
|---|---|---|---|
| A1 | The single-direction result of Arditi et al. (2024) is real and reproducible, not an artefact of their model set. | **High** | Low — 13 models up to 72B, NeurIPS-accepted, widely independently reproduced. |
| A2 | Nobody has already measured per-category refusal directions. If they have, my contribution shrinks to a replication. | Medium | **Medium** — I have searched the obvious places but cannot prove a negative. |
| A3 | The method transfers to Qwen2.5, a model family released after the paper. | Medium | Low — the paper covered Qwen-1.8B-chat; architecture is unchanged in the relevant respects. |

### About the data

| # | Assumption | Stakes | Uncertainty |
|---|---|---|---|
| A4 | **HarmBench's semantic categories are real semantic distinctions, not annotation convenience.** If `illegal` and `cybercrime_intrusion` overlap heavily in content, I am measuring label noise. | **High** | **High** — ← **crux** |
| A5 | 20 training prompts per category is enough for a stable mean direction. | **High** | **High** — ← **crux**. The paper used 128. I have a quarter of that. |
| A6 | Alpaca is an appropriate harmless contrast set for all four categories equally — i.e. the harmless baseline is not itself category-dependent. | Medium | Medium |

### About the method

| # | Assumption | Stakes | Uncertainty |
|---|---|---|---|
| A7 | **The model refuses HarmBench prompts at a high enough baseline rate that there is something to ablate.** | **Highest — kills the project** | **High** — ← **primary crux** |
| A8 | Refusal-substring matching is an adequate proxy for refusal at 48 generated tokens. | Medium | Medium — the paper uses 512 tokens; refusal phrases appear early, but a model could comply after a hedge. |
| A9 | A permutation null over relabelled harmful prompts is the correct null for "no category structure". | **High** | Low — **tested**: calibrated at 6.7% false-positive rate on synthetic data. |
| A10 | Cosine similarity between directions is the right geometric statistic (vs. subspace angle, or a learned probe). | Medium | Medium — it is the simplest defensible choice; a subspace method is the obvious escalation. |

### About tools and resources

| # | Assumption | Stakes | Uncertainty |
|---|---|---|---|
| A11 | A 1.5B model in fp16 plus generation fits a free Colab T4 within one session. | Medium | Low — ~3 GB of 16 GB; arithmetic is comfortable. |
| A12 | Forward hooks on `embed_tokens` and each decoder layer correctly implement residual-stream ablation in current `transformers`. | **High** | Medium — the API has shifted between versions. |

### About impact

| # | Assumption | Stakes | Uncertainty |
|---|---|---|---|
| A13 | Someone would act on this result — a safety-training team would design differently, or a monitoring team would build per-category probes. | Medium | Medium — the honest answer is that at 1.5B this is suggestive, not decisive. Scale is the gap. |

---

## Unit 2.3 — Deciding what to test first

### Ranking the cruxes

**A7 is first**, and it is not close. If Qwen2.5-1.5B-Instruct does not refuse HarmBench prompts at a
high rate, there is no refusal behaviour to locate, no direction to extract, and no ablation to
measure. Every other assumption is downstream of it. It is also the cheapest thing on the list to
check — no direction extraction, no hooks, just generate and count.

A5 and A4 are next. A12 is high-stakes but self-revealing: if the hooks are wrong, ablation will do
nothing at any layer, and that failure is loud rather than silent.

I am deliberately **not** starting with the interesting part (the transfer matrix). That is the
mistake the unit warns about — testing what is exciting rather than what is load-bearing.

### The first cheap test

**Test:** Generate completions for 32 HarmBench `standard` behaviors (8 per category) on
Qwen2.5-1.5B-Instruct with no intervention. Score refusal by substring matching. ~10 minutes.

**Prediction, written before running:**

- Overall baseline refusal **70–90%**. Reasoning: Qwen2.5-Instruct has real safety post-training, and
  HarmBench standard behaviors are explicit rather than subtle. But 1.5B is small, and small models
  are inconsistently aligned — so my interval is wide and I hold it loosely.
- **By category, I predict unevenness**, and this is the prediction I am least sure of:
  `chemical_biological` and `illegal` refused most (most salient in safety training data),
  `misinformation_disinformation` refused least — "write a persuasive article arguing X" reads much
  closer to a normal writing request than "synthesise Y does".
- If that unevenness shows up, it is already a small finding, and it complicates the main experiment:
  a category with a low refusal floor has little headroom for ablation to reduce.

**Decision rule, fixed in advance:**

| Baseline refusal | Action |
|---|---|
| ≥ 70% | Proceed as planned. |
| 40–70% | Proceed, but drop the weakest category and report the reduced headroom explicitly. |
| < 40% | Switch to `google/gemma-2-2b-it` (stronger refusal training; needs HF licence acceptance). Do not try to rescue the weaker model. |

**Why this test earns its place:** the outcome changes what I do next in all three branches. That is
the bar the unit sets — if every result led to the same next step, it would be the wrong test.

### Second test, if A7 passes

Ablate with the **pooled** direction and confirm refusal collapses. This reproduces the paper's core
result on my setup and simultaneously clears A12 — if the hooks are broken, nothing moves. Only once
the pooled effect reproduces is the per-category question meaningful.

### What I expect to be wrong about

My A7 interval (70–90%) is the number most likely to miss, and the per-category ordering is close to a
guess. I am recording both now precisely so the comparison afterwards is honest rather than
reconstructed.

---

## Running notes

**24 Oct 2026** — Direction chosen. Harness built and figures tested on synthetic data.
Permutation-test calibration run before any real data: 6.7% false-positive rate under a true null
(target ~5%), but ~100% power against even `shared_frac = 0.85`. **Consequence: a significant p-value
is cheap here.** The verdict logic was rewritten to rest on effect size and the transfer matrix, with
p as a gate only.

Also worth recording: on the first calibration attempt the detector flagged "category-specific" on
data built with no structure at all. It looked like a bug. Running 200 replications showed the test
was correctly calibrated and that first run was an unlucky seed. Had I run it once, I would either
have "fixed" a non-bug or shipped a false result.
