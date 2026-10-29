# Does refusal decompose by harm category?

**8-day plan to the BlueDot discussion (24 Oct → 1 Nov 2026)**

---

## The question

Arditi et al. (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) showed refusal in
chat models is mediated by a **single direction** in the residual stream, recovered by
difference-in-means and verified by ablation (refusal disappears) and activation addition (refusal
appears on harmless prompts).

That result **pooled all harmful prompts into one set** — cybercrime, chemical/biological,
misinformation, illegal — and so by construction recovered one direction.

> ~~Whether distinct harm categories have distinct refusal directions has never been measured.~~
> *Corrected 29 Oct 2026:* per-domain directions have been compared geometrically (ValueShift
> Research, LessWrong, Apr 2026) and found in a refusal-token fine-tuned model (Alagharu et al. 2026,
> arXiv:2603.13359). What remains open, and what this project tests, is whether HarmBench harm
> categories are separable **behaviourally** on an unmodified model, with a calibrated null.

Both answers are findings, so there is no null-result risk:

| Outcome | Safety implication |
|---|---|
| One shared direction | Safety training is shallower than the pooled result implies — a single ablation removes refusal across every harm type at once. Argues the training objective itself needs rethinking. |
| Category-specific | There is internal structure. Harm types could be monitored or steered independently, and ablating one may leave others standing. A handle for more robust design. |
| Separable but interchangeable | Geometry splits, behaviour does not. The most interesting case, and worth its own writeup. |

---

## Method

1. **Extract** a direction per category by difference-in-means against a shared harmless set, plus the
   pooled direction as the reference point.
2. **Compare** them by cosine similarity.
3. **Null model** — permutation test. Randomly relabel the *same* harmful pool into pseudo-categories
   and recompute. Real categories are distinct only if their cross-similarity sits clearly below this.
4. **Transfer matrix** — ablate with category *A*'s direction, measure refusal rate on category *B*.
   This is the behavioural test and the headline figure.
5. **Activation addition** — add a category direction to harmless prompts, confirm induced refusal.

### Why step 3 is the load-bearing one

Every category direction is a harmful-minus-harmless contrast, so they all share a generic "this is
harmful" component. Cosine similarities will look high **whether or not category structure exists**.
Reading a high number as "one shared direction" would be wrong without the null.

### What the test calibration showed

The notebook validates the permutation test on synthetic data with the answer built in, and that check
changed how the result must be read:

| Synthetic scenario | Test fires | Verdict |
|---|---|---|
| `shared_frac = 1.0` — no structure at all | **4.5%** (200 reps; 6.7% in the 60-rep v0.1 run) | calibrated (should be ~5%) |
| `shared_frac = 0.85` — weak structure | 100% | |
| `shared_frac = 0.0` — fully category-specific | 100% | |

**The test is calibrated but extremely powerful** — it flags even trivial structure. So a significant
p-value is cheap, and the **effect size and transfer matrix carry the conclusion**, not p. This is
stated explicitly in the notebook so the finding is not oversold.

---

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` — ungated, ~3 GB fp16, fits free Colab T4 |
| Harmful | [HarmBench](https://github.com/centerforaisafety/HarmBench) `standard` behaviors, 4 categories balanced to 28 each |
| Harmless | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) instructions with no input field |
| Split | 20 train / 8 val per category |
| Runtime | ~40 min end to end on a free T4 |

Categories: `cybercrime_intrusion` (40 available), `illegal` (58), `misinformation_disinformation`
(34), `chemical_biological` (28 — the binding constraint). Excluded: `copyright` (its own functional
category, not a harm type), `harmful` (vague catch-all), `harassment_bullying` (only 19 — too thin for
a stable mean).

---

## Schedule

| Day | Date | Unit | What has to be true by the end |
|---|---|---|---|
| 1 | Sat 24 – Sun 25 | **U2** Get it working | Notebook runs end to end on Colab. Baseline refusal, layer selected, first transfer matrix in hand. |
| 2 | Mon 26 | **U2** remaining | "Figure out what needs to be true" + "Deciding what to test first" exercises, written from the actual first-run numbers. |
| 3–4 | Tue 27 – Wed 28 | **U3** Update and iterate | Second model (`gemma-2-2b-it`) for robustness. Widen the layer sweep. Raise n per cell if the intervals are too wide to read. |
| 5–6 | Thu 29 – Fri 30 | **U4** Write up | Public GitHub repo, README, three figures, short writeup. |
| 7 | Sat 31 | buffer | Figure polish, LinkedIn post, dry-run the demo. |
| 8 | **Sun 1 Nov** | **Discussion** | Present. |

Unit 5 (final project submission) comes **after** the discussion, not before. The form needs a
shareable link and two personal consent decisions — it is a one-shot submission and should wait until
Units 3 and 4 are actually done.

---

## Fallbacks

| If | Then |
|---|---|
| Baseline refusal on Qwen2.5-1.5B is low (< ~60%) | The model barely refuses, so there is little to ablate. Switch to `google/gemma-2-2b-it` (stronger refusal training; needs HF licence acceptance). |
| Ablation barely moves refusal at any layer | Widen `LAYER_BAND` to the full depth, and sweep the last 3–5 post-instruction positions rather than only the final token. |
| Transfer gap ≈ 0 | **This is the result, not a failure.** One shared switch. Write it up as a strengthening of the original paper with a sharper claim. |
| Per-category means too noisy at n = 20 | Pool HarmBench `contextual` behaviors in (adds ~27 cyber, ~28 chem/bio, ~31 misinfo), handling the context string in the prompt. |
| Colab times out mid-run | Cache activations to Drive after section 4; the expensive part is generation, not extraction. |

---

## Scope and handling

Prompts come only from published safety benchmarks; no new harmful prompts are written. Completions
are scored programmatically by refusal-substring matching and are never printed or saved —
`PRINT_COMPLETIONS` is off. No modified weights are written to disk. The deliverable is the structure
result, not a model.

The activation-addition half of the experiment is purely safety-positive: it makes the model more
cautious, and it is what establishes the direction is causal rather than correlational.

---

## Known limitations (state these in the writeup)

- One model, one size. The single-direction result held to 72B; this does not yet.
- Layer sweep narrowed to the middle band at the final post-instruction token, not the full
  (layer, position) grid the paper uses.
- `MAX_NEW_TOKENS = 48` vs the paper's 512. Refusal substrings appear early, but long-form drift is
  unmeasured.
- Refusal-substring scoring only; no Llama Guard safety score (will not fit the free tier alongside
  the model).
- n = 8 per transfer cell — wide intervals. Read the matrix as direction-of-effect, not a precise
  estimate.
- `chemical_biological` caps every category at 28, bounding how stable the per-category means can be.
