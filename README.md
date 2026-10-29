# Does refusal decompose by harm category?

**A mechanistic-interpretability probe of refusal-direction structure in instruct-tuned language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Status (29 Oct 2026): preliminary, one model.** The best-supported reading is **one shared refusal
> switch, which misinformation prompts push about half as hard** — not a separate misinformation
> switch. Removing only the misinformation-specific part of its direction never removed refusal, at any
> of four candidate layers. **Open problem:** no candidate direction passes Arditi et al.'s KL < 0.1
> check on harmless prompts, so the exact transfer numbers are not yet trustworthy. Next: the full
> layer × position search. See [Results (v0.2)](#results-v02).

---

## The question

Arditi et al. (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) showed that refusal
in chat models is mediated by a **single direction** in the residual stream. Recover it by
difference-in-means, ablate it and the model stops refusing; add it and the model refuses harmless
requests. The result held across 13 models up to 72B.

That experiment pooled all harmful prompts into one set — cybercrime, chemical/biological,
misinformation, illegal — and so, by construction, recovered one direction.

Put plainly: is there one main switch for the whole house, or does each room have its own?

### Related work, and what this adds

Others have started to look inside that pool:

- **ValueShift Research (LessWrong, April 2026)** extracted refusal directions for five harm domains
  on Llama-2-7b-chat and compared them by cosine similarity and PCA, concluding there is a shared core
  plus domain-specific parts. They had not yet run cross-domain ablation.
- **Alagharu et al. (2026, [arXiv:2603.13359](https://arxiv.org/abs/2603.13359))** found
  category-aligned directions in a Llama 3 8B *fine-tuned to emit refusal-type tokens*, with categories
  that are refusal types (e.g. "incomplete request") rather than harm types, and used them for steering.

This project adds, for HarmBench harm categories on an **unmodified** instruct model:

1. a permutation null for the geometry, **calibrated on synthetic data** before use;
2. a **behavioural** test — ablate one category's direction, measure refusal on the others;
3. a test of the simpler explanation for any category that behaves differently (v0.2).

## Why either answer is worth having

| Finding | Implication |
|---|---|
| **One shared direction** | Safety training is shallower than the pooled result implies — a single ablation removes refusal across every harm type at once. Argues the training objective itself needs rethinking. |
| **Category-specific** | There is internal structure. Harm types could be monitored or steered independently, and ablating one may leave the others standing. A handle for more robust design. |
| **Separable but interchangeable** | Geometry splits, behaviour does not. The most interesting case, and worth its own writeup. |

There is no null-result risk. Every outcome is a finding.

---

## Method

1. **Extract** a refusal direction *per harm category* by difference-in-means against a shared
   harmless set, plus the pooled direction as the reference point.

   ```
   r_c = mean(activations | harmful prompts in category c) − mean(activations | harmless prompts)
   ```

2. **Compare** the directions by cosine similarity.

3. **Null model** — a permutation test, and the step the result lives or dies on.

4. **Cross-category ablation transfer** — ablate with category *A*'s direction, measure refusal on
   category *B*. The behavioural test, and the headline figure.

5. **Activation addition** — add a category direction to harmless prompts and confirm induced refusal.

6. **Separate switch or weaker refusal?** *(v0.2)* — partial-strength ablation of the pooled
   direction, and ablation of only the category-specific part of a direction (its pooled component
   projected out).

### Why step 3 is load-bearing

Every category direction is a harmful-minus-harmless contrast, so all of them share a generic
"this request is harmful" component. Cosine similarities will look high **whether or not category
structure exists**. Reading a high number as "one shared direction" would be wrong.

So the same harmful pool is randomly relabelled into pseudo-categories and the statistic recomputed.
That is the value expected when there is *no* category structure at all. Real categories count as
distinct only if they sit clearly below it.

**What it cannot show.** Each direction also carries its prompts' *topic* — cybercrime prompts share
computing words, chem/bio prompts share chemistry words. Real categories can sit below the null even if
refusal is one switch. So this step shows the categories **differ**; only the behavioural tests (4 and
6) show whether **refusal** differs.

### What calibrating the test changed

The permutation test was validated on synthetic activations with the answer built in, before any real
data was touched:

| Synthetic scenario | Test fires (200 replications × 150 permutations) | |
|---|---|---|
| `shared_frac = 1.0` — no structure at all | **4.5%** | calibrated (target ~5%) |
| `shared_frac = 0.85` — weak structure | 100% | |
| `shared_frac = 0.0` — fully category-specific | 100% | |

(The v0.1 `results.json` records 6.7%, from a shorter 60-replication run — 4 of 60. The 200-replication
figure above is from the v0.2 notebook's calibration cell, which runs on CPU in seconds.)

The test is calibrated **but extremely powerful** — it flags even trivial structure. So a significant
p-value is cheap here, and the **effect size and transfer matrix carry the conclusion**, not `p`. The
verdict logic in the notebook is written that way deliberately.

On the first calibration run the detector flagged structure in data built with none. It looked like a
bug; 200 replications showed the test was correctly calibrated and that run was an unlucky seed. A
single run would have led to either chasing a non-existent bug or shipping a false positive.

---

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` — ungated, ~3 GB fp16, fits a free Colab T4 |
| Harmful | [HarmBench](https://github.com/centerforaisafety/HarmBench) `standard` behaviors, 4 categories balanced to 28 each |
| Harmless | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) instructions with no input field |
| Split | 20 train / 8 test per category, seed `20261024`; layer chosen on a separate 40-prompt selection set (v0.2) |
| Runtime | ~40 min end to end (v0.1); v0.2 adds section 11b |

Categories: `cybercrime_intrusion`, `illegal`, `misinformation_disinformation`,
`chemical_biological`. Excluded as categories: `copyright` (a functional category, not a harm type),
`harmful` (catch-all), `harassment_bullying` (only 19 standard behaviors — too thin for a stable mean).
In v0.2, `harmful` + `harassment_bullying` (40 prompts) are used **only** to choose the layer, so the
test prompts are never seen before the final measurements.

### Running it

Open `notebooks/refusal_categories.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**,
and Run All. No API keys required; the datasets are fetched from their official repositories at
runtime.

---

## Results (v0.2)

Run on 29 Oct 2026, Colab T4. Layer chosen on the separate selection set: **20** (layers 20, 22, 23
all reached 0% on it; ties go to the earliest; layer 16 gave 5%).

**Geometry.** Mean cross-category cosine **0.839** vs null **0.983** (95% range 0.971–0.990); 0 of 300
random splits as low, **p = 0.0033** (the floor), effect **0.144**. Calibration: 4.5% false-positive
rate over 200 replications.

**Transfer at layer 20.** Every direction, misinformation's included, removed refusal on every
category (all cells 0%). Activation addition on harmless prompts: cyber 53%, illegal 81%,
misinformation 53%, chem/bio 53%, pooled 59%.

**Separate switch or weaker refusal (section 11b, layer 20).**

| Check | Result |
|---|---|
| How hard each category pushes the pooled direction | cyber +38.9, illegal +32.3, chem/bio +32.5, **misinfo +17.8** |
| Partial pooled ablation (25 / 50 / 75%) | 0% everywhere — uninformative at this layer |
| Ablate only the category-specific part | refusal stays: cyber 100%, illegal 100%, **misinfo 100%**, chem/bio 88% |

**Robustness across layers (section 11c).**

| Layer | KL on harmless (want < 0.1) | Ablate misinfo → others refused | Misinfo-specific part only → misinfo refused |
|---|---|---|---|
| 16 | 0.386 | 67% | 100% |
| 20 | 1.677 | 0% | 100% |
| 22 | 0.132 | 8% | 100% |
| 23 | 0.150 | 4% | 100% |

### Reading

- **Solid across all four layers:** the misinformation-specific component carries no refusal of its
  own. Misinformation refusal lives on the shared direction.
- **Misinformation pushes that shared direction about half as hard** as the other categories. That
  explains the v0.1 table: misinformation's direction only partly overlaps the shared one, so removing
  it removes enough of the switch to stop the weak misinformation refusal, but not the stronger ones.
- **Not yet trustworthy:** the exact transfer numbers. They swing with the layer (67% at 16, 0% at
  20), and **no candidate passes the KL < 0.1 filter** — removing these directions also disturbs how
  the model answers harmless questions, so some "0% refusal" may be damaged output rather than
  removed refusal. Substring scoring cannot tell the two apart.
- **Next:** search every layer × the last five post-instruction positions, as Arditi et al. do, keep
  only directions passing KL, and re-run the misinformation tests on that direction. Then a second
  model.

---

## Results (v0.1, superseded)

Run on 24 Oct 2026 (`results.json`, `fig1`–`fig3` in this repo are from this run). Selected layer:
**16** of 28. Kept for the record.

> **Why superseded.** v0.1 chose the layer using the same 8 test prompts per category that the
> results below are scored on, and its layer-16 direction fails the KL check (0.386). Its headline —
> that misinformation does not transfer — is explained in v0.2 by misinformation's weaker refusal,
> not a separate switch.

**Baseline.** The model refuses **97%** of held-out harmful prompts and **3%** of harmless ones, so
there is real refusal to study (my pre-registered prediction was 70–90%).

**Geometry.** Cybercrime, illegal and chem/bio directions are close (cosine 0.89–0.92). Misinformation
sits apart (0.54–0.62 with the rest). Mean cross-category cosine is **0.74** against a null of
**0.965** (95% range 0.943–0.979); none of 300 random splits was as low, **p < 0.0033** (the floor
for 300 draws), effect size **0.23**. *Per the caveat above, this shows the categories differ — not yet
that refusal does.*

**Behaviour — refusal on held-out prompts after ablating each direction** (n = 8 per cell):

| Ablated ↓ / measured → | cyber | illegal | misinfo | chem/bio |
|---|---|---|---|---|
| none | 100% | 100% | 100% | 88% |
| cyber | 0% | 12% | 0% | 0% |
| illegal | 0% | 0% | 0% | 0% |
| **misinfo** | **75%** | **50%** | 12% | **75%** |
| chem/bio | 0% | 0% | 0% | 0% |
| pooled | 0% | 0% | 0% | 0% |

Ablating cyber, illegal or chem/bio removes refusal on **every** category, misinformation included.
Ablating misinformation removes refusal on misinformation but leaves the others refused **16 of 24**
times, against **1 of 72** for the other single-category rows. (No p-value is given here: every row is
scored on the same 8 prompts per column, so the cells are not independent samples.)

**Activation addition** (refusal induced on harmless prompts): illegal 100%, chem/bio 88%, pooled 88%,
cyber 81%, **misinformation 56%**.

### Reading (as of v0.1)

Cybercrime, illegal and chem/bio behaved as one shared switch; misinformation's direction did not
transfer. v0.1 already noted this could be weaker refusal on one switch rather than a separate switch;
v0.2 (above) supports the weaker-refusal reading.

---

## Repository

```
notebooks/refusal_categories.ipynb   the full experiment (v0.2)
results.json                         v0.1 run, 24 Oct 2026 (v0.2 numbers are in the README and LOG)
fig1–fig3 *.png                      v0.1 figures
docs/PLAN.md                         scope, schedule, fallbacks, limitations
docs/LOG.md                          assumptions, cruxes, predictions recorded before testing
```

### Changelog

**v0.2 (29 Oct 2026)** — fixes from review; run on Colab T4 the same day.

- Layer now chosen on a separate selection set, not the test prompts.
- New section 11b: separate switch vs weaker refusal.
- New section 11c: misinformation tests repeated at every candidate layer, with Arditi et al.'s
  KL < 0.1 check on harmless prompts. No candidate passed.
- Permutation p-value reported as (k+1)/(n+1), never 0.
- Calibration uses 200 replications in the notebook (v0.1 notebook used 60).
- Removed the refusal substring `"i'm an"`, which also matches non-refusals ("I'm an expert…").
- Stated that the cosine test cannot separate topic from refusal.
- Related work added; the "never been measured" claim corrected.

---

## Scope and handling

Prompts come only from published safety benchmarks; no new harmful prompts were written. Completions
are scored programmatically by refusal-substring matching and are **never printed or saved** —
`PRINT_COMPLETIONS` is off by default and should stay off. No modified weights are written to disk or
distributed.

The deliverable is the structure result, not a model. The activation-addition half of the experiment
is purely safety-positive: it makes the model *more* cautious, and it is what establishes the
direction is causal rather than merely correlated.

---

## Limitations

- One model, one size. The single-direction result held to 72B; this does not yet. A second model and
  a larger one are part of the minimum claim, not an extension.
- Layer sweep narrowed to the middle band at the final post-instruction token, not the full
  (layer, position) grid.
- `MAX_NEW_TOKENS = 48` against the paper's 512. Refusal substrings appear early, but long-form drift
  is unmeasured.
- Refusal-substring scoring only; no Llama Guard safety score.
- n = 8 per transfer cell — wide intervals. The matrix shows direction of effect, not a precise
  estimate.
- `chemical_biological` caps every category at 28, bounding how stable the per-category means can be.
- The cosine test detects any difference between categories, including topic vocabulary; it cannot by
  itself show that refusal differs.
- v0.1 chose the layer on the test prompts (fixed in v0.2). The v0.2 selection set is a different
  prompt distribution, so its chosen layer may be slightly less than optimal for the four categories.
- **No direction found so far passes the KL < 0.1 check.** Ablation also disturbs harmless behaviour,
  and substring scoring cannot separate removed refusal from damaged output. Transfer numbers are
  provisional until the full (layer, position) search finds a clean direction.

---

## Citation

```bibtex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel
          and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```

Also: [HarmBench](https://arxiv.org/abs/2402.04249) (Mazeika et al., 2024);
[Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al., 2023).

Related: Alagharu et al., *From Refusal Tokens to Refusal Control: Discovering and Steering
Category-Specific Refusal Directions* (2026, [arXiv:2603.13359](https://arxiv.org/abs/2603.13359));
ValueShift Research, *Experiments on refusal shape in LLMs* (LessWrong, April 2026,
[link](https://www.lesswrong.com/posts/qRuTqqyHhwpwbdzMf/experiments-on-refusal-shape-in-llms)).

## Author

**Mateenah Jahan** — [GitHub](https://github.com/MateenahJAHAN) ·
[Portfolio](https://mateenahjahan.github.io) · [LinkedIn](https://linkedin.com/in/mateenah-jahan)

Developed during the [BlueDot Impact](https://bluedot.org) Technical AI Safety Project sprint,
October 2026.

## License

MIT — see [LICENSE](LICENSE). The datasets are the work of their original authors under their own
licenses; this repository redistributes none of them.
