# Does refusal decompose by harm category?

**A mechanistic-interpretability probe of refusal-direction structure in instruct-tuned language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Status: harness built and validated. Results pending.**
> The measurement instrument has been calibrated against known ground truth. The model runs are next.

---

## The question

Arditi et al. (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) showed that refusal
in chat models is mediated by a **single direction** in the residual stream. Recover it by
difference-in-means, ablate it and the model stops refusing; add it and the model refuses harmless
requests. The result held across 13 models up to 72B.

That experiment pooled all harmful prompts into one set — cybercrime, chemical/biological,
misinformation, illegal — and so, by construction, recovered one direction.

> **Whether distinct harm categories have distinct refusal directions has never been measured.**

Put plainly: is there one main switch for the whole house, or does each room have its own?

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

### Why step 3 is load-bearing

Every category direction is a harmful-minus-harmless contrast, so all of them share a generic
"this request is harmful" component. Cosine similarities will look high **whether or not category
structure exists**. Reading a high number as "one shared direction" would be wrong.

So the same harmful pool is randomly relabelled into pseudo-categories and the statistic recomputed.
That is the value expected when there is *no* category structure at all. Real categories count as
distinct only if they sit clearly below it.

### What calibrating the test changed

The permutation test was validated on synthetic activations with the answer built in, before any real
data was touched:

| Synthetic scenario | Test fires | |
|---|---|---|
| `shared_frac = 1.0` — no structure at all | **6.7%** | calibrated (target ~5%) |
| `shared_frac = 0.85` — weak structure | 100% | |
| `shared_frac = 0.0` — fully category-specific | 100% | |

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
| Split | 20 train / 8 val per category, seed `20261024` |
| Runtime | ~40 min end to end |

Categories: `cybercrime_intrusion`, `illegal`, `misinformation_disinformation`,
`chemical_biological`. Excluded: `copyright` (a functional category, not a harm type), `harmful`
(catch-all), `harassment_bullying` (only 19 standard behaviors — too thin for a stable mean).

### Running it

Open `notebooks/refusal_categories.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**,
and Run All. No API keys required; the datasets are fetched from their official repositories at
runtime.

---

## Repository

```
notebooks/refusal_categories.ipynb   the full experiment, 18 cells
docs/PLAN.md                         scope, schedule, fallbacks, limitations
docs/LOG.md                          assumptions, cruxes, predictions recorded before testing
```

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

## Author

**Mateenah Jahan** — [GitHub](https://github.com/MateenahJAHAN) ·
[Portfolio](https://mateenahjahan.github.io) · [LinkedIn](https://linkedin.com/in/mateenah-jahan)

Developed during the [BlueDot Impact](https://bluedot.org) Technical AI Safety Project sprint,
October 2026.

## License

MIT — see [LICENSE](LICENSE). The datasets are the work of their original authors under their own
licenses; this repository redistributes none of them.
