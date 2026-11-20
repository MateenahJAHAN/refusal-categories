# Does refusal decompose by harm category?

An exploratory mechanistic-interpretability study of refusal-related representations in Qwen2.5-1.5B-Instruct, undertaken for the BlueDot Impact Project Sprint.

> **Corrected v0.2 results (29 October 2026).** All four candidate directions failed the prespecified quality check on harmless prompts. The study did not establish either a shared or a category-specific refusal mechanism. [Archived v0.2 measurements](https://huggingface.co/datasets/Mateenah/refusal-categories-results/blob/917856ed158f4dcb70ade8b5a49f0cf4accce71c/results_v0.2.json)

## Behavioral follow-up · November 2026

A separate pilot tested 24 benign tasks under plain-language, risk-associated, and irrelevant labels on two evaluable checkpoints. Both models returned zero explicit refusals across the 144 completed trials. The follow-up includes a matched design, raw responses, failure diagnostics, and figures. [Read the experiment and results](followup_v1_benign_wording/README.md)

## Question and design

This study asks whether refusal-related representations vary across four harm categories. It compared category-level geometry against a permutation null and evaluated behavior following interventions. V0.2 used a separate selection set and reserved eight prompts per category for behavioral evaluation. The general approach builds on [Arditi et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.11717).

## What the corrected run supports

- **Category-associated geometry.** The observed mean cross-category cosine similarity was 0.8391, compared with a mean of 0.9829 under the permutation null. The add-one permutation p-value was 0.0033 across 300 permutations. Topic and wording can contribute to that difference.
- **Failed intervention quality check.** The four harmless-prompt KL values were 0.3861, 1.6765, 0.1324, and 0.1499 against a threshold of less than 0.1. The qualifying set was empty. The exported exploratory selection identifies a candidate; it does not indicate that the candidate qualified.
- **Behavioral results are provisional.** The study uses one model, repeated prompts across candidates, small held-out groups, and refusal-phrase scoring. That scoring cannot distinguish lost refusal from degraded output. The synthetic calibration only characterizes the permutation test under the simulated conditions.

A future validation study would need a frozen protocol, topic-matched evaluation examples, and independent scoring of output quality. An independent model would be needed for a cross-model claim.

## Research artifacts

| Artifact | Contents |
| --- | --- |
| [Corrected v0.2 results](https://huggingface.co/datasets/Mateenah/refusal-categories-results/blob/917856ed158f4dcb70ade8b5a49f0cf4accce71c/results_v0.2.json) | Dated measurements and recorded quality checks |
| [Source notebook](notebooks/refusal_categories.ipynb) | Versioned study code |
| [Historical v0.1 results](results.json) and the unsuffixed figures | The 24 October run, superseded because candidate selection reused the evaluation prompts |
| [Study plan](docs/PLAN.md) and [log](docs/LOG.md) | Design history, assumptions, and changes |

The public artifacts contain aggregate measurements. The project evaluated prompts from [HarmBench](https://arxiv.org/abs/2402.04249) and [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

[MIT license](LICENSE)
