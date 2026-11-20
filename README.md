# Does refusal decompose by harm category?

An exploratory mechanistic-interpretability study of refusal-related representations in Qwen2.5-1.5B-Instruct, undertaken for the BlueDot Impact Project Sprint.

> **Corrected v0.2 results (29 October 2026).** All four candidates failed the harmless-prompt KL check added in a later robustness analysis. The mechanism remains unresolved. [Archived v0.2 measurements](https://huggingface.co/datasets/Mateenah/refusal-categories-results/blob/917856ed158f4dcb70ade8b5a49f0cf4accce71c/results_v0.2.json) · [When the check was added](https://github.com/MateenahJAHAN/refusal-categories/commit/cfed840211f54de0fd277e623e404918d8882a0b)

## KL gate replay

The [executable check](quality_gate/check.py) applied the archived cutoff of KL < 0.1 to all four recorded benign-output measurements. Zero candidates passed. The criterion was added after candidate selection and inspection of the behavioral results. Those results leave the mechanism unresolved. [Input measurements](quality_gate/archived_kl.json) · [Replay output](quality_gate/replay.json)

Run `python3 quality_gate/check.py` from the repository root. The report requires one finite, nonnegative KL measurement per archived layer and stops on incomplete or malformed input. The input contains only the four KL measurements extracted from the pinned v0.2 archive above, whose SHA-256 is `84c81a69e84b7b23856257811f9fa82e38b5406b74b7219b747be6e06fd93e55`.

## Behavioral follow-up · November 2026

A separate pilot tested 24 benign tasks under plain-language, risk-associated, and irrelevant labels on two evaluable checkpoints. Both models returned zero explicit refusals across the 144 completed trials. The follow-up includes a matched design, raw responses, failure diagnostics, and figures. [Read the experiment and results](followup_v1_benign_wording/README.md)

![BlueDot follow-up counts of refusals and correct short answers](followup_v1_benign_wording/outcomes.png)

[Matched wording design](followup_v1_benign_wording/design.png)

## Question and design

This study asks whether refusal-related representations vary across four harm categories. It compared category-level geometry against a permutation null and evaluated behavior following interventions. V0.2 used a separate selection set and reserved eight prompts per category for behavioral evaluation. The general approach builds on [Arditi et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.11717).

## What the corrected run supports

- **Category-associated geometry.** The observed mean cross-category cosine similarity was 0.8391, compared with a mean of 0.9829 under the permutation null. The add-one permutation p-value was 0.0033 across 300 permutations. Topic and wording can contribute to that difference.
- **Interpretation limits.** The behavioral measurements used one model, repeated prompts across candidates, small held-out groups, and refusal-phrase scoring. The scoring could count degraded output as a change in refusal. The synthetic calibration characterizes the permutation test under the simulated conditions.

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
