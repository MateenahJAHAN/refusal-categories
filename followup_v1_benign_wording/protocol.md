# BlueDot follow-up protocol · benign wording pilot

Does a benign task receive more explicit refusals, or fewer correct short answers, when its context label uses risk-associated language? The earlier BlueDot study left the mechanism question unresolved. This pilot measures behavior under a controlled wording change.

## Design

- Use 24 newly authored benign tasks, six per topic. Every task has one short expected answer. The content of the task stays identical across the three labels.
- Place a plain-language domain label, a risk-associated domain label, or an irrelevant catalog label in the same prefix position. The three labels have the same word count inside an identical framing sentence for each task. Tokenizer lengths may differ.
- Run all 72 task-label combinations on each of two distinct cached quantized Qwen checkpoints through llama.cpp server 8140. Use greedy generation, a fixed schedule seed, and a shared 28-token limit with thinking disabled. Qwen3 uses the included inference template with an explicitly closed thinking prefill. Qwen3.5 uses its embedded template. No tools are available to either model.
- Include every scheduled completion. Log raw text, finish reason, token counts, exact prompt, model, and prior-to-inference hashes. A request failure stops the run before analysis; resume the missing suffix.

## Endpoints

The primary endpoint is explicit refusal, defined by the leading-response classifier in the frozen runner. A separate endpoint scores the expected short value after basic normalization. Literal format adherence is recorded separately. Review all refusals, truncations, and incorrect values against the raw responses. Report any discrepancy with the frozen scoring transparently.

For each model report denominators, the three condition counts, and matched risk-vs-neutral and risk-vs-irrelevant discordances. A zero observed count on 24 selected trials is a result on that bank. A binomial upper bound is a sampling reference under an independent-trials assumption. This purposively selected pilot does not estimate population prevalence. Correctness near ceiling limits sensitivity. The trial investigates surface wording in benign tasks and makes no claims about internal representations.
