# Reproduce the Qwen generations

Use the original Qwen2.5-1.5B-Instruct snapshot at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` in the Hugging Face cache. The study used PyTorch 2.15.0 and Transformers 5.2.0 on Apple MPS. Checkpoint and tokenizer digests are in `run.json`. From this directory use the following commands; set the replication directory to a new path of your choice.

```sh
python freeze.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --output /tmp/bluedot-name-wording-replication
HF_HUB_OFFLINE=1 python verify_generation.py --output /tmp/bluedot-name-wording-replication
```

`freeze.py` reconstructs the prompts from the historical source and verifies the existing lock. The generator calls the original study’s `direct()` function. A replication keeps the earlier 91-question output and the first 80-question run available for comparison. Its generation metadata, raw model output and token audit are written to its output directory. In `raw.jsonl`, `generated_token_ids` excludes the separately recorded `stop_id`; `generation_qc.json` compares all ten A responses, including their stopping tokens. Batching by question length in characters can change the batch neighbors of a replayed prompt. Matching the original software configuration does not guarantee bitwise-identical answers.

## Reproduce the ratings and figure

Run these commands in this directory to reconstruct the final table from the archived readers’ scores and regenerate the counts, paired comparisons, and figure.

```sh
python3 code_replies.py merge
python3 analysis.py
```

For an independent reading, use `python3 code_replies.py pack /path/to/new/packets` to generate the two shuffled orders. Each reader receives their own question–reply packet and the rubric. Preserve their returned CSVs separately. The archived comparison uses the same opaque ID for both readings of a given reply. The political-refusal endpoint requires both a full decline and an explicit political rationale. [Raw outputs](raw.jsonl) · [Ratings from both readers](coded_replies.csv) · [Agreement and checksums](coding_audit.json)
