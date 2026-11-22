# Reproduce the SmolLM2 replies

The [question bank](bank.json) is byte-identical to the [Qwen bank](../fresh_names/bank.json). The [protocol](protocol.md) defines the paired endpoint and across-checkpoint test. The [design lock](freeze.json) records the input and code digests. Keep the adjacent BlueDot study folders alongside this directory.

Use Python 3.12.11 with PyTorch 2.15.0, Transformers 5.2.0 and an Apple Metal device. The original experiment ran unquantized FP32 weights in batches of eight with the SDPA implementation. Cache the full [SmolLM2-1.7B-Instruct snapshot](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) at the pinned revision before running offline. Run these commands inside this directory, using a fresh path for the replication outputs.

```sh
python freeze.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --output /tmp/bluedot-smol-replication
HF_HUB_OFFLINE=1 python verify_generation.py --output /tmp/bluedot-smol-replication
```

The generator calls the original study's shared direct-response routine. It records one JSON line per completed question, with its reply, answer tokens, stop token, stop reason and continuation count. Batches are written incrementally and an interrupted run can resume from its saved IDs under the same software and frozen design. The token audit reconstructs the native chat input lengths, decodes every stored answer, checks stopping behavior, verifies checkpoint hashes and confirms the match to the Qwen bank.


## Recompute the comparison

From this directory, run `python score.py merge` to join both archived readings, then `python compare_models.py` to reproduce the per-name transitions, within-checkpoint results, cross-checkpoint paired test and figure. The analysis checks the raw reply hashes, rubric hashes and exact prompt IDs and text across the two runs. To repeat blinded scoring on these outputs, run `python score.py pack /path/to/new-packets` and give each reader only their own packet and rubric. Repeating generation with new replies requires a new reading. [Frozen rubric](../name_wording/coding_rubric.txt) · [All final classifications](coded_replies.csv) · [Paired results](interaction.csv)
