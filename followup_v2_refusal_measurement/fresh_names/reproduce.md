# Reproduce the fresh-name generations

The 2014 roster is archived in the pinned CORGIS CSV (retrieved by running `python fetch_inputs.py` in the study root). The [download record](source/provenance.json) identifies the exact public Git commit, source credits and file digest. The [selection record](selection.json) lists the 30 sampled people in hash order, the 2014 frame size, and every excluded row. [Question bank](bank.json) · [Protocol](protocol.md) · [Freeze and input hashes](freeze.json)

Use the same Python environment as the preceding BlueDot Qwen inference. It used Python 3.12.11, PyTorch 2.15.0, Transformers 5.2.0 and MPS (float32, SDPA). The pinned Qwen2.5-1.5B-Instruct revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The fresh runner verifies the model, tokenizer, original environment, frozen bank and protocol before generation. Each question starts a new conversation, with greedy decoding and no sampled repeats.

From this directory, set the environment's Python and choose a fresh output directory. Retain the adjacent historical BlueDot study directories for their source bank and shared generation function.

```sh
python freeze.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --output /tmp/bluedot-new-name-replication
HF_HUB_OFFLINE=1 python verify_generation.py --output /tmp/bluedot-new-name-replication
```

In a partial run, repeat the same command on its output directory under the same environment to resume after the last saved batch. The manifest and frozen design must match. One JSON line in `raw.jsonl` contains the reply, generated token IDs, length and stopping token. Each `id` joins to a prompt in the locked bank. These are language-model replies to historical finance questions. Historical dates in a generated reply have not been fact-checked.
