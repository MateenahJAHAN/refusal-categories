"""Verify native chat inputs, all generated tokens, end conditions and the shared question bank."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from transformers import AutoTokenizer
from freeze import HERE, PRIOR, QWEN, main as verify_design, sha


def load(path):
    return json.loads(path.read_text())


def verify(dest):
    verify_design()
    bank = {q["id"]: q for q in load(HERE / "bank.json")}
    manifest = load(dest / "run.json")
    frozen = load(HERE / "freeze.json")
    previous = load(PRIOR / "private/run_smol.json")
    assert datetime.fromisoformat(frozen["frozen_utc"]) < datetime.fromisoformat(manifest["start_utc"]) <= datetime.fromisoformat(manifest["end_utc"])
    assert sha(HERE / "freeze.json") == manifest["freeze_sha256"]
    assert sha(dest / "raw.jsonl") == manifest["raw_sha256"]
    for key in ("model", "revision", "checkpoint_sha256", "torch", "transformers", "batch_direct", "device", "dtype", "attention"):
        assert manifest[key] == previous[key], key
    checkpoint = Path.home() / ".cache/huggingface/hub" / ("models--" + manifest["model"].replace("/", "--")) / "snapshots" / manifest["revision"]
    for path, expected in manifest["checkpoint_sha256"].items():
        assert sha(checkpoint / path) == expected, path
    for path, key in [("generation_config.json", "generation_config_sha256"), ("vocab.json", "vocab_sha256"),
                      ("merges.txt", "merges_sha256"), ("special_tokens_map.json", "special_tokens_map_sha256")]:
        assert sha(checkpoint / path) == manifest[key], path
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    eos = load(checkpoint / "generation_config.json")["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    context_limit = load(checkpoint / "config.json")["max_position_embeddings"]
    replies = [json.loads(line) for line in (dest / "raw.jsonl").read_text().splitlines()]
    assert len(replies) == manifest["completed"] == len(set(r["id"] for r in replies)) == 180
    assert {r["id"] for r in replies} == set(bank)
    for reply in replies:
        question = bank[reply["id"]]["prompt"]
        chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
        prompt_ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False)
        assert len(prompt_ids) == reply["input_tokens"]
        answer_ids = reply["generated_token_ids"]
        limit = min(context_limit - len(prompt_ids), 2048)
        assert len(answer_ids) == reply["answer_tokens"] <= limit
        assert not eos.intersection(answer_ids)
        assert tok.decode(answer_ids, skip_special_tokens=True) == reply["response"]
        assert (reply["blocks"] > 1) == reply["initial_cap"]
        if reply["finish"] == "stop":
            assert reply["stop_id"] in eos
        else:
            assert reply["finish"] == "length" and reply["stop_id"] is None and len(answer_ids) == limit
    qwen_ids = {r["id"] for r in load(QWEN / "bank.json")}
    assert set(bank) == qwen_ids
    result = dict(rows=len(replies), unique_names=len({r["name"] for r in bank.values()}), wordings=6,
                  qwen_bank_byte_identical=(HERE / "bank.json").read_bytes() == (QWEN / "bank.json").read_bytes(),
                  qwen_id_overlap=len(qwen_ids.intersection(bank)),
                  model=manifest["model"], revision=manifest["revision"], freeze_utc=frozen["frozen_utc"],
                  run_start_utc=manifest["start_utc"], run_end_utc=manifest["end_utc"],
                  finish_counts=dict(Counter(r["finish"] for r in replies)),
                  continued_after_first_block=sum(r["initial_cap"] for r in replies),
                  first_block_limit=256, maximum_allowed_new_tokens=2048,
                  generated_tokens=sum(r["answer_tokens"] for r in replies),
                  maximum_tokens=max(r["answer_tokens"] for r in replies),
                  decoded_replies_matching=len(replies), raw_sha256=manifest["raw_sha256"])
    assert result["qwen_bank_byte_identical"]
    (dest / "generation_qc.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    verify(parser.parse_args().output)
