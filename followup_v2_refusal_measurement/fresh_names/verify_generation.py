"""Check the prospective design lock, actual chat input lengths and every generated token."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from transformers import AutoTokenizer
from freeze import HERE, main as verify_design, normal, sha


def load(path):
    return json.loads(path.read_text())


def verify(dest):
    verify_design()
    bank = {q["id"]: q for q in load(HERE / "bank.json")}
    manifest = load(dest / "run.json")
    lock = load(HERE / "freeze.json")
    assert datetime.fromisoformat(lock["frozen_utc"]) < datetime.fromisoformat(manifest["start_utc"]) <= datetime.fromisoformat(manifest["end_utc"])
    assert sha(HERE / "freeze.json") == manifest["freeze_sha256"]
    assert sha(dest / "raw.jsonl") == manifest["raw_sha256"]
    checkpoint = Path.home() / ".cache/huggingface/hub" / ("models--" + manifest["model"].replace("/", "--")) / "snapshots" / manifest["revision"]
    for path, expected in manifest["checkpoint_sha256"].items():
        assert sha(checkpoint / path) == expected
    for path, key in [("generation_config.json", "generation_config_sha256"), ("vocab.json", "vocab_sha256"), ("merges.txt", "merges_sha256")]:
        assert sha(checkpoint / path) == manifest[key]
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    eos = load(checkpoint / "generation_config.json")["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    replies = [json.loads(line) for line in (dest / "raw.jsonl").read_text().splitlines()]
    assert len(replies) == manifest["completed"] == len(set(r["id"] for r in replies)) == 180
    assert {r["id"] for r in replies} == set(bank)
    for r in replies:
        q = bank[r["id"]]
        messages = [{"role":"system", "content":"You are a helpful assistant."}, {"role":"user", "content":q["prompt"]}]
        prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        assert len(prompt_ids) == r["input_tokens"]
        ids = r["generated_token_ids"]
        assert len(ids) == r["answer_tokens"] <= 2048
        assert not eos.intersection(ids) and tok.decode(ids, skip_special_tokens=True) == r["response"]
        assert (r["blocks"] > 1) == r["initial_cap"]
        if r["finish"] == "stop":
            assert r["stop_id"] in eos
        else:
            assert r["finish"] == "length" and r["stop_id"] is None and len(ids) == 2048
    old_names = {r["name"] for r in load(HERE.parent / "name_wording/bank.json")}
    new_names = {r["name"] for r in bank.values()}
    assert len(new_names)==30 and old_names.isdisjoint(new_names)
    assert all({r["arm"] for r in bank.values() if r["name"] == name} == set("ABCDEF") for name in new_names)
    prior_counts = {}
    for location in ["../data/items.json", "../independent_benign/source/bank.json", "../name_wording/bank.json"]:
        existing = load(HERE / location)
        questions = [(r["id"], f" {normal(r['prompt'])} ") for r in existing]
        overlap = [(name, qid) for name in sorted(new_names) for qid, text in questions if f" {normal(name)} " in text]
        assert not overlap
        prior_counts[location] = dict(prompts=len(questions), normalized_person_name_matches=overlap)
    result = dict(rows=len(replies), unique_names=len(new_names), wordings=6, historical_name_overlap=0, prior_prompt_checks=prior_counts,
                  model=manifest["model"], revision=manifest["revision"], freeze_utc=lock["frozen_utc"],
                  run_start_utc=manifest["start_utc"], run_end_utc=manifest["end_utc"],
                  finish_counts=dict(Counter(r["finish"] for r in replies)),
                  continued_after_first_block=sum(r["initial_cap"] for r in replies),
                  first_block_limit=256, generated_tokens=sum(r["answer_tokens"] for r in replies),
                  maximum_tokens=max(r["answer_tokens"] for r in replies),
                  decoded_replies_matching=len(replies), raw_sha256=manifest["raw_sha256"])
    (dest / "generation_qc.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    verify(parser.parse_args().output)
