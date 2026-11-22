"""Verify all new tokens against the tokenizer and compare all ten originals with the prior run."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer
from freeze import HERE, PRIOR, main as verify_freeze, sha


def rows(path):
    data=[json.loads(line) for line in path.read_text().splitlines()]
    assert len({r["id"] for r in data})==len(data)
    return {r["id"]:r for r in data}


def verify(dest):
    verify_freeze()
    bank={r["id"]:r for r in json.loads((HERE/"bank.json").read_text())}
    manifest=json.loads((dest/"run.json").read_text())
    freeze=json.loads((HERE/"freeze.json").read_text())
    assert datetime.fromisoformat(freeze["frozen_utc"]) < datetime.fromisoformat(manifest["start_utc"]) <= datetime.fromisoformat(manifest["end_utc"])
    assert sha(HERE/"freeze.json")==manifest["freeze_sha256"]
    assert sha(dest/"raw.jsonl")==manifest["raw_sha256"]
    checkpoint=Path.home()/".cache/huggingface/hub"/("models--"+manifest["model"].replace("/","--"))/"snapshots"/manifest["revision"]
    for fname,expected in manifest["checkpoint_sha256"].items():
        assert sha(checkpoint/fname)==expected
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    eos=json.loads((checkpoint/"generation_config.json").read_text())["eos_token_id"]
    eos={eos} if isinstance(eos,int) else set(eos)
    raw=rows(dest/"raw.jsonl")
    assert len(raw)==80 and set(bank)==set(raw)
    previous=rows(PRIOR/"private/replies_qwen.jsonl")
    prior_bank={r["id"]:r["prompt"] for r in json.loads((PRIOR/"source/bank.json").read_text())}
    comparisons=[]
    for rid,b in bank.items():
        r=raw[rid]
        input_ids=tokenizer.apply_chat_template([{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":b["prompt"]}],tokenize=True,add_generation_prompt=True,return_dict=False)
        assert len(input_ids)==r["input_tokens"]
        ids=r["generated_token_ids"]
        assert tokenizer.decode(ids,skip_special_tokens=True)==r["response"]
        assert len(ids)==r["answer_tokens"]<=2048 and not eos.intersection(ids)
        assert (r["finish"]=="stop" and r["stop_id"] in eos) or (r["finish"]=="length" and r["stop_id"] is None and len(ids)==2048)
        assert (r["blocks"]>1)==r["initial_cap"]
        if b["arm"]=="A":
            assert prior_bank[b["source_id"]]==b["prompt"]
            earlier=previous[b["source_id"]]
            assert earlier["input_tokens"]==r["input_tokens"]
            first_difference=next((idx for idx,(a,z) in enumerate(zip(earlier["generated_token_ids"],ids)) if a!=z),None)
            if first_difference is None and len(earlier["generated_token_ids"])!=len(ids):
                first_difference=min(len(earlier["generated_token_ids"]),len(ids))
            comparisons.append(dict(id=rid,source_id=b["source_id"],name=b["name"],prior_tokens=earlier["answer_tokens"],
                                    rerun_tokens=r["answer_tokens"],answer_tokens_identical=ids==earlier["generated_token_ids"],
                                    stop_id_identical=r["stop_id"]==earlier["stop_id"],text_identical=r["response"]==earlier["response"],
                                    first_differing_answer_token_zero_based=first_difference))
    assert len(comparisons)==10 and len({r["name"] for r in comparisons})==10
    summaries=dict(rows=len(raw),unique_names=len({r['name'] for r in bank.values()}),unique_wordings=len({r['arm'] for r in bank.values()}),
                   generated_tokens=sum(r["answer_tokens"] for r in raw.values()),
                   natural_endings=sum(r["finish"]=="stop" for r in raw.values()),continued_after_first_block=sum(r["initial_cap"] for r in raw.values()),
                   first_block_limit=256,maximum_tokens=max(r["answer_tokens"] for r in raw.values()),
                   decoded_full_replies_matching=80,matching_baseline_tokens_including_stop=sum(c["answer_tokens_identical"] and c["stop_id_identical"] for c in comparisons),
                   baseline_comparisons=comparisons)
    (dest/"generation_qc.json").write_text(json.dumps(summaries,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summaries.items() if k!="baseline_comparisons"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=HERE)
    verify(p.parse_args().output)
