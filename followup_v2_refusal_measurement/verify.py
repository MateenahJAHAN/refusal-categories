"""Independent completeness/accounting checks of every scored presentation."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
def blob(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    res=json.loads((ROOT/"results.json").read_text())
    bank=json.loads((DATA/"items.json").read_text())
    assert sorted((sum(x["group"]==g for x in bank),g) for g in ("controls","xstest","wording"))==[(24,"controls"),(72,"wording"),(250,"xstest")]
    with (ROOT/"item_contrasts.csv").open() as h:
        contrast={(r["model"],r["id"]):r for r in csv.DictReader(h)}
    for key in ("qwen","smol"):
        rows=[json.loads(l) for l in (DATA/f"raw_{key}.jsonl").read_text().splitlines()]
        by=defaultdict(dict)
        for r in rows:
            assert r["input_tokens"]>0
            chosen=max(("A","B"),key=lambda c:r["logp_"+c])
            by[r["id"]][r["help_first"],r["help_letter"]]=(r["help_letter"]==chosen)
        assert len(by)==346 and all(len(v)==4 for v in by.values())
        count=defaultdict(lambda:[0,0])
        for item in bank:
            action=by[item["id"]]
            a=int(action[True,"A"]!=action[True,"B"])+int(action[False,"A"]!=action[False,"B"])
            b=int(action[True,"A"]!=action[False,"A"])+int(action[True,"B"]!=action[False,"B"])
            compared=contrast[key,item["id"]]
            assert (a,b,sum(action.values()))==tuple(int(compared[k]) for k in ("code_flips","order_flips","help_count"))
            count[item["group"]][0]+=a;count[item["group"]][1]+=b
        for group,(a,b) in count.items():
            assert (a,b)==tuple(res[key][group][k] for k in ("code_flip_pairs","order_flip_pairs"))
    direct=json.loads((ROOT/"direct_results.json").read_text())
    assert sum(c["items"] for c in direct["response_categories"].values())==125
    cont=json.loads((ROOT/"continued_results.json").read_text())
    rows=[json.loads(l) for l in (DATA/"continued_qwen.jsonl").read_text().splitlines()]
    assert len(rows)==len({r["id"] for r in rows})==cont["n_extended"]
    assert sum(r["finish"]=="length" for r in rows)==cont["final_censored"]
    with (ROOT/"direct_paired.csv").open() as h:
        paired=list(csv.DictReader(h))
    assert len(paired)==len({r["id"] for r in paired})==125
    assert sum(r["original_type"]!=r["extended_type"] for r in paired)==cont["n_changed"]
    assert all(sum(r["extended_type"]==k for r in paired)==v for k,v in cont["counts"].items())
    files=["continuation_protocol.md","continue_direct.py","compare_continued.py","continued_results.json","direct_paired.csv",
           "data/continuation_run.json","data/continued_qwen.jsonl","data/continued_blind.jsonl","data/continued_codes.csv","protocol.md","direct_protocol.md","prepare.py","run.py","direct.py","analyze.py","compare_direct.py","verify.py",
           "replay_prior.py","data/prior_study_replay.json","data/sources.json","data/items.json","data/run_qwen.json","data/run_smol.json","data/raw_qwen.jsonl","data/raw_smol.jsonl",
           "data/direct_run.json","data/direct_qwen.jsonl","data/blind_reading.jsonl","data/response_codes.csv","results.json",
           "direct_results.json","data/annotation_record.json","item_contrasts.csv","benchmark_types.csv","figure1.png"]
    (ROOT/"verified_sha256.json").write_text(json.dumps({p:blob(ROOT/p) for p in files},indent=2)+"\n")
    print("All 2,768 probe presentations 125 direct answers and 104 exact-prefix continuations accounted for.")

if __name__=="__main__":
    main()
