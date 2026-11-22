"""Join the frozen direct-response readings to the counterbalanced probe after coding."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    data=ROOT/"data"
    direct=[json.loads(l) for l in (data/"direct_qwen.jsonl").read_text().splitlines()]
    meta=json.loads((data/"direct_run.json").read_text())
    assert len(direct)==125 and meta["raw_sha256"]==sha(data/"direct_qwen.jsonl")
    assert all(v==sha(ROOT/k) for k,v in meta["sha256"].items())
    with (data/"response_codes.csv").open() as f:
        labels={int(r["blind_id"]):r["response_type"] for r in csv.DictReader(f)}
    assert set(labels)==set(range(1,126)) and set(labels.values())<= {"engaged","decline","mixed","indeterminate"}
    with (ROOT/"item_contrasts.csv").open() as f:
        probed={r["id"]:r for r in csv.DictReader(f) if r["model"]=="qwen" and r["group"]=="xstest"}
    ordered=["help_A_first","help_B_first","help_A_last","help_B_last"]
    strata=sorted(set(labels.values()))
    stats={s:dict(items=0,any_classifier_decline=0,stable_classifier_decline=0,stable_classifier_help=0,
                  any_code_flip=0,limit_reached=0, classifier_declines=0, classifier_presentations=0) for s in strata}
    agree={k:Counter() for k in ordered}
    by_type={}
    for r in direct:
        cat=labels[r["blind_id"]]; s=stats[cat]; p=probed[r["id"]]; h=int(p["help_count"])
        s["items"]+=1; s["any_classifier_decline"]+=h<4; s["stable_classifier_decline"]+=h==0
        s["stable_classifier_help"]+=h==4;s["any_code_flip"]+=int(p["code_flips"])>0
        s["limit_reached"]+=r["finish"]=="length";s["classifier_declines"]+=4-h;s["classifier_presentations"]+=4
        by_type.setdefault(p["family"],Counter())[cat]+=1
        for k in ordered:
            if cat in ("decline","engaged"):
                agree[k][cat+"_"+("help" if int(p[k]) else "decline")]+=1
    res=dict(n=len(direct),length_caps=sum(r["finish"]=="length" for r in direct),response_categories=stats,
             separate_codebooks={k:dict(v) for k,v in agree.items()},by_type={k:dict(v) for k,v in by_type.items()},
             annotation_sha256=sha(data/"response_codes.csv"),blind_input_sha256=sha(data/"blind_reading.jsonl"))
    (ROOT/"direct_results.json").write_text(json.dumps(res,indent=2)+"\n")

if __name__=="__main__":
    main()
