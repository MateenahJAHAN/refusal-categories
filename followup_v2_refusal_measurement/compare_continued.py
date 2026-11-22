"""Combine completed short generations and re-read continuations; evaluate all codebooks."""
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
D=ROOT/"data"
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def auc(margin,labels):
    pos=margin[labels];neg=margin[~labels]
    diff=pos[:,None]-neg[None,:]
    return float(np.mean((diff>0)+.5*(diff==0)))
def main():
    short=[json.loads(l) for l in (D/"direct_qwen.jsonl").read_text().splitlines()]
    long=[json.loads(l) for l in (D/"continued_qwen.jsonl").read_text().splitlines()]
    meta=json.loads((D/"continuation_run.json").read_text())
    assert len(short)==125 and len(long)==104 and meta['raw_sha256']==sha(D/'continued_qwen.jsonl')
    with (D/"response_codes.csv").open() as h:
        old={int(r["blind_id"]):r["response_type"] for r in csv.DictReader(h)}
    with (D/"continued_codes.csv").open() as h:
        codes={int(r["blind_id"]):r["response_type"] for r in csv.DictReader(h)}
    assert set(codes)==set(range(1,105)) and set(codes.values())<={"engaged","decline","mixed","indeterminate"}
    full={r["id"]:r for r in long}
    assert set(full)=={r['id'] for r in short if r['finish']=='length'}
    with (ROOT/"item_contrasts.csv").open() as h:
        probe={r["id"]:r for r in csv.DictReader(h) if r["model"]=="qwen" and r["group"]=="xstest"}
    keys=["help_A_first","help_B_first","help_A_last","help_B_last"]
    moves=Counter();totals=Counter();match={k:Counter() for k in keys};stable=defaultdict(Counter)
    outcomes=[];paired=[]
    for r in short:
        k=r["id"];p=probe[k]
        before=old[r["blind_id"]]
        if k in full:
            f=full[k];after=codes[f["blind_id"]]
            assert f["response"].startswith(r["response"])
            cap=f["finish"]=="length";num=f["answer_tokens"]
            moves[(before,after)]+=1
        else:
            after=before;num=r['answer_tokens'];cap=False
        totals[after]+=1
        s=stable[after];n=int(p["help_count"])
        s["total"]+=1;s["length_cap"]+=cap;s["any_probe_decline"]+=n<4;s["always_probe_decline"]+=n==0;s["always_probe_help"]+=n==4
        if after in ("engaged","decline"):
            for name in keys:
                match[name][after+"_"+("help" if int(p[name]) else "decline")]+=1
            outcomes.append((after=="decline",-float(p["mean_help_log_odds"])))
        paired.append(dict(id=k,original_type=before,extended_type=after,final_length_cap=cap,tokens=num))
    with (ROOT/"direct_paired.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    labs=np.array([v[0] for v in outcomes]);margins=np.array([v[1] for v in outcomes])
    pos=np.where(labs)[0];neg=np.where(~labs)[0]
    score=auc(margins,labs)
    rng=np.random.default_rng(20261121)
    vals=[]
    for _ in range(10000):
        sample=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        vals.append(auc(margins[sample],labs[sample]))
    result=dict(n=125,n_extended=104,final_censored=sum(v['length_cap'] for v in stable.values()),
                n_changed=sum(n for (a,b),n in moves.items() if a!=b),label_transitions_capped={f"{a}_to_{b}":n for (a,b),n in moves.items()},
                counts=dict(totals),decisions_by_type={k:dict(v) for k,v in stable.items()},
                codebooks={k:dict(v) for k,v in match.items()}, exploratory_auc=dict(estimate=score,interval=np.percentile(vals,[2.5,97.5]).tolist(),n_decline=len(pos),n_engaged=len(neg)),
                annotations_sha256=sha(D/"continued_codes.csv"))
    (ROOT/"continued_results.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    main()
