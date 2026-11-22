"""Recompute every archived and follow-up cell before estimating the planned paired contrast."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDITIONS = [("appropriate","letter"),("predicted","letter"),("appropriate","action"),("predicted","action")]

def load(path):
    return json.loads(path.read_text())

def rows(path):
    return list(map(json.loads,path.read_text().splitlines()))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare_ranks(positives, negatives):
    return (positives[:,None] > negatives[None,:]).astype(float) + 0.5*(positives[:,None] == negatives[None,:])

def norm(frame,kind):
    return frame+"_"+kind

def read_model(key, bank):
    original = ROOT/f"data/raw_{key}.jsonl"
    new = HERE/f"raw_{key}.jsonl"
    assert sha(original)==load(ROOT/f"data/run_{key}.json")["raw_sha256"]
    manifest = load(HERE/f"run_{key}.json")
    assert sha(new)==manifest["raw_sha256"]
    assert all(sha(HERE/f)==h for f,h in manifest["inputs"].items())
    assert len(rows(original))==1384 and len(rows(new))==2192
    table = defaultdict(dict)
    for row in rows(original):
        id_=row["id"]
        if id_ not in bank: continue
        label=row["help_letter"]
        alt={"A":"B","B":"A"}[label]
        k=("appropriate","letter",label,row["help_first"])
        table[id_][k]=(row["logp_"+alt]-row["logp_"+label], row["top_text"] in (label,alt))
    for row in rows(new):
        k=(row["frame"],row["kind"],row["help_label"],row["help_first"])
        assert k not in table[row["id"]]
        margin=row["logp_decline"]-row["logp_help"]
        assert np.isfinite(margin) and margin != 0
        table[row["id"]][k]=(margin,row["top_text"] in (row["help_label"],row["decline_label"]))
    assert set(bank)==set(table) and all(len(x)==12 for x in table.values())
    benchmark=[i for i in bank if bank[i]["group"]=="xstest"]
    control=[i for i in bank if bank[i]["group"]=="controls"]
    results={"conditions":{},"frame_flips":{}}
    for frame,kind in CONDITIONS:
        name=norm(frame,kind)
        labels=("A","B") if kind=="letter" else ("Answer",)
        opts=[(frame,kind,lab,first) for first in (True,False) for lab in labels]
        def declines(key_,ident):
            return table[ident][key_][0]>0
        props={}
        for opt in opts:
            lab,first=opt[2:]
            counts=Counter((bank[i]["expected"], "decline" if declines(opt,i) else "help") for i in control)
            props[lab+("_first" if first else "_last")]=dict(help=sum(not declines(opt,i) for i in benchmark),top_target=sum(table[i][opt][1] for i in benchmark),
                        help_control_correct=counts[("help","help")],decline_control_correct=counts[("decline","decline")])
        results["conditions"][name]=dict(forms=props,order_flips=sum(declines((frame,kind,lab,True),i)!=declines((frame,kind,lab,False),i) for i in benchmark for lab in labels),order_pairs=len(benchmark)*len(labels))
        if frame=="appropriate":
            results["frame_flips"][kind]={lab+("_first" if first else "_last"):sum(declines(("appropriate",kind,lab,first),i)!=declines(("predicted",kind,lab,first),i) for i in benchmark) for lab in labels for first in (True,False)}
    return table,results

def main():
    bank={r["id"]:r for r in load(ROOT/"data/items.json") if r["group"] in {"xstest","controls"}}
    assert Counter(x["group"] for x in bank.values())=={"xstest":250,"controls":24}
    tables={}
    results={}
    for key in ("qwen","smol"):
        tables[key],results[key]=read_model(key,bank)
    with (ROOT/"direct_paired.csv").open() as fh:
        coded=list(csv.DictReader(fh))
    assert Counter(r["extended_type"] for r in coded)=={"decline":7,"engaged":107,"mixed":11}
    coded.sort(key=lambda r:int(r["id"].split("-")[1]))
    binary=[r for r in coded if r["extended_type"]!="mixed"]
    positive=np.array([r["extended_type"]=="decline" for r in binary])
    consensus={}
    matrix={}
    individual=[]
    for frame,kind in CONDITIONS:
        name=norm(frame,kind)
        scores={}
        conf=defaultdict(Counter)
        for r in coded:
            id_=r["id"]
            values=[v[0] for k,v in tables["qwen"][id_].items() if k[:2]==(frame,kind)]
            score=float(np.mean(values))
            assert score!=0
            scores[id_]=score
            conf[r["extended_type"]]["decline" if score>0 else "help"]+=1
            individual.append(dict(id=id_,outcome=r["extended_type"],frame=frame,kind=kind,mean_decline_log_odds=score))
        arr=np.array([scores[r["id"]] for r in binary])
        matrix[name]=compare_ranks(arr[positive],arr[~positive])
        consensus[name]={outcome:{"help":c["help"],"decline":c["decline"]} for outcome,c in conf.items()}
    rng=np.random.default_rng(2301)
    boot=defaultdict(list)
    for _ in range(100):
        pos=rng.integers(7,size=(100,7))
        neg=rng.integers(107,size=(100,107))
        for name,pairs in matrix.items():
            boot[name].extend(pairs[pos[:,:,None],neg[:,None,:]].mean(axis=(1,2)))
    base="appropriate_letter";test="predicted_action"
    results["auc"]={name:{"estimate":float(m.mean()),"interval":np.percentile(boot[name],[2.5,97.5]).tolist()} for name,m in matrix.items()}
    results["primary"]={"contrast":f"{test} minus {base}", "delta":float((matrix[test]-matrix[base]).mean()),"interval":np.percentile(np.array(boot[test])-np.array(boot[base]),[2.5,97.5]).tolist(),"seed":2301,"bootstrap":10000,"declines":7,"engaged":107}
    assert abs(results["auc"][base]["estimate"] - load(ROOT/"continued_results.json")["exploratory_auc"]["estimate"])<1e-10
    results["consensus"]=consensus
    (HERE/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    with (HERE/"free_response_scores.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(individual[0]));writer.writeheader();writer.writerows(individual)
    print(json.dumps(dict(primary=results["primary"],auc=results["auc"],consensus=consensus),indent=2))

if __name__=="__main__":
    main()
