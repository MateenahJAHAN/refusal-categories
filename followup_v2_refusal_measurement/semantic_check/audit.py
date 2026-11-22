"""Independently recount keyed raw outputs, free-text codes, continuations and discrimination."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
D=ROOT/"data"
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):
    return json.loads(p.read_text())
def lines(p):
    return list(map(json.loads,p.read_text().splitlines()))
def read_csv(p):
    with p.open() as fh:return list(csv.DictReader(fh))

def main():
    original= j(ROOT/"results.json")
    results=j(HERE/"results.json")
    bank={r["id"]:r for r in j(D/"items.json")}
    assert Counter(i["group"] for i in bank.values()) == {"xstest":250,"controls":24,"wording":72}
    pairs=read_csv(ROOT/"direct_paired.csv")
    short=lines(D/"direct_qwen.jsonl"); long=lines(D/"continued_qwen.jsonl")
    assert sha(D/"direct_qwen.jsonl")==j(D/"direct_run.json")["raw_sha256"]
    assert sha(D/"continued_qwen.jsonl")==j(D/"continuation_run.json")["raw_sha256"]
    old={int(r["blind_id"]):r["response_type"] for r in read_csv(D/"response_codes.csv")}
    latest={int(r["blind_id"]):r["response_type"] for r in read_csv(D/"continued_codes.csv")}
    completed={r["id"]:r for r in long}
    types={}
    assert len(short)==len({r["id"] for r in short})==125
    assert set(completed)=={r["id"] for r in short if r["finish"]=="length"}
    paired={r["id"]:r for r in pairs}
    for r in short:
        id_=r["id"]
        assert len(r["generated_token_ids"])==r["answer_tokens"]
        response=r
        initial=old[r["blind_id"]]
        if id_ in completed:
            response=completed[id_]
            assert response["finish"]!="length" and response["response"].startswith(r["response"])
            assert len(r["generated_token_ids"])+len(response["added_token_ids"])==response["answer_tokens"]
            types[id_]=latest[response["blind_id"]]
        else:types[id_]=initial
        assert (paired[id_]["original_type"],paired[id_]["extended_type"],int(paired[id_]["tokens"]))==(initial,types[id_],response["answer_tokens"])
    assert Counter(types.values())=={"engaged":107,"mixed":11,"decline":7}
    benchmark={i for i,r in bank.items() if r["group"]=="xstest"}
    controls={i for i,r in bank.items() if r["group"]=="controls"}
    output={"direct":dict(complete=len(types),extended=len(completed),max_tokens=max(int(r["tokens"]) for r in pairs),codes=Counter(types.values())),"models":{}}
    keyed=defaultdict(list)
    for key in ("qwen","smol"):
        archived=lines(D/f"raw_{key}.jsonl")
        fresh=lines(HERE/f"raw_{key}.jsonl")
        assert sha(D/f"raw_{key}.jsonl")==j(D/f"run_{key}.json")["raw_sha256"]
        assert sha(HERE/f"raw_{key}.jsonl")==j(HERE/f"run_{key}.json")["raw_sha256"]
        expected={(i,fr,kind,l,o) for i in benchmark|controls for fr in ("appropriate","predicted") for kind in ("action","letter") for l in (("Answer",) if kind=="action" else ("A","B")) for o in (True,False)}
        tokens=j(HERE/f"run_{key}.json")["token_ids"]
        raw={}
        by=defaultdict(dict)
        for r in archived:
            id_=r["id"];l=r["help_letter"];o=r["help_first"]
            alt="B" if l=="A" else "A"
            assert (r["top_id"] in (tokens[l],tokens[alt]))==(r["top_text"] in (l,alt))
            margin=r["logp_"+alt]-r["logp_"+l]
            by[id_][(l,o)]=margin
            if id_ in benchmark|controls:raw[id_,"appropriate","letter",l,o]=margin
        assert len(archived)==1384 and len(by)==346 and all(len(x)==4 for x in by.values())
        for r in fresh:
            labels=r["help_label"],r["decline_label"]
            assert (r["top_id"] in [tokens[lab] for lab in labels])==(r["top_text"] in labels)
            condition=r["id"],r["frame"],r["kind"],r["help_label"],r["help_first"]
            assert condition not in raw
            raw[condition]=r["logp_decline"]-r["logp_help"]
        assert len(fresh)==2192 and set(raw)==expected
        for grp in ("xstest","controls","wording"):
            letter_switches=0;order_switches=0;ids=set()
            for id_,v in by.items():
                if bank[id_]["group"]!=grp:continue
                flips=sum((v["A",o]>0)!=(v["B",o]>0) for o in (True,False))
                letter_switches+=flips
                if flips:ids.add(id_)
                order_switches+=sum((v[l,False]>0)!=(v[l,True]>0) for l in ("A","B"))
            assert (letter_switches,order_switches,len(ids))==tuple(original[key][grp][m] for m in ["code_flip_pairs","order_flip_pairs","code_flip_items"])
        by_type={}
        for i,descriptor in bank.items():
            if descriptor["group"]!="xstest":continue
            values=by[i]
            by_type.setdefault(descriptor["family"],[]).append(sum((values["A",o]>0)!=(values["B",o]>0) for o in (True,False)))
        rng=np.random.default_rng(20261121)
        total=np.zeros(10000)
        for family in sorted(by_type):
            changes=np.asarray(by_type[family])
            total+=changes[rng.integers(25,size=(10000,25))].sum(axis=1)
        interval=np.percentile(total/500,[2.5,97.5]).tolist()
        assert np.allclose(interval,original[key]["xstest"]["composition_interval"])
        control_numeric=[]
        for i in controls:
            for kind,labels in [("letter",("A","B")),("action",("Answer",))]:
                for l in labels:
                    for o in (True,False):
                        reference=raw[i,"appropriate",kind,l,o];replicate=raw[i,"predicted",kind,l,o]
                        control_numeric.append((reference,replicate))
        dev=max(abs(a-b) for a,b in control_numeric)
        disagreement=sum((a>0)!=(b>0) for a,b in control_numeric)
        output["models"][key]={"fresh":len(fresh),"original":len(archived),"distinct_followup_keyed_cells":len(expected),"identical_control_rechecks":len(control_numeric),"control_decisions_different":disagreement,"largest_control_margin_difference":dev,"original_stratified_interval":interval}
        if key=="qwen":
            for (i,frame,kind,l,o),m in raw.items():
                if i in types:keyed[i,frame+"_"+kind].append(m)
    audit_scores=read_csv(HERE/"free_response_scores.csv")
    assert len(audit_scores)==len(keyed)==500
    for r in audit_scores:
        i=r["id"];k=r["frame"]+"_"+r["kind"];score=np.mean(keyed[i,k])
        assert abs(score-float(r["mean_decline_log_odds"])) < 1e-10 and r["outcome"]==types[i]
    auc={}
    win_matrices={}
    for k in results["auc"]:
        pos=[np.mean(keyed[i,k]) for i in types if types[i]=="decline"]
        neg=[np.mean(keyed[i,k]) for i in types if types[i]=="engaged"]
        wins=sum(float(p>n)+.5*float(p==n) for p in pos for n in neg)
        ordered=sorted(types,key=lambda i:int(i.split("-")[1]))
        a=np.array([np.mean(keyed[i,k]) for i in ordered if types[i]=="decline"])
        b=np.array([np.mean(keyed[i,k]) for i in ordered if types[i]=="engaged"])
        win_matrices[k]=(a[:,None]>b[None,:]).astype(float)+.5*(a[:,None]==b[None,:])
        auc[k]={"favorable_pairs":wins,"all_pairs":len(pos)*len(neg),"auroc":wins/(len(pos)*len(neg))}
        assert abs(auc[k]["auroc"]-results["auc"][k]["estimate"])<1e-12
        for label in ("decline","engaged","mixed"):
            declines=sum(np.mean(keyed[i,k])>0 for i in types if types[i]==label)
            assert int(declines)==results["consensus"][k][label]["decline"]
    rng=np.random.default_rng(2301)
    bs=defaultdict(list)
    for _ in range(100):
        pi=rng.integers(7,size=(100,7))
        ni=rng.integers(107,size=(100,107))
        pc=np.stack([np.bincount(r,minlength=7) for r in pi])
        nc=np.stack([np.bincount(r,minlength=107) for r in ni])
        for k,v in win_matrices.items():
            bs[k].extend(np.einsum('ri,ij,rj->r',pc,v,nc)/(7*107))
    for k,v in bs.items():
        assert np.allclose(np.percentile(v,[2.5,97.5]),results["auc"][k]["interval"])
    paired=np.asarray(bs["predicted_action"])-np.asarray(bs["appropriate_letter"])
    assert np.allclose(np.percentile(paired,[2.5,97.5]),results["primary"]["interval"])
    output["intervals_rechecked_via_multiplicity_weights"]=True
    output["rank_checks"]=auc
    output["prespecified_paired_difference"]=auc["predicted_action"]["auroc"]-auc["appropriate_letter"]["auroc"]
    assert abs(output["prespecified_paired_difference"]-results["primary"]["delta"])<1e-12
    inputs=["PLAN.md","run_semantic.py","analyze_semantic.py","response_sensitivity.py","response_sensitivity.csv","figures.py","audit.py","raw_qwen.jsonl","raw_smol.jsonl","run_qwen.json","run_smol.json","results.json","free_response_scores.csv"]
    original_hashes=j(ROOT/"verified_sha256.json")
    assert all(sha(ROOT/p)==digest for p,digest in original_hashes.items())
    output["sha256"]={f:sha(HERE/f) for f in inputs}
    output["original_manifest_sha256"]=sha(ROOT/"verified_sha256.json")
    figure_manifest = j(HERE / "original_figure_hashes.json")
    assert len(figure_manifest) == 7
    output["figure_sha256"] = {name: sha(ROOT / "figures" / name) for name in figure_manifest}
    assert output["figure_sha256"] == figure_manifest
    (HERE/"audit.json").write_text(json.dumps(output,indent=2)+"\n")
    print("Checked 7,152 forward-pass records, 125 complete replies, 4 AUROCs and 7 figures.")

if __name__=="__main__":main()
