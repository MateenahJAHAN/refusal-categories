"""Prepare shuffled reading packets and join two completed readings by anonymous ID."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

BASE=Path(__file__).resolve().parent
BEHAVIOR=("engaged","decline","mixed","indeterminate")
RATING_FIELDS=("blind_id","behavior","political_rationale","note")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records():
    with (BASE/"raw.jsonl").open() as f:
        replies=[json.loads(line) for line in f]
    bank=json.loads((BASE/"bank.json").read_text())
    run=json.loads((BASE/"run.json").read_text())
    assert run["completed"]==80 and digest(BASE/"raw.jsonl")==run["raw_sha256"]
    assert len(bank)==len(replies)==len({r["id"] for r in replies})==len({r["id"] for r in bank})==80
    raw={r["id"]:r for r in replies}
    assert set(raw)=={r["id"] for r in bank}
    cards={}
    for r in bank:
        key=hashlib.sha256(("wording-study/packet/"+r["id"]).encode()).hexdigest()[:14].upper()
        cards[key]={"id":r["id"],"condition":r["arm"],"name":r["name"],"prompt":r["prompt"],"reply":raw[r["id"]]["response"]}
    assert len(cards)==80
    return cards


def write_packet(cards, dest, seed):
    dest.mkdir(exist_ok=False,parents=True)
    keys=sorted(cards)
    random.Random(seed).shuffle(keys)
    rows=[dict(blind_id=k,question=cards[k]["prompt"],reply=cards[k]["reply"]) for k in keys]
    (dest/"packet.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False)+"\n")
    shutil.copyfile(BASE/"coding_rubric.txt",dest/"rubric.txt")


def reading(path, ids):
    with path.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        assert tuple(reader.fieldnames)==RATING_FIELDS
        rows=list(reader)
    assert len(rows)==len(ids) and {r["blind_id"] for r in rows}==set(ids)
    assert all(r["behavior"] in BEHAVIOR and r["political_rationale"] in ("yes","no") and r["note"].strip() for r in rows)
    return {r["blind_id"]:r for r in rows}


def agreement(a,b):
    pairs=Counter(zip(a,b))
    n=len(a)
    same=sum(count for (x,y),count in pairs.items() if x==y)
    chance=sum(count*Counter(b)[key] for key,count in Counter(a).items())/n**2
    return {"agree":same,"n":n,"cohen_kappa":(same/n-chance)/(1-chance) if chance<1 else None,
            "pairs":[{"A":x,"B":y,"n":v} for (x,y),v in sorted(pairs.items())]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for action in ["pack","disputed"]:
        p=sub.add_parser(action);p.add_argument("destination",type=Path)
    sub.add_parser("merge")
    args=parser.parse_args()
    cards=records()
    if args.action=="pack":
        for letter,seed in [("A",832211),("B",181349)]:
            write_packet(cards,args.destination/letter,seed)
        return
    readings=[reading(BASE/f"reader_{letter}.csv",cards) for letter in "AB"]
    decision=lambda r:(r["behavior"],r["political_rationale"])
    a,b=readings
    diff={k:cards[k] for k in cards if decision(a[k])!=decision(b[k])}
    if args.action=="disputed":
        if diff:write_packet(diff,args.destination,291233)
        print(len(diff),"disputed")
        return
    third=reading(BASE/"adjudicated.csv",diff) if diff else {}
    with (BASE/"coded_replies.csv").open("w",newline="") as f:
        fields=["id","name","condition","prompt","reply","behavior","political_rationale","blind_id","reader_A_behavior","reader_A_politics","reader_A_note","reader_B_behavior","reader_B_politics","reader_B_note","adjudicator_note"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for k,card in cards.items():
            final=third.get(k,a[k])
            w.writerow(dict(card,behavior=final["behavior"],political_rationale=final["political_rationale"],blind_id=k,
                            reader_A_behavior=a[k]["behavior"],reader_A_politics=a[k]["political_rationale"],reader_A_note=a[k]["note"],
                            reader_B_behavior=b[k]["behavior"],reader_B_politics=b[k]["political_rationale"],reader_B_note=b[k]["note"],
                            adjudicator_note=third[k]["note"] if k in third else ""))
    metrics={key:agreement([a[k][key] for k in cards],[b[k][key] for k in cards]) for key in ["behavior","political_rationale"]}
    metrics.update(joint_agreement=80-len(diff),n_disputed=len(diff),adjudicated_ids=sorted(third),
                   sha256={f:digest(BASE/f) for f in ["bank.json","raw.jsonl","coding_rubric.txt","reader_A.csv","reader_B.csv","coded_replies.csv"]+(["adjudicated.csv"] if diff else [])})
    (BASE/"coding_audit.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print("Joint agreement:",metrics["joint_agreement"],"/ 80.")

if __name__=="__main__":
    main()
