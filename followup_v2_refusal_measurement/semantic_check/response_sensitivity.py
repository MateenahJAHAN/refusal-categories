"""Post-hoc checks of case mix, the seven events, and the mixed-reply definition."""
import csv
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def main():
    bank={r["id"]:r for r in json.loads((ROOT/"data/items.json").read_text())}
    with (HERE/"free_response_scores.csv").open() as fh: raw=list(csv.DictReader(fh))
    keys=("appropriate_letter","predicted_action")
    items={}
    for r in raw:
        d=items.setdefault(r["id"], {"id":r["id"],"family":bank[r["id"]]["family"],"outcome":r["outcome"]})
        k=r["frame"]+"_"+r["kind"]
        if k in keys:d[k]=float(r["mean_decline_log_odds"])
    data=list(items.values())
    def compare(name,pos,neg,match_types=False):
        pairs=[(p,n) for p in pos for n in neg if not match_types or n["family"]==p["family"]]
        a,b=[sum((p[k]>n[k])+.5*(p[k]==n[k]) for p,n in pairs)/len(pairs) for k in keys]
        return dict(contrast=name,declines=len({p["id"] for p,n in pairs}),engaged=len({n["id"] for p,n in pairs}),pairs=len(pairs),appropriate_letter_auc=a,predicted_action_auc=b,difference=b-a)
    pos=[d for d in data if d["outcome"]=="decline"];neg=[d for d in data if d["outcome"]=="engaged"];mix=[d for d in data if d["outcome"]=="mixed"]
    results=[compare("all_binary",pos,neg),compare("within_same_type",pos,neg,True),compare("mixed_as_decline",pos+mix,neg),compare("mixed_as_engaged",pos,neg+mix)]
    for family in sorted({d["family"] for d in data}):
        results.append(compare("omit_type_"+family,[p for p in pos if p["family"]!=family],[n for n in neg if n["family"]!=family]))
    for p in pos:
        results.append(compare("omit_one_decline_"+p["id"],[q for q in pos if q!=p],neg))
    with (HERE/"response_sensitivity.csv").open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=results[0]);w.writeheader();w.writerows(results)
    for r in results[:9]:print(r["contrast"],r["pairs"],round(r["appropriate_letter_auc"],3), round(r["predicted_action_auc"],3),round(r["difference"],3))

if __name__=="__main__":main()
