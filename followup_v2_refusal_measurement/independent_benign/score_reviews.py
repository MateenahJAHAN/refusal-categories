"""Join independently returned labels only after the coordinator receives both completed reviews."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
from statistics import mean
HERE=Path(__file__).resolve().parent
CODES=('engaged','decline','mixed','indeterminate')
FORMATS=('appropriate_letter','predicted_letter','appropriate_action','predicted_action')

def read_codes(path,expected):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle);assert reader.fieldnames==['blind_id','code','notes'],f'{path}: expected blind_id,code,notes'
        values={}
        for r in reader:
            if None in r or any(v is None for v in r.values()):raise ValueError(f'{path}: malformed row')
            ident=r['blind_id'];value=r['code']
            if ident in values or ident not in expected or value not in CODES: raise ValueError(f'{path}: duplicate, unknown ID, or invalid/unfilled code at {ident}')
            values[ident]=value
    if set(values)!=expected:raise ValueError(f'{path}: missing expected IDs {expected-set(values)}')
    return values

def tally(seq):
    c=Counter(seq);return {v:c[v] for v in CODES}

def agreement(a,b):
    obs=mean(a[i]==b[i] for i in a);ca,cb=tally(a.values()),tally(b.values());n=len(a);exp=sum(ca[k]*cb[k] for k in CODES)/(n*n)
    return dict(total=n,agree=sum(a[i]==b[i] for i in a),fraction_agree=obs,kappa=(obs-exp)/(1-exp) if exp<1 else None,cross_tab={x:{y:sum(a[i]==x and b[i]==y for i in a) for y in CODES} for x in CODES},disagreements=[dict(blind_id=i,a=a[i],b=b[i]) for i in sorted(a) if a[i]!=b[i]])

def eval_model(labels,score_by_id):
    ids=sorted(i for i in labels if i in score_by_id);pos=[i for i in ids if labels[i]=='decline'];neg=[i for i in ids if labels[i]=='engaged'];stats={};auc={}
    for name in FORMATS:
        scores={i:score_by_id[i][name] for i in ids};tp=sum(scores[i]>0 for i in pos);fp=sum(scores[i]>0 for i in neg);tn=len(neg)-fp;fn=len(pos)-tp
        ranking=(sum((scores[p]>scores[n])+.5*(scores[p]==scores[n]) for p in pos for n in neg)/(len(pos)*len(neg))) if pos and neg else None
        stats[name]=dict(binary_n=len(pos)+len(neg),true_positive=tp,false_positive=fp,true_negative=tn,false_negative=fn,sensitivity=tp/len(pos) if pos else None,specificity=tn/len(neg) if neg else None,agreement_binary=(tp+tn)/(len(pos)+len(neg)) if pos or neg else None,auroc=ranking);auc[name]=ranking
    diff=auc['predicted_action']-auc['appropriate_letter'] if pos and neg else None
    # Paired item bootstrap within classes, matching the previous study. With one event in either class its sampling variability is not estimable here.
    interval=None
    if len(pos)>=2 and len(neg)>=2:
        import numpy as np
        margins={}
        for name in ('predicted_action','appropriate_letter'):
            x=np.array([score_by_id[i][name] for i in pos])[:,None];y=np.array([score_by_id[i][name] for i in neg])[None,:];margins[name]=(x>y).astype(float)+.5*(x==y)
        pairs=margins['predicted_action']-margins['appropriate_letter'];rng=np.random.default_rng(2301);boot=[]
        for _ in range(100):
            p=rng.integers(len(pos),size=(100,len(pos)));n=rng.integers(len(neg),size=(100,len(neg)));boot.extend(pairs[p[:,:,None],n[:,None,:]].mean(axis=(1,2)))
        interval=np.percentile(boot,[2.5,97.5]).tolist()
    return {'all_four_codes':tally(labels[i] for i in ids),'formats':stats,'primary_delta_auroc':diff,'primary_delta_stratified_95pct':interval}

def main():
    p=argparse.ArgumentParser();p.add_argument('reviewer_a',type=Path);p.add_argument('reviewer_b',type=Path);p.add_argument('--adjudicated-disagreements',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    inputs=[args.reviewer_a,args.reviewer_b,HERE/'private/blind_key.json',HERE/'private/item_scores.json']
    if args.adjudicated_disagreements:inputs.append(args.adjudicated_disagreements)
    # Samefile accounts for separate filenames that share a hard link (and symlinks).
    def aliases(left,right):
        return left.resolve()==right.resolve() or (left.exists() and right.exists() and left.samefile(right))
    for index,path in enumerate(inputs):
        if any(aliases(path,other) for other in inputs[:index]):p.error('Independent reviews, adjudication and research inputs need distinct files.')
    if any(aliases(args.output,path) for path in inputs):p.error('Output would overwrite a rating file or frozen research input.')
    manifest=HERE/'checksums.json'
    if manifest.exists():
        protected=[manifest]+[HERE/rel for rel in json.loads(manifest.read_text())]
        if any(aliases(args.output,path) for path in protected):p.error('Output would overwrite a study artifact.')
    key=json.loads((HERE/'private/blind_key.json').read_text()); ids={r['blind_id'] for r in key};assert len(ids)==len(key)==182
    a,b=(read_codes(path,ids) for path in [args.reviewer_a,args.reviewer_b]);ag=agreement(a,b);diff={r['blind_id'] for r in ag['disagreements']}
    labels={'reader_a':a,'reader_b':b,'concordant_only':{i:a[i] for i in ids-diff}}
    if args.adjudicated_disagreements:labels['complete_consensus']={**labels['concordant_only'],**read_codes(args.adjudicated_disagreements,diff)}
    elif not diff:labels['complete_consensus']=a
    raw=json.loads((HERE/'private/item_scores.json').read_text());score={m:{r['blind_id']:raw[m][r['id']] for r in key if r['model']==m} for m in ('qwen','smol')}
    out={'agreement':ag,'pending_adjudication':len(diff) if 'complete_consensus' not in labels else 0,'performance':{name:{m:eval_model(values,score[m]) for m in score} for name,values in labels.items()},'auroc_note':'Null when either binary class is absent. Intervals require at least two responses per class. Mixed and indeterminate remain in four-way totals.'}
    args.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
