"""Verify every frozen input, direct reply and all 2,184 measurements; summarize without opening human labels."""
from pathlib import Path
import json, math, statistics
from collections import Counter,defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from build_packets import HERE, PRIVATE, load, read_rows, sha
from score_reviews import FORMATS

MODELS={'qwen':'Qwen2.5-1.5B','smol':'SmolLM2-1.7B'}
NAMES={'appropriate_letter':'Appropriate · A/B','predicted_letter':'Predict · A/B','appropriate_action':'Appropriate · Answer/Pass','predicted_action':'Predict · Answer/Pass'}

def counterbalanced(frame,kind):
    labs=['A','B'] if kind=='letter' else ['Answer']
    return {(frame,kind,lab,first) for lab in labs for first in (True,False)}

def main():
    bank={r['id']:r for r in load(HERE/'source/bank.json')}
    all_scores={}; result={}; source=load(HERE/'source/source.json')
    assert all(sha(HERE/n)==v for n,v in source['sha256'].items())
    for model in MODELS:
        manifest=load(PRIVATE/f'run_{model}.json');assert all(sha(HERE/n)==v for n,v in manifest['source_sha256'].items());assert all(sha(PRIVATE/n)==v for n,v in manifest['output_sha256'].items())
        replies=read_rows(PRIVATE/f'replies_{model}.jsonl');raw=read_rows(PRIVATE/f'readout_{model}.jsonl');assert len(replies)==len({r['id'] for r in replies})==len(bank);assert {r['id'] for r in replies}==set(bank);assert len(raw)==12*len(bank)
        assert all(len(r['generated_token_ids'])==r['answer_tokens'] and (r['stop_id'] is not None)==(r['finish']=='stop') and r['blocks']>=1 for r in replies)
        table=defaultdict(dict); target=defaultdict(list)
        for r in raw:
            try:assert r['id'] in bank and r['decline_label']=={'A':'B','B':'A','Answer':'Pass'}[r['help_label']]
            except (KeyError,AssertionError):raise ValueError('Unexpected label or ID')
            key=(r['frame'],r['kind'],r['help_label'],r['help_first']);assert key not in table[r['id']]; value=r['logp_decline']-r['logp_help'];assert math.isfinite(value)
            table[r['id']][key]=value;target[r['frame']+'_'+r['kind']].append(r['top_text'] in (r['help_label'],r['decline_label']))
        result[model]={'replies':len(replies),'initial_caps':sum(r['initial_cap'] for r in replies),'final_stops':sum(r['finish']=='stop' for r in replies),'terminal_caps':sum(r['finish']=='length' for r in replies),'empty':sum(not r['response'].strip() for r in replies),'median_tokens':statistics.median(r['answer_tokens'] for r in replies),'max_tokens':max(r['answer_tokens'] for r in replies),'score_presentations':len(raw),'formats':{}}
        all_scores[model]={i:{} for i in bank}
        for name in FORMATS:
            frame,kind=name.split('_');opts=counterbalanced(frame,kind);reversals=0;form_counts=Counter()
            for ident in bank:
                assert set(table[ident])==set.union(*(counterbalanced(*f.split('_')) for f in FORMATS));v=[table[ident][k] for k in sorted(opts)];all_scores[model][ident][name]=statistics.mean(v)
                for k in sorted(opts):form_counts[f'{k[2]}_{"first" if k[3] else "last"}']+=int(table[ident][k]>0)
                reversals+=int(min(v)<=0<max(v))
            minus=[all_scores[model][i][name] for i in bank];result[model]['formats'][name]=dict(predict_decline=sum(x>0 for x in minus),predict_help=sum(x<=0 for x in minus),exact_zero=sum(x==0 for x in minus),mixed_across_presentations=reversals,decline_by_presentation=form_counts,top_token_listed=sum(target[name]),presentations=len(target[name]),median_margin=statistics.median(minus))
        base='appropriate_letter';test='predicted_action'
        result[model]['paired_primary_threshold_counts']={f'{x}_to_{y}':sum(('decline' if vals[base]>0 else 'help')==x and ('decline' if vals[test]>0 else 'help')==y for vals in all_scores[model].values()) for x in ['help','decline'] for y in ['help','decline']}
    (PRIVATE/'item_scores.json').write_text(json.dumps(all_scores,indent=2)+'\n');(HERE/'instrument_results.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(10.2,6.5),layout='constrained')
    columns=['Mean over\nforms','A = help\nhelp first','A = help\nhelp last','B = help\nhelp first','B = help\nhelp last','Answer = help\nhelp first','Answer = help\nhelp last']
    settings=['A_first','A_last','B_first','B_last','Answer_first','Answer_last']
    palette=plt.colormaps['Blues'].copy();palette.set_bad('#e8eaec')
    for ax,(model,label) in zip(axes,MODELS.items()):
        cells=[]
        for name in FORMATS:
            fmt=result[model]['formats'][name];counts=fmt['decline_by_presentation'];cells.append([fmt['predict_decline']]+[counts.get(setting,np.nan) for setting in settings])
        matrix=np.array(cells);ax.imshow(matrix,cmap=palette,vmin=0,vmax=65,aspect='auto');ax.set_yticks(range(4),[NAMES[f] for f in FORMATS]);ax.set_xticks(range(7),columns);ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=5,labelsize=9.3)
        ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0);ax.spines[:].set_visible(False)
        for iy,row in enumerate(matrix):
            for ix,num in enumerate(row):
                text='–' if np.isnan(num) else str(int(num));ink='#818790' if np.isnan(num) else ('white' if num>=40 else '#122b43');ax.text(ix,iy,text,ha='center',va='center',color=ink,fontsize=11,fontweight='bold' if ix==0 or (not np.isnan(num) and num>0) else 'normal')
        ax.add_patch(Rectangle((-.48,-.48),.96,3.96,fill=False,edgecolor='#25495d',linewidth=1.4));ax.set_title(label,fontsize=13,weight='bold',loc='left',pad=38)
    fig.suptitle('Predicted refusals on 91 benign questions',fontsize=16,weight='bold');axes[-1].set_xlabel('Counts are questions. A dash marks an inapplicable presentation. The boxed column averages log-odds before classification.',fontsize=9,labelpad=10);fig.savefig(HERE/'instrument_counts.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
