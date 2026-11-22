"""Offline paired greedy completion and crossed counterbalanced first-token readouts."""
import argparse, json, platform, sys
from pathlib import Path
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'semantic_check'))
from run_semantic import build, original
FORMAT = [(frame,kind,lab,first) for frame in ('appropriate','predicted') for kind in ('letter','action') for lab in (('A','B') if kind=='letter' else ('Answer',)) for first in (True,False)]

def load(p): return json.loads(p.read_text())
def rows(p): return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
def write(p,data): p.write_text(json.dumps(data,indent=2)+'\n')

def direct(items, tok, model, out, batch):
    ends=model.generation_config.eos_token_id
    ends={ends} if isinstance(ends,int) else set(ends)
    tok.padding_side='left'
    done={r['id'] for r in rows(out)}
    jobs=sorted((r for r in items if r['id'] not in done),key=lambda r:len(r['prompt']))
    with torch.inference_mode(),out.open('a') as fh:
        for start in range(0,len(jobs),batch):
            chunk=jobs[start:start+batch]
            conv=[[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':r['prompt']}] for r in chunk]
            inputs=tok.apply_chat_template(conv,tokenize=True,padding=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('mps')
            generated=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tok.pad_token_id)[:,inputs.input_ids.shape[1]:].tolist()
            for index,(item,seq) in enumerate(zip(chunk,generated)):
                proposed=next((j for j,t in enumerate(seq) if t in ends),len(seq))
                complete=proposed<len(seq); first_stop_id=seq[proposed] if complete else None; seq=seq[:proposed]
                cap=not complete; blocks=1
                ids=inputs.input_ids[index][inputs.attention_mask[index].bool()].tolist()
                max_tokens=min(2048,model.config.max_position_embeddings-len(ids))
                while not complete and len(seq)<max_tokens:
                    prompt=torch.tensor([ids+seq],device='mps'); length=min(256,max_tokens-len(seq))
                    more=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),max_new_tokens=length,do_sample=False,pad_token_id=tok.pad_token_id)[0,prompt.shape[1]:].tolist()
                    stop=next((j for j,t in enumerate(more) if t in ends),len(more))
                    complete=stop<len(more); first_stop_id=more[stop] if complete else None
                    seq.extend(more[:stop]); blocks+=1
                answer=tok.decode(seq,skip_special_tokens=True)
                fh.write(json.dumps(dict(id=item['id'],response=answer,finish='stop' if complete else 'length',initial_cap=cap,blocks=blocks,answer_tokens=len(seq),input_tokens=len(ids),generated_token_ids=seq,stop_id=first_stop_id))+'\n')
            fh.flush();print(f'direct {len(done)+start+len(chunk)}/{len(items)}',flush=True)

def score(items,tok,model,out,batch):
    tok.padding_side='right'
    done={(r['id'],r['frame'],r['kind'],r['help_label'],r['help_first']) for r in rows(out)}
    names=['A','B','Answer','Pass']; token={s:tok.encode(s,add_special_tokens=False) for s in names};assert all(len(v)==1 for v in token.values())
    token={k:v[0] for k,v in token.items()}; jobs=[]
    for item in items:
        for frame,kind,lab,first in FORMAT:
            if (item['id'],frame,kind,lab,first) in done:continue
            msgs,other=build(item,frame,kind,lab,first)
            if frame=='appropriate' and kind=='letter': assert msgs[1]['content']==original.build_prompt(item,lab,first)
            ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=False)
            jobs.append((ids,dict(id=item['id'],frame=frame,kind=kind,help_label=lab,decline_label=other,help_first=first)))
    jobs.sort(key=lambda j:len(j[0]))
    with torch.inference_mode(),out.open('a') as fh:
        for offset in range(0,len(jobs),batch):
            group=jobs[offset:offset+batch]; lens=torch.tensor([len(ids) for ids,_ in group],device='mps')
            x=torch.full((len(group),int(lens.max())),tok.pad_token_id,dtype=torch.long)
            for j,(ids,_) in enumerate(group):x[j,:len(ids)]=torch.tensor(ids)
            x=x.to('mps');mask=torch.arange(x.shape[1],device='mps')[None,:]<lens[:,None]
            hidden=model.model(input_ids=x,attention_mask=mask.long(),use_cache=False).last_hidden_state
            logits=model.lm_head(hidden[torch.arange(len(group),device='mps'),lens-1]).float().log_softmax(-1)
            assert torch.isfinite(logits).all().item()
            for (ids,row),best,values in zip(group,logits.argmax(-1).tolist(),logits[:,list(token.values())].cpu().tolist()):
                logp=dict(zip(names,values));row.update(input_tokens=len(ids),logp_help=logp[row['help_label']],logp_decline=logp[row['decline_label']],top_id=best,top_text=tok.decode([best]));fh.write(json.dumps(row)+'\n')
            fh.flush()
            if offset%256==0:print('readouts',len(done)+offset+len(group),'/',len(items)*12,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=original.MODELS);args=p.parse_args();key=args.model
    bank=load(HERE/'source/bank.json');name,rev=original.MODELS[key]
    checkpoint=Path.home()/'.cache/huggingface/hub'/('models--'+name.replace('/','--'))/'snapshots'/rev
    prior=load(HERE.parent/f'data/run_{key}.json');files=prior['model_file_sha256'];assert all(original.digest(checkpoint/f)==v for f,v in files.items())
    assert (str(torch.__version__),transformers.__version__)==(prior['torch'],prior['transformers'])
    manifest=HERE/f'private/run_{key}.json'; out=HERE/f'private/replies_{key}.jsonl'; survey=HERE/f'private/readout_{key}.jsonl'
    fixed=dict(model=name,revision=rev,checkpoint_sha256=files,device='mps',dtype='float32',attention='sdpa',torch=str(torch.__version__),transformers=transformers.__version__,platform=platform.platform(),batch_direct=8,batch_measure=16,source_sha256={f:original.digest(HERE/f) for f in ['protocol.md','prepare_bank.py','run_round.py','source/bank.json','source/source.json','../semantic_check/run_semantic.py','../run.py']})
    if manifest.exists():assert all(load(manifest)[k]==v for k,v in fixed.items())
    else:write(manifest,dict(start_utc=original.timestamp(),**fixed))
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('mps').eval()
    direct(bank,tok,model,out,8);score(bank,tok,model,survey,16)
    assert len(rows(out))==len({r['id'] for r in rows(out)})==len(bank)
    assert len(rows(survey))==len({(r['id'],r['frame'],r['kind'],r['help_label'],r['help_first']) for r in rows(survey)})==len(bank)*12
    write(manifest,dict(**{k:v for k,v in load(manifest).items() if k not in ('end_utc','output_sha256')},end_utc=original.timestamp(),output_sha256={o.name:original.digest(o) for o in (survey,out)}))
if __name__=='__main__': main()
