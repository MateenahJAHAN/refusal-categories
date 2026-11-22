"""Continue all and only the direct responses that reached the first length cap."""
import argparse,hashlib,json,random
from pathlib import Path
from datetime import datetime,timezone
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent
D=ROOT/"data"
def h(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
def now():
    return datetime.now(timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--batch",type=int,default=12)
    args=ap.parse_args()
    meta=json.loads((D/"direct_run.json").read_text())
    assert h(D/"direct_qwen.jsonl")==meta["raw_sha256"]
    assert h(args.model_dir/"model.safetensors")==meta["weights_sha256"]
    old=[json.loads(l) for l in (D/"direct_qwen.jsonl").read_text().splitlines()]
    bank={r["id"]:r for r in json.loads((D/"items.json").read_text())}
    capped={r["id"]:r for r in old if r["finish"]=="length"}
    assert len(old)==125 and len(capped)==104
    order=sorted(capped)
    random.Random(20261122).shuffle(order)
    ids={value:i+1 for i,value in enumerate(order)}
    data=D/"continued_qwen.jsonl"
    runpath=D/"continuation_run.json"
    facts=dict(original_run=h(D/"direct_run.json"),old_labels=h(D/"response_codes.csv"),
            protocol=h(ROOT/"continuation_protocol.md"),script=h(ROOT/"continue_direct.py"),batch=args.batch,
            added_token_cap=416, model_revision=meta["revision"], weights_sha256=meta["weights_sha256"], dtype="float32",device="mps")
    if runpath.exists():
        oldmeta=json.loads(runpath.read_text())
        assert all(oldmeta[k]==v for k,v in facts.items())
    else:
        runpath.write_text(json.dumps(dict(start_utc=now(),**facts),indent=2)+"\n")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.float32,attn_implementation="sdpa").to("mps").eval()
    eos=model.generation_config.eos_token_id
    ends=set([eos] if isinstance(eos,int) else eos)
    existing=[json.loads(l) for l in data.read_text().splitlines()] if data.exists() else []
    done=set(r["id"] for r in existing)
    assert len(existing)==len(done)
    plans=[]
    for k,r in capped.items():
        assert len(r["generated_token_ids"])==96
        if k in done: continue
        prompt=tok.apply_chat_template([dict(role="system",content=meta["system"]),dict(role="user",content=bank[k]["prompt"])], tokenize=True,add_generation_prompt=True,return_dict=False)
        plans.append((prompt+r["generated_token_ids"],k))
    plans.sort(key=lambda v:len(v[0]))
    with torch.inference_mode(), data.open("a") as stream:
        for i in range(0,len(plans),args.batch):
            group=plans[i:i+args.batch]
            width=max(len(p) for p,k in group)
            toks=torch.full((len(group),width),tok.pad_token_id,dtype=torch.long)
            mask=torch.zeros_like(toks)
            for j,(p,k) in enumerate(group):
                toks[j,-len(p):]=torch.tensor(p)
                mask[j,-len(p):]=1
            seqs=model.generate(input_ids=toks.to("mps"),attention_mask=mask.to("mps"),
                max_new_tokens=416,do_sample=False,pad_token_id=tok.pad_token_id)[:,width:].tolist()
            for (_,k),seq in zip(group,seqs):
                n=next((j for j,t in enumerate(seq) if t in ends),len(seq))
                total=capped[k]["generated_token_ids"]+seq[:n]
                s=tok.decode(total,skip_special_tokens=True)
                assert s.startswith(capped[k]["response"])
                stream.write(json.dumps(dict(id=k,blind_id=ids[k],original_id=capped[k]["blind_id"],response=s,
                    finish="stop" if n<len(seq) else "length",answer_tokens=len(total),added_token_ids=seq[:n]))+"\n")
            stream.flush()
            print(i+len(group),"of",len(plans),flush=True)
    final=[json.loads(l) for l in data.read_text().splitlines()]
    assert len(final)==len(set(r["id"] for r in final))==104
    blind=[dict(blind_id=r["blind_id"],request=bank[r["id"]]["prompt"],reply=r["response"]) for r in sorted(final,key=lambda r:r["blind_id"])]
    (D/"continued_blind.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in blind))
    saved=json.loads(runpath.read_text());saved.update(end_utc=now(),raw_sha256=h(data))
    runpath.write_text(json.dumps(saved,indent=2)+"\n")
if __name__=="__main__":
    main()
