"""Greedy free responses, collected without action codes or action-probe scores."""
import argparse
import hashlib
import json
import random
from datetime import datetime,timezone
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
TYPES={"homonyms","figurative_language","definitions","nons_group_real_discr","real_group_nons_discr"}
def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--batch",type=int,default=8)
    args=p.parse_args()
    data=ROOT/"data"
    rawfile=data/"direct_qwen.jsonl"
    stamp=data/"direct_run.json"
    bank=[r for r in json.loads((data/"items.json").read_text()) if r["group"]=="xstest" and r["family"] in TYPES]
    assert len(bank)==125
    rng=random.Random(212611)
    rng.shuffle(bank)
    records={r["id"]:i+1 for i,r in enumerate(bank)}
    comp={r["id"]:r for r in bank}
    assert len(comp)==125
    config={"model":"Qwen/Qwen2.5-1.5B-Instruct","revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "weights_sha256":digest(args.model_dir/"model.safetensors"),"torch":str(torch.__version__),
            "transformers":transformers.__version__,"device":"mps","dtype":"float32","batch":args.batch,
            "greedy":True,"max_new_tokens":96,"system":"You are a helpful assistant.",
            "sha256":{f:digest(ROOT/f) for f in ["direct_protocol.md","direct.py","data/items.json"]}}
    assert config["weights_sha256"]=="dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
    if stamp.exists():
        old=json.loads(stamp.read_text())
        assert all(old[k]==v for k,v in config.items())
    else:
        stamp.write_text(json.dumps(dict(start_utc=now(),**config),indent=2)+"\n")
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.float32,attn_implementation="sdpa").to("mps").eval()
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    tok.padding_side="left"
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    eos=model.generation_config.eos_token_id
    ends=set([eos] if isinstance(eos,int) else eos)
    oldrows=[json.loads(l) for l in rawfile.read_text().splitlines()] if rawfile.exists() else []
    done={r["id"] for r in oldrows}
    assert len(done)==len(oldrows)
    remaining=sorted([r for r in bank if r["id"] not in done],key=lambda r:len(r["prompt"]))
    with torch.inference_mode(), rawfile.open("a") as h:
        for i in range(0,len(remaining),args.batch):
            group=remaining[i:i+args.batch]
            conversations=[[{"role":"system","content":config["system"]},{"role":"user","content":r["prompt"]}] for r in group]
            inputs=tok.apply_chat_template(conversations,tokenize=True,padding=True,add_generation_prompt=True,return_tensors="pt",return_dict=True).to("mps")
            outs=model.generate(**inputs,max_new_tokens=96,do_sample=False,pad_token_id=tok.pad_token_id)[:,inputs.input_ids.shape[1]:].tolist()
            for r,seq in zip(group,outs):
                first_stop=next((j for j,x in enumerate(seq) if x in ends),len(seq))
                text=tok.decode(seq[:first_stop],skip_special_tokens=True)
                finish="stop" if first_stop < len(seq) else "length"
                h.write(json.dumps(dict(id=r["id"],blind_id=records[r["id"]],response=text,finish=finish,answer_tokens=first_stop,generated_token_ids=seq[:first_stop]))+"\n")
            h.flush()
            print(i+len(group),"of",len(remaining),flush=True)
    complete=[json.loads(l) for l in rawfile.read_text().splitlines()]
    assert len(complete)==len({r["id"] for r in complete})==125
    blind=[{"blind_id":r["blind_id"],"request":comp[r["id"]]["prompt"],"reply":r["response"]} for r in sorted(complete,key=lambda row:row["blind_id"])]
    (data/"blind_reading.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in blind))
    stamp.write_text(json.dumps(dict(**json.loads(stamp.read_text()), end_utc=now(), raw_sha256=digest(rawfile)),indent=2)+"\n")

if __name__=="__main__":
    main()
