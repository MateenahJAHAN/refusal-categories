"""Run independent greedy Qwen chats with the exact previous direct-response code path."""
import argparse
from datetime import datetime, timezone
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from freeze import HERE, PRIOR, main as verify_freeze, sha
sys.path.insert(0,str(PRIOR))
from run_round import direct


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE, help="Use a fresh output directory to replicate the original run.")
    dest = parser.parse_args().output
    dest.mkdir(exist_ok=True,parents=True)
    verify_freeze()
    old=json.loads((PRIOR/"private/run_qwen.json").read_text())
    ckpt=Path.home()/".cache/huggingface/hub"/("models--"+old["model"].replace("/","--"))/"snapshots"/old["revision"]
    for name,h in old["checkpoint_sha256"].items():
        assert sha(ckpt/name)==h
    assert str(torch.__version__)==old["torch"] and transformers.__version__==old["transformers"]
    assert torch.backends.mps.is_available()
    fixed = {k:old[k] for k in ["model","revision","checkpoint_sha256","torch","transformers","device","dtype","attention","batch_direct"]}
    fixed.update(platform=platform.platform(), freeze_sha256=sha(HERE/"freeze.json"), runner_sha256=sha(Path(__file__)),
                 generation_config_sha256=sha(ckpt/"generation_config.json"), vocab_sha256=sha(ckpt/"vocab.json"),
                 merges_sha256=sha(ckpt/"merges.txt"), python=sys.version.split()[0], local_files_only=True)
    manifest = dest/"run.json"
    out = dest/"raw.jsonl"
    if manifest.exists():
        record=json.loads(manifest.read_text())
        assert all(record[k]==v for k,v in fixed.items())
        if "raw_sha256" in record:
            assert sha(out)==record["raw_sha256"]
            print("Run already complete",flush=True)
            return
    else:
        assert not out.exists()
        record={"start_utc":now(),**fixed}
        with manifest.open("x") as f: json.dump(record,f,indent=2)
    bank=json.loads((HERE/"bank.json").read_text())
    tok=AutoTokenizer.from_pretrained(ckpt,local_files_only=True)
    tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(ckpt,local_files_only=True,dtype=torch.float32,attn_implementation="sdpa").to("mps").eval()
    direct(bank,tok,model,out,old["batch_direct"])
    rows=[json.loads(s) for s in out.read_text().splitlines()]
    assert len(rows)==len(bank)==len({r["id"] for r in rows})
    assert {r["id"] for r in rows}=={r["id"] for r in bank}
    record.update(end_utc=now(),completed=len(rows),raw_sha256=sha(out))
    tmp=manifest.with_suffix(".json.tmp");tmp.write_text(json.dumps(record,indent=2)+"\n");tmp.replace(manifest)


if __name__=="__main__":
    main()
