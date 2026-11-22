"""Verify the saved baseline summaries against the original versioned analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REVISION = "42f57fcb3980cbc60f6c7f6944aab94398d743ee"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout",type=Path)
    args = parser.parse_args()
    repo=args.checkout.resolve()
    commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    assert commit == REVISION
    kl=repo/"quality_gate"
    sys.path.insert(0,str(kl))
    from check import evaluate
    gate=evaluate(json.loads((kl/"archived_kl.json").read_text()))
    assert gate == json.loads((kl/"replay.json").read_text())
    follow=repo/"followup_v1_benign_wording"
    # Older analysis modules import each other by bare names; establish the source path.
    sys.path.insert(0,str(follow))
    from analyze import verify,summarize
    records=verify()
    saved=(follow/"results.json").read_bytes()
    assert summarize(records)==json.loads(saved)
    out={"public_revision":commit,"source_results_reproduced":True,
         "wording":{"raw_trials":len(records),"summary_matches_archived":True,
                    "archived_results_sha256":hashlib.sha256(saved).hexdigest()},"kl":gate}
    (ROOT/"data/prior_study_replay.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":
    main()
