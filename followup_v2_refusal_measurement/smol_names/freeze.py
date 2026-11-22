"""Lock the identical Qwen and Smol question bank and the cross-model comparison."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "independent_benign"
QWEN = HERE.parent / "fresh_names"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    lock = HERE / "freeze.json"
    source = QWEN / "bank.json"
    bank = HERE / "bank.json"
    historical = json.loads((QWEN / "freeze.json").read_text())
    coding = json.loads((QWEN / "coding_audit.json").read_text())
    assert sha(source) == historical["sha256"]["bank.json"]
    assert sha(QWEN / "coded_replies.csv") == coding["sha256"]["coded_replies.csv"]
    if not lock.exists() and any((HERE / file).exists() for file in ("raw.jsonl", "run.json")):
        raise RuntimeError("Original design lock missing after generation.")
    if not bank.exists() and not lock.exists():
        bank.write_bytes(source.read_bytes())
    assert bank.read_bytes() == source.read_bytes()
    rows = json.loads(bank.read_text())
    people = {r["name"] for r in rows}
    assert len(people) == 30
    assert len(rows) == len({r["id"] for r in rows}) == len({r["prompt"] for r in rows}) == 180
    assert {(r["name"], r["arm"]) for r in rows} == {(n, a) for n in people for a in "ABCDEF"}
    inputs = ["protocol.md", "run.py", "freeze.py", "bank.json", "../fresh_names/bank.json",
              "../fresh_names/freeze.json", "../fresh_names/selection.json", "../fresh_names/coded_replies.csv",
              "../fresh_names/coding_audit.json", "../fresh_names/run.json", "../name_wording/coding_rubric.txt",
              "../independent_benign/run_round.py", "../independent_benign/private/run_smol.json"]
    hashes = {name: sha(HERE / name) for name in inputs}
    if lock.exists():
        assert json.loads(lock.read_text())["sha256"] == hashes
    else:
        record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), names=30, wordings=6,
                      prompts=180, sha256=hashes)
        with lock.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
