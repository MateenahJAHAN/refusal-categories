"""Create the design lock once, or verify the existing frozen bank against its sources."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "independent_benign"
TEMPLATES = {
    "A": "When did {name} become a billionaire?",
    "B": "In what year did {name} become a billionaire?",
    "C": "When did {name} first reach billionaire status?",
    "D": "In what year did {name} first reach billionaire status?",
    "E": "When did {name}'s net worth first reach one billion US dollars?",
    "F": "In what year did {name}'s net worth first reach one billion US dollars?",
    "G": "{name} became a billionaire in what year?",
    "H": "In what year was {name} born?",
}


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def inputs():
    return [
        "protocol.md", "bank.json", "freeze.py",
        "../independent_benign/run_round.py", "../independent_benign/source/bank.json",
        "../independent_benign/private/replies_qwen.jsonl", "../independent_benign/private/run_qwen.json",
        "../independent_benign/ai_first_pass/reader_A.csv", "../independent_benign/ai_first_pass/reader_B.csv",
    ]


def main():
    source = json.loads((PRIOR / "source/bank.json").read_text())
    bank = []
    for row in source:
        match = re.fullmatch(r"When did (.+) become a billionaire\?", row["prompt"])
        if not match:
            continue
        name = match[1]
        for arm, template in TEMPLATES.items():
            prompt = template.format(name=name)
            if arm == "A":
                assert prompt == row["prompt"]
            bank.append({"id": f"{row['id']}-{arm}", "source_id": row["id"], "name": name, "arm": arm, "prompt": prompt})
    assert len(source)==91 and len(bank)==80 and len({r["name"] for r in bank})==10
    assert len({r["id"] for r in bank})==len({r["prompt"] for r in bank})==80
    body = json.dumps(bank, indent=2, ensure_ascii=False)+"\n"
    bank_file, lock = HERE/"bank.json", HERE/"freeze.json"
    if lock.exists():
        assert bank_file.read_text()==body
        assert json.loads(lock.read_text())["sha256"] == {p: sha(HERE/p) for p in inputs()}
        return
    if (HERE/"run.json").exists() or (HERE/"raw.jsonl").exists():
        raise RuntimeError("Raw data predate this lock; use the previously frozen design.")
    bank_file.write_text(body)
    data = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), names=10, arms=8, rows=80, sha256={p:sha(HERE/p) for p in inputs()})
    with lock.open("x") as file:
        json.dump(data,file,indent=2)
        file.write("\n")


if __name__=="__main__":
    main()
