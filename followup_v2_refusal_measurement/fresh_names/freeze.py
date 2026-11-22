"""Select the panel once from the archived 2014 roster and verify its immutable design."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "independent_benign"
PREFIX = "BlueDot-wealth-names-v1|"
GROUP = re.compile(r"&|/|\b(and|famil(?:y|ies)|brothers|siblings|heirs|sisters|partners)\b", re.I)
ADDITIONAL_OLD_SPELLINGS = {"Ding Lei", "Lei Ding", "Balkrishna", "Beto Sicupira", "Carlos Sicupira",
                           "Cheng Yutung", "Roman Arkadyevich Abramovich"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normal(name):
    text = unicodedata.normalize("NFKD", name.casefold())
    plain = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", plain))


def encoded(data):
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def design():
    source = HERE / "source/billionaires.csv"
    source_record = json.loads((HERE / "source/provenance.json").read_text())
    assert sha(source) == source_record["sha256"]
    old_bank = json.loads((PRIOR / "source/bank.json").read_text())
    old_prompts = [f" {normal(r['prompt'])} " for r in old_bank]
    known = {normal(n) for n in ADDITIONAL_OLD_SPELLINGS}
    wording_rows = json.loads((HERE.parent / "name_wording/bank.json").read_text())
    known.update(normal(r["name"]) for r in wording_rows)
    forms = {r["arm"]: r["prompt"].replace(r["name"], "{name}") for r in wording_rows if r["arm"] in "ABCDEF"}
    assert len(forms) == 6
    ranked = {}
    excluded = []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        snapshot = [r for r in csv.DictReader(stream) if r["year"] == "2014"]
    for row in snapshot:
        name = row["name"].strip()
        norm = normal(name)
        if float(row["wealth.worth in billions"]) < 1:
            reason = "recorded wealth below one billion"
        elif GROUP.search(name):
            reason = "joint or family label"
        elif "(" in name or ")" in name:
            reason = "multiple display names or alias"
        elif norm in known or any(f" {norm} " in q for q in old_prompts):
            reason = "previous person or prompt"
        elif norm in ranked:
            reason = "duplicate normalized name"
        else:
            key = hashlib.sha256((PREFIX + norm).encode()).hexdigest()
            ranked[norm] = dict(name=name, normalized=norm, selection_hash=key,
                                source_id="p2014-" + key[:12], source_rank=int(row["rank"]))
            continue
        excluded.append(dict(name=name, reason=reason))
    roster = sorted(ranked.values(), key=lambda n: (n["selection_hash"], n["normalized"]))[:30]
    bank = [dict(id=r["source_id"]+"-"+arm, source_id=r["source_id"], name=r["name"], arm=arm,
                 prompt=forms[arm].format(name=r["name"])) for r in roster for arm in "ABCDEF"]
    assert len(bank) == len({r["id"] for r in bank}) == len({r["prompt"] for r in bank}) == 180
    assert len({r["name"] for r in bank}) == 30
    assert all(normal(r["name"]) not in known for r in roster)
    return {"selection.json": encoded(dict(source_rows_2014=len(snapshot), eligible_people=len(ranked),
                                            excluded=excluded, n=30, members=roster)),
            "bank.json": encoded(bank)}


def main():
    output = design()
    lock = HERE / "freeze.json"
    if not lock.exists() and any((HERE / p).exists() for p in ("raw.jsonl", "run.json")):
        raise RuntimeError("Cannot create the first freeze after generation.")
    for filename, text in output.items():
        path = HERE / filename
        if lock.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
    inputs = ["source/billionaires.csv", "source/provenance.json", "protocol.md", "freeze.py", "selection.json", "bank.json",
              "../name_wording/bank.json", "../name_wording/coding_rubric.txt", "../independent_benign/source/bank.json",
              "../independent_benign/run_round.py", "../independent_benign/private/run_qwen.json", "run.py"]
    hashes = {p: sha(HERE / p) for p in inputs}
    if lock.exists():
        assert json.loads(lock.read_text())["sha256"] == hashes
    else:
        with lock.open("x") as stream:
            stream.write(encoded(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                                      names=30, wordings=6, prompts=180, sha256=hashes)))


if __name__ == "__main__":
    main()
