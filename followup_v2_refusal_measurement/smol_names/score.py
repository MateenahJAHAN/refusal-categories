"""Score the SmolLM2 replication with the Qwen rubric and a new pair of blinded ID mappings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "name_wording"
sys.path.insert(0, str(ORIGINAL))
from code_replies import agreement, digest, reading

SEEDS = {"A": 940172, "B": 410379, "C": 713299}
VARIABLES = ("behavior", "political_rationale")


def responses():
    bank = json.loads((HERE / "bank.json").read_text())
    manifest = json.loads((HERE / "run.json").read_text())
    raw_path = HERE / "raw.jsonl"
    assert manifest["raw_sha256"] == digest(raw_path)
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    by_id = {row["id"]: row for row in raw}
    assert len(bank) == len(raw) == len(by_id) == manifest["completed"] == 180
    assert {row["id"] for row in bank} == set(by_id)
    assert len({(row["name"], row["arm"]) for row in bank}) == 180
    return {row["id"]: {"id": row["id"], "name": row["name"], "condition": row["arm"],
                        "prompt": row["prompt"], "reply": by_id[row["id"]]["response"]}
            for row in bank}


def packet(records, reader):
    ids = sorted(records)
    random.Random(SEEDS[reader]).shuffle(ids)
    key = {}
    rows = []
    for identity in ids:
        blinded = hashlib.sha256(f"second-checkpoint/{reader}/{identity}".encode()).hexdigest()[:14].upper()
        key[blinded] = identity
        record = records[identity]
        rows.append(dict(blind_id=blinded, question=record["prompt"], reply=record["reply"]))
    assert len(key) == len(records)
    return key, rows


def as_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_packet(rows, folder):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "packet.json").write_text(as_json(rows))
    (folder / "rubric.txt").write_bytes((ORIGINAL / "coding_rubric.txt").read_bytes())


def read_ratings(records, reader):
    key, rows = packet(records, reader)
    data = reading(HERE / f"reader_{reader}.csv", key)
    assert list(data) == [row["blind_id"] for row in rows], f"Different packet order for {reader}"
    return {key[k]: v for k, v in data.items()}


def compare(records):
    a, b = (read_ratings(records, r) for r in "AB")
    disputed = {k: row for k, row in records.items() if any(a[k][f] != b[k][f] for f in VARIABLES)}
    return a, b, disputed


def merge(records):
    a, b, disputed = compare(records)
    third = read_ratings(disputed, "C") if disputed else {}
    joined = []
    for k, row in records.items():
        final = dict(row)
        for f in VARIABLES:
            final[f] = a[k][f] if a[k][f] == b[k][f] else third[k][f]
        for label, entry in [("A", a[k]), ("B", b[k]), ("C", third.get(k, {}))]:
            for field in (*VARIABLES, "note", "blind_id"):
                final[f"reader_{label}_{field}"] = entry.get(field, "")
        joined.append(final)
    out = HERE / "coded_replies.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    metrics = {f: agreement([a[k][f] for k in records], [b[k][f] for k in records]) for f in VARIABLES}
    files = ["bank.json", "raw.jsonl", "reader_A.csv", "reader_B.csv", out.name]
    packets = {r: packet(records, r)[1] for r in "AB"}
    if disputed:
        files.append("reader_C.csv")
        packets["C"] = packet(disputed, "C")[1]
    metrics.update(n=len(records), joint_agreement=len(records) - len(disputed),
                   adjudicated_ids=sorted(disputed), rubric_sha256=digest(ORIGINAL / "coding_rubric.txt"),
                   packet_sha256={r: hashlib.sha256(as_json(rows).encode()).hexdigest() for r, rows in packets.items()},
                   sha256={f: digest(HERE / f) for f in files})
    (HERE / "coding_audit.json").write_text(as_json(metrics))
    print(f"{len(records)} replies. Initial agreement on both variables: {metrics['joint_agreement']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for step in ("pack", "disputed"):
        sub.add_parser(step).add_argument("destination", type=Path)
    sub.add_parser("merge")
    args = parser.parse_args()
    records = responses()
    if args.action == "pack":
        for label in "AB":
            write_packet(packet(records, label)[1], args.destination / label)
    elif args.action == "disputed":
        _, _, disputed = compare(records)
        if disputed:
            write_packet(packet(disputed, "C")[1], args.destination / "C")
        print(f"{len(disputed)} replies with a disagreement.")
    else:
        merge(records)
