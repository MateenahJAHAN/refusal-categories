"""Fetch the two pinned source datasets and rebuild the archived CoCoNot reader-ID join."""
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def fetch(relative, url, expected):
    target = ROOT / relative
    if target.exists():
        data = target.read_bytes()
    else:
        with urlopen(Request(url, headers={"User-Agent": "refusal-measurement-study/1"}), timeout=90) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected file digest for {relative}")
    if not target.exists():
        target.write_bytes(data)
    print(f"Verified {relative}")


def restore_reading_join():
    corpus = ROOT / "independent_benign"
    with (corpus / "ai_first_pass/all_ratings.csv").open(newline="") as handle:
        ratings = {(r["model"], r["question_id"]): r for r in csv.DictReader(handle)}
    key = []
    for short, model in [("qwen", "Qwen2.5-1.5B-Instruct"), ("smol", "SmolLM2-1.7B-Instruct")]:
        for line in (corpus / f"private/replies_{short}.jsonl").read_text().splitlines():
            raw = json.loads(line)
            scored = ratings[model, raw["id"]]
            assert scored["reply"] == raw["response"]
            key.append(dict(blind_id=scored["blind_id"], model=short, id=raw["id"]))
    assert len(key) == len({r["blind_id"] for r in key}) == 182
    body = (json.dumps(key, indent=2) + "\n").encode()
    assert hashlib.sha256(body).hexdigest() == "e93d8f5bd4508874365f7cc5977168385d4d389f18998c27e7306af7fb86c180"
    (corpus / "private/blind_key.json").write_bytes(body)


if __name__ == "__main__":
    roster = json.loads((ROOT / "fresh_names/source/provenance.json").read_text())
    fetch("fresh_names/source/billionaires.csv", roster["download"], roster["sha256"])
    contrast = json.loads((ROOT / "independent_benign/source/source.json").read_text())
    item = "source/test-00000-of-00001.parquet"
    fetch("independent_benign/" + item, contrast["source_url"], contrast["sha256"][item])
    card_url = f"https://huggingface.co/datasets/allenai/coconot/resolve/{contrast['revision']}/README.md"
    fetch("independent_benign/source/README.md", card_url, contrast["sha256"]["source/README.md"])
    restore_reading_join()
