"""Run the prospectively amended BlueDot comparison on the cached Qwen2.5 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from run import CONDITIONS, OPTIONS, ROOT, SYSTEM, classify, frozen_hashes, messages, read_cases, utc

MODEL = "qwen25-1.5b"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUTPUT = ROOT / f"raw_{MODEL}.jsonl"
MANIFEST = ROOT / "freeze_qwen25.json"
CONTROLS = ROOT / "qwen25_runtime_controls.json"
DEVICE = "mps"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest(model_dir: Path) -> dict:
    return {
        "model": MODEL, "revision": REVISION, "backend": DEVICE,
        "torch": str(torch.__version__), "transformers": transformers_version,
        "weight_sha256": file_hash(model_dir / "model.safetensors"),
        "tokenizer_sha256": file_hash(model_dir / "tokenizer_config.json"),
        "original_freeze_sha256": file_hash(ROOT / "freeze.json"),
        "amendment_sha256": file_hash(ROOT / "AMENDMENT_QWEN25.md"),
        "runner_sha256": file_hash(ROOT / "run_qwen25.py"),
    }


def complete(model, tokenizer, conversation: list[dict[str, str]]) -> tuple[str, str, dict[str, int]]:
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    n_input = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=OPTIONS["max_tokens"])[0][n_input:]
    end_ids = model.generation_config.eos_token_id
    end_ids = [end_ids] if isinstance(end_ids, int) else (end_ids or [])
    finish = "stop" if len(tokens) < OPTIONS["max_tokens"] or int(tokens[-1]) in end_ids else "length"
    return tokenizer.decode(tokens, skip_special_tokens=True), finish, {"prompt_tokens": n_input, "completion_tokens": len(tokens)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((ROOT / "freeze.json").read_text())
    assert original["sha256"] == frozen_hashes()
    facts = manifest(args.model_dir)
    if args.action == "freeze":
        with MANIFEST.open("x") as stream:
            json.dump({"frozen_utc": utc(), **facts}, stream, indent=2)
            stream.write("\n")
        return
    saved = json.loads(MANIFEST.read_text())
    assert {key: saved[key] for key in facts} == facts
    if not torch.backends.mps.is_available():
        raise SystemExit("Expected the recorded MPS runtime.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.float16, attn_implementation="eager"
    ).to(DEVICE).eval()

    if not CONTROLS.exists():
        checks = []
        for prompt, expected in (("What is 3 plus 4?", "7"), ("The sample says Green is A and Blue is B. What color is B?", "Blue")):
            conversation = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            answer, finish, usage = complete(model, tokenizer, conversation)
            checks.append({
                "prompt": prompt, "expected": expected, "content": answer,
                "score": classify(answer, expected), "finish_reason": finish, "usage": usage,
            })
        CONTROLS.write_text(json.dumps(checks, indent=2) + "\n")
    checks = json.loads(CONTROLS.read_text())
    assert len(checks) == 2 and all(x["score"]["value_correct"] and x["finish_reason"] == "stop" for x in checks)

    existing = [json.loads(line) for line in OUTPUT.read_text().splitlines()] if OUTPUT.exists() else []
    done = {(row["case_id"], row["condition"]) for row in existing}
    assert len(existing) == len(done)
    schedule = [(case, condition) for case in read_cases() for condition in CONDITIONS]
    random.Random(20261120).shuffle(schedule)
    with OUTPUT.open("a") as stream:
        for index, (case, condition) in enumerate(schedule, 1):
            if (case["id"], condition) in done:
                continue
            conversation = messages(case, condition)
            value, finish, usage = complete(model, tokenizer, conversation)
            row = {
                "model": MODEL, "case_id": case["id"], "topic": case["topic"], "condition": condition,
                "order": index, "timestamp_utc": utc(), "messages": conversation, "expected": case["expected"],
                "content": value, "usage": usage, "finish_reason": finish, "score": classify(value, case["expected"]),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 12 == 0:
                print(MODEL, f"{index}/{len(schedule)}", flush=True)


if __name__ == "__main__":
    main()
