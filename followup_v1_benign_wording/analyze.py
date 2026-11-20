"""Verify the frozen BlueDot dataset and render the observed outcomes and design."""
import csv
import hashlib
from itertools import product
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from run import CONDITIONS, MODELS, classify, frozen_hashes, messages, read_cases

ROOT = Path(__file__).resolve().parent
LABELS = {"neutral": "Plain wording", "risk_cued": "Risk-associated", "irrelevant": "Irrelevant label"}
DIAGNOSTIC_MODEL = "qwen3-4b"
ADDED_MODEL = "qwen25-1.5b"
VALID_MODELS = ("qwen35-4b", ADDED_MODEL)
DISPLAY_NAMES = {"qwen35-4b": "Qwen3.5 4B", ADDED_MODEL: "Qwen2.5 1.5B"}
COLORS = {"neutral": "#26786d", "risk_cued": "#934575", "irrelevant": "#6b7691"}
METRICS = ("explicit_refusal", "value_correct", "literal_format")


def verify() -> dict:
    freeze = json.loads((ROOT / "freeze.json").read_text())
    assert freeze["sha256"] == frozen_hashes()
    fallback = json.loads((ROOT / "freeze_qwen25_fp32.json").read_text())
    for key, name in (
        ("original_freeze_sha256", "freeze.json"),
        ("amendment_sha256", "AMENDMENT_QWEN25_FP32.md"),
        ("runner_sha256", "run_qwen25.py"),
    ):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == fallback[key]
    checks = json.loads((ROOT / "qwen25_runtime_controls_fp32.json").read_text())
    assert len(checks) == 2 and all(check["score"]["value_correct"] for check in checks)
    cases = {row["id"]: row for row in read_cases()}
    records = {}
    for model in (*MODELS, ADDED_MODEL):
        model_freeze = fallback if model == ADDED_MODEL else freeze
        path = ROOT / f"raw_{model}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        keys = {(row["case_id"], row["condition"]) for row in rows}
        assert len(rows) == len(keys) == 72
        assert keys == set(product(cases, CONDITIONS))
        for row in rows:
            case = cases[row["case_id"]]
            cond = row["condition"]
            assert row["model"] == model and row["topic"] == case["topic"]
            assert row["messages"] == messages(case, cond)
            assert row["expected"] == case["expected"]
            assert row["score"] == classify(row["content"], case["expected"])
            assert row["timestamp_utc"] > model_freeze["frozen_utc"]
            records[model, row["case_id"], cond] = row
    return records


def exact_sign_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(discordant, i) for i in range(min(gains, losses) + 1)) / (2 ** discordant))


def summarize(records: dict) -> dict:
    case_ids = sorted({key[1] for key in records})
    cells = []
    contrasts = []
    token_balance = []
    flagged = []
    diagnostic = [row for (model, _, _), row in records.items() if model == DIAGNOSTIC_MODEL]
    assert len(diagnostic) == 72
    assert all(row["finish_reason"] == "length" and not row["score"]["value_correct"] for row in diagnostic)
    for model in VALID_MODELS:
        for cond in CONDITIONS:
            rows = [records[model, id_, cond] for id_ in case_ids]
            counts = {metric: sum(row["score"][metric] for row in rows) for metric in METRICS}
            cells.append({
                "model": model, "condition": cond, "n": len(rows),
                **counts,
                "truncated": sum(row["finish_reason"] == "length" for row in rows),
                "zero_refusal_95pct_binomial_upper": 1 - 0.05 ** (1 / len(rows)) if counts["explicit_refusal"] == 0 else None,
            })
            flagged.extend(row for row in rows if row["score"]["explicit_refusal"] or not row["score"]["value_correct"] or row["finish_reason"] == "length")
        for control, metric in product(("neutral", "irrelevant"), ("explicit_refusal", "value_correct")):
            pairs = [(records[model, id_, "risk_cued"]["score"][metric], records[model, id_, control]["score"][metric]) for id_ in case_ids]
            gains = sum(first and not second for first, second in pairs)
            losses = sum(second and not first for first, second in pairs)
            contrasts.append({
                "model": model, "control": control, "metric": metric, "pairs": len(pairs),
                "risk_only": gains, "control_only": losses,
                "risk_minus_control_pp": 100 * (gains - losses) / len(pairs),
                "exact_paired_p_descriptive": exact_sign_p(gains, losses),
            })
        for control in ("neutral", "irrelevant"):
            differences = [
                records[model, id_, "risk_cued"]["usage"]["prompt_tokens"]
                - records[model, id_, control]["usage"]["prompt_tokens"]
                for id_ in case_ids
            ]
            token_balance.append({
                "model": model, "control": control, "equal_token_pairs": differences.count(0),
                "risk_minus_control_tokens_min": min(differences),
                "risk_minus_control_tokens_max": max(differences),
            })
    return {
        "records": len(records),
        "behavioral_comparison_records": len(records) - len(diagnostic),
        "runtime_diagnostic": {"model": DIAGNOSTIC_MODEL, "n": len(diagnostic), "truncated_without_answer": len(diagnostic)},
        "cells": cells,
        "paired_contrasts": contrasts,
        "token_balance": token_balance,
        "flagged_rows": flagged,
    }


def plot_outcomes(cells: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.6), layout="constrained", sharey=True)
    for axis, metric, title in zip(
        axes, ("explicit_refusal", "value_correct"), ("Explicit refusals", "Correct short answers")
    ):
        for model_index, model in enumerate(VALID_MODELS):
            for cond_index, cond in enumerate(CONDITIONS):
                row = next(r for r in cells if r["model"] == model and r["condition"] == cond)
                x = model_index + (cond_index - 1) * 0.24
                count = row[metric]
                axis.bar(x, count, width=0.21, color=COLORS[cond], label=LABELS[cond] if model_index == 0 else "")
                axis.text(x, count + 0.4, f"{count}/{row['n']}", ha="center", va="bottom", fontsize=8)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks([0, 1], [DISPLAY_NAMES[model] for model in VALID_MODELS])
        axis.set_ylim(0, 27.5)
        axis.set_yticks([0, 6, 12, 18, 24])
        axis.grid(axis="y", color="#e9eaf1")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Tasks")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("BlueDot follow-up: the same benign task under three labels", x=0.035, ha="left", fontweight="bold")
    fig.savefig(ROOT / "outcomes.png", dpi=180)
    plt.close(fig)


def plot_design() -> None:
    fig, axis = plt.subplots(figsize=(10.5, 4.3), layout="constrained")
    axis.set_xlim(0, 10.55)
    axis.set_ylim(0, 4.3)
    axis.axis("off")
    boxes = [
        (0.15, 1.47, 2.05, 1.12, "24 benign tasks\nSame task in each triplet", "#daeae7"),
        (2.82, 2.92, 2.25, 0.65, "Plain-language label", "#daeae7"),
        (2.82, 1.70, 2.25, 0.65, "Risk-associated label", "#f5e7ef"),
        (2.82, 0.48, 2.25, 0.65, "Irrelevant label", "#e9ecf4"),
        (5.72, 1.47, 2.0, 1.12, "Two evaluated checkpoints\n72 generations each", "#e9ecf4"),
        (8.32, 1.47, 2.05, 1.12, "Refusal, correct value\nand exact formatting", "#daeae7"),
    ]
    for x, y, width, height, title, color in boxes:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.065,rounding_size=0.10", facecolor=color, edgecolor="#8b99a7", linewidth=0.9))
        axis.text(x + width / 2, y + height / 2, title, ha="center", va="center", fontsize=10, color="#1d2633")
    for y in (3.245, 2.025, 0.805):
        axis.add_patch(FancyArrowPatch((2.23, 2.03), (2.78, y), arrowstyle="-|>", mutation_scale=12, color="#6b7691"))
        axis.add_patch(FancyArrowPatch((5.11, y), (5.68, 2.03), arrowstyle="-|>", mutation_scale=12, color="#6b7691"))
    axis.add_patch(FancyArrowPatch((7.78, 2.03), (8.28, 2.03), arrowstyle="-|>", mutation_scale=12, color="#6b7691"))
    axis.text(0.12, 4.06, "Matched wording evaluation", fontsize=16, fontweight="bold", color="#1d2633")
    axis.text(2.82, 0.14, "Same position and word count within each task", fontsize=8, color="#47535f")
    fig.savefig(ROOT / "design.png", dpi=180)
    plt.close(fig)


def main() -> None:
    result = summarize(verify())
    with (ROOT / "outcomes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["cells"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["cells"])
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot_outcomes(result["cells"])
    plot_design()
    print(json.dumps({"records": result["records"], "counts": result["cells"], "flagged": len(result["flagged_rows"])}, indent=2))


if __name__ == "__main__":
    main()
