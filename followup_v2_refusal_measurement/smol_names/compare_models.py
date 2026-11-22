"""Compare matched opening effects in Qwen and SmolLM2 on one frozen bank."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fresh_names.analysis import OUTCOMES, PAIRS, exact_p, load_cohort, outcome

MODELS = {"Qwen": ("Qwen/Qwen2.5-1.5B-Instruct", HERE.parent / "fresh_names"),
          "SmolLM2": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", HERE)}
ENDPOINTS = ("political_refusal", "all_full_refusals")
MILESTONE_LABELS = ("Become a billionaire", "First reach billionaire status", "Net worth first reach US$1B")


def save_table(filename, rows):
    with (HERE / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_panels():
    panels = {}
    for model, (checkpoint, directory) in MODELS.items():
        run = json.loads((directory / "run.json").read_text())
        assert run["model"] == checkpoint
        names, panel = load_cohort(directory, 30, "ABCDEF")
        panels[model] = panel
    assert (HERE / "bank.json").read_bytes() == (HERE.parent / "fresh_names/bank.json").read_bytes()
    assert panels["Qwen"].keys() == panels["SmolLM2"].keys()
    for key, qwen in panels["Qwen"].items():
        smol = panels["SmolLM2"][key]
        assert (qwen["id"], qwen["prompt"]) == (smol["id"], smol["prompt"])
    audits = [json.loads((directory / "coding_audit.json").read_text()) for _, directory in MODELS.values()]
    assert audits[0]["rubric_sha256"] == audits[1]["rubric_sha256"]
    return names, panels


def condition_counts(names, panels):
    rows = []
    for model, panel in panels.items():
        for arm in "ABCDEF":
            replies = [panel[name, arm] for name in names]
            tally = Counter(outcome(row) for row in replies)
            forms = {row["prompt"].replace(row["name"], "{name}") for row in replies}
            assert len(forms) == 1 and sum(tally.values()) == len(names)
            rows.append(dict(model=model, arm=arm, template=forms.pop(), names=len(names),
                             **{label: tally[label] for label in OUTCOMES},
                             all_full_refusals=tally["political_refusal"] + tally["other_refusal"],
                             any_political_rationale=sum(row["political_rationale"] == "yes" for row in replies)))
    return rows


def sign_flip_test(changes):
    """Exact two-sided test of sign symmetry, conditional on the absolute person differences."""
    distribution = Counter({0: 1})
    for magnitude in (abs(d) for d in changes if d):
        updated = Counter()
        for total, count in distribution.items():
            updated[total - magnitude] += count
            updated[total + magnitude] += count
        distribution = updated
    observed = abs(sum(changes))
    tail = sum(count for total, count in distribution.items() if abs(total) >= observed)
    assignments = sum(distribution.values())
    assert assignments == 2 ** sum(d != 0 for d in changes)
    return tail, assignments


def paired_tables(names, panels):
    individuals, contrasts, interactions = [], [], []
    for when, year, milestone in PAIRS:
        role = "primary" if when == "A" else "secondary"
        names_by_pair = []
        for name in names:
            row = dict(name=name, when_arm=when, year_arm=year,
                       when_id=panels["Qwen"][name, when]["id"], year_id=panels["Qwen"][name, year]["id"])
            for model, panel in panels.items():
                first, second = panel[name, when], panel[name, year]
                assert first["prompt"].replace("When did ", "In what year did ", 1) == second["prompt"]
                for opening, record in (("when", first), ("year", second)):
                    state = outcome(record)
                    row[f"{model}_{opening}_behavior"] = record["behavior"]
                    row[f"{model}_{opening}_rationale"] = record["political_rationale"]
                    row[f"{model}_{opening}_political_refusal"] = int(state == "political_refusal")
                    row[f"{model}_{opening}_all_full_refusals"] = int(record["behavior"] == "decline")
                for endpoint in ENDPOINTS:
                    row[f"{model}_reduction_{endpoint}"] = row[f"{model}_when_{endpoint}"] - row[f"{model}_year_{endpoint}"]
            for endpoint in ENDPOINTS:
                row[f"Qwen_minus_SmolLM2_reduction_{endpoint}"] = row[f"Qwen_reduction_{endpoint}"] - row[f"SmolLM2_reduction_{endpoint}"]
            names_by_pair.append(row)
        individuals.extend(names_by_pair)

        for endpoint in ENDPOINTS:
            for model in MODELS:
                counts = Counter((r[f"{model}_when_{endpoint}"], r[f"{model}_year_{endpoint}"])
                                 for r in names_by_pair)
                lost, gained = counts[1, 0], counts[0, 1]
                primary_test = role == "primary" and endpoint == "political_refusal"
                contrasts.append(dict(model=model, when_arm=when, year_arm=year, milestone=milestone,
                                      role=role, endpoint=endpoint, names=len(names),
                                      when_count=lost + counts[1, 1], year_count=gained + counts[1, 1],
                                      when_only=lost, year_only=gained, both=counts[1, 1], neither=counts[0, 0],
                                      when_minus_year_pp=100 * (lost - gained) / len(names),
                                      two_sided_mcnemar_p=exact_p(lost, gained) if primary_test else ""))
            differences = [r[f"Qwen_minus_SmolLM2_reduction_{endpoint}"] for r in names_by_pair]
            tally = Counter(differences)
            assert set(tally).issubset({-2, -1, 0, 1, 2})
            primary_test = role == "primary" and endpoint == "political_refusal"
            tail, assignments = sign_flip_test(differences) if primary_test else (None, None)
            interactions.append(dict(when_arm=when, year_arm=year, milestone=milestone, role=role,
                                     endpoint=endpoint, names=len(names),
                                     **{f"n_difference_{d}": tally[d] for d in range(-2, 3)},
                                     Qwen_minus_SmolLM2_reduction_pp=100 * sum(differences) / len(names),
                                     null_tail_assignments=tail, null_total_assignments=assignments,
                                     two_sided_sign_flip_p=tail / assignments if primary_test else ""))
    return individuals, contrasts, interactions


def draw_figure(contrasts):
    colors = ("#99492f", "#285d71")
    ink = "#233740"
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 4.8))
    fig.subplots_adjust(left=.253, right=.987, top=.85, bottom=.12, hspace=.46, wspace=.15)
    for i, (endpoint, title) in enumerate(zip(ENDPOINTS, ("Full refusals citing politics", "All full refusals"))):
        for j, (model, model_title) in enumerate(zip(MODELS, ("Qwen2.5-1.5B", "SmolLM2-1.7B"))):
            ax = axes[i, j]
            rows = [r for r in contrasts if r["model"] == model and r["endpoint"] == endpoint]
            assert len(rows) == 3
            for y, row in zip((2, 1, 0), rows):
                for offset, count, color in zip((.16, -.16), (row["when_count"], row["year_count"]), colors):
                    ax.barh(y + offset, count, height=.23, color=color, zorder=2)
                    ax.text(count + .35, y + offset, f"{count}/30", va="center", ha="left",
                            fontsize=9.1, color=ink)
            ax.set(xlim=(0, 34.5), ylim=(-.53, 2.53), xticks=(0, 10, 20, 30), yticks=(2, 1, 0))
            ax.set_yticklabels(MILESTONE_LABELS if j == 0 else [])
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#d9e3e5", lw=.75)
            ax.tick_params(axis="both", length=0, labelsize=9, colors=ink)
            ax.tick_params(axis="y", pad=9)
            ax.spines[:].set_visible(False)
            if i == 0:
                ax.set_title(model_title, fontsize=12, loc="left", weight="bold", color=ink, pad=12)
        fig.text(.012, .883 if i == 0 else .467, title, fontsize=10.5, color=ink, weight="bold")
    handles = [Patch(facecolor=color, label=label)
               for color, label in zip(colors, ("When did …?", "In what year did …?"))]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.245, -.003),
               frameon=False, ncol=2, fontsize=10, columnspacing=1.6, handlelength=1.35)
    for filetype in ("png", "pdf"):
        fig.savefig(HERE / f"model_comparison.{filetype}", dpi=260, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def main():
    names, panels = load_panels()
    counts = condition_counts(names, panels)
    individual, contrasts, interaction = paired_tables(names, panels)
    save_table("condition_counts.csv", counts)
    save_table("person_pairs.csv", individual)
    save_table("contrasts.csv", contrasts)
    save_table("interaction.csv", interaction)
    draw_figure(contrasts)
    for row in interaction:
        if row["endpoint"] == "political_refusal":
            print(row["when_arm"], row["year_arm"], row["Qwen_minus_SmolLM2_reduction_pp"], row["two_sided_sign_flip_p"])


if __name__ == "__main__":
    main()
