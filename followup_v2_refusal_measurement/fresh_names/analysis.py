"""Count political refusals in matched pairs of Qwen replies and draw the replication figure."""
import csv
import hashlib
import json
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent
COHORTS = (("original", ROOT.parent / "name_wording", 10, "ABCDEFGH"),
           ("fresh", ROOT, 30, "ABCDEF"))
PAIRS = (("A", "B", "Become a billionaire"),
         ("C", "D", "First reach billionaire status"),
         ("E", "F", "Net worth first reach one billion US dollars"))
OUTCOMES = ("engaged", "political_refusal", "other_refusal", "mixed", "indeterminate")
PAIR_LABELS = {(True, False): "when_only", (False, True): "year_only",
               (True, True): "both", (False, False): "neither"}


def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_table(filename, rows):
    with (ROOT / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_cohort(directory, size, conditions):
    bank = json.loads((directory / "bank.json").read_text())
    raw_file = directory / "raw.jsonl"
    raw = [json.loads(line) for line in raw_file.read_text().splitlines()]
    archive = json.loads((directory / "run.json").read_text())
    assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == archive["raw_sha256"]
    coded_file = directory / "coded_replies.csv"
    audit = json.loads((directory / "coding_audit.json").read_text())
    for path in (coded_file, raw_file, directory / "bank.json"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["sha256"][path.name]
    coded = read_table(coded_file)
    expected = size * len(conditions)
    prompts = {row["id"]: row for row in bank}
    replies = {row["id"]: row["response"] for row in raw}
    assert len(raw) == len(bank) == len(coded) == archive["completed"] == expected
    assert prompts.keys() == replies.keys() == {row["id"] for row in coded}
    assert len(prompts) == expected
    blind_keys = ("blind_id",) if "blind_id" in coded[0] else ("reader_A_blind_id", "reader_B_blind_id")
    for key in blind_keys:
        assert len({row[key] for row in coded}) == expected
    for row in coded:
        item = prompts[row["id"]]
        assert row["prompt"] == item["prompt"] and row["reply"] == replies[row["id"]]
        assert row["name"] == item["name"] and row["condition"] == item["arm"]
        assert row["behavior"] in ("engaged", "decline", "mixed", "indeterminate")
        assert row["political_rationale"] in ("yes", "no")
    names = sorted({row["name"] for row in coded})
    lookup = {(row["name"], row["condition"]): row for row in coded}
    assert len(names) == size and lookup.keys() == {(n, c) for n in names for c in conditions}
    return names, lookup


def outcome(row):
    if row["behavior"] == "decline":
        return "political_refusal" if row["political_rationale"] == "yes" else "other_refusal"
    return row["behavior"]


def exact_p(lost, gained):
    """Two-sided exact McNemar test, conditional on the number of discordant names."""
    discordant = lost + gained
    tail = sum(comb(discordant, x) for x in range(min(lost, gained) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def summarize():
    counts, contrasts, individual = [], [], []
    templates = {}
    previous_names = set()
    for group, directory, size, conditions in COHORTS:
        names, items = load_cohort(directory, size, conditions)
        assert previous_names.isdisjoint({n.casefold() for n in names})
        previous_names.update(n.casefold() for n in names)
        for condition in "ABCDEF":
            records = [items[n, condition] for n in names]
            forms = {r["prompt"].replace(r["name"], "{name}") for r in records}
            assert len(forms) == 1
            template = forms.pop()
            assert templates.setdefault(condition, template) == template
            tally = Counter(outcome(r) for r in records)
            counts.append({"cohort": group, "condition": condition, "template": template,
                           "names": size, **{key: tally[key] for key in OUTCOMES},
                           "all_full_refusals": tally["political_refusal"] + tally["other_refusal"],
                           "any_political_rationale": sum(r["political_rationale"] == "yes" for r in records)})
        for when, year, milestone in PAIRS:
            transitions = Counter()
            for name in names:
                first, second = items[name, when], items[name, year]
                assert first["prompt"].startswith("When did ")
                assert first["prompt"].replace("When did ", "In what year did ", 1) == second["prompt"]
                first_outcome, second_outcome = outcome(first), outcome(second)
                key = PAIR_LABELS[first_outcome == "political_refusal", second_outcome == "political_refusal"]
                transitions[key] += 1
                individual.append({"cohort": group, "name": name, "when_arm": when, "year_arm": year,
                                   "when_id": first["id"], "year_id": second["id"],
                                   "when_outcome": first_outcome, "year_outcome": second_outcome,
                                   "political_refusal_transition": key})
            lost, gained = transitions["when_only"], transitions["year_only"]
            assert sum(transitions.values()) == size
            inference = "exploratory" if group == "original" else ("primary" if when == "A" else "secondary")
            contrasts.append({"cohort": group, "when_arm": when, "year_arm": year,
                              "milestone": milestone, "analysis": inference, "names": size,
                              "when_political_refusals": lost + transitions["both"],
                              "year_political_refusals": gained + transitions["both"],
                              **{key: transitions[key] for key in PAIR_LABELS.values()},
                              "year_minus_when_percentage_points": 100 * (gained - lost) / size,
                              "exact_two_sided_mcnemar_p": exact_p(lost, gained) if group == "fresh" else ""})
    write_table("condition_counts.csv", counts)
    write_table("contrasts.csv", contrasts)
    write_table("name_pairs.csv", individual)
    return contrasts


def draw(contrasts):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    colors = ("#99492f", "#285d71")
    ink = "#233740"
    fig = plt.figure(figsize=(10, 3.65))
    grid = fig.add_gridspec(1, 3, left=.170, right=.993, top=.78, bottom=.24,
                            width_ratios=(1, 1, 1.14), wspace=.20)
    tick_percent = (0, 50, 100)
    centers = (2, 1, 0)
    for column, (cohort, _, size, _) in enumerate(COHORTS):
        ax = fig.add_subplot(grid[0, column])
        rows = [r for r in contrasts if r["cohort"] == cohort]
        for y, row in zip(centers, rows):
            for dy, key, color in zip((.165, -.165), ("when_political_refusals", "year_political_refusals"), colors):
                count = row[key]
                edge = count * 100 / size
                ax.barh(y + dy, edge, height=.22, color=color, zorder=2)
                ax.text(edge + 2, y + dy, f"{count}/{size}", ha="left", va="center", color=ink, fontsize=10)
        ax.set(xlim=(0, 121), ylim=(-.5, 2.5), xticks=tick_percent,
               xticklabels=[f"{n}%" for n in tick_percent])
        cohort_title = "Original" if cohort == "original" else "Fresh"
        ax.set_title(f"{cohort_title} names (n = {size})", fontsize=11.4, fontweight="bold", loc="left", pad=12)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, length=0, colors="#48616b")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d9e3e5", lw=.75)
        ax.set_yticks(centers)
        ax.set_yticklabels(("Become a\nbillionaire", "First reach\nbillionaire status",
                            "Net worth first reach\nUS$1 billion") if column == 0 else [])
        ax.tick_params(axis="y", labelsize=10, length=0, pad=9, colors=ink)
    ax = fig.add_subplot(grid[0, 2])
    ax.set(xlim=(0, 4), ylim=(-.5, 2.5))
    ax.axis("off")
    ax.set_title("Fresh names by response pair", loc="center", fontsize=11.4, fontweight="bold", pad=29)
    headers = ("When\nonly", "Year\nonly", "Both", "Neither")
    for i, header in enumerate(headers):
        ax.text(i + .5, 2.58, header, fontsize=9.5, color=ink, ha="center", va="center", linespacing=1)
    for y, row in zip(centers, [r for r in contrasts if r["cohort"] == "fresh"]):
        for i, (key, fill) in enumerate(zip(("when_only", "year_only", "both", "neither"),
                                             ("#f5e6df", "#e0eef2", "#e9e7ee", "#f1f2ef"))):
            ax.add_patch(Rectangle((i+.035, y-.36), .93, .72, facecolor=fill, edgecolor="white", lw=.6))
            ax.text(i + .5, y, str(row[key]), ha="center", va="center", fontsize=12, color=ink)
    handles = [Patch(facecolor=c, label=s) for c, s in zip(colors, ("When did …?", "In what year did …?"))]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.16,.02), ncol=2,
               columnspacing=1.55, handlelength=1.25, handletextpad=.5,
               frameon=False, fontsize=10)
    fig.text(.54, .09, "Bars show full refusals explicitly citing politics.", color=ink, va="center", fontsize=9.6)
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"replication.{extension}", dpi=260, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


if __name__ == "__main__":
    results = summarize()
    draw(results)
    for row in results:
        print(row["cohort"], row["when_arm"] + "/" + row["year_arm"],
              row["when_political_refusals"], row["year_political_refusals"], row["exact_two_sided_mcnemar_p"])
