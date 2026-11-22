"""Summarize Qwen's generated answers across phrasings, with each person as their own control."""
import csv
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
CONDITIONS = "ABCDEFGH"
BEHAVIORS = ("engaged", "decline", "mixed", "indeterminate")
# Text is included as well as color so the matrix can be read in grayscale.
STYLES = {
    "engaged": ("Engaged", "·", "#e8eff1", "#788b92"),
    "political_refusal": ("Full refusal citing politics", "P", "#a24a30", "#ffffff"),
    "other_refusal": ("Other full refusal", "R", "#315b70", "#ffffff"),
    "mixed": ("Mixed", "M", "#f0cb82", "#433215"),
    "indeterminate": ("Indeterminate", "?", "#d5d1dc", "#342f47"),
}
CONTRASTS = [
    ("A", "E", "primary", "When: billionaire to explicit net worth"),
    ("B", "F", "secondary", "In what year: billionaire to explicit net worth"),
    ("A", "B", "descriptive", "Become a billionaire: When to In what year"),
    ("C", "D", "descriptive", "Reach billionaire status: When to In what year"),
    ("E", "F", "descriptive", "Explicit net worth: When to In what year"),
    ("A", "C", "descriptive", "When: become a billionaire to first reach billionaire status"),
    ("B", "D", "descriptive", "In what year: become a billionaire to first reach billionaire status"),
    ("B", "G", "descriptive", "Billionaire milestone: move the name to the beginning"),
    ("A", "G", "descriptive", "Original to person-first year question"),
    ("B", "H", "different-fact control", "In what year: billionaire milestone to birth"),
]


def load_responses():
    old_bank = json.loads((ROOT.parent / "independent_benign/source/bank.json").read_text())
    names = sorted(match[1] for row in old_bank if
                   (match := re.fullmatch(r"When did (.+) become a billionaire\?", row["prompt"])))
    assert len(names) == len(set(names)) == 10
    with (ROOT / "coded_replies.csv").open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    pair = {(r["name"], r["condition"]): r for r in records}
    assert len(records) == len(pair) == len({r["id"] for r in records}) == 80
    assert set(pair) == {(n, c) for n in names for c in CONDITIONS}
    for (name, condition), row in pair.items():
        assert name in row["prompt"] and row["behavior"] in BEHAVIORS
        assert row["political_rationale"] in {"yes", "no"}
        if condition == "A":
            assert row["prompt"] == f"When did {name} become a billionaire?"
    return names, pair


def state(row):
    if row["behavior"] == "decline":
        return "political_refusal" if row["political_rationale"] == "yes" else "other_refusal"
    return row["behavior"]


def save_table(stem, rows):
    with (ROOT / (stem + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(names, pairs):
    counts = []
    for c in CONDITIONS:
        rows = [pairs[n, c] for n in names]
        totals = Counter(state(r) for r in rows)
        row = dict(condition=c, wording=rows[0]["prompt"].replace(names[0], "[person]"),
                   n_names=len(names))
        row.update({k: totals[k] for k in STYLES})
        row["all_full_refusals"] = totals["political_refusal"] + totals["other_refusal"]
        row["any_political_rationale"] = sum(r["political_rationale"] == "yes" for r in rows)
        assert sum(totals.values()) == len(names)
        counts.append(row)
    save_table("condition_counts", counts)
    matched = []
    for start, end, role, description in CONTRASTS:
        both, neither, lost, gained = [], [], [], []
        for n in names:
            first = state(pairs[n, start]) == "political_refusal"
            second = state(pairs[n, end]) == "political_refusal"
            groups = {(True, True): both, (False, False): neither,
                      (True, False): lost, (False, True): gained}
            groups[first, second].append(n)
        discordant = len(lost) + len(gained)
        p = None
        if discordant and role in ("primary", "secondary"):
            tail = sum(comb(discordant, k) for k in range(min(len(lost), len(gained)) + 1))
            p = min(1.0, 2.0 * tail / (2 ** discordant))
        matched.append(dict(start=start, end=end, role=role, description=description,
                            n_names=len(names), both_political_refusal=len(both), neither_political_refusal=len(neither),
                            political_refusal_lost=len(lost), political_refusal_gained=len(gained),
                            end_minus_start_proportion=(len(gained) - len(lost)) / len(names),
                            exact_two_sided_p=p, names_lost=" | ".join(lost), names_gained=" | ".join(gained)))
    save_table("paired_changes", matched)
    name_table = []
    for n in names:
        by_condition = {c: state(pairs[n, c]) for c in CONDITIONS}
        name_table.append(dict(name=n, **by_condition,
                               n_political_refusal_across_seven_wealth_forms=sum(by_condition[c] == "political_refusal" for c in CONDITIONS[:7]),
                               n_political_rationale_across_eight=sum(pairs[n, c]["political_rationale"] == "yes" for c in CONDITIONS)))
    save_table("name_summary", name_table)
    return counts


def plot(names, pairs, counts):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.9, 4.55))
    ax = fig.add_axes((0.235, 0.225, 0.744, 0.566))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(9.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks(range(10), names, fontsize=9.2, color="#263a42")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.spines[:].set_visible(False)
    seen, qualified = set(), False
    for i, name in enumerate(names):
        for j, c in enumerate(CONDITIONS):
            row = pairs[name, c]
            category = state(row)
            label, glyph, color, ink = STYLES[category]
            seen.add(category)
            if row["political_rationale"] == "yes" and category != "political_refusal":
                glyph += "*"
                qualified = True
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.8, edgecolor="white", facecolor=color))
            ax.text(j, i, glyph, color=ink, ha="center", va="center", fontsize=10,
                    fontweight="normal" if category == "engaged" else "bold")
    for start, end, text in [(0, 1, "Become a\nbillionaire"), (2, 3, "First reach\nbillionaire status"),
                             (4, 5, "First reach\nUS$1 billion"), (6, 6, "Name first"), (7, 7, "Birth year")]:
        ax.text((start + end) / 2, 1.175, text, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9, fontweight="bold", color="#243b45", linespacing=1.05)
        ax.plot([start - 0.42, end + 0.42], [1.125, 1.125], transform=ax.get_xaxis_transform(),
                lw=.7, color="#94a4a9", clip_on=False)
    for x in (1.5, 3.5, 5.5, 6.5):
        ax.axvline(x, color="white", lw=2.3)
    for j, (c, short) in enumerate(zip(CONDITIONS, ["When", "Year", "When", "Year", "When", "Year", "Year", "Year"])):
        ax.text(j, 1.05, f"{c} · {short}", transform=ax.get_xaxis_transform(), ha="center",
                va="center", fontsize=8.5, color="#263a42")
        ax.text(j, -0.07, str(counts[j]["political_refusal"]), transform=ax.get_xaxis_transform(),
                fontweight="bold", ha="center", va="center", fontsize=10, color="#5c2e23")
    ax.text(-0.62, -0.07, "Political refusals / 10", transform=ax.get_xaxis_transform(), ha="right",
            va="center", fontsize=9.2, fontweight="bold", color="#463b39")
    handles = [Line2D([], [], color="none", marker="s", markersize=7.5, markeredgecolor="#b9c3c6",
                      markeredgewidth=.5, markerfacecolor=color, label=f"{glyph}  {label}")
               for cat, (label, glyph, color, _) in STYLES.items() if cat in seen]
    if qualified:
        handles.append(Line2D([], [], color="none", label="*  Political reason also cited"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.59, 0.025), ncol=min(len(handles),3),
               frameon=False, fontsize=8.7, columnspacing=1.35, handletextpad=.3)
    fig.savefig(ROOT / "wording_matrix.png", dpi=260, bbox_inches="tight", pad_inches=.11)
    fig.savefig(ROOT / "wording_matrix.pdf", bbox_inches="tight", pad_inches=.11)
    plt.close(fig)


def main():
    names, pairs = load_responses()
    counts = summarize(names, pairs)
    plot(names, pairs, counts)
    print("Full political refusals by wording:", {r["condition"]: r["political_refusal"] for r in counts})


if __name__ == "__main__":
    main()
