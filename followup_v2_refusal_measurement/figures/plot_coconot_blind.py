"""Plot factual-question refusals and the overlap with B-first predictive choices."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "independent_benign"
scores = json.loads((DATA / "ai_first_pass/scored.json").read_text())
assert scores["pending_adjudication"] == 0
names = {"qwen":"Qwen2.5-1.5B", "smol":"SmolLM2-1.7B"}
with (DATA/"ai_first_pass/all_ratings.csv").open(newline="") as handle:
    reviews=list(csv.DictReader(handle))
observed_color, extra_color = "#255e72", "#cf8a46"
fig, axes = plt.subplots(1,2,figsize=(8.5,2.65),sharex=True,sharey=True)
for ax,(key,name) in zip(axes,names.items()):
    direct={r["question_id"]:r["final_AI_code"] for r in reviews if r["model"].startswith(name+"-")}
    assert len(direct)==91 and set(direct.values())<={"decline","engaged"}
    selected={}
    for line in (DATA/"private"/f"readout_{key}.jsonl").read_text().splitlines():
        r=json.loads(line)
        if r["frame"]=="predicted" and r["kind"]=="letter" and r["help_label"]=="B" and r["help_first"]:
            selected[r["id"]]=r["logp_decline"]>r["logp_help"]
    assert set(selected)==set(direct)
    choices=json.loads((DATA/"private/item_scores.json").read_text())[key]
    for y,pred in enumerate([{i:b=="decline" for i,b in direct.items()},selected,
                              {i:choices[i]["predicted_action"]>0 for i in direct}]):
        hit=sum(v and direct[i]=="decline" for i,v in pred.items())
        extra=sum(v and direct[i]=="engaged" for i,v in pred.items())
        ax.barh(y,hit,height=.42,color=observed_color)
        ax.barh(y,extra,height=.42,left=hit,color=extra_color)
        if hit+extra==0:
            ax.plot(0,y,"o",ms=4,mfc="white",mec="#666",clip_on=False)
        ax.text(hit+extra+1.65,y,str(hit+extra),va="center",fontsize=9)
    ax.set_title(name,loc="left",fontsize=11,fontweight='semibold')
    ax.set_xlim(-.5,65);ax.set_xticks([0,20,40,60]);ax.set_ylim(2.55,-.55)
    ax.grid(axis="x",color="#aab3b8",alpha=.26,lw=.6);ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False);ax.spines["bottom"].set_color("#888")
    ax.tick_params(axis="y",length=0,pad=5);ax.tick_params(axis="x",labelsize=8)
axes[0].set_yticks([0,1,2],["Free reply","B-first letter probe","Averaged Answer/Pass"],fontsize=9)
fig.supxlabel("Questions out of 91",fontsize=9,y=.144)
fig.legend(handles=[Patch(color=observed_color,label="Full refusal in free reply"),Patch(color=extra_color,label="Engaged free reply flagged by probe")],
           loc="lower center",ncol=2,frameon=False,fontsize=8.5)
fig.subplots_adjust(left=.201,right=.985,top=.85,bottom=.34,wspace=.15)
fig.savefig(HERE/"05_coconot_behavior.png",dpi=210,bbox_inches="tight",pad_inches=.07)
