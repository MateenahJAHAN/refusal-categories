"""Make the paper's four main figures and three diagnostic panels from recorded scores."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT/"figures"
OUT.mkdir(exist_ok=True)
INK="#172e3d"; TEAL="#00787a"; RUST="#a54f34"; SKY="#d4ebed"; GREY="#536773"; PALE="#e6ebed"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelcolor":INK,"text.color":INK,"xtick.color":GREY,"ytick.color":INK,"axes.edgecolor":"#abb9bf","axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,"axes.axisbelow":True})
def save(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=230,bbox_inches="tight",pad_inches=.1,facecolor="white")
    plt.close(fig)

def design():
    fig,ax=plt.subplots(figsize=(9.5,3.25))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    def box(x,y,w,h,title,desc,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.007,rounding_size=.015",fc=fill,ec="#8ca2ac",lw=.9))
        ax.text(x+w/2,y+h*.70,title,ha="center",va="center",weight="bold",fontsize=10)
        ax.text(x+w/2,y+h*.33,desc,ha="center",va="center",fontsize=9,linespacing=1.23)
    def arrow(x,y,x2,y2):
        ax.annotate("",(x2,y2),(x,y),arrowprops={"arrowstyle":"-|>","color":GREY,"lw":1.3,"shrinkA":2.5,"shrinkB":3})
    box(.02,.40,.21,.37,"250 safe requests","10 XSTest types\n25 questions per type",SKY)
    box(.32,.59,.39,.35,"Forced choice · two checkpoints","Appropriate or predicted action\nLetters A/B or words Answer/Pass", "#eaf3f3")
    box(.32,.08,.39,.35,"Free answer · Qwen only","125 questions across five types\nOne AI reader scored each reply","#f7f0ea")
    box(.79,.39,.19,.38,"Paired evaluation","7 full declines\n107 engaged replies", "#eaf0f3")
    arrow(.235,.60,.312,.76);arrow(.235,.52,.312,.26);arrow(.718,.74,.785,.62);arrow(.718,.26,.785,.50)
    ax.text(.89,.28,"11 mixed replies\nreported separately",ha="center",va="center",fontsize=8.7,color=GREY)
    save(fig,"01_design")

def swaps():
    r=json.loads((ROOT/"results.json").read_text())
    fig,(left,right)=plt.subplots(1,2,figsize=(9.6,3.0),gridspec_kw={"width_ratios":[1.35,1]},layout="constrained")
    labs=["A · first","B · first","A · last","B · last"];ks=["help_A_first","help_B_first","help_A_last","help_B_last"]
    for i,(key,color,label,mk) in enumerate([("qwen",TEAL,"Qwen","o"),("smol",RUST,"SmolLM2","s")]):
        v=r[key]["xstest"]; ys=np.arange(4)+(i-.5)*.18; n=[v[k] for k in ks]
        left.scatter(n,ys,color=color,marker=mk,s=31,label=label,zorder=3)
        x=v["code_flip_pairs"]/5; lo,hi=np.array(v["composition_interval"])*100
        right.errorbar(x,i,xerr=np.array([[x-lo],[hi-x]]),fmt=mk,color=color,capsize=4,lw=1.8,ms=7)
        right.text(x,i-.22,f'{v["code_flip_pairs"]}/500',ha="center",fontsize=9)
    left.set_yticks(range(4),labs);left.set_ylim(3.5,-.5);left.set_xlim(90,261);left.set_xticks([100,150,200,250])
    left.set_xlabel("Help decisions out of 250 questions");left.set_title("A   Letter assigned to help and its position",loc="left",size=10,weight="bold")
    left.legend(frameon=False,fontsize=8.5,loc="lower left",ncol=2)
    right.set_yticks([0,1],["Qwen","SmolLM2"]);right.set(xlim=(0,21.5),ylim=(1.47,-.51),xlabel="Decisions reversed (%)");right.set_xticks([0,5,10,15,20])
    right.set_title("B   Swapping letters at fixed order",loc="left",size=10,weight="bold")
    for a in (left,right): a.grid(axis="x",color=PALE); a.tick_params(axis="y",length=0)
    save(fig,"02_letters")

def exact_matrix():
    r=json.loads((ROOT/"continued_results.json").read_text())["codebooks"]
    key=["help_A_first","help_B_first","help_A_last","help_B_last"]
    mat=np.array([[r[k].get("decline_decline",0) for k in key],[r[k].get("engaged_decline",0) for k in key]])
    den=np.array([7,107])[:,None]
    fig,ax=plt.subplots(figsize=(8.5,2.0),layout="constrained")
    cm=LinearSegmentedColormap.from_list("bl",["#f1f4f5","#93c7ca","#1e6f75"])
    ax.imshow(mat/den,vmin=0,vmax=.65,cmap=cm,aspect="auto")
    for i,j in np.ndindex(mat.shape):
        ax.text(j,i,f"{mat[i,j]}/{den[i,0]}  ({100*mat[i,j]/den[i,0]:.0f}%)",ha="center",va="center",fontsize=10,weight="bold")
    ax.set_yticks([0,1],["Full decline","Engaged"]);ax.tick_params(axis="both",length=0)
    ax.set_xticks(range(4),["Help = A, first","Help = B, first","Help = A, last","Help = B, last"]);ax.xaxis.tick_top()
    for sp in ax.spines.values():sp.set_visible(False)
    for j in [.5,1.5,2.5]:ax.axvline(j,color="white",lw=2)
    ax.axhline(.5,color="white",lw=2)
    save(fig,"03_direct")

def contexts():
    data=list(csv.DictReader((ROOT/"benchmark_types.csv").open()))
    short={"definitions":"Definitions","figurative_language":"Figurative language","historical_events":"Historical events","homonyms":"Homonyms","nons_group_real_discr":"Fictitious group, real predicate","privacy_fictional":"Privacy, fictional subject","privacy_public":"Privacy, public information","real_group_nons_discr":"Real group, fictitious predicate","safe_contexts":"Safe contexts","safe_targets":"Safe targets"}
    labels=list(short)
    fig,ax=plt.subplots(figsize=(8.6,3.7),layout="constrained")
    for i,(key,clr,marker,label) in enumerate([("qwen",TEAL,"o","Qwen"),("smol",RUST,"s","SmolLM2")]):
        d={r["family"]:int(r["code_flip_pairs"]) for r in data if r["model"]==key}
        ax.scatter([d[k] for k in labels],np.arange(10)+(i-.5)*.26,color=clr,marker=marker,label=label,s=32,zorder=3)
    ax.set_yticks(range(10),[short[k] for k in labels],fontsize=9);ax.invert_yaxis();ax.set_xlim(-.6,21.4);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel("Letter swaps that reversed the decoded action (50 pairs per type)")
    ax.grid(axis="x",color=PALE);ax.tick_params(axis="y",length=0);ax.legend(frameon=False,fontsize=9)
    save(fig,"S1_types")

def length_check():
    data=list(csv.DictReader((ROOT/"direct_paired.csv").open()))
    fig,ax=plt.subplots(figsize=(8.6,2.65),layout="constrained")
    for j,(category,label,color) in enumerate([("engaged","Engaged",TEAL),("mixed","Mixed",RUST),("decline","Full decline","#435c83")]):
        values=np.array(sorted(int(r["tokens"]) for r in data if r["extended_type"]==category))
        offsets=((np.arange(len(values))*0.61803398875)%1-.5)*.47
        ax.scatter(values,j+offsets,alpha=.73,s=16,facecolor=color,edgecolor="white",linewidth=.35)
    ax.axvline(96,color=GREY,lw=1,ls="--");ax.text(97,-.56,"Initial limit: 96",fontsize=8.7,va="top")
    ax.set(ylim=(2.55,-.65),xlim=(0,418),xlabel="Generated tokens at completion");ax.set_yticks(range(3),["Engaged (107)","Mixed (11)","Full decline (7)"])
    ax.grid(axis="x",color=PALE);ax.tick_params(axis="y",length=0)
    save(fig,"S2_length")

def auc(r=None):
    if r is None:
        r=json.loads((HERE/"results.json").read_text())
    fig,(a,b)=plt.subplots(1,2,figsize=(9.5,2.9),gridspec_kw={"width_ratios":[1,1.02]},layout="constrained")
    keys=["appropriate_letter","predicted_letter","appropriate_action","predicted_action"]
    labels=["Appropriate · A/B","Predicted · A/B","Appropriate · words","Predicted · words"]
    for i,key in enumerate(keys):
        rr=r["auc"][key];v=rr["estimate"];lo,hi=rr["interval"];c=TEAL if "action" in key else GREY
        a.errorbar(v,i,xerr=np.array([[v-lo],[hi-v]]),fmt="o",color=c,capsize=3)
    a.axvline(.5,ls="--",lw=1,color=RUST);a.set(xlim=(0,1),ylim=(3.6,-.6),xlabel="Area under the ROC curve");a.set_xticks([0,.25,.5,.75,1]);a.set_yticks(range(4),labels,fontsize=9)
    a.set_title("A   Ranking declines against engaged answers",loc="left",size=9.7,weight="bold");a.grid(axis="x",color=PALE);a.tick_params(axis="y",length=0)
    for j,(outcome,label,den,marker,c) in enumerate([("decline","Full declines",7,"o",TEAL),("engaged","Engaged",107,"s",RUST)]):
        vals=[100*r["consensus"][k][outcome]["decline"]/den for k in keys]
        b.scatter(vals,np.arange(4)+(j-.5)*.18,color=c,marker=marker,s=30,label=label,zorder=2)
    b.set(xlim=(-3,103),ylim=(3.6,-.6),xlabel="Classified as decline (%)");b.set_yticks(range(4),[""]*4);b.set_xticks([0,25,50,75,100]);b.grid(axis="x",color=PALE);b.tick_params(axis="y",length=0)
    b.set_title("B   Decisions at a fixed zero margin",loc="left",size=9.7,weight="bold");b.legend(frameon=False,fontsize=8,loc="upper right")
    save(fig,"04_semantics")

def case_mix():
    with (HERE/"response_sensitivity.csv").open() as fh:
        d={r["contrast"]:r for r in csv.DictReader(fh)}
    display=[("all_binary","All full declines vs engaged"),("within_same_type","Within question type"),("mixed_as_decline","Mixed grouped with full declines"),("mixed_as_engaged","Mixed grouped with engaged"),("omit_type_definitions","Omit definitions"),("omit_type_figurative_language","Omit figurative language"),("omit_type_homonyms","Omit homonyms"),("omit_type_nons_group_real_discr","Omit fictitious-group questions"),("omit_type_real_group_nons_discr","Omit real-group questions")]
    fig,ax=plt.subplots(figsize=(8.5,3.2),layout="constrained")
    for i,(k,label) in enumerate(display):
        row=d[k];v=[float(row["appropriate_letter_auc"]),float(row["predicted_action_auc"])]
        ax.plot(v,[i,i],color="#bdc9c9",lw=1.8)
        ax.scatter(v[0],i,s=29,marker="s",color=GREY,label="Appropriate · A/B" if i==0 else None,zorder=3)
        ax.scatter(v[1],i,s=34,marker="o",color=TEAL,label="Predicted · words" if i==0 else None,zorder=3)
    ax.axvline(.5,color=RUST,ls="--",lw=.9);ax.axhline(3.5,color=PALE,lw=1)
    ax.set(xlim=(.25,.91),ylim=(8.65,-.6),xlabel="Area under the ROC curve")
    ax.set_xticks([.3,.4,.5,.6,.7,.8,.9]);ax.set_yticks(range(len(display)),[s for _,s in display],size=9)
    ax.tick_params(axis="y",length=0);ax.grid(axis="x",color=PALE);ax.legend(frameon=False,fontsize=8,loc="center right")
    save(fig,"S3_case_mix")

if __name__=="__main__":
    design();swaps();exact_matrix();contexts();length_check()
    if (HERE/"results.json").exists():auc()
    if (HERE/"response_sensitivity.csv").exists():case_mix()
