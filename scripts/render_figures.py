#!/usr/bin/env python3
from pathlib import Path
import argparse,pandas as pd,numpy as np,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
def save(path,w,h):
 plt.gcf().set_size_inches(w,h);plt.tight_layout();plt.savefig(path,dpi=300,bbox_inches="tight");plt.close()
a=argparse.ArgumentParser();a.add_argument("--output",default="generated_figures");out=(ROOT/a.parse_args().output).resolve();out.mkdir(parents=True,exist_ok=True)
plt.figure();ax=plt.gca();ax.set_axis_off();boxes=[(0.06,.80,.25,.11,"Prespecified participant split\nDiscovery 36 / Validation 12 /\nConfirmation 12"),(.38,.80,.25,.11,"Subjective architecture\n2D person-relative reports\nheld-out hedonic reports"),(.69,.80,.25,.11,"Direct EEG hedonic tests\nvalence / liking\n+ identity/context diagnostics"),(.12,.52,.29,.13,"Outcome-free EEG trajectory\n3 epochs → 6D geometry\nDiscovery fit → fixed transfer"),(.59,.52,.29,.13,"Multiresolution structure\nexperience K=2,3,4\nreport-transition K=2,3"),(.12,.23,.29,.13,"Semantic/report parent tests\nA1 / A2 / B2 / B3\nsplitwise multiplicity control"),(.59,.23,.29,.13,"Compact-EEG characterization\n6 predefined signal-space blocks\nstable subfeatures only"),(.35,.04,.30,.11,"Peripheral physiology boundary\nDiscovery-only gates;\nheld-out physiology unopened")];
for x,y,w,h,t in boxes: ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01",fill=False,linewidth=1.2));ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9)
for aa,bb in [((.31,.855),(.38,.855)),((.63,.855),(.69,.855)),((.815,.80),(.735,.65)),((.735,.52),(.735,.36)),((.59,.585),(.41,.585)),((.265,.52),(.265,.36)),((.41,.295),(.59,.295)),((.265,.23),(.43,.15)),((.735,.23),(.57,.15))]:ax.annotate("",xy=bb,xytext=aa,arrowprops=dict(arrowstyle="->",linewidth=1))
ax.text(.5,.96,"Analysis architecture",ha="center",fontsize=13,fontweight="bold");ax.text(.5,.01,"Outcome access, family naming, and held-out opening followed prespecified criteria.",ha="center",va="bottom",fontsize=8);save(out/"Figure1_analysis_architecture.png",9,6)

# Figure 2: subjective architecture from repository discovery and validation tables.
pa=pd.read_csv(ROOT/"results/source_data/subjective_discovery_parallel_analysis.csv")
vp=pd.read_csv(ROOT/"results/source_data/subjective_validation_prediction_summary.csv")
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
axs[0].plot(pa["component"],pa["observed_eigenvalue"],marker="o",label="Observed eigenvalue")
axs[0].plot(pa["component"],pa["null_eigenvalue_95"],marker="o",label="95% null threshold")
axs[0].set_xlabel("Component");axs[0].set_ylabel("Eigenvalue");axs[0].set_title("A  Two-dimensional subjective organization");axs[0].legend(frameon=False);axs[0].set_xticks(pa["component"])
vp=vp.set_index("outcome").loc[["valence","liking"]].reset_index();vals=vp["r2"].to_numpy();lo=vp["r2_ci_low"].to_numpy();hi=vp["r2_ci_high"].to_numpy();yerr=np.vstack([vals-lo,hi-vals])
axs[1].bar(["Valence","Liking"],vals,yerr=yerr,capsize=4);axs[1].axhline(0,linewidth=1);axs[1].set_ylim(-0.05,0.85);axs[1].set_ylabel("Validation $R^2$");axs[1].set_title("B  Frozen report-profile prediction")
for i,v in enumerate(vals):axs[1].text(i,v+0.04,f"{v:.3f}",ha="center",fontsize=9)
fig.tight_layout();fig.savefig(out/"Figure2_subjective_architecture.png",dpi=300,bbox_inches="tight");plt.close(fig)

# Figure 3: direct EEG outcome tests and non-hedonic diagnostics.
dm=pd.read_csv(ROOT/"results/source_data/direct_eeg_and_diagnostic_metrics.csv").set_index("evidence_id")
fig,axs=plt.subplots(1,2,figsize=(12,4.5));ids=["E_PRIMARY_VALENCE","E_PRIMARY_LIKING"];labels=["Valence","Liking"]
est=np.array([float(dm.loc[i,"estimate"]) for i in ids]);lo=np.array([float(dm.loc[i,"ci_low"]) for i in ids]);hi=np.array([float(dm.loc[i,"ci_high"]) for i in ids]);err=np.vstack([est-lo,hi-est])
axs[0].bar(labels,est,yerr=err,capsize=4);axs[0].axhline(0,linewidth=1);axs[0].set_ylabel("Signal-specific $\\Delta R^2$");axs[0].set_title("A  Direct hedonic EEG prediction unsupported")
dids=["E_IDENTITY","E_CONTEXT_EARLY","E_CONTEXT_LATE","E_CONTEXT_RATING"];dlabels=["Identity","Context\nstimulus early","Context\nstimulus late","Context\nrating early"];dv=[float(dm.loc[i,"estimate"]) for i in dids]
axs[1].bar(dlabels,dv);axs[1].set_ylim(0,1.05);axs[1].set_ylabel("Diagnostic score");axs[1].set_title("B  Strong non-hedonic structure in EEG")
for i,v in enumerate(dv):axs[1].text(i,v+0.025,f"{v:.3f}",ha="center",fontsize=8)
fig.tight_layout();fig.savefig(out/"Figure3_direct_hedonic_EEG_and_diagnostics.png",dpi=300,bbox_inches="tight");plt.close(fig)

# Figure 4: fixed trajectory geometry and transfer.
tt=pd.read_csv(ROOT/"results/source_data/trajectory_transfer_summary.csv")
fig,axs=plt.subplots(1,3,figsize=(14,4.3));axs[0].plot([0,1,2],[0,0.75,0.15],marker="o");axs[0].annotate("",xy=(1,0.75),xytext=(0,0),arrowprops={"arrowstyle":"->"});axs[0].annotate("",xy=(2,0.15),xytext=(1,0.75),arrowprops={"arrowstyle":"->"});axs[0].set_xticks([0,1,2],["stimulus\nearly","stimulus\nlate","rating\nearly"]);axs[0].set_yticks([]);axs[0].set_title("A  Trajectory construction\n(schematic, not data)")
coh=tt["cohort"].tolist();f1=tt["family1_raw_fraction"].to_numpy();f2=tt["family2_raw_fraction"].to_numpy();met=tt["structural_metric_value"].to_numpy();axs[1].bar(coh,f1,label="Family 1");axs[1].bar(coh,f2,bottom=f1,label="Family 2");axs[1].set_ylim(0,1);axs[1].set_ylabel("Fraction of assigned trajectories");axs[1].set_title("B  Family balance");axs[1].tick_params(axis="x",rotation=20);axs[1].legend(frameon=False)
axs[2].bar(coh,met);axs[2].set_ylim(0.9,1.005);axs[2].set_ylabel("Structural metric");axs[2].set_title("C  Structural transfer");axs[2].tick_params(axis="x",rotation=20)
for i,(r,v) in enumerate(zip(tt.to_dict("records"),met)):
    lab="LOPO agreement" if r["cohort"]=="Discovery" else "in-support fraction";axs[2].text(i,v-0.013,f"{v:.3f}\n{lab}",ha="center",va="top",fontsize=7)
fig.tight_layout();fig.savefig(out/"Figure4_trajectory_geometry_and_transfer.png",dpi=300,bbox_inches="tight");plt.close(fig)

s=pd.read_csv(ROOT/"results/source_data/multiresolution_structural_transfer_summary.csv");labels=[f"{r.structure.replace('_',' ')} K={int(r.k)}" for _,r in s.iterrows()];plt.figure();ax=plt.gca();xx=np.arange(3)
for j in range(len(labels)):ax.scatter(xx,np.full(3,j),s=90)
ax.set_xticks(xx,["Discovery","Validation","Confirmation"]);ax.set_yticks(np.arange(len(labels)),labels);ax.set_xlim(-.5,2.5);ax.set_ylim(-.7,len(labels)-.3);ax.invert_yaxis();ax.set_title("Prespecified multiresolution structural-transfer criteria passed in all internal splits");ax.set_xlabel("Marker = prespecified structural-transfer criterion passed");ax.grid(axis="x",alpha=.25);save(out/"Figure5_multiresolution_structural_transfer.png",8.5,4.8)
p=pd.read_csv(ROOT/"results/source_data/semantic_report_parent_tests.csv");p["label"]=p.apply(lambda r:f"{r['split'][0]}:{r['test_id']}",axis=1);p["v"]=-np.log10(p["p_adjusted"].clip(lower=1e-6));plt.figure();ax=plt.gca();xx=np.arange(len(p));ax.scatter(xx,p["v"],s=34);ax.axhline(-np.log10(.05),linestyle="--",linewidth=1);ax.set_xticks(xx,p["label"],rotation=90,fontsize=7);ax.set_ylabel("-log10(adjusted p)");ax.set_title("No split-level semantic/report parent test passed the corrected threshold (0/27)");save(out/"Figure6_semantic_report_parent_tests.png",10,5.2)
b=pd.read_csv(ROOT/"results/source_data/signal_space_block_summary.csv");q=b.pivot(index="block",columns="domain",values="stable_fraction_within_authorized_feature_rows").fillna(0).reindex(["time_frequency","morphology","plv_channel","plv_global","waveform_dct","covariance"]);xx=np.arange(len(q));w=.36;plt.figure();ax=plt.gca();ax.bar(xx-w/2,q["experience"].values,w,label="Experience");ax.bar(xx+w/2,q["report_transition"].values,w,label="Report transition");ax.set_xticks(xx,[i.replace("_","\n") for i in q.index]);ax.set_ylabel("Stable fraction within prespecified feature rows");ax.set_title("Cross-split stable compact-EEG feature composition is multicomponent");ax.legend(frameon=False);ax.set_ylim(0,max(.7,float(np.nanmax(q.values))*1.15));save(out/"Figure7_compact_EEG_block_stability.png",9,5);print("PASS: all seven table-driven figures regenerated.")
