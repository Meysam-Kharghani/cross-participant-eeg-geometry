#!/usr/bin/env python3
from pathlib import Path
import csv,sys,math
r=Path(__file__).resolve().parents[1]/"results/source_data"
def rows(n):
 with (r/n).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def b(v):return str(v).strip().lower() in {"1","true","t","yes","y"}
def f(v):return float(v) if str(v).strip() not in {"","nan","NA"} else float('nan')
struct=rows("multiresolution_structural_transfer_summary.csv"); pt=rows("semantic_report_parent_tests.csv"); dom=rows("signal_space_domain_summary.csv")
pa=rows("subjective_discovery_parallel_analysis.csv"); vp=rows("subjective_validation_prediction_summary.csv")
dm={x["evidence_id"]:x for x in rows("direct_eeg_and_diagnostic_metrics.csv")}; tt=rows("trajectory_transfer_summary.csv")
local=rows("local_state_alignment_decision.csv"); topo=rows("topography_extension_summary.csv"); phys=rows("physiology_boundary_summary.csv")
checks=[
 len(struct)==5 and all(b(x["structurally_replicated_across_all_three_splits"]) for x in struct),
 len(pt)==27 and sum(b(x["reject_adjusted_0p05"]) for x in pt)==0,
 sum(int(float(x["n_authorized_contrast_block_pairs"])) for x in dom)==31,
 sum(int(float(x["n_authorized_feature_instances"])) for x in dom)==938,
 sum(int(float(x["n_stable_subfeature_instances"])) for x in dom)==277,
 len(pa)==7 and [b(x["selected_by_parallel_analysis"]) for x in pa]==[True,True,False,False,False,False,False],
 len(vp)==2 and abs(f(next(x for x in vp if x['outcome']=='valence')['r2'])-0.678397)<1e-5 and abs(f(next(x for x in vp if x['outcome']=='liking')['r2'])-0.723933)<1e-5,
 abs(f(dm['E_PRIMARY_VALENCE']['estimate'])+0.0045778528091537)<1e-12 and abs(f(dm['E_PRIMARY_LIKING']['estimate'])-0.0152336083502162)<1e-12,
 f(dm['E_IDENTITY']['estimate'])>.96 and f(dm['E_CONTEXT_EARLY']['estimate'])>.99 and f(dm['E_CONTEXT_LATE']['estimate'])>.97,
 len(tt)==3 and all(b(x['structural_gate_passed']) for x in tt) and all(f(x['structural_metric_value'])>.98 for x in tt),
 len(local)>=1 and any(str(x.get('stable_discrete_family_solution','')).lower()=='false' for x in local),
 len(topo)>=5 and any('gate FAIL' in str(x.get('result','')) for x in topo),
 len(phys)>=1 and any('0/47' in str(x) for x in phys),
]
print("PASS: repository result tables satisfy the scientific checks." if all(checks) else "FAIL: scientific checks.")
if not all(checks):
 for i,v in enumerate(checks,1):
  if not v: print(f"FAILED_CHECK_{i}")
sys.exit(0 if all(checks) else 1)
