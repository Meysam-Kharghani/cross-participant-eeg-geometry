#!/usr/bin/env python3
"""Post-hoc context-preserving robustness analyses for primary analysis trajectory geometry.

This script is explicitly POST HOC. It does not alter the fixed primary analysis.
It requires the released/reproducibility-input trajectory tables and trial manifest.

Outputs:
  09_context_stratified_deranged_null_gain_<N>rep.csv
  09b_context_stratified_deranged_null_summary_<N>rep.csv
  09c_context_stratified_deranged_null_detailed_<M>rep.csv
  09d_context_counts.csv
  09e_observed_family_by_context.csv
  10_context_stratified_null_LOPO_R8like_<R>rep.csv
  10b_context_stratified_null_LOPO_R8like_summary.csv
  11_continuous_axis_LOPO_6D.csv
  11b_continuous_axis_LOPO_primitive3D.csv
  11c_continuous_axis_LOPO_summary.csv

Nulls preserve participant and elicitation context (imagery/video) but destroy
same-trial temporal correspondence. No-fixed-point permutations are used so the
intended epoch relation is destroyed for every row in each stratum.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

FEATURES6 = [
    'early_to_late_norm','late_to_rating_norm','early_to_rating_direct_norm',
    'total_path_length','directness_ratio','segment_cosine'
]
PRIM3 = ['early_to_late_norm','late_to_rating_norm','segment_cosine']
FIRST4 = set(FEATURES6[:4])
E_COLS = [f'stimulus_early_pc{i}' for i in range(1,9)]
L_COLS = [f'stimulus_late_pc{i}' for i in range(1,9)]
R_COLS = [f'rating_early_pc{i}' for i in range(1,9)]


def participant_weights(df):
    vc = df['subject_id'].astype(str).value_counts()
    return df['subject_id'].astype(str).map(lambda s: 1.0/vc[s]).to_numpy(float)


def transform_and_scale(df, features):
    X = df[features].to_numpy(float).copy()
    for j, f in enumerate(features):
        if f in FIRST4:
            X[:, j] = np.log1p(X[:, j])
        elif f == 'directness_ratio':
            x = np.clip(X[:, j], 1e-6, 1-1e-6)
            X[:, j] = np.log(x/(1-x))
        elif f == 'segment_cosine':
            x = np.clip(X[:, j], -1+1e-6, 1-1e-6)
            X[:, j] = np.arctanh(x)
        else:
            raise ValueError(f)
    Z = np.empty_like(X)
    subs = df['subject_id'].astype(str).to_numpy()
    for s in pd.unique(subs):
        ix = np.flatnonzero(subs == s)
        xs = X[ix]
        med = np.median(xs, axis=0)
        iqr = np.percentile(xs, 75, axis=0) - np.percentile(xs, 25, axis=0)
        bad = ~np.isfinite(iqr) | (iqr <= 1e-8)
        if bad.any():
            pooled = np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0)
            iqr[bad] = np.where(pooled[bad] > 1e-8, pooled[bad], 1.0)
        Z[ix] = (xs-med)/iqr
    return np.clip(Z, -6, 6)


def fit_k2(Z, w, seed, n_init=10):
    km = KMeans(n_clusters=2, random_state=seed, n_init=n_init,
                max_iter=500, tol=1e-7, algorithm='lloyd')
    km.fit(Z, sample_weight=w)
    return km.labels_.astype(int), km.cluster_centers_.copy()


def align_labels(candidate, reference, weights=None):
    if weights is None:
        weights = np.ones(len(candidate))
    tab = np.zeros((2,2), float)
    for a,b,w in zip(candidate, reference, weights): tab[int(a),int(b)] += w
    rr, cc = linear_sum_assignment(-tab)
    mp = {int(a):int(b) for a,b in zip(rr,cc)}
    return np.array([mp[int(x)] for x in candidate], int), mp


def relative_sse_gain(Z, labels, centers, w):
    mu = np.average(Z, axis=0, weights=w)
    s1 = np.sum(w*np.sum((Z-mu)**2,axis=1))
    s2 = np.sum(w*np.sum((Z-centers[labels])**2,axis=1))
    return float((s1-s2)/s1)


def derangement(n, rng):
    if n < 2:
        raise ValueError('A context stratum with <2 rows cannot be deranged.')
    base = np.arange(n)
    # Rejection sampling gives an ordinary random permutation conditional on no fixed points.
    # With n>=15 in the present data this is fast.
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != base): return p
    # deterministic safe fallback
    shift = int(rng.integers(1, n))
    return np.roll(base, shift)


def build_destroyed(df, null_type, rng):
    parts=[]
    for (_, _), g in df.groupby(['subject_id','context'], sort=False):
        g=g.reset_index(drop=True)
        E=g[E_COLS].to_numpy(float); L=g[L_COLS].to_numpy(float); R=g[R_COLS].to_numpy(float)
        n=len(g)
        if null_type == 'rating_only':
            L2=L; R2=R[derangement(n,rng)]
        elif null_type == 'late_rating_pair':
            q=derangement(n,rng); L2=L[q]; R2=R[q]
        elif null_type == 'late_rating_independent':
            L2=L[derangement(n,rng)]; R2=R[derangement(n,rng)]
        else: raise ValueError(null_type)
        v1=L2-E; v2=R2-L2; vd=R2-E
        d1=np.linalg.norm(v1,axis=1); d2=np.linalg.norm(v2,axis=1); dd=np.linalg.norm(vd,axis=1)
        path=d1+d2
        cosine=np.sum(v1*v2,axis=1)/np.maximum(d1*d2,1e-15)
        direct=dd/np.maximum(path,1e-15)
        parts.append(pd.DataFrame({
            'subject_id':g['subject_id'].astype(str), 'context':g['context'],
            'early_to_late_norm':d1, 'late_to_rating_norm':d2,
            'early_to_rating_direct_norm':dd, 'total_path_length':path,
            'directness_ratio':direct, 'segment_cosine':np.clip(cosine,-1,1)
        }))
    return pd.concat(parts,ignore_index=True)


def prespecified_reference(traj, assignments, lopo):
    Z=transform_and_scale(traj, FEATURES6); w=participant_weights(traj)
    fr=assignments['trajectory_family_1based'].to_numpy(int)-1
    lab,ctr=fit_k2(Z,w,seed=540,n_init=100)
    mapped,mp=align_labels(lab,fr,w)
    if not np.all(mapped==fr):
        raise RuntimeError('Could not reproduce released K=2 assignment exactly.')
    ctrf=np.zeros_like(ctr)
    for old,new in mp.items(): ctrf[new]=ctr[old]
    gain=relative_sse_gain(Z,fr,ctrf,w)
    agree=lopo['assignment_agrees_with_full_fit'].astype(bool)
    thr=float(np.quantile(lopo.loc[agree,'assigned_squared_distance'].to_numpy(float),.995))
    return Z,w,fr,ctrf,gain,thr


def null_metrics(dn, prespecified_labels, prespecified_centers, support_thr, seed):
    Z=transform_and_scale(dn,FEATURES6); w=participant_weights(dn)
    lab,ctr=fit_k2(Z,w,seed=seed,n_init=5)
    gain=relative_sse_gain(Z,lab,ctr,w)
    dist=((Z[:,None,:]-prespecified_centers[None,:,:])**2).sum(axis=2)
    fassign=dist.argmin(axis=1); mind=dist.min(axis=1)
    tmp=pd.DataFrame({'subject_id':dn['subject_id'].astype(str), 'family':fassign, 'support':mind<=support_thr})
    pb_support=float(tmp.groupby('subject_id')['support'].mean().mean())
    pb_f1=float(tmp.groupby('subject_id')['family'].apply(lambda x:(x==0).mean()).mean())
    aligned,_=align_labels(fassign,prespecified_labels,w)
    return {
        'relative_SSE_gain':gain,
        'pb_support':pb_support,
        'pb_family1':pb_f1,
        'mean_min_sq_dist':float(np.average(mind,weights=w)),
        'ARI_vs_observed':float(adjusted_rand_score(prespecified_labels,fassign)),
        'aligned_agreement_vs_observed':float(np.mean(aligned==prespecified_labels)),
    }


def lopo_stability(df, full_labels, seed, n_init=5):
    subs=df['subject_id'].astype(str).to_numpy(); okall=np.zeros(len(df),bool); rows=[]
    for i,s in enumerate(sorted(pd.unique(subs))):
        tr=subs!=s; te=~tr
        dtr=df.loc[tr].reset_index(drop=True); dte=df.loc[te].reset_index(drop=True)
        ztr=transform_and_scale(dtr,FEATURES6); zte=transform_and_scale(dte,FEATURES6)
        wtr=participant_weights(dtr)
        lab,ctr=fit_k2(ztr,wtr,seed+i,n_init=n_init)
        _,mp=align_labels(lab,full_labels[tr],wtr)
        raw=np.argmin(((zte[:,None,:]-ctr[None,:,:])**2).sum(axis=2),axis=1)
        mapped=np.array([mp[int(x)] for x in raw],int)
        ok=mapped==full_labels[te]; okall[np.flatnonzero(te)]=ok
        rows.append((s,float(ok.mean())))
    ps=pd.DataFrame(rows,columns=['subject_id','agreement'])
    return float(okall.mean()),float(ps.agreement.min()),int((ps.agreement>=.95).sum())


def continuous_axis_lopo(traj, features, seed=540):
    Z=transform_and_scale(traj,features); w=participant_weights(traj)
    full_lab,full_ctr=fit_k2(Z,w,seed=seed,n_init=100)
    full_axis=full_ctr[1]-full_ctr[0]; full_axis/=np.linalg.norm(full_axis)
    subs=traj['subject_id'].astype(str).to_numpy(); rows=[]
    for i,s in enumerate(sorted(pd.unique(subs))):
        tr=subs!=s
        dtr=traj.loc[tr].reset_index(drop=True)
        ztr=transform_and_scale(dtr,features); wtr=participant_weights(dtr)
        lab,ctr=fit_k2(ztr,wtr,seed=seed+i+1,n_init=30)
        # align fold labels to full-fit labels on training rows
        _,mp=align_labels(lab,full_lab[tr],wtr)
        aligned_ctr=np.zeros_like(ctr)
        for old,new in mp.items(): aligned_ctr[new]=ctr[old]
        axis=aligned_ctr[1]-aligned_ctr[0]; axis/=np.linalg.norm(axis)
        cos=float(np.clip(np.dot(axis,full_axis),-1,1)); angle=float(np.degrees(np.arccos(cos)))
        rows.append((s,cos,angle))
    out=pd.DataFrame(rows,columns=['held_out_subject_id','axis_cosine_to_full','axis_angle_deg'])
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trajectories',required=True)
    ap.add_argument('--assignments',required=True)
    ap.add_argument('--lopo',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=20261017)
    ap.add_argument('--null-reps',type=int,default=400)
    ap.add_argument('--detailed-reps',type=int,default=100)
    ap.add_argument('--null-lopo-reps',type=int,default=20)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    traj=pd.read_csv(args.trajectories,low_memory=False)
    ass=pd.read_csv(args.assignments,low_memory=False)
    lopo=pd.read_csv(args.lopo,low_memory=False)
    man=pd.read_csv(args.manifest,low_memory=False)
    keys=['subject_id','raw_behavior_row_id','trial_number']
    x=traj.merge(man[keys+['context']],on=keys,how='left',validate='one_to_one')
    if x.context.isna().any(): raise RuntimeError('Missing context after manifest merge')
    Z,w,fr,ctr,obs_gain,support_thr=prespecified_reference(x,ass,lopo)

    # diagnostics/context counts
    x.groupby('context').size().rename('n').reset_index().to_csv(out/'09d_context_counts.csv',index=False)
    pd.DataFrame({'context':x.context,'family':fr+1,'subject_id':x.subject_id}).groupby(['context','family']).size().rename('n').reset_index().to_csv(out/'09e_observed_family_by_context.csv',index=False)

    rng=np.random.default_rng(args.seed)
    null_types=['late_rating_pair','late_rating_independent','rating_only']
    rows=[]
    for nt in null_types:
        for r in range(args.null_reps):
            dn=build_destroyed(x,nt,rng)
            met=null_metrics(dn,fr,ctr,support_thr,seed=args.seed+1000*r+17*null_types.index(nt))
            rows.append({'null_type':nt,'replicate':r,**met})
    tab=pd.DataFrame(rows)
    tab[['null_type','replicate','relative_SSE_gain']].to_csv(out/f'09_context_stratified_deranged_null_gain_{args.null_reps}rep.csv',index=False)
    tab[tab.replicate < args.detailed_reps].to_csv(out/f'09c_context_stratified_deranged_null_detailed_{args.detailed_reps}rep.csv',index=False)
    sm=[]
    for nt,g in tab.groupby('null_type',sort=False):
        sm.append({
            'null_type':nt,'reps':len(g),'observed_relative_SSE_gain':obs_gain,
            'mean_gain':g.relative_SSE_gain.mean(),'sd_gain':g.relative_SSE_gain.std(ddof=1),
            'p95_gain':g.relative_SSE_gain.quantile(.95),'p99_gain':g.relative_SSE_gain.quantile(.99),
            'empirical_p_gain_ge_observed':(1+(g.relative_SSE_gain>=obs_gain).sum())/(len(g)+1),
            'mean_pb_support':g.pb_support.mean(),'sd_pb_support':g.pb_support.std(ddof=1),
            'mean_pb_family1':g.pb_family1.mean(),'mean_min_sq_dist':g.mean_min_sq_dist.mean(),
            'mean_ARI_vs_observed':g.ARI_vs_observed.mean(),
            'mean_aligned_agreement_vs_observed':g.aligned_agreement_vs_observed.mean(),
        })
    pd.DataFrame(sm).to_csv(out/f'09b_context_stratified_deranged_null_summary_{args.null_reps}rep.csv',index=False)

    # Diagnostic null LOPO: fewer replications by design due computational expense.
    rng2=np.random.default_rng(args.seed+991)
    lrows=[]
    for nt in null_types:
        for r in range(args.null_lopo_reps):
            dn=build_destroyed(x,nt,rng2)
            zd=transform_and_scale(dn,FEATURES6); wd=participant_weights(dn)
            lab,_=fit_k2(zd,wd,seed=args.seed+50000+r+31*null_types.index(nt),n_init=10)
            lo,mn,n95=lopo_stability(dn,lab,seed=args.seed+70000+r+31*null_types.index(nt),n_init=5)
            # R8-like diagnostic gate mirrors stability thresholds, not the fixed primary gate's status.
            both=int(dn.groupby('subject_id').apply(lambda g: len(np.unique(lab[g.index]))==2, include_groups=False).sum())
            gate=(lo>=.99 and mn>=.90 and n95>=30 and both==dn.subject_id.nunique())
            lrows.append({'null_type':nt,'replicate':r,'lopo_agreement':lo,'min_subject':mn,'n_subjects_ge_0.95':n95,'n_subjects_both_families':both,'R8like_gate_pass':gate})
    ltab=pd.DataFrame(lrows); ltab.to_csv(out/f'10_context_stratified_null_LOPO_R8like_{args.null_lopo_reps}rep.csv',index=False)
    ls=[]
    for nt,g in ltab.groupby('null_type',sort=False):
        ls.append({'null_type':nt,'mean_lopo':g.lopo_agreement.mean(),'sd_lopo':g.lopo_agreement.std(ddof=1),
                   'p10_lopo':g.lopo_agreement.quantile(.10),'p90_lopo':g.lopo_agreement.quantile(.90),
                   'mean_min_subject':g.min_subject.mean(),'mean_n95':g['n_subjects_ge_0.95'].mean(),
                   'mean_both':g.n_subjects_both_families.mean(),'gate_pass_rate':g.R8like_gate_pass.mean()})
    pd.DataFrame(ls).to_csv(out/'10b_context_stratified_null_LOPO_R8like_summary.csv',index=False)

    # Continuous-axis LOPO stability.
    a6=continuous_axis_lopo(x,FEATURES6); a3=continuous_axis_lopo(x,PRIM3)
    a6.to_csv(out/'11_continuous_axis_LOPO_6D.csv',index=False); a3.to_csv(out/'11b_continuous_axis_LOPO_primitive3D.csv',index=False)
    ss=[]
    for name,a in [('prespecified6D',a6),('primitive3D',a3)]:
        ss.append({'representation':name,'mean_axis_cosine':a.axis_cosine_to_full.mean(),'min_axis_cosine':a.axis_cosine_to_full.min(),
                   'mean_axis_angle_deg':a.axis_angle_deg.mean(),'median_axis_angle_deg':a.axis_angle_deg.median(),'max_axis_angle_deg':a.axis_angle_deg.max()})
    pd.DataFrame(ss).to_csv(out/'11c_continuous_axis_LOPO_summary.csv',index=False)

    print('Observed relative SSE gain:',obs_gain)
    print(pd.DataFrame(sm).to_string(index=False))
    print(pd.DataFrame(ls).to_string(index=False))
    print(pd.DataFrame(ss).to_string(index=False))

if __name__=='__main__': main()
