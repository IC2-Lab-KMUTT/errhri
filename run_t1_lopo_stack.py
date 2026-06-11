"""T1 — regenerate tabular stream OOFs under LOPO (36 folds) + owned HSE (already LOPO),
isotonic-calibrated greedy fuse under LOPO meta-CV, official video macro-F1.
Tests: does LOPO lift our stack like it lifted HSE (5-fold 0.696 ref)?
"""
import sys, os, re, subprocess, tempfile
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from errhri_features import FeatureBank, metrics as M
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR
track = 1
MODS = {"au_xgb":"au","aug_xgb":"au_graph","gaze":"gaze","pose":"pose","blend":"blend","audio":"audio"}
XGB = dict(n_estimators=400,max_depth=4,learning_rate=0.04,subsample=0.8,colsample_bytree=0.6,
           reg_lambda=3.0,min_child_weight=5,eval_metric="logloss",n_jobs=4,tree_method="hist")
ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk=dict(zip(keys,ref.y)); gk=dict(zip(keys,ref.groups)); nk=dict(zip(keys,ref.n_frames))
def qid(v): return re.sub(r"_.*","",str(v))
ourkey={(p,qid(v)):(p,v) for (p,v) in keys}

P={}
for name,mod in MODS.items():
    fb=FeatureBank(track,[mod]).load()
    fkeys=list(zip(fb.df.participant.astype(str),fb.df.video.astype(str)))
    X,y,g=fb.matrix(normalize=True,leak_clean=True); g=np.array(g)
    spw=float((y==0).sum())/max((y==1).sum(),1)
    oof=np.full(len(fkeys),np.nan)
    for u in np.unique(g):
        tr,va=g!=u,g==u
        clf=XGBClassifier(**XGB,scale_pos_weight=spw).fit(X[tr],y[tr])
        oof[va]=clf.predict_proba(X[va])[:,1]
    P[name]=dict(zip(fkeys,oof))
    print(f"[{name}] LOPO OOF done",flush=True)
hdf=pd.read_csv(CACHE_DIR/"hse_his_oof_t1.csv")
o={}
for p,v,x in zip(hdf.participant.astype(str),hdf.video.astype(str),hdf.oof_hse_his):
    ok=ourkey.get((p,qid(v)))
    if ok is not None: o[ok]=float(x)
P["HSE_owned"]=o; print(f"[HSE_owned] joined {len(o)}",flush=True)

common=[k for k in keys if all(k in od for od in P.values())]
S={n:np.array([od[k] for k in common]) for n,od in P.items()}
yv=np.array([yk[k] for k in common]); gv=np.array([gk[k] for k in common]); nv=np.array([nk[k] for k in common])
print(f"clips {len(common)}",flush=True)
def lopo_iter():
    for u in np.unique(gv):
        yield np.where(gv!=u)[0], np.where(gv==u)[0]
def fuse(names):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    pf=np.zeros(len(yv))
    for tr,va in lopo_iter():
        ctr,cva=[],[]
        for n in names:
            iso=IsotonicRegression(out_of_bounds="clip").fit(S[n][tr],yv[tr])
            ctr.append(iso.transform(S[n][tr])); cva.append(iso.transform(S[n][va]))
        Atr,Ava=np.column_stack(ctr),np.column_stack(cva)
        pf[va]=Ava[:,0] if len(names)==1 else LogisticRegression(max_iter=2000).fit(Atr,yv[tr]).predict_proba(Ava)[:,1]
    return pf
def clipF(pf):
    pred=np.zeros(len(yv),int)
    for tr,va in lopo_iter():
        thr=M.tune_threshold(yv[tr],pf[tr]); pred[va]=(pf[va]>=thr).astype(int)
    return M.primary(track,yv,pred,pf)
def official(pf,tag):
    pred=np.zeros(len(yv),int)
    for tr,va in lopo_iter():
        thr=M.tune_threshold(yv[tr],pf[tr]); pred[va]=(pf[va]>=thr).astype(int)
    gt,sub=[],[]
    for (pid,vid),yt,yp,prob,nf in zip(common,yv,pred,pf,nv):
        nf=int(max(nf,WS[track]))
        for f in range(1,nf+1): gt.append((pid,vid,f,int(yt)))
        for w in range(max((nf-WS[track])//SLIDE[track]+1,1)): sub.append((pid,vid,w,int(yp),float(1-prob),float(prob)))
    d=tempfile.mkdtemp(prefix="t1lopo_"); gtp,subp=os.path.join(d,"gt.csv"),os.path.join(d,"sub.csv")
    pd.DataFrame(gt,columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp,index=False)
    pd.DataFrame(sub,columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp,index=False)
    out=subprocess.run([sys.executable,REPO_EVAL,"--gt",gtp,"--pred",subp,"--track",str(track),"--fps",str(FPS[track]),"--window_size",str(WS[track]),"--slide",str(SLIDE[track])],capture_output=True,text=True)
    vid=out.stdout.split("n=1319")[-1] if "n=1319" in out.stdout else out.stdout
    f1=[l for l in vid.splitlines() if "F1 macro" in l]
    print(f"  {tag:30} video {f1[0].split(':')[-1].strip() if f1 else '?'}",flush=True)
print("solo LOPO clip-F1:",{n:round(clipF(S[n]),3) for n in S},flush=True)
rem=list(S); ch=[]; best=-1
while rem:
    sc=sorted([(clipF(fuse(ch+[s])),s) for s in rem],reverse=True)
    if sc[0][0]<=best+1e-4: break
    best=sc[0][0]; ch.append(sc[0][1]); rem.remove(sc[0][1])
    print(f"  greedy + {sc[0][1]:12} -> {best:.3f}",flush=True)
print(f"greedy: {ch} ({best:.3f})",flush=True)
official(fuse(ch),"LOPO_stack_greedy")
official(fuse([n for n in ch if n!="HSE_owned"]) if "HSE_owned" in ch else fuse(ch),"LOPO_no_HSE")
pd.DataFrame({"participant":[k[0] for k in common],"video":[k[1] for k in common],
  **{f"oof_{n}":S[n] for n in S}}).to_csv(CACHE_DIR/"lopo_oof_t1.csv",index=False)
print("saved lopo_oof_t1.csv | refs: 5fold our 0.696, owned-HSE-fused 0.729, philix-ref 0.748",flush=True)
