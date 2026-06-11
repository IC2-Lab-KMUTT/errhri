"""T1 — apply philix's window-table template to OUR dense sequences (au_seq 48x41 + gaze_seq 48x10):
per-participant z-norm -> window stats (mean/std/min/max/q25/q75/slope/vel/pos) -> class-weighted
LGBM -> mean-agg -> LOPO. Produces an owned orthogonal stream (different representation than HSE).
"""
import re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier
from errhri_features import FeatureBank
from errhri_features.config import CACHE_DIR
track=1; WS,SLIDE=25,5   # 48-frame seqs -> 5 windows of 25
LGBM=dict(objective="binary",n_estimators=300,learning_rate=0.03,num_leaves=31,max_depth=6,
          min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
          reg_lambda=5.0,reg_alpha=1.0,n_jobs=4,random_state=42,verbosity=-1)
ref=FeatureBank(track,["au"]).load()
keys=list(zip(ref.df.participant.astype(str),ref.df.video.astype(str)))
yk=dict(zip(keys,ref.y)); gk=dict(zip(keys,ref.groups))
seqs={}
for mod in ("au_seq","gaze_seq"):
    m=pd.read_csv(CACHE_DIR/f"{mod}_t{track}.csv")
    m["participant"]=m.participant.astype(str); m["video"]=m.video.astype(str)
    tcols=sorted([c for c in m.columns if re.match(r"^t\d+__",c)],
                 key=lambda c:(int(re.match(r"^t(\d+)__",c).group(1)),c))
    chans=sorted(set(re.sub(r"^t\d+__","",c) for c in tcols))
    T=max(int(re.match(r"^t(\d+)__",c).group(1)) for c in tcols)+1
    arrs={}
    for _,r in m.iterrows():
        A=np.zeros((T,len(chans)),np.float32)
        for ci,ch in enumerate(chans):
            for t in range(T):
                col=f"t{t:02d}__{ch}"
                if col in m.columns: A[t,ci]=r[col]
        arrs[(r.participant,r.video)]=A
    seqs[mod]=arrs
    print(f"[{mod}] {len(arrs)} clips T={T} C={len(chans)}",flush=True)
common=[k for k in keys if all(k in seqs[mod] for mod in seqs)]
X3={k:np.concatenate([seqs["au_seq"][k],seqs["gaze_seq"][k]],axis=1) for k in common}
yv=np.array([yk[k] for k in common]); gv=np.array([gk[k] for k in common])
C=next(iter(X3.values())).shape[1]
print(f"clips {len(common)} channels {C}",flush=True)
# per-participant z-norm over frames
for u in np.unique(gv):
    ks=[k for k,g in zip(common,gv) if g==u]
    allf=np.concatenate([X3[k] for k in ks],axis=0)
    mu,sd=allf.mean(0),allf.std(0)+1e-6
    for k in ks: X3[k]=(X3[k]-mu)/sd
def wstats(w,s,n):
    q25,q75=np.quantile(w,.25,0),np.quantile(w,.75,0)
    vel=np.abs(np.diff(w,axis=0)).mean(0) if len(w)>1 else np.zeros(w.shape[1],np.float32)
    return np.concatenate([w.mean(0),w.std(0),w.min(0),w.max(0),q25,q75,w[-1]-w[0],vel,
                           np.array([s/max(n,1),(s+len(w))/max(n,1)],np.float32)]).astype(np.float32)
Xw,vm=[],[]
for vi,k in enumerate(common):
    f=X3[k]; n=len(f)
    for s in (list(range(0,n-WS+1,SLIDE)) or [0]):
        Xw.append(wstats(f[s:s+WS] if n>=WS else f,s,n)); vm.append(vi)
Xw=np.asarray(Xw,np.float32); vm=np.asarray(vm)
wy=yv[vm]; wg=gv[vm]; spw=float((wy==0).sum())/max((wy==1).sum(),1)
print(f"windows {len(Xw)} x {Xw.shape[1]}",flush=True)
oof=np.zeros(len(Xw),np.float32)
uu=np.unique(wg)
for i,u in enumerate(uu):
    tr,va=wg!=u,wg==u
    clf=LGBMClassifier(**{**LGBM,"scale_pos_weight":spw}).fit(Xw[tr],wy[tr])
    oof[va]=clf.predict_proba(Xw[va])[:,1]
    if (i+1)%9==0: print(f"  fold {i+1}/{len(uu)}",flush=True)
vscore=np.array([oof[vm==vi].mean() for vi in range(len(common))])
auc=roc_auc_score(yv,vscore)
bf=max(f1_score(yv,(vscore>=t).astype(int),average="macro") for t in np.quantile(vscore,np.linspace(.02,.98,49)))
print(f"\nourseq-wlgbm (au_seq+gaze_seq, his template): AUC {auc:.4f}  bestF1 {bf:.4f}",flush=True)
print("refs: HSE_owned 0.820 AUC; our old temporal streams ~0.72 AUC",flush=True)
pd.DataFrame({"participant":[k[0] for k in common],"video":[k[1] for k in common],
              "oof_ourseq":vscore}).to_csv(CACHE_DIR/"ourseq_wlgbm_oof_t1.csv",index=False)
print("saved ourseq_wlgbm_oof_t1.csv",flush=True)
