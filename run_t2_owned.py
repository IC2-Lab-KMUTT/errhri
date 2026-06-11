"""T2 owned: HSEmotion tail-window stream from our copy of philix's t2 features, honest LOPO,
agg=mean fixed a priori. Then fuse (raw + pz-ranked) with our T2 streams, greedy by video-AUC
under LOPO. Refs: our 0.576, his honest single 0.604 raw / 0.624 pz, his nested fusion 0.612.
"""
import re, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from errhri_features import FeatureBank
from pipelines.recipes import _oof_by_key, Stream
from errhri_features.config import CACHE_DIR
SEED=42; np.random.seed(SEED)
track=2; WS,SLIDE,PCA_K=10,2,128
LGBM=dict(objective="binary",n_estimators=300,learning_rate=0.03,num_leaves=31,max_depth=6,
          min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
          reg_lambda=5.0,reg_alpha=1.0,n_jobs=4,random_state=SEED,verbosity=-1)
ref=FeatureBank(track,["au"]).load()
keys=list(zip(ref.df.participant.astype(str),ref.df.video.astype(str)))
yk=dict(zip(keys,ref.y)); gk=dict(zip(keys,ref.groups))
def qid(v): return re.sub(r"_\d+$","",str(v))
ourkey={(p,qid(v)):(p,v) for (p,v) in keys}

z=np.load("/home/ic2/research/errhri/philix_feats/t2_feats.npz",allow_pickle=True)
pid=z["participant_ids"].astype(str); vid=z["official_video_ids"].astype(str)
fn=z["frame_nums"].astype(int)
MP=int(z["mp_dim"][0]); HSE=int(z["hse_dim"][0]); EMB=HSE-10
F=z["features"].astype(np.float32)
def pzc(x,p):
    o=np.empty_like(x)
    for u in np.unique(p):
        m=p==u; o[m]=(x[m]-x[m].mean(0))/(x[m].std(0)+1e-6)
    return o
F=pzc(F,pid)
emb=F[:,MP:MP+EMB]
fit=np.random.choice(len(emb),min(120000,len(emb)),replace=False)
pca=PCA(n_components=PCA_K,svd_solver="randomized",random_state=SEED).fit(emb[fit])
F=np.concatenate([F[:,:MP],pca.transform(emb).astype(np.float32),F[:,MP+EMB:]],axis=1)
print(f"frame matrix {F.shape}",flush=True)
rows=defaultdict(list)
for i,(p,v) in enumerate(zip(pid,vid)):
    ok=ourkey.get((p,qid(v)))
    if ok is not None: rows[ok].append(i)
for k in rows:
    a=np.array(rows[k]); rows[k]=a[np.argsort(fn[a])]
common=[k for k in keys if k in rows]
yv=np.array([yk[k] for k in common]); gv=np.array([gk[k] for k in common])
print(f"matched {len(common)}/{len(keys)}",flush=True)
def wstats(w,s,n):
    vel=np.abs(np.diff(w,axis=0)).mean(0) if len(w)>1 else np.zeros(w.shape[1],np.float32)
    return np.concatenate([w.mean(0),w.std(0),w.min(0),w.max(0),w[-1]-w[0],vel,
                           np.array([s/max(n,1),(s+len(w))/max(n,1)],np.float32)]).astype(np.float32)
hse_oof={}
for segname,seglen in (("tail60",60),("tail90",90)):
    Xw,vm=[],[]
    for vi,k in enumerate(common):
        idx=rows[k][-min(seglen,len(rows[k])):]
        f=F[idx]; n=len(f)
        for s in (list(range(0,n-WS+1,SLIDE)) or [0]):
            Xw.append(wstats(f[s:s+WS] if n>=WS else f,s,n)); vm.append(vi)
    Xw=np.asarray(Xw,np.float32); vm=np.asarray(vm)
    wy=yv[vm]; wg=gv[vm]
    oof=np.zeros(len(Xw),np.float32)
    for u in np.unique(wg):
        tr,va=wg!=u,wg==u
        clf=LGBMClassifier(**LGBM).fit(Xw[tr],wy[tr])
        oof[va]=clf.predict_proba(Xw[va])[:,1]
    vs=np.array([oof[vm==vi].mean() for vi in range(len(common))])
    hse_oof[f"HSE_{segname}"]=vs
    print(f"[HSE_{segname}] windows {len(Xw)} AUC {roc_auc_score(yv,vs):.4f}",flush=True)

P={}
ZOO={"au_xgb":Stream(("au",),model="xgb"),"blend":Stream(("blend",),model="xgb"),
     "gaze":Stream(("gaze",),model="xgb")}
for n,s in ZOO.items():
    try:
        o=_oof_by_key(track,s,5)
        if o: P[n]=np.array([o[k] for k in common])
    except Exception: pass
td=pd.read_csv(CACHE_DIR/f"temporal_dense_oof_t{track}.csv")
td["participant"]=td.participant.astype(str); td["video"]=td.video.astype(str)
tmap={(p,v):i for i,(p,v) in enumerate(zip(td.participant,td.video))}
for c in ("oof_gru_attn","oof_gru_attn_bc"):
    if c in td.columns:
        P["TD_"+c[4:]]=np.array([float(td[c].iloc[tmap[k]]) if k in tmap else np.nan for k in common])
P.update(hse_oof)
def pzr(s):
    o=np.empty_like(s)
    for u in np.unique(gv):
        m=gv==u; o[m]=(s[m]-s[m].mean())/(s[m].std()+1e-9)
    return o
for n in list(P): P[n+"_pz"]=pzr(P[n])
def A(s): return roc_auc_score(yv,s)
print("\nsolo video-AUC:",flush=True)
for n in sorted(P): print(f"  {n:18} {A(P[n]):.4f}",flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
def fuse(names):
    pf=np.zeros(len(yv))
    for u in np.unique(gv):
        tr,va=np.where(gv!=u)[0],np.where(gv==u)[0]
        ctr,cva=[],[]
        for n in names:
            iso=IsotonicRegression(out_of_bounds="clip").fit(P[n][tr],yv[tr])
            ctr.append(iso.transform(P[n][tr])); cva.append(iso.transform(P[n][va]))
        Atr,Ava=np.column_stack(ctr),np.column_stack(cva)
        pf[va]=Ava[:,0] if len(names)==1 else LogisticRegression(max_iter=2000).fit(Atr,yv[tr]).predict_proba(Ava)[:,1]
    return pf
rem=list(P); ch=[]; best=-1
while rem:
    sc=sorted([(A(fuse(ch+[s])),s) for s in rem],reverse=True)
    if sc[0][0]<=best+1e-4: break
    best=sc[0][0]; ch.append(sc[0][1]); rem.remove(sc[0][1])
    print(f"  greedy + {sc[0][1]:18} -> {best:.4f}",flush=True)
print(f"\nT2 OWNED greedy: {ch}  video-AUC {best:.4f}",flush=True)
print("refs: our 0.576 | his honest single 0.604/0.624pz | his nested fusion 0.612",flush=True)
pd.DataFrame({"participant":[k[0] for k in common],"video":[k[1] for k in common],
  **{f"oof_{n}":P[n] for n in hse_oof}}).to_csv(CACHE_DIR/"hse_owned_oof_t2.csv",index=False)
print("saved hse_owned_oof_t2.csv",flush=True)
