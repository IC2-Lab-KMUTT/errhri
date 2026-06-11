"""Model the saved scanpath features + fuse into the T2 owned stack (vs 0.6262)."""
import re, numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from errhri_features import FeatureBank
from pipelines.recipes import _oof_by_key, Stream
from errhri_features.config import CACHE_DIR
track=2
ref=FeatureBank(track,["au"]).load()
keys=list(zip(ref.df.participant.astype(str),ref.df.video.astype(str)))
yk=dict(zip(keys,ref.y)); gk=dict(zip(keys,ref.groups))
df=pd.read_csv(CACHE_DIR/"scanpath_t2.csv")
df["participant"]=df.participant.astype(str); df["video"]=df.video.astype(str)
cols=[c for c in df.columns if c not in ("participant","video")]
sk=[(p,v) for p,v in zip(df.participant,df.video)]
yv=np.array([yk[k] for k in sk]); gv=np.array([gk[k] for k in sk])
X=df[cols].to_numpy(np.float32).copy()
X=np.nan_to_num(X)
for u in np.unique(gv):
    m=gv==u; X[m]=(X[m]-X[m].mean(0))/(X[m].std(0)+1e-6)
LGBM=dict(objective="binary",n_estimators=200,learning_rate=0.05,num_leaves=15,max_depth=4,
          min_child_samples=30,subsample=0.8,colsample_bytree=0.7,reg_lambda=3.0,
          n_jobs=4,random_state=42,verbosity=-1)
oof=np.zeros(len(yv))
for u in np.unique(gv):
    tr,va=gv!=u,gv==u
    oof[va]=LGBMClassifier(**LGBM).fit(X[tr],yv[tr]).predict_proba(X[va])[:,1]
def pzr(s,g):
    o=np.empty_like(s)
    for u in np.unique(g):
        m=g==u; o[m]=(s[m]-s[m].mean())/(s[m].std()+1e-9)
    return o
print(f"SCANPATH solo AUC {roc_auc_score(yv,oof):.4f}  pz {roc_auc_score(yv,pzr(oof,gv)):.4f}",flush=True)
pd.DataFrame({"participant":[k[0] for k in sk],"video":[k[1] for k in sk],
              "oof_scanpath":oof}).to_csv(CACHE_DIR/"scanpath_oof_t2.csv",index=False)
# fuse into owned T2 stack
hs=pd.read_csv(CACHE_DIR/"hse_owned_oof_t2.csv")
hs["participant"]=hs.participant.astype(str); hs["video"]=hs.video.astype(str)
hmap={(p,v):i for i,(p,v) in enumerate(zip(hs.participant,hs.video))}
smap={k:i for i,k in enumerate(sk)}
common=[k for k in keys if k in hmap and k in smap]
yv2=np.array([yk[k] for k in common]); gv2=np.array([gk[k] for k in common])
P={}
for c in ("oof_HSE_tail60","oof_HSE_tail90"):
    P[c[4:]]=np.array([float(hs[c].iloc[hmap[k]]) for k in common])
P["scanpath"]=np.array([oof[smap[k]] for k in common])
ZOO={"gaze":Stream(("gaze",),model="xgb"),"au_xgb":Stream(("au",),model="xgb"),"blend":Stream(("blend",),model="xgb")}
for n,s in ZOO.items():
    try:
        o=_oof_by_key(track,s,5)
        if o: P[n]=np.array([o[k] for k in common])
    except Exception: pass
td=pd.read_csv(CACHE_DIR/f"temporal_dense_oof_t{track}.csv")
td["participant"]=td.participant.astype(str); td["video"]=td.video.astype(str)
tmap={(p,v):i for i,(p,v) in enumerate(zip(td.participant,td.video))}
if "oof_gru_attn" in td.columns:
    P["TD_gru_attn"]=np.array([float(td["oof_gru_attn"].iloc[tmap[k]]) for k in common if k in tmap])
for n in list(P): P[n+"_pz"]=pzr(P[n],gv2)
def A(s): return roc_auc_score(yv2,s)
def fuse(names):
    pf=np.zeros(len(yv2))
    for u in np.unique(gv2):
        tr,va=np.where(gv2!=u)[0],np.where(gv2==u)[0]
        ctr,cva=[],[]
        for n in names:
            iso=IsotonicRegression(out_of_bounds="clip").fit(P[n][tr],yv2[tr])
            ctr.append(iso.transform(P[n][tr])); cva.append(iso.transform(P[n][va]))
        Atr,Ava=np.column_stack(ctr),np.column_stack(cva)
        pf[va]=Ava[:,0] if len(names)==1 else LogisticRegression(max_iter=2000).fit(Atr,yv2[tr]).predict_proba(Ava)[:,1]
    return pf
rem=list(P); ch=[]; best=-1
while rem:
    sc=sorted([(A(fuse(ch+[s])),s) for s in rem],reverse=True)
    if sc[0][0]<=best+1e-4: break
    best=sc[0][0]; ch.append(sc[0][1]); rem.remove(sc[0][1])
    print(f"  greedy + {sc[0][1]:18} -> {best:.4f}",flush=True)
print(f"\nT2 owned+scanpath greedy: {ch}  AUC {best:.4f}  (prev owned best 0.6262)",flush=True)
