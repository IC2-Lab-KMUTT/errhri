"""Slim replica of philix's EXACT T1 lgbm_ws25 honest OOF, on our copied features.
Uses his window_stats(+q25/q75), single-PCA(128), leak_guard, scale_pos_weight,
video_MEAN agg, LOPO. Saves OOF for fusion + prints honest AUC/F1. Confirms 0.851 repro.
"""
import os, re, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier
from errhri_features import FeatureBank
from errhri_features.config import CACHE_DIR
SEED=42; np.random.seed(SEED)
WS,SLIDE,PCA_K,LEAK=25,10,128,0.30
LGBM=dict(objective="binary",n_estimators=300,learning_rate=0.03,num_leaves=31,
          max_depth=6,min_child_samples=60,subsample=0.8,subsample_freq=1,
          colsample_bytree=0.5,reg_lambda=5.0,reg_alpha=1.0,n_jobs=4,random_state=SEED,verbosity=-1)
z=np.load("/home/ic2/research/errhri/philix_feats/t1_feats.npz",allow_pickle=True)
pid=z["participant_ids"].astype(str); vid=z["official_video_ids"].astype(str)
fn=z["frame_nums"].astype(int); lab=z["labels"].astype(int)
MP=int(z["mp_dim"][0]); HSE=int(z["hse_dim"][0]); EMB=HSE-10
feats=z["features"].astype(np.float32)
def pzc(x,p):
    o=np.empty_like(x)
    for u in np.unique(p):
        m=p==u; mu,sd=x[m].mean(0),x[m].std(0)+1e-6; o[m]=(x[m]-mu)/sd
    return o
feats=pzc(feats,pid)
emb=feats[:,MP:MP+EMB]
fit=np.random.choice(len(emb),min(120000,len(emb)),replace=False)
pca=PCA(n_components=PCA_K,svd_solver="randomized",random_state=SEED).fit(emb[fit])
red=pca.transform(emb).astype(np.float32)
F=np.concatenate([feats[:,:MP],red,feats[:,MP+EMB:]],axis=1)
print(f"frame matrix {F.shape}",flush=True)
rows=defaultdict(list)
for i in range(len(pid)): rows[(pid[i],vid[i])].append(i)
videos=[]
for k,idx in rows.items():
    idx=np.array(idx); idx=idx[np.argsort(fn[idx])]
    videos.append(dict(pid=k[0],vid=k[1],idx=idx,label=int(lab[idx[0]]),n=len(idx)))
videos.sort(key=lambda v:(v["pid"],v["vid"]))
vy=np.array([v["label"] for v in videos]); vp=np.array([v["pid"] for v in videos])
vn=np.array([v["n"] for v in videos])
def wstats(w,s,n):
    q25,q75=np.quantile(w,0.25,0),np.quantile(w,0.75,0)
    vel=np.abs(np.diff(w,0)).mean(0) if len(w)>1 else np.zeros(w.shape[1],np.float32)
    return np.concatenate([w.mean(0),w.std(0),w.min(0),w.max(0),q25,q75,w[-1]-w[0],vel,
                           np.array([s/max(n,1),(s+len(w))/max(n,1)],np.float32)]).astype(np.float32)
def table():
    X,vm=[],[]
    for vi,v in enumerate(videos):
        f=F[v["idx"]]; n=len(f); st=list(range(0,n-WS+1,SLIDE)) or [0]
        for s in st: X.append(wstats(f[s:s+WS] if n>=WS else f,s,n)); vm.append(vi)
    return np.asarray(X,np.float32),np.asarray(vm)
X,vm=table()
wn=vn[vm].astype(np.float32); Xc=X-X.mean(0); yc=wn-wn.mean()
corr=(Xc*yc[:,None]).sum(0)/(np.sqrt((Xc**2).sum(0))*np.sqrt((yc**2).sum())+1e-9)
keep=np.abs(corr)<=LEAK; X=X[:,keep]
print(f"windows {len(X)} kept {keep.sum()}/{len(keep)}",flush=True)
wy=vy[vm]; wg=vp[vm]; spw=float((wy==0).sum())/max((wy==1).sum(),1)
oof=np.zeros(len(X),np.float32)
uu=np.unique(wg)
for i,u in enumerate(uu):
    tr,va=wg!=u,wg==u
    clf=LGBMClassifier(**{**LGBM,"scale_pos_weight":spw}).fit(X[tr],wy[tr])
    oof[va]=clf.predict_proba(X[va])[:,1]
    if (i+1)%6==0: print(f"  fold {i+1}/{len(uu)}",flush=True)
vscore=np.array([oof[vm==vi].mean() for vi in range(len(videos))])
auc=roc_auc_score(vy,vscore)
bf=max(f1_score(vy,(vscore>=t).astype(int),average="macro") for t in np.quantile(vscore,np.linspace(.02,.98,49)))
print(f"\nlgbm_ws25 (his exact recipe): video-AUC {auc:.4f}  best-global-F1 {bf:.4f}",flush=True)
print("philix saved OOF ref: AUC 0.851",flush=True)
pd.DataFrame({"participant":[v["pid"] for v in videos],"video":[v["vid"] for v in videos],
             "oof_hse_his":vscore}).to_csv(CACHE_DIR/"hse_his_oof_t1.csv",index=False)
print("saved hse_his_oof_t1.csv",flush=True)
