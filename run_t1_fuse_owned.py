"""T1 — fuse our FULLY-OWNED HSEmotion stream (hse_his_oof_t1.csv, his exact recipe on our
feature copy, AUC 0.820) into our calibrated stack. No dependence on philix's OOF dump.
Compares: our-stack+owned-HSE vs (reference) our-stack+philix-OOF vs our-stack alone.
"""
import sys, os, re, subprocess, tempfile
import numpy as np, pandas as pd
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR
track = 1
ZOO = {"au_xgb": Stream(("au",),model="xgb"),"aug_xgb":Stream(("au_graph",),model="xgb"),
       "gaze":Stream(("gaze",),model="xgb"),"pose":Stream(("pose",),model="xgb"),
       "blend":Stream(("blend",),model="xgb"),"audio":Stream(("audio",),model="xgb")}
ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk=dict(zip(keys,ref.y)); gk=dict(zip(keys,ref.groups)); nk=dict(zip(keys,ref.n_frames))
def qid(v): return re.sub(r"_.*","",str(v))
ourkey={(p,qid(v)):(p,v) for (p,v) in keys}
oofs={}
for n,s in ZOO.items():
    o=_oof_by_key(track,s,5)
    if o: oofs[n]=o
tdf=pd.read_csv(CACHE_DIR/f"temporal_oof_t{track}.csv")
tdf["participant"]=tdf.participant.astype(str); tdf["video"]=tdf.video.astype(str)
for c in [c for c in tdf.columns if c.startswith("oof_")]:
    oofs["T_"+c.replace("oof_","")]={(p,v):float(x) for p,v,x in zip(tdf.participant,tdf.video,tdf[c])}
# owned HSE (his exact recipe) - csv keyed by (participant, QID no-suffix) -> remap to our keys
hdf=pd.read_csv(CACHE_DIR/"hse_his_oof_t1.csv")
o={}
for p,v,x in zip(hdf.participant.astype(str),hdf.video.astype(str),hdf.oof_hse_his):
    ok=ourkey.get((p,qid(v)))
    if ok is not None: o[ok]=float(x)
oofs["HSE_owned"]=o; print(f"owned HSE joined {len(o)}/{len(keys)}",flush=True)
# philix OOF for reference
pz=np.load("/tmp/p_track1_oof.npz",allow_pickle=True)
po={}
for pp,vv,sc in zip(pz["pids"].astype(str),pz["vids"].astype(str),pz["lgbm_ws25"]):
    ok=ourkey.get((pp,qid(vv)))
    if ok is not None: po[ok]=float(sc)
oofs["P_ws25"]=po
common=[k for k in keys if all(k in od for od in oofs.values())]
P={n:np.array([od[k] for k in common]) for n,od in oofs.items()}
yv=np.array([yk[k] for k in common]); gv=np.array([gk[k] for k in common]); nv=np.array([nk[k] for k in common])
print(f"clips {len(common)} streams {list(P)}",flush=True)
def fuse(names):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    fid=subject_folds(gv,5); pf=np.zeros(len(yv))
    for trm,vam in iter_folds(fid):
        tr,va=np.where(trm)[0],np.where(vam)[0]; ctr,cva=[],[]
        for n in names:
            iso=IsotonicRegression(out_of_bounds="clip").fit(P[n][tr],yv[tr])
            ctr.append(iso.transform(P[n][tr])); cva.append(iso.transform(P[n][va]))
        Atr,Ava=np.column_stack(ctr),np.column_stack(cva)
        pf[va]=Ava[:,0] if len(names)==1 else LogisticRegression(max_iter=2000).fit(Atr,yv[tr]).predict_proba(Ava)[:,1]
    return pf
def clipF(pf):
    fid=subject_folds(gv,5); pred=np.zeros(len(yv),int)
    for trm,vam in iter_folds(fid):
        tr,va=np.where(trm)[0],np.where(vam)[0]
        thr=M.tune_threshold(yv[tr],pf[tr]); pred[va]=(pf[va]>=thr).astype(int)
    return M.primary(track,yv,pred,pf)
def official(pf,tag):
    fid=subject_folds(gv,5); pred=np.zeros(len(yv),int)
    for trm,vam in iter_folds(fid):
        tr,va=np.where(trm)[0],np.where(vam)[0]
        thr=M.tune_threshold(yv[tr],pf[tr]); pred[va]=(pf[va]>=thr).astype(int)
    gt,sub=[],[]
    for (pid,vid),yt,yp,prob,nf in zip(common,yv,pred,pf,nv):
        nf=int(max(nf,WS[track]))
        for f in range(1,nf+1): gt.append((pid,vid,f,int(yt)))
        for w in range(max((nf-WS[track])//SLIDE[track]+1,1)): sub.append((pid,vid,w,int(yp),float(1-prob),float(prob)))
    d=tempfile.mkdtemp(prefix="t1own_"); gtp,subp=os.path.join(d,"gt.csv"),os.path.join(d,"sub.csv")
    pd.DataFrame(gt,columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp,index=False)
    pd.DataFrame(sub,columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp,index=False)
    out=subprocess.run([sys.executable,REPO_EVAL,"--gt",gtp,"--pred",subp,"--track",str(track),"--fps",str(FPS[track]),"--window_size",str(WS[track]),"--slide",str(SLIDE[track])],capture_output=True,text=True)
    vid=out.stdout.split("n=1319")[-1] if "n=1319" in out.stdout else out.stdout
    f1=[l for l in vid.splitlines() if "F1 macro" in l]
    print(f"  {tag:34} video {f1[0].split(':')[-1].strip() if f1 else '?'}",flush=True)
def greedy(pool):
    rem=list(pool); ch=[]; best=-1
    while rem:
        sc=sorted([(clipF(fuse(ch+[s])),s) for s in rem],reverse=True)
        if sc[0][0]<=best+1e-4: break
        best=sc[0][0]; ch.append(sc[0][1]); rem.remove(sc[0][1])
    return ch,best
our=[k for k in P if not k.startswith("P_") and k!="HSE_owned"]
print("solo:",{n:round(clipF(P[n]),3) for n in ["HSE_owned","P_ws25"]},flush=True)
co,bo=greedy(our);                         print(f"\nOUR stack alone:        {bo:.3f} {co}",flush=True); official(fuse(co),"our_alone")
ch,bh=greedy(our+["HSE_owned"]);           print(f"OUR + owned HSE:        {bh:.3f} {ch}",flush=True); official(fuse(ch),"our+owned_HSE")
cp,bp=greedy(our+["P_ws25"]);              print(f"OUR + philix OOF (ref): {bp:.3f} {cp}",flush=True); official(fuse(cp),"our+philix_ref")
