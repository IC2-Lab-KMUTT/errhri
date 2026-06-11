"""T2 scanpath: reading/cognitive-load dynamics from full-fps iris tracking over the clip TAIL.
Per clip: last 5s @ native fps -> FaceMesh(refine) iris center + EAR -> saccade/fixation/blink
features (n_saccades, regressions, fixation durations, path length, H/V ratio, blink count,
vel stats). LGBM LOPO -> video-AUC. New representation: reading behavior, not emotion/geometry.
"""
import os, re, glob, numpy as np, pandas as pd, cv2
import mediapipe as mp
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank
from errhri_features.config import CACHE_DIR
track=2; TAIL_S=5.0
ref=FeatureBank(track,["au"]).load()
keys=list(zip(ref.df.participant.astype(str),ref.df.video.astype(str)))
yk=dict(zip(keys,ref.y)); gk=dict(zip(keys,ref.groups))
ROOT="/home/ic2/research/errhri/raw/d1/trainval"
fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,
   max_num_faces=1,min_detection_confidence=0.3,min_tracking_confidence=0.3)
L_IRIS=[468,469,470,471,472]; R_IRIS=[473,474,475,476,477]
L_EYE=[33,133,159,145]; R_EYE=[362,263,386,374]
def ear(lm,ids):
    p=lambda i:np.array([lm[i].x,lm[i].y])
    return (np.linalg.norm(p(ids[2])-p(ids[3])))/(np.linalg.norm(p(ids[0])-p(ids[1]))+1e-9)
def track_clip(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): return None
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start=max(0,nf-int(TAIL_S*fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    xs,ys,ears=[],[],[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        res=fm.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark
            ix=np.mean([lm[i].x for i in L_IRIS+R_IRIS]); iy=np.mean([lm[i].y for i in L_IRIS+R_IRIS])
            # gaze relative to eye corners (head-motion invariant-ish)
            cx=np.mean([lm[i].x for i in (33,133,362,263)]); cy=np.mean([lm[i].y for i in (33,133,362,263)])
            xs.append(ix-cx); ys.append(iy-cy)
            ears.append((ear(lm,L_EYE)+ear(lm,R_EYE))/2)
        else:
            xs.append(np.nan); ys.append(np.nan); ears.append(np.nan)
    cap.release()
    return np.array(xs),np.array(ys),np.array(ears),fps
def feats(xs,ys,ears,fps):
    m=~np.isnan(xs)
    if m.sum()<10: return None
    x,y,e=xs[m],ys[m],ears[m]
    vx,vy=np.diff(x),np.diff(y)
    v=np.hypot(vx,vy)
    thr=np.median(v)*3+1e-6
    sac=v>thr
    nsac=int(np.sum(np.diff(sac.astype(int))==1))
    # regressions: sign flips of x-velocity during saccades (re-reading)
    sx=np.sign(vx[sac]) if sac.any() else np.array([0])
    nreg=int(np.sum(np.diff(sx)!=0))
    # fixations: runs below threshold
    fix=~sac; runs=[]; c=0
    for b in fix:
        if b: c+=1
        elif c: runs.append(c); c=0
    if c: runs.append(c)
    runs=np.array(runs)/fps if len(runs) else np.array([0.0])
    blink=int(np.sum((e[:-1]>0.2)&(e[1:]<=0.2)))
    return np.array([nsac/len(x)*fps, nreg/max(nsac,1), runs.mean(), runs.max(),
                     v.sum(), np.abs(vx).sum()/(np.abs(vy).sum()+1e-9),
                     v.mean()*fps, v.max(), blink/len(x)*fps, x.std(), y.std(),
                     e.mean(), e.std(), 1.0-m.mean()],dtype=np.float32)
rows=[]
done=0
for (p,v) in keys:
    # raw file may carry a take suffix already in our key
    cand=glob.glob(f"{ROOT}/{p}/{v}.mp4") or glob.glob(f"{ROOT}/{p}/{v}_*.mp4")
    if not cand: continue
    r=track_clip(cand[0])
    if r is None: continue
    f=feats(*r)
    if f is None: continue
    rows.append((p,v,*f))
    done+=1
    if done%50==0: print(f"  {done} clips",flush=True)
cols=["nsac_ps","reg_ratio","fix_mean","fix_max","path_len","hv_ratio","vel_mean","vel_max",
      "blink_ps","x_std","y_std","ear_mean","ear_std","miss_rate"]
df=pd.DataFrame(rows,columns=["participant","video"]+cols)
df.to_csv(CACHE_DIR/"scanpath_t2.csv",index=False)
print(f"extracted {len(df)} clips -> scanpath_t2.csv",flush=True)
sk=[(p,v) for p,v in zip(df.participant,df.video)]
yv=np.array([yk[k] for k in sk]); gv=np.array([gk[k] for k in sk])
X=df[cols].to_numpy(np.float32)
# per-participant z
for u in np.unique(gv):
    m=gv==u; X[m]=(X[m]-X[m].mean(0))/(X[m].std(0)+1e-6)
LGBM=dict(objective="binary",n_estimators=200,learning_rate=0.05,num_leaves=15,max_depth=4,
          min_child_samples=30,subsample=0.8,colsample_bytree=0.7,reg_lambda=3.0,
          n_jobs=4,random_state=42,verbosity=-1)
oof=np.zeros(len(yv))
for u in np.unique(gv):
    tr,va=gv!=u,gv==u
    oof[va]=LGBMClassifier(**LGBM).fit(X[tr],yv[tr]).predict_proba(X[va])[:,1]
print(f"\nSCANPATH solo video-AUC: {roc_auc_score(yv,oof):.4f} (n={len(yv)})",flush=True)
def pzr(s):
    o=np.empty_like(s)
    for u in np.unique(gv):
        m=gv==u; o[m]=(s[m]-s[m].mean())/(s[m].std()+1e-9)
    return o
print(f"SCANPATH pz video-AUC:   {roc_auc_score(yv,pzr(oof)):.4f}",flush=True)
pd.DataFrame({"participant":[k[0] for k in sk],"video":[k[1] for k in sk],
              "oof_scanpath":oof}).to_csv(CACHE_DIR/"scanpath_oof_t2.csv",index=False)
print("saved scanpath_oof_t2.csv",flush=True)
