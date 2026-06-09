"""End-to-end example a teammate can copy: load features -> CV per stream -> late fusion ->
official submission. Shows exactly the three things you control: features, model, fusion.

    python -m scripts.example_pipeline --track 1
"""
import argparse
import numpy as np
from xgboost import XGBClassifier
from errhri_features import FeatureBank, CVEvaluator, late_fusion, submission


def xgb(spw=1.0):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=2.0, scale_pos_weight=spw,
                         eval_metric="logloss", tree_method="hist", n_jobs=4, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=1)
    a = ap.parse_args()
    modalities = ["au", "audio", "embed"] if a.track == 1 else ["au"]   # drop weak streams on T2
    bank = FeatureBank(track=a.track, modalities=modalities).load()
    y, groups, nfr = bank.y, bank.groups, bank.n_frames
    spw = max((y == 0).sum(), 1) / max((y == 1).sum(), 1)
    ev = CVEvaluator(track=a.track)

    # 1) per-modality streams (each leak-cleaned, signal-tier features)
    oof = {}
    for mod in modalities:
        b = FeatureBank(track=a.track, modalities=[mod]).load()
        rep = ev.run(lambda: xgb(spw), b, select="signal", leak_clean=True)
        oof[mod] = rep.oof_prob
        print(f"  {mod:<6} {rep}")

    # 2) late fusion (the validated win on T1)
    if len(oof) > 1:
        fused = late_fusion(a.track, oof, y, groups, nfr, method="stack")
        print(f"  FUSION {fused}")
        probs, preds = fused.oof_prob, fused.oof_pred
    else:
        rep = ev.run(lambda: xgb(spw), bank, select="signal", leak_clean=True)
        probs, preds = rep.oof_prob, rep.oof_pred

    # 3) official windowed submission + video-level score (sanity-check vs CV)
    keys = list(zip(bank.df.participant, bank.df.video))
    sub = submission.write_submission(a.track, keys, probs, preds, nfr,
                                      path=f"submission_track{a.track}.csv")
    gt = bank.df[["participant", "video", "label"]]
    print("  official video-level:", submission.official_score(a.track, sub, gt))


if __name__ == "__main__":
    main()
