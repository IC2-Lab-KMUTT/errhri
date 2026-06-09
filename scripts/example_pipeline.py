"""End-to-end example a teammate can copy: per-stream CV -> late fusion -> official submission.

Everything model-side goes through the `pipelines` recipe layer, so the core signal pipeline is
never touched — you change behaviour by changing the `Stream` configs / fusion params below.

    python -m scripts.example_pipeline --track 1
"""
import argparse
from errhri_features import FeatureBank, submission
from pipelines.recipes import Stream, run_stream, run_fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=1)
    a = ap.parse_args()

    # --- the part you tweak: which streams, which models, which params ---------------------
    if a.track == 1:
        streams = [Stream(("au",)), Stream(("audio",)), Stream(("embed",))]   # weak-but-orthogonal
    else:
        streams = [Stream(("au",))]                                            # T2: drop weak streams

    # 1) per-stream group-eval
    for s in streams:
        print(f"  {s.name:<14} {run_stream(a.track, s)}")

    # 2) late fusion (validated T1 win). include_temporal=True adds the GRU if the `traj` cache exists.
    if len(streams) > 1:
        fused = run_fusion(a.track, streams, method="stack")
        print(f"  FUSION         {fused}")
        probs, preds = fused.oof_prob, fused.oof_pred
    else:
        rep = run_stream(a.track, streams[0])
        probs, preds = rep.oof_prob, rep.oof_pred

    # 3) official windowed submission + video-level score (sanity-check vs CV)
    bank = FeatureBank(track=a.track, modalities=streams[0].modalities).load()
    keys = list(zip(bank.df.participant, bank.df.video))
    sub = submission.write_submission(a.track, keys, probs, preds, bank.n_frames,
                                      path=f"submission_track{a.track}.csv")
    gt = bank.df[["participant", "video", "label"]]
    print("  official video-level:", submission.official_score(a.track, sub, gt))


if __name__ == "__main__":
    main()
