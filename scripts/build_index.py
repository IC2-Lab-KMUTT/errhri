"""Build the clip index `index_t<track>.csv` from the raw videos (the first thing to run).

Scans  RAW_ROOTS[track]/<participant>/<video>.mp4, derives the label from the filename
convention, and counts frames. Output schema: participant,video,label,n_frames.

  Track 1 "BAD":      <QID...>_<label>.mp4    label 1=failure, 0=control
  Track 2 "Bad Idea": q_<n>_main_<label>.mp4  label 1=poorly, 0=well
Both encode the label as the trailing _<0|1> token, so we read that.

    python -m scripts.build_index --track all
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from errhri_features.config import RAW_ROOTS, CACHE_DIR


def _count_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def _one(args):
    pid, vid, path = args
    try:
        label = int(vid.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return dict(participant=pid, video=vid, label=label, n_frames=_count_frames(path))


def build(track, workers=8):
    root = RAW_ROOTS[track]
    jobs = []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mp4 in sorted(pdir.glob("*.mp4")):
            jobs.append((pdir.name, mp4.stem, mp4))
    if not jobs:
        raise FileNotFoundError(f"no videos under {root} — set ERRHRI_T{track}_ROOT")
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_one, jobs) if r is not None]
    df = pd.DataFrame(rows).sort_values(["participant", "video"]).reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"index_t{track}.csv"
    df.to_csv(out, index=False)
    print(f"track {track}: {len(df)} clips, {df.participant.nunique()} subjects, "
          f"{int(df.label.sum())} pos / {int((df.label == 0).sum())} neg -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="all", help="1 | 2 | all")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for t in ([1, 2] if a.track == "all" else [int(a.track)]):
        build(t, a.workers)


if __name__ == "__main__":
    main()
