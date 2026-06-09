"""Extract every (or selected) feature modality to the cache, for one or both tracks.

    python -m scripts.build_index  --track all                     # FIRST: build the clip index
    python -m scripts.extract_all  --track 1 --modalities au audio embed blend traj
    python -m scripts.extract_all  --track all --modalities au

Caches land in ERRHRI_CACHE (see errhri_features/config.py). Re-running resumes from cache.
`scripts.build_index` must run once first — every extractor reads `index_t<track>.csv` for the
clip list + labels. Heads-up: blend/au/embed/traj want CPU-only torch (see README install caveat);
run them from the CPU-torch venv. audio needs ffmpeg + opensmile. `traj` (per-frame trajectory for
the temporal GRU) needs the same FaceLandmarker model as `blend`.
"""
import argparse
from errhri_features.extractors import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="1", help="1 | 2 | all")
    ap.add_argument("--modalities", nargs="+", default=["au", "audio", "embed"],
                    choices=list(REGISTRY))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--s", type=int, default=None, help="frames/clip (extractor default if unset)")
    a = ap.parse_args()
    tracks = [1, 2] if a.track == "all" else [int(a.track)]
    for t in tracks:
        for mod in a.modalities:
            cls = REGISTRY[mod]
            kw = {"workers": a.workers}
            if a.s is not None and mod != "audio":
                kw["s"] = a.s
            cls(**kw).run(t)


if __name__ == "__main__":
    main()
