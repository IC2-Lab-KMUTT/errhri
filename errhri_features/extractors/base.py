"""BaseExtractor: parallel, resumable, cached per-clip feature extraction.

Subclass and implement `init_worker()` (load your model once per process) and
`extract_clip(track, participant, video) -> dict`. `run(track)` handles frame budget, the
process pool, resume-from-cache, and writing `<name>_t<track>.csv` keyed by participant,video.
"""
from __future__ import annotations
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from ..config import CACHE_DIR

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GLOG_minloglevel", "3")

_EXTR = None  # per-process singleton


def _init_global(extractor):
    global _EXTR
    _EXTR = extractor
    _EXTR.init_worker()


def _run_one(args):
    track, pid, vid = args
    try:
        row = _EXTR.extract_clip(track, pid, vid)
    except Exception as e:  # never kill the pool on one bad clip
        row = {"_error": str(e)[:120]}
    row.update(participant=pid, video=vid)
    return row


class BaseExtractor:
    name = "base"

    def __init__(self, s: int = 8, workers: int = 8, cache_dir: Path = None):
        self.s = s
        self.workers = workers
        self.cache_dir = Path(cache_dir or CACHE_DIR)

    # ---- override these -----------------------------------------------------
    def init_worker(self):
        """Load the model into `self` once per worker process."""

    def extract_clip(self, track, participant, video) -> dict:
        raise NotImplementedError

    # ---- driver -------------------------------------------------------------
    def out_path(self, track):
        return self.cache_dir / f"{self.name}_t{track}.csv"

    def run(self, track: int, index_df: pd.DataFrame = None):
        from ..datasets import load_index
        idx = index_df if index_df is not None else load_index(track)
        out = self.out_path(track)
        done = set()
        if out.exists():
            d = pd.read_csv(out)
            done = set(map(tuple, d[["participant", "video"]].astype(str).values))
        todo = [(track, str(r.participant), str(r.video)) for r in idx.itertuples()
                if (str(r.participant), str(r.video)) not in done]
        print(f"[{self.name}] track {track}: {len(todo)} to do ({len(done)} cached), "
              f"{self.workers}w x {self.s}f", flush=True)
        if not todo:
            return out
        rows = []
        with ProcessPoolExecutor(self.workers, initializer=_init_global, initargs=(self,)) as ex:
            for n, fut in enumerate(as_completed([ex.submit(_run_one, a) for a in todo]), 1):
                rows.append(fut.result())
                if n % 200 == 0:
                    self._flush(rows, out); rows = []
                    print(f"[{self.name}] {n}/{len(todo)}", flush=True)
        self._flush(rows, out)
        final = pd.read_csv(out)
        print(f"[{self.name}] DONE track {track}: {len(final)} rows, {final.shape[1]-2} feats -> {out}")
        return out

    def _flush(self, rows, out):
        if not rows:
            return
        new = pd.DataFrame(rows)
        if out.exists():
            new = pd.concat([pd.read_csv(out), new], ignore_index=True)
            new = new.drop_duplicates(subset=["participant", "video"], keep="last")
        out.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(out, index=False)
