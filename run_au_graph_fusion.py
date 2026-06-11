"""Measure whether dense OpenGraphAU (au_graph) lifts the officially-ranked video macro-F1.

Runs the official eval.py bridge for several stream sets on a track. au_graph is the fast
single-forward 41-AU dense stream (s=48), compared against / added to the proven baseline
(xgb_au + rf_au + blend) and the current best facial mix (+faceemb +pose).
"""
import sys
from pipelines.official import official_report, DEFAULT_STREAMS
from pipelines.recipes import Stream

BASE = dict(DEFAULT_STREAMS)                              # xgb_au, rf_au, blend
AUG = {"au_graph": Stream(("au_graph",), model="xgb")}
AUG_RF = {"au_graph_rf": Stream(("au_graph",), model="rf")}
FEMB = {"faceemb": Stream(("faceemb",), model="xgb")}
POSE = {"pose": Stream(("pose",), model="xgb")}

CONFIGS = {
    "au_graph_solo":          AUG,
    "baseline":               BASE,
    "baseline+au_graph":      {**BASE, **AUG},
    "baseline+au_graph(x+rf)":{**BASE, **AUG, **AUG_RF},
    "base+faceemb+pose":      {**BASE, **FEMB, **POSE},
    "base+faceemb+pose+aug":  {**BASE, **FEMB, **POSE, **AUG},
}

track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
only = sys.argv[2] if len(sys.argv) > 2 else None
for name, streams in CONFIGS.items():
    if only and only not in name:
        continue
    print(f"\n############## {name} ##############", flush=True)
    try:
        official_report(track, streams=streams, objective="macro")
    except Exception as e:
        import traceback; traceback.print_exc()
