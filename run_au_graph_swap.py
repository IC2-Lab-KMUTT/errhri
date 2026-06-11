"""Test the REPLACE hypothesis: does the higher-quality dense au_graph help when it SUBSTITUTES the
old 10-frame py-feat AU, instead of being stacked on top of it (which only dilutes/duplicates)?

Baseline core = {xgb_au, rf_au, blend} where `au` is the OLD 10-frame py-feat AU.
Swap core      = {xgb_au_graph, rf_au_graph, blend} — old AU replaced by dense OpenGraphAU.
"""
import sys
from pipelines.official import official_report
from pipelines.recipes import Stream

BLEND = {"blend": Stream(("blend",), model="xgb")}
OLD_AU = {"xgb_au": Stream(("au",), model="xgb"), "rf_au": Stream(("au",), model="rf")}
AUG = {"xgb_aug": Stream(("au_graph",), model="xgb"), "rf_aug": Stream(("au_graph",), model="rf")}
FEMB = {"faceemb": Stream(("faceemb",), model="xgb")}
POSE = {"pose": Stream(("pose",), model="xgb")}

CONFIGS = {
    "baseline (old au)":          {**OLD_AU, **BLEND},
    "SWAP au->au_graph":          {**AUG, **BLEND},
    "SWAP + faceemb + pose":      {**AUG, **BLEND, **FEMB, **POSE},
    "SWAP + old au too (all AU)": {**OLD_AU, **AUG, **BLEND},
}

track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for name, streams in CONFIGS.items():
    print(f"\n############## {name} ##############", flush=True)
    try:
        official_report(track, streams=streams, objective="macro")
    except Exception:
        import traceback; traceback.print_exc()
