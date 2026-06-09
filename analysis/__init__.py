"""Analysis pipelines — the experiments that produced the curated signal map.

These are runnable reproductions of the studies behind SIGNAL_INVENTORY.md / signal_map.py, so
teammates can re-derive (and challenge) every verdict on their own cache rather than trust a table:

  feature_report      per-feature univariate strength table -> FEATURE_STRENGTH.md (both tracks)
  dimension_breakdown per-semantic-dimension AUC, static vs dynamics split
  timing_features     granular timing-only signal check (onset/peak/magnitude)
  complementarity     cross-stream fusion potential (error decorrelation, oracle, late fusion)

Run any as a module, e.g.  `python -m analysis.feature_report`.
"""
