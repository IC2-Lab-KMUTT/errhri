# Feature strength — granular per-feature correlation breakdown

Univariate signal strength of **every engineered feature** on both tracks, so you can make
your own keep/drop calls instead of trusting the curated `signal_map.py` verdicts. Generated
by `analysis/feature_report.py` from the feature caches — re-run it on your own data.

- **sep** = univariate separability, `max(AUC, 1-AUC)` of the per-subject-normalized feature
  vs the label (0.50 = noise, higher = more discriminative *alone*; fusion can still rescue a
  low-sep feature if it is *orthogonal* — see `analysis/complementarity.py`).
- **dir** = `+` feature is higher in the error class (failure / poorly-handled), `-` lower.
- **leak** = `|corr(feature, n_frames)|`; **> 0.30 is a duration proxy — do not use on T1**
  (the `leak_clean=True` guard strips these automatically).
- **fam** = static level / dynamics (std·range·slope·velocity) / timing (onset·peak·magnitude).

> Track-1 reference bars: official baseline macro-F1 **0.502**, our honest fusion **0.674**,
> the forbidden duration-only leak **0.702**. Track-2: baseline AUC **0.564** (signal is weak;
> most per-feature `sep` sit near 0.50 — that is the finding, not a bug).


## expression  (10 feats, T1 best sep 0.701)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `expr_entropy` | dynamics | 0.701 | + | 0.00 | 0.521 | + |
| `expr_neutral` | static | 0.687 | - | 0.02 | 0.545 | - |
| `expr_happiness` | static | 0.686 | + | 0.06 | 0.547 | + |
| `expr_nswitch` | dynamics | 0.671 | + | 0.09 | 0.512 | - |
| `expr_surprise` | static | 0.525 | + | 0.02 | 0.511 | - |
| `expr_anger` | static | 0.513 | - | 0.09 | 0.509 | + |
| `expr_sadness` | static | 0.501 | + | 0.02 | 0.514 | + |
| `expr_fear` | static | 0.501 | + | 0.01 | 0.503 | - |
| `expr_contempt` | static | 0.500 | + | 0.00 | 0.500 | + |
| `expr_disgust` | static | 0.500 | + | 0.00 | 0.500 | + |

## smile  (45 feats, T1 best sep 0.661)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `au6int_iqr` | dynamics | 0.661 | + | 0.05 | 0.530 | - |
| `au12int_vel` | dynamics | 0.657 | + | 0.04 | 0.529 | - |
| `au6int_vel` | dynamics | 0.653 | + | 0.01 | 0.533 | - |
| `au12int_std` | dynamics | 0.650 | + | 0.00 | 0.531 | - |
| `au6int_std` | dynamics | 0.646 | + | 0.03 | 0.544 | - |
| `au12int_range` | dynamics | 0.645 | + | 0.00 | 0.526 | - |
| `au6int_range` | dynamics | 0.645 | + | 0.03 | 0.539 | - |
| `au12int_max` | static | 0.633 | + | 0.01 | 0.538 | - |
| `au12int_p90` | static | 0.633 | + | 0.01 | 0.542 | - |
| `au12int_iqr` | dynamics | 0.627 | + | 0.00 | 0.502 | - |
| `au6int_max` | static | 0.626 | + | 0.03 | 0.533 | - |
| `au6int_p90` | static | 0.622 | + | 0.03 | 0.553 | - |
| `geolipcorner_asym_std` | dynamics | 0.615 | + | 0.08 | 0.545 | + |
| `geolipcorner_asym_vel` | dynamics | 0.615 | + | 0.09 | 0.546 | + |
| `au12_rate` | static | 0.613 | + | 0.01 | 0.509 | - |
| `au12int_mean` | static | 0.613 | + | 0.01 | 0.543 | - |
| `geolipcorner_asym_range` | dynamics | 0.610 | + | 0.08 | 0.546 | + |
| `au6int_mean` | static | 0.609 | + | 0.03 | 0.545 | - |
| `au6int_slope` | dynamics | 0.608 | + | 0.08 | 0.570 | + |
| `au12int_slope` | dynamics | 0.608 | + | 0.08 | 0.559 | + |
| `au6_rate` | static | 0.595 | + | 0.02 | 0.510 | - |
| `geolipcorner_asym_slope` | dynamics | 0.594 | + | 0.01 | 0.512 | - |
| `geolipcorner_asym_max` | static | 0.593 | + | 0.06 | 0.550 | + |
| `au14_rate` | static | 0.592 | + | 0.02 | 0.520 | - |
| `geolipcorner_asym_iqr` | dynamics | 0.588 | + | 0.08 | 0.558 | + |
| `au12int_delta` | dynamics | 0.584 | + | 0.08 | 0.531 | + |
| `geolipcorner_asym_p90` | static | 0.582 | + | 0.05 | 0.553 | + |
| `geolipcorner_asym_delta` | dynamics | 0.582 | + | 0.02 | 0.502 | - |
| `au6int_delta` | dynamics | 0.578 | + | 0.07 | 0.540 | + |
| `au12_ever` | static | 0.577 | + | 0.00 | 0.515 | - |
| `au12_ntrans` | dynamics | 0.576 | + | 0.03 | 0.509 | - |
| `au6_ntrans` | dynamics | 0.574 | + | 0.02 | 0.503 | - |
| `au6_ever` | static | 0.573 | + | 0.02 | 0.502 | - |
| `au12int_median` | static | 0.564 | + | 0.00 | 0.516 | - |
| `au6int_median` | static | 0.559 | + | 0.02 | 0.534 | - |
| `geolipcorner_asym_mean` | static | 0.555 | + | 0.03 | 0.537 | + |
| `au14_ever` | static | 0.543 | + | 0.01 | 0.520 | + |
| `geolipcorner_asym_median` | static | 0.543 | + | 0.02 | 0.524 | + |
| `au6int_p10` | static | 0.540 | + | 0.02 | 0.517 | - |
| `au6int_min` | static | 0.529 | + | 0.02 | 0.505 | - |
| `au14_ntrans` | dynamics | 0.527 | + | 0.04 | 0.502 | - |
| `au12int_p10` | static | 0.527 | + | 0.01 | 0.518 | - |
| `geolipcorner_asym_min` | static | 0.523 | - | 0.01 | 0.510 | + |
| `geolipcorner_asym_p10` | static | 0.504 | + | 0.01 | 0.519 | + |
| `au12int_min` | static | 0.501 | + | 0.01 | 0.510 | - |

## mouth_jaw  (72 feats, T1 best sep 0.691)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `geomouth_width_std` | dynamics | 0.691 | + | 0.05 | 0.584 | + |
| `geomouth_width_range` | dynamics | 0.691 | + | 0.04 | 0.573 | + |
| `geomouth_width_vel` | dynamics | 0.688 | + | 0.01 | 0.579 | + |
| `geomouth_width_max` | static | 0.687 | + | 0.05 | 0.542 | + |
| `geomouth_width_p90` | static | 0.681 | + | 0.04 | 0.550 | + |
| `geonose_lip_iqr` | dynamics | 0.671 | + | 0.05 | 0.567 | + |
| `geomouth_width_delta` | dynamics | 0.669 | + | 0.10 | 0.553 | + |
| `geomouth_width_slope` | dynamics | 0.668 | + | 0.10 | 0.574 | + |
| `geonose_lip_min` | static | 0.667 | - | 0.03 | 0.582 | - |
| `geonose_lip_std` | dynamics | 0.666 | + | 0.01 | 0.599 | + |
| `geomouth_open_std` | dynamics | 0.665 | + | 0.02 | 0.527 | + |
| `geomouth_width_iqr` | dynamics | 0.663 | + | 0.07 | 0.593 | + |
| `geomouth_open_range` | dynamics | 0.662 | + | 0.02 | 0.524 | + |
| `geonose_lip_range` | dynamics | 0.661 | + | 0.00 | 0.595 | + |
| `geonose_lip_p10` | static | 0.659 | - | 0.03 | 0.571 | - |
| `geojaw_open_std` | dynamics | 0.653 | + | 0.06 | 0.579 | + |
| `geojaw_open_slope` | dynamics | 0.652 | + | 0.07 | 0.568 | + |
| `geojaw_open_range` | dynamics | 0.650 | + | 0.06 | 0.578 | + |
| `geomouth_open_max` | static | 0.650 | + | 0.03 | 0.536 | + |
| `geonose_lip_vel` | dynamics | 0.645 | + | 0.07 | 0.561 | + |
| `au26int_std` | dynamics | 0.645 | + | 0.00 | 0.514 | + |
| `geomouth_open_p90` | static | 0.645 | + | 0.02 | 0.542 | + |
| `au26int_range` | dynamics | 0.645 | + | 0.01 | 0.509 | + |
| `geojaw_open_vel` | dynamics | 0.643 | + | 0.09 | 0.562 | + |
| `au26int_vel` | dynamics | 0.642 | + | 0.02 | 0.529 | - |
| `geojaw_open_iqr` | dynamics | 0.639 | + | 0.02 | 0.563 | + |
| `geomouth_width_mean` | static | 0.637 | + | 0.03 | 0.532 | + |
| `geomouth_open_vel` | dynamics | 0.637 | + | 0.07 | 0.503 | + |
| `au26int_iqr` | dynamics | 0.636 | + | 0.03 | 0.512 | + |
| `au25int_range` | dynamics | 0.632 | + | 0.03 | 0.541 | - |
| `au26int_max` | static | 0.628 | + | 0.01 | 0.516 | + |
| `au25int_std` | dynamics | 0.626 | + | 0.02 | 0.538 | - |
| `geomouth_open_mean` | static | 0.626 | + | 0.00 | 0.542 | + |
| `geojaw_open_max` | static | 0.626 | + | 0.01 | 0.522 | + |
| `geojaw_open_delta` | dynamics | 0.624 | + | 0.08 | 0.547 | + |
| `au25int_vel` | dynamics | 0.623 | + | 0.06 | 0.545 | - |
| `geonose_lip_mean` | static | 0.614 | - | 0.04 | 0.536 | - |
| `geomouth_width_median` | static | 0.614 | + | 0.01 | 0.529 | + |
| `au25int_max` | static | 0.613 | + | 0.03 | 0.547 | - |
| `geomouth_open_iqr` | dynamics | 0.613 | + | 0.02 | 0.532 | + |
| `au26int_p90` | static | 0.613 | + | 0.01 | 0.529 | + |
| `geojaw_open_p90` | static | 0.609 | + | 0.02 | 0.517 | + |
| `geomouth_open_slope` | dynamics | 0.597 | + | 0.06 | 0.584 | + |
| `au25int_iqr` | dynamics | 0.594 | + | 0.01 | 0.552 | + |
| `au25int_p90` | static | 0.594 | + | 0.02 | 0.551 | - |
| `geonose_lip_median` | static | 0.591 | - | 0.04 | 0.532 | - |
| `au26int_mean` | static | 0.589 | + | 0.01 | 0.521 | - |
| `geonose_lip_slope` | dynamics | 0.589 | - | 0.10 | 0.550 | - |
| `au26int_slope` | dynamics | 0.585 | + | 0.03 | 0.563 | + |
| `geojaw_open_mean` | static | 0.579 | + | 0.05 | 0.505 | + |
| `geomouth_open_delta` | dynamics | 0.577 | + | 0.08 | 0.572 | + |
| `au25int_mean` | static | 0.577 | + | 0.02 | 0.546 | - |
| `au25int_slope` | dynamics | 0.571 | + | 0.06 | 0.540 | + |
| `geonose_lip_delta` | dynamics | 0.564 | - | 0.07 | 0.575 | - |
| `au26int_delta` | dynamics | 0.558 | + | 0.01 | 0.548 | + |
| `geojaw_open_median` | static | 0.555 | + | 0.05 | 0.503 | + |
| `au25int_delta` | dynamics | 0.552 | + | 0.07 | 0.526 | + |
| `geomouth_open_median` | static | 0.549 | + | 0.02 | 0.531 | + |
| `au25int_min` | static | 0.544 | + | 0.03 | 0.511 | - |
| `geomouth_width_p10` | static | 0.538 | + | 0.01 | 0.517 | - |
| `au26int_median` | static | 0.533 | + | 0.01 | 0.500 | - |
| `geomouth_open_min` | static | 0.532 | - | 0.03 | 0.512 | - |
| `geojaw_open_min` | static | 0.520 | - | 0.06 | 0.530 | - |
| `au25int_median` | static | 0.519 | + | 0.03 | 0.538 | - |
| `au26int_min` | static | 0.517 | + | 0.00 | 0.504 | - |
| `geonose_lip_p90` | static | 0.517 | - | 0.04 | 0.500 | - |
| `au25int_p10` | static | 0.516 | + | 0.02 | 0.520 | - |
| `geomouth_width_min` | static | 0.504 | - | 0.01 | 0.518 | - |
| `geojaw_open_p10` | static | 0.503 | - | 0.06 | 0.526 | + |
| `au26int_p10` | static | 0.503 | + | 0.00 | 0.510 | - |
| `geonose_lip_max` | static | 0.502 | + | 0.03 | 0.510 | - |
| `geomouth_open_p10` | static | 0.501 | - | 0.00 | 0.530 | - |

## brow  (81 feats, T1 best sep 0.631)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `au2int_iqr` | dynamics | 0.631 | + | 0.01 | 0.533 | - |
| `au2int_std` | dynamics | 0.626 | + | 0.02 | 0.546 | - |
| `geobrow_eye_l_slope` | dynamics | 0.624 | + | 0.02 | 0.506 | + |
| `au2int_p90` | static | 0.622 | + | 0.00 | 0.528 | - |
| `au1int_iqr` | dynamics | 0.621 | + | 0.01 | 0.520 | - |
| `au1int_std` | dynamics | 0.619 | + | 0.05 | 0.541 | - |
| `au2int_range` | dynamics | 0.617 | + | 0.02 | 0.540 | - |
| `au2int_max` | static | 0.614 | + | 0.02 | 0.536 | - |
| `geobrow_eye_l_std` | dynamics | 0.610 | + | 0.00 | 0.546 | + |
| `au1int_p90` | static | 0.610 | + | 0.02 | 0.526 | - |
| `au2int_vel` | dynamics | 0.608 | + | 0.04 | 0.539 | - |
| `geobrow_eye_l_delta` | dynamics | 0.605 | + | 0.03 | 0.501 | + |
| `au1int_range` | dynamics | 0.605 | + | 0.05 | 0.546 | - |
| `geobrow_eye_l_range` | dynamics | 0.601 | + | 0.00 | 0.542 | + |
| `geobrow_eye_r_iqr` | dynamics | 0.600 | + | 0.00 | 0.542 | + |
| `au1int_max` | static | 0.599 | + | 0.05 | 0.536 | - |
| `au1int_vel` | dynamics | 0.598 | + | 0.07 | 0.545 | - |
| `geobrow_eye_l_iqr` | dynamics | 0.596 | + | 0.00 | 0.539 | + |
| `geobrow_eye_r_std` | dynamics | 0.595 | + | 0.01 | 0.548 | + |
| `au4int_std` | dynamics | 0.594 | + | 0.05 | 0.537 | + |
| `geobrow_eye_r_p10` | static | 0.594 | - | 0.07 | 0.521 | - |
| `au4int_vel` | dynamics | 0.594 | + | 0.08 | 0.535 | + |
| `au2int_mean` | static | 0.592 | + | 0.01 | 0.525 | - |
| `geobrow_eye_l_vel` | dynamics | 0.592 | + | 0.04 | 0.534 | + |
| `au4int_iqr` | dynamics | 0.590 | + | 0.01 | 0.554 | + |
| `geobrow_eye_r_min` | static | 0.590 | - | 0.07 | 0.527 | - |
| `geoinner_brow_p10` | static | 0.589 | - | 0.00 | 0.506 | - |
| `geobrow_eye_r_range` | dynamics | 0.588 | + | 0.01 | 0.552 | + |
| `au4int_range` | dynamics | 0.586 | + | 0.06 | 0.529 | + |
| `au1int_mean` | static | 0.586 | + | 0.02 | 0.524 | - |
| `geobrow_eye_r_slope` | dynamics | 0.586 | + | 0.02 | 0.515 | + |
| `geobrow_eye_r_vel` | dynamics | 0.585 | + | 0.04 | 0.537 | + |
| `geobrow_eye_l_p10` | static | 0.585 | - | 0.09 | 0.533 | - |
| `geobrow_eye_l_min` | static | 0.581 | - | 0.09 | 0.528 | - |
| `geoinner_brow_min` | static | 0.579 | - | 0.01 | 0.511 | - |
| `au1int_slope` | dynamics | 0.573 | + | 0.00 | 0.537 | + |
| `geoinner_brow_iqr` | dynamics | 0.573 | + | 0.01 | 0.528 | + |
| `geobrow_eye_r_delta` | dynamics | 0.568 | + | 0.03 | 0.515 | + |
| `au2int_slope` | dynamics | 0.567 | + | 0.01 | 0.544 | + |
| `geoinner_brow_std` | dynamics | 0.567 | + | 0.02 | 0.572 | + |
| `geoinner_brow_range` | dynamics | 0.565 | + | 0.02 | 0.581 | + |
| `geobrow_eye_l_max` | static | 0.560 | + | 0.07 | 0.505 | + |
| `geoinner_brow_vel` | dynamics | 0.557 | + | 0.04 | 0.543 | + |
| `geobrow_eye_r_median` | static | 0.556 | - | 0.07 | 0.503 | - |
| `au1int_delta` | dynamics | 0.555 | + | 0.02 | 0.525 | + |
| `au4int_p90` | static | 0.552 | + | 0.03 | 0.535 | + |
| `au4int_max` | static | 0.552 | + | 0.05 | 0.524 | + |
| `au2int_median` | static | 0.551 | + | 0.03 | 0.503 | - |
| `geoinner_brow_mean` | static | 0.547 | - | 0.00 | 0.517 | - |
| `au1int_median` | static | 0.546 | + | 0.01 | 0.506 | - |
| `geoinner_brow_median` | static | 0.546 | - | 0.00 | 0.510 | - |
| `geobrow_eye_l_median` | static | 0.546 | - | 0.08 | 0.517 | - |
| `au2int_delta` | dynamics | 0.544 | + | 0.02 | 0.533 | + |
| `geobrow_eye_r_max` | static | 0.544 | + | 0.06 | 0.511 | - |
| `geobrow_eye_l_p90` | static | 0.539 | + | 0.06 | 0.505 | + |
| `au2_ntrans` | dynamics | 0.538 | + | 0.03 | 0.501 | - |
| `au4int_min` | static | 0.533 | - | 0.00 | 0.505 | - |
| `geobrow_eye_r_mean` | static | 0.532 | - | 0.07 | 0.507 | - |
| `au2_rate` | static | 0.529 | + | 0.01 | 0.506 | - |
| `au2_ever` | static | 0.528 | + | 0.01 | 0.501 | - |
| `au4int_mean` | static | 0.528 | + | 0.01 | 0.534 | + |
| `au4_rate` | static | 0.525 | - | 0.04 | 0.515 | + |
| `au4int_delta` | dynamics | 0.524 | + | 0.02 | 0.505 | - |
| `geoinner_brow_p90` | static | 0.524 | - | 0.00 | 0.535 | + |
| `au1_ever` | static | 0.523 | + | 0.00 | 0.501 | - |
| `au4int_slope` | dynamics | 0.523 | + | 0.03 | 0.501 | - |
| `au4int_p10` | static | 0.521 | - | 0.00 | 0.502 | + |
| `au1_ntrans` | dynamics | 0.521 | + | 0.02 | 0.506 | - |
| `au1_rate` | static | 0.520 | + | 0.00 | 0.501 | - |
| `geobrow_eye_r_p90` | static | 0.514 | + | 0.05 | 0.508 | - |
| `geobrow_eye_l_mean` | static | 0.513 | - | 0.08 | 0.521 | - |
| `geoinner_brow_delta` | dynamics | 0.512 | + | 0.03 | 0.506 | + |
| `au4int_median` | static | 0.512 | - | 0.00 | 0.525 | + |
| `au1int_p10` | static | 0.511 | + | 0.01 | 0.503 | - |
| `geoinner_brow_max` | static | 0.509 | + | 0.01 | 0.553 | + |
| `au4_ever` | static | 0.508 | + | 0.04 | 0.509 | + |
| `geoinner_brow_slope` | dynamics | 0.506 | + | 0.01 | 0.523 | + |
| `au1int_min` | static | 0.506 | + | 0.01 | 0.504 | - |
| `au4_ntrans` | dynamics | 0.506 | + | 0.02 | 0.520 | + |
| `au2int_p10` | static | 0.504 | + | 0.03 | 0.506 | - |
| `au2int_min` | static | 0.500 | + | 0.02 | 0.503 | - |

## eye  (51 feats, T1 best sep 0.624)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `au5int_iqr` | dynamics | 0.624 | + | 0.02 | 0.537 | - |
| `au5int_std` | dynamics | 0.618 | + | 0.02 | 0.551 | + |
| `au5int_range` | dynamics | 0.612 | + | 0.02 | 0.548 | + |
| `geoeye_open_l_std` | dynamics | 0.609 | + | 0.03 | 0.544 | + |
| `geoeye_open_r_iqr` | dynamics | 0.608 | + | 0.02 | 0.519 | + |
| `geoeye_open_l_iqr` | dynamics | 0.605 | + | 0.03 | 0.566 | + |
| `au5int_p90` | static | 0.603 | + | 0.01 | 0.548 | - |
| `geoeye_open_l_range` | dynamics | 0.602 | + | 0.03 | 0.544 | + |
| `geoeye_open_r_slope` | dynamics | 0.599 | - | 0.07 | 0.513 | - |
| `au5int_vel` | dynamics | 0.599 | + | 0.02 | 0.543 | + |
| `geoeye_open_l_slope` | dynamics | 0.598 | - | 0.06 | 0.517 | - |
| `au5int_max` | static | 0.598 | + | 0.02 | 0.557 | + |
| `geoeye_open_l_max` | static | 0.594 | + | 0.03 | 0.550 | + |
| `geoeye_open_l_p90` | static | 0.592 | + | 0.02 | 0.541 | + |
| `geoeye_open_l_vel` | dynamics | 0.584 | + | 0.02 | 0.549 | + |
| `geoeye_open_r_std` | dynamics | 0.581 | + | 0.01 | 0.525 | + |
| `geoeye_open_r_range` | dynamics | 0.578 | + | 0.00 | 0.536 | + |
| `geoeye_open_r_delta` | dynamics | 0.577 | - | 0.04 | 0.525 | - |
| `au5int_mean` | static | 0.577 | + | 0.01 | 0.540 | - |
| `au7_rate` | static | 0.570 | + | 0.03 | 0.500 | - |
| `au5int_slope` | dynamics | 0.569 | + | 0.02 | 0.550 | + |
| `geoeye_open_r_vel` | dynamics | 0.564 | + | 0.04 | 0.534 | + |
| `geoeye_open_l_delta` | dynamics | 0.563 | - | 0.03 | 0.519 | - |
| `geoeye_open_l_min` | static | 0.562 | - | 0.04 | 0.502 | + |
| `geoeye_open_r_max` | static | 0.557 | + | 0.03 | 0.529 | + |
| `geoeye_open_r_min` | static | 0.552 | - | 0.02 | 0.513 | - |
| `geoeye_asym_std` | dynamics | 0.552 | + | 0.02 | 0.552 | + |
| `geoeye_open_r_p10` | static | 0.550 | - | 0.01 | 0.502 | + |
| `geoeye_asym_range` | dynamics | 0.547 | + | 0.02 | 0.539 | + |
| `geoeye_open_l_p10` | static | 0.547 | - | 0.02 | 0.524 | - |
| `au7_ntrans` | dynamics | 0.547 | + | 0.03 | 0.503 | - |
| `geoeye_asym_min` | static | 0.545 | - | 0.02 | 0.539 | - |
| `au5int_delta` | dynamics | 0.543 | + | 0.01 | 0.533 | - |
| `geoeye_open_r_p90` | static | 0.538 | + | 0.04 | 0.515 | + |
| `au7_ever` | static | 0.535 | + | 0.01 | 0.504 | - |
| `geoeye_asym_p10` | static | 0.534 | - | 0.03 | 0.524 | - |
| `geoeye_asym_max` | static | 0.533 | + | 0.02 | 0.536 | + |
| `au5int_median` | static | 0.532 | + | 0.00 | 0.513 | - |
| `geoeye_asym_delta` | dynamics | 0.527 | - | 0.02 | 0.509 | + |
| `geoeye_asym_vel` | dynamics | 0.526 | + | 0.00 | 0.566 | + |
| `geoeye_open_r_median` | static | 0.526 | - | 0.05 | 0.505 | + |
| `geoeye_open_l_median` | static | 0.520 | - | 0.04 | 0.509 | + |
| `geoeye_asym_median` | static | 0.519 | - | 0.02 | 0.521 | + |
| `geoeye_asym_slope` | dynamics | 0.516 | - | 0.05 | 0.507 | - |
| `geoeye_asym_p90` | static | 0.512 | + | 0.03 | 0.541 | + |
| `geoeye_asym_mean` | static | 0.511 | - | 0.02 | 0.538 | + |
| `geoeye_open_r_mean` | static | 0.510 | - | 0.04 | 0.507 | + |
| `au5int_min` | static | 0.509 | - | 0.00 | 0.503 | - |
| `geoeye_asym_iqr` | dynamics | 0.508 | + | 0.02 | 0.539 | + |
| `au5int_p10` | static | 0.503 | + | 0.00 | 0.506 | - |
| `geoeye_open_l_mean` | static | 0.501 | - | 0.04 | 0.508 | + |

## nose  (15 feats, T1 best sep 0.637)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `au9int_iqr` | dynamics | 0.637 | + | 0.00 | 0.545 | + |
| `au9int_std` | dynamics | 0.631 | + | 0.02 | 0.564 | + |
| `au9int_range` | dynamics | 0.628 | + | 0.02 | 0.560 | + |
| `au9int_vel` | dynamics | 0.612 | + | 0.05 | 0.560 | + |
| `au9int_max` | static | 0.608 | + | 0.02 | 0.550 | - |
| `au9int_p90` | static | 0.595 | + | 0.01 | 0.546 | - |
| `au9int_slope` | dynamics | 0.583 | + | 0.06 | 0.533 | + |
| `au10_rate` | static | 0.578 | + | 0.01 | 0.507 | - |
| `au9int_mean` | static | 0.572 | + | 0.02 | 0.543 | - |
| `au10_ntrans` | dynamics | 0.564 | + | 0.02 | 0.526 | + |
| `au10_ever` | static | 0.562 | + | 0.01 | 0.521 | + |
| `au9int_delta` | dynamics | 0.556 | + | 0.03 | 0.520 | + |
| `au9int_median` | static | 0.534 | + | 0.01 | 0.533 | - |
| `au9int_min` | static | 0.523 | + | 0.02 | 0.515 | - |
| `au9int_p10` | static | 0.508 | + | 0.02 | 0.524 | - |

## head_pose  (36 feats, T1 best sep 0.656)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `roll_std` | dynamics | 0.656 | + | 0.00 | 0.566 | + |
| `roll_range` | dynamics | 0.651 | + | 0.01 | 0.555 | + |
| `roll_iqr` | dynamics | 0.642 | + | 0.00 | 0.593 | + |
| `roll_vel` | dynamics | 0.638 | + | 0.05 | 0.570 | + |
| `pitch_range` | dynamics | 0.608 | + | 0.05 | 0.523 | + |
| `pitch_std` | dynamics | 0.605 | + | 0.04 | 0.517 | + |
| `pitch_iqr` | dynamics | 0.604 | + | 0.02 | 0.524 | + |
| `pitch_vel` | dynamics | 0.599 | + | 0.09 | 0.552 | + |
| `yaw_std` | dynamics | 0.594 | + | 0.05 | 0.550 | + |
| `yaw_vel` | dynamics | 0.588 | + | 0.08 | 0.542 | + |
| `yaw_range` | dynamics | 0.587 | + | 0.06 | 0.544 | + |
| `yaw_iqr` | dynamics | 0.585 | + | 0.05 | 0.556 | + |
| `pitch_p10` | static | 0.583 | - | 0.05 | 0.530 | - |
| `pitch_min` | static | 0.580 | - | 0.05 | 0.532 | - |
| `roll_max` | static | 0.569 | + | 0.01 | 0.534 | + |
| `pitch_median` | static | 0.565 | - | 0.05 | 0.506 | - |
| `pitch_mean` | static | 0.563 | - | 0.05 | 0.502 | - |
| `yaw_p10` | static | 0.555 | - | 0.02 | 0.530 | - |
| `roll_min` | static | 0.554 | - | 0.01 | 0.504 | + |
| `roll_p90` | static | 0.551 | + | 0.00 | 0.534 | + |
| `roll_p10` | static | 0.548 | - | 0.00 | 0.513 | - |
| `yaw_min` | static | 0.544 | - | 0.03 | 0.541 | - |
| `roll_delta` | dynamics | 0.527 | + | 0.01 | 0.521 | + |
| `pitch_delta` | dynamics | 0.521 | - | 0.03 | 0.509 | + |
| `pitch_slope` | dynamics | 0.520 | - | 0.06 | 0.507 | + |
| `roll_slope` | dynamics | 0.517 | + | 0.03 | 0.511 | + |
| `yaw_p90` | static | 0.517 | + | 0.01 | 0.501 | + |
| `yaw_max` | static | 0.516 | + | 0.00 | 0.510 | + |
| `yaw_mean` | static | 0.515 | - | 0.00 | 0.518 | - |
| `yaw_median` | static | 0.514 | - | 0.01 | 0.506 | - |
| `yaw_delta` | dynamics | 0.513 | - | 0.05 | 0.507 | - |
| `pitch_p90` | static | 0.511 | - | 0.04 | 0.514 | - |
| `yaw_slope` | dynamics | 0.507 | - | 0.04 | 0.517 | + |
| `pitch_max` | static | 0.507 | + | 0.02 | 0.519 | - |
| `roll_median` | static | 0.505 | + | 0.01 | 0.516 | + |
| `roll_mean` | static | 0.504 | + | 0.00 | 0.521 | + |

## gaze  (24 feats, T1 best sep 0.612)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `gaze_pitch_slope` | dynamics | 0.612 | - | 0.00 | 0.505 | + |
| `gaze_pitch_std` | dynamics | 0.605 | + | 0.00 | 0.531 | - |
| `gaze_pitch_range` | dynamics | 0.603 | + | 0.00 | 0.539 | - |
| `gaze_pitch_iqr` | dynamics | 0.598 | + | 0.01 | 0.535 | + |
| `gaze_pitch_max` | static | 0.597 | + | 0.01 | 0.531 | - |
| `gaze_pitch_vel` | dynamics | 0.595 | + | 0.01 | 0.553 | - |
| `gaze_pitch_delta` | dynamics | 0.581 | - | 0.02 | 0.500 | + |
| `gaze_yaw_mean` | static | 0.576 | + | 0.03 | 0.534 | - |
| `gaze_pitch_p90` | static | 0.576 | + | 0.04 | 0.512 | - |
| `gaze_yaw_p90` | static | 0.575 | + | 0.04 | 0.503 | - |
| `gaze_yaw_median` | static | 0.571 | + | 0.07 | 0.527 | - |
| `gaze_yaw_p10` | static | 0.568 | + | 0.02 | 0.545 | - |
| `gaze_yaw_max` | static | 0.558 | + | 0.01 | 0.509 | - |
| `gaze_pitch_min` | static | 0.558 | - | 0.05 | 0.517 | - |
| `gaze_yaw_delta` | dynamics | 0.547 | - | 0.05 | 0.525 | + |
| `gaze_pitch_p10` | static | 0.540 | - | 0.04 | 0.528 | - |
| `gaze_yaw_min` | static | 0.538 | - | 0.00 | 0.531 | + |
| `gaze_pitch_median` | static | 0.534 | - | 0.05 | 0.518 | - |
| `gaze_yaw_slope` | dynamics | 0.531 | - | 0.01 | 0.533 | - |
| `gaze_yaw_iqr` | dynamics | 0.525 | + | 0.03 | 0.557 | + |
| `gaze_pitch_mean` | static | 0.524 | + | 0.05 | 0.513 | - |
| `gaze_yaw_std` | dynamics | 0.519 | + | 0.00 | 0.535 | - |
| `gaze_yaw_range` | dynamics | 0.513 | + | 0.00 | 0.530 | - |
| `gaze_yaw_vel` | dynamics | 0.500 | + | 0.00 | 0.526 | - |

## audio  (93 feats, T1 best sep 0.602)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `loudness_sma3_stddevNorm` | static | 0.602 | + | 0.01 | 0.548 | + |
| `MeanUnvoicedSegmentLength` | static | 0.594 | - | 0.33⚠ | 0.569 | - |
| `rms_db_p95` | static | 0.592 | + | 0.03 | 0.537 | + |
| `silent_frac` | static | 0.590 | - | 0.04 | 0.510 | + |
| `F1amplitudeLogRelF0_sma3nz_amean` | static | 0.584 | + | 0.05 | 0.516 | - |
| `spectralFluxUV_sma3nz_amean` | static | 0.582 | + | 0.05 | 0.542 | - |
| `F1amplitudeLogRelF0_sma3nz_stddevNorm` | static | 0.580 | - | 0.06 | 0.516 | + |
| `StddevVoicedSegmentLengthSec` | static | 0.580 | + | 0.09 | 0.507 | - |
| `F2amplitudeLogRelF0_sma3nz_amean` | static | 0.578 | + | 0.06 | 0.514 | - |
| `F3amplitudeLogRelF0_sma3nz_amean` | static | 0.578 | + | 0.05 | 0.517 | - |
| `MeanVoicedSegmentLengthSec` | static | 0.577 | + | 0.08 | 0.503 | - |
| `equivalentSoundLevel_dBp` | static | 0.577 | + | 0.05 | 0.527 | + |
| `spectralFlux_sma3_amean` | static | 0.574 | + | 0.04 | 0.530 | - |
| `F2amplitudeLogRelF0_sma3nz_stddevNorm` | static | 0.574 | - | 0.07 | 0.519 | + |
| `loudness_sma3_pctlrange0-2` | static | 0.574 | + | 0.01 | 0.522 | - |
| `loudness_sma3_stddevFallingSlope` | static | 0.573 | - | 0.00 | 0.549 | + |
| `F3amplitudeLogRelF0_sma3nz_stddevNorm` | static | 0.571 | - | 0.06 | 0.519 | + |
| `dyn_range_db` | static | 0.570 | + | 0.09 | 0.519 | + |
| `F3bandwidth_sma3nz_stddevNorm` | static | 0.568 | + | 0.11 | 0.505 | - |
| `loudnessPeaksPerSec` | static | 0.567 | - | 0.12 | 0.502 | - |
| `mfcc1_sma3_amean` | static | 0.566 | + | 0.04 | 0.518 | - |
| `shimmerLocaldB_sma3nz_stddevNorm` | static | 0.566 | + | 0.11 | 0.528 | + |
| `spectralFlux_sma3_stddevNorm` | static | 0.565 | + | 0.09 | 0.520 | + |
| `alphaRatioV_sma3nz_stddevNorm` | static | 0.564 | - | 0.01 | 0.532 | + |
| `loudness_sma3_meanFallingSlope` | static | 0.563 | - | 0.01 | 0.515 | + |
| `loudness_sma3_amean` | static | 0.562 | + | 0.00 | 0.522 | - |
| `rms_db_mean` | static | 0.562 | + | 0.04 | 0.517 | - |
| `loudness_sma3_percentile80.0` | timing | 0.555 | + | 0.00 | 0.503 | - |
| `mfcc3V_sma3nz_stddevNorm` | static | 0.555 | - | 0.01 | 0.522 | + |
| `HNRdBACF_sma3nz_amean` | static | 0.554 | + | 0.10 | 0.507 | - |
| `spectralFluxV_sma3nz_amean` | static | 0.554 | + | 0.01 | 0.509 | + |
| `VoicedSegmentsPerSec` | static | 0.552 | + | 0.03 | 0.519 | - |
| `F3bandwidth_sma3nz_amean` | static | 0.552 | + | 0.10 | 0.515 | + |
| `spectralFluxV_sma3nz_stddevNorm` | static | 0.552 | + | 0.10 | 0.523 | + |
| `logRelF0-H1-A3_sma3nz_stddevNorm` | static | 0.551 | + | 0.04 | 0.511 | + |
| `mfcc1V_sma3nz_stddevNorm` | static | 0.550 | - | 0.04 | 0.522 | + |
| `mfcc1V_sma3nz_amean` | static | 0.550 | + | 0.07 | 0.508 | - |
| `F3frequency_sma3nz_stddevNorm` | static | 0.548 | + | 0.14 | 0.504 | - |
| `hammarbergIndexV_sma3nz_amean` | static | 0.547 | + | 0.09 | 0.511 | - |
| `F0semitoneFrom27.5Hz_sma3nz_stddevNorm` | timing | 0.546 | + | 0.09 | 0.516 | + |
| `mfcc4_sma3_amean` | static | 0.546 | - | 0.03 | 0.508 | - |
| `loudness_sma3_stddevRisingSlope` | static | 0.546 | - | 0.04 | 0.537 | + |
| `F0semitoneFrom27.5Hz_sma3nz_percentile80.0` | timing | 0.543 | + | 0.11 | 0.515 | + |
| `F2frequency_sma3nz_amean` | static | 0.543 | + | 0.10 | 0.531 | + |
| `loudness_sma3_percentile50.0` | timing | 0.542 | + | 0.02 | 0.505 | - |
| `hammarbergIndexV_sma3nz_stddevNorm` | static | 0.542 | - | 0.04 | 0.519 | - |
| `F1bandwidth_sma3nz_amean` | static | 0.542 | + | 0.09 | 0.505 | - |
| `logRelF0-H1-H2_sma3nz_amean` | static | 0.541 | + | 0.02 | 0.522 | - |
| `slopeUV500-1500_sma3nz_amean` | static | 0.541 | - | 0.03 | 0.506 | + |
| `F1frequency_sma3nz_amean` | static | 0.540 | + | 0.11 | 0.540 | + |
| `jitterLocal_sma3nz_stddevNorm` | static | 0.540 | + | 0.15 | 0.500 | - |
| `mfcc4_sma3_stddevNorm` | static | 0.540 | + | 0.00 | 0.506 | - |
| `loudness_sma3_meanRisingSlope` | static | 0.539 | - | 0.01 | 0.536 | + |
| `F3frequency_sma3nz_amean` | static | 0.538 | + | 0.10 | 0.529 | + |
| `mfcc2_sma3_stddevNorm` | static | 0.537 | - | 0.02 | 0.514 | + |
| `F0semitoneFrom27.5Hz_sma3nz_percentile50.0` | timing | 0.536 | + | 0.10 | 0.504 | + |
| `F0semitoneFrom27.5Hz_sma3nz_amean` | timing | 0.535 | + | 0.10 | 0.516 | + |
| `alphaRatioV_sma3nz_amean` | static | 0.533 | - | 0.06 | 0.516 | + |
| `F2frequency_sma3nz_stddevNorm` | static | 0.533 | + | 0.13 | 0.506 | - |
| `F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2` | timing | 0.533 | + | 0.08 | 0.522 | + |
| `mfcc3_sma3_stddevNorm` | static | 0.533 | + | 0.05 | 0.513 | + |
| `mfcc2_sma3_amean` | static | 0.532 | - | 0.00 | 0.533 | - |
| `F0semitoneFrom27.5Hz_sma3nz_percentile20.0` | timing | 0.532 | + | 0.09 | 0.506 | + |
| `logRelF0-H1-H2_sma3nz_stddevNorm` | static | 0.531 | - | 0.01 | 0.505 | + |
| `slopeUV0-500_sma3nz_amean` | static | 0.531 | - | 0.02 | 0.565 | + |
| `logRelF0-H1-A3_sma3nz_amean` | static | 0.530 | + | 0.04 | 0.521 | - |
| `slopeV0-500_sma3nz_amean` | static | 0.530 | + | 0.03 | 0.502 | + |
| `F1bandwidth_sma3nz_stddevNorm` | static | 0.530 | + | 0.11 | 0.542 | + |
| `HNRdBACF_sma3nz_stddevNorm` | static | 0.528 | - | 0.02 | 0.530 | + |
| `F1frequency_sma3nz_stddevNorm` | static | 0.527 | + | 0.12 | 0.506 | - |
| `slopeV0-500_sma3nz_stddevNorm` | static | 0.526 | - | 0.06 | 0.506 | - |
| `mfcc1_sma3_stddevNorm` | static | 0.526 | - | 0.06 | 0.531 | - |
| `F2bandwidth_sma3nz_stddevNorm` | static | 0.522 | + | 0.11 | 0.535 | + |
| `loudness_sma3_percentile20.0` | timing | 0.522 | + | 0.02 | 0.502 | - |
| `mfcc3_sma3_amean` | static | 0.521 | + | 0.01 | 0.559 | + |
| `F2bandwidth_sma3nz_amean` | static | 0.521 | + | 0.10 | 0.514 | - |
| `shimmerLocaldB_sma3nz_amean` | static | 0.519 | + | 0.07 | 0.521 | + |
| `voiced_frac` | static | 0.517 | - | 0.12 | 0.523 | - |
| `slopeV500-1500_sma3nz_stddevNorm` | static | 0.517 | + | 0.01 | 0.503 | + |
| `StddevUnvoicedSegmentLength` | static | 0.515 | - | 0.28 | 0.512 | - |
| `alphaRatioUV_sma3nz_amean` | static | 0.513 | + | 0.03 | 0.518 | + |
| `hammarbergIndexUV_sma3nz_amean` | static | 0.513 | - | 0.03 | 0.523 | - |
| `mfcc2V_sma3nz_amean` | static | 0.512 | + | 0.02 | 0.511 | - |
| `mfcc3V_sma3nz_amean` | static | 0.512 | + | 0.02 | 0.513 | - |
| `F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope` | timing | 0.510 | - | 0.12 | 0.513 | + |
| `mfcc4V_sma3nz_stddevNorm` | static | 0.510 | + | 0.07 | 0.511 | - |
| `slopeV500-1500_sma3nz_amean` | static | 0.509 | + | 0.03 | 0.505 | + |
| `F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope` | timing | 0.508 | - | 0.09 | 0.503 | - |
| `mfcc2V_sma3nz_stddevNorm` | static | 0.506 | - | 0.05 | 0.502 | + |
| `F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope` | timing | 0.506 | - | 0.11 | 0.524 | - |
| `mfcc4V_sma3nz_amean` | static | 0.505 | + | 0.01 | 0.512 | - |
| `F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope` | timing | 0.503 | - | 0.10 | 0.510 | - |
| `jitterLocal_sma3nz_amean` | static | 0.502 | + | 0.07 | 0.520 | - |

## other_lipmouth  (48 feats, T1 best sep 0.596)

| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |
|---|---|---|---|---|---|---|
| `au15int_iqr` | dynamics | 0.596 | + | 0.01 | 0.538 | - |
| `au20int_range` | dynamics | 0.588 | + | 0.01 | 0.539 | + |
| `au20int_std` | dynamics | 0.584 | + | 0.02 | 0.530 | + |
| `au20int_iqr` | dynamics | 0.584 | + | 0.01 | 0.513 | + |
| `au20int_vel` | dynamics | 0.581 | + | 0.03 | 0.537 | + |
| `au20int_p90` | static | 0.581 | + | 0.01 | 0.521 | + |
| `au17int_vel` | dynamics | 0.580 | + | 0.02 | 0.526 | + |
| `au20int_max` | static | 0.578 | + | 0.01 | 0.524 | + |
| `au17int_std` | dynamics | 0.577 | + | 0.02 | 0.519 | + |
| `au17int_range` | dynamics | 0.573 | + | 0.01 | 0.521 | + |
| `au20int_mean` | static | 0.572 | + | 0.01 | 0.516 | - |
| `au17int_max` | static | 0.570 | + | 0.01 | 0.510 | + |
| `au15int_std` | dynamics | 0.569 | + | 0.00 | 0.526 | - |
| `au15int_range` | dynamics | 0.563 | + | 0.00 | 0.521 | - |
| `au17int_p90` | static | 0.563 | + | 0.02 | 0.506 | + |
| `au17int_iqr` | dynamics | 0.561 | + | 0.02 | 0.516 | + |
| `au20int_median` | static | 0.559 | + | 0.02 | 0.512 | - |
| `au17int_mean` | static | 0.558 | + | 0.02 | 0.505 | + |
| `au20int_slope` | dynamics | 0.558 | + | 0.01 | 0.530 | + |
| `au15int_vel` | dynamics | 0.555 | + | 0.01 | 0.519 | - |
| `au17_ntrans` | dynamics | 0.544 | + | 0.04 | 0.505 | - |
| `au17int_median` | static | 0.543 | + | 0.01 | 0.509 | + |
| `au15int_p90` | static | 0.541 | + | 0.01 | 0.525 | - |
| `au20int_p10` | static | 0.541 | + | 0.02 | 0.507 | - |
| `au17_ever` | static | 0.538 | + | 0.08 | 0.505 | + |
| `au15int_max` | static | 0.536 | + | 0.01 | 0.522 | - |
| `au24_ever` | static | 0.534 | + | 0.04 | 0.501 | + |
| `au15int_mean` | static | 0.533 | - | 0.01 | 0.510 | - |
| `au24_ntrans` | dynamics | 0.532 | + | 0.03 | 0.518 | - |
| `au20int_delta` | dynamics | 0.530 | + | 0.02 | 0.508 | + |
| `au15int_delta` | dynamics | 0.526 | + | 0.02 | 0.506 | - |
| `au15int_slope` | dynamics | 0.525 | - | 0.00 | 0.524 | - |
| `au20int_min` | static | 0.520 | - | 0.01 | 0.511 | - |
| `au15_ever` | static | 0.517 | + | 0.00 | 0.504 | + |
| `au15_rate` | static | 0.516 | + | 0.02 | 0.513 | + |
| `au15_ntrans` | dynamics | 0.516 | + | 0.01 | 0.503 | - |
| `au17int_slope` | dynamics | 0.516 | + | 0.01 | 0.514 | - |
| `au15int_min` | static | 0.513 | - | 0.04 | 0.501 | - |
| `au23_rate` | static | 0.511 | + | 0.01 | 0.510 | + |
| `au15int_median` | static | 0.509 | - | 0.02 | 0.502 | - |
| `au24_rate` | static | 0.508 | + | 0.03 | 0.528 | + |
| `au17int_p10` | static | 0.508 | - | 0.01 | 0.513 | - |
| `au17int_min` | static | 0.507 | - | 0.02 | 0.518 | - |
| `au23_ever` | static | 0.506 | + | 0.01 | 0.501 | - |
| `au17_rate` | static | 0.506 | + | 0.06 | 0.519 | - |
| `au23_ntrans` | dynamics | 0.505 | - | 0.01 | 0.501 | - |
| `au15int_p10` | static | 0.505 | - | 0.03 | 0.508 | - |
| `au17int_delta` | dynamics | 0.504 | - | 0.01 | 0.501 | - |
