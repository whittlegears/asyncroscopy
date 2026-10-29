# Tune vs acquire: 7a27d4e, T = 120 s of instrument time

Twin: 2900 particles on 1.2 um, map tiles 256 px at 100 nm, dwell 4 us; 0.5 s overhead per frame, 1 s per move. Information: unique ground-truth particles found (twin only). 2 repeat(s) per point.

## small_error: C1 30 nm, |A1| 10 nm, drift 0 nm/s

| budget | tune images | tune s | tiles | found | found in focus | residual after tune nm | median focus error nm | in-focus frames | rule: tune? (found / in focus) | paid off? |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.0 | 67 | 1352.5 | 0.0 | 31.6 | 31.6 | 0.00 | - | - |
| focus_only | 10 | 5.2 | 64 | 1299.0 | 0.0 | 24.8 | 24.8 | 0.00 | no / no | no / no |
| xs | 26 | 13.4 | 59 | 1184.5 | 0.0 | 24.4 | 24.4 | 0.00 | no / no | no / no |
| s | 38 | 19.6 | 56 | 1133.0 | 1133.0 | 12.4 | 12.4 | 1.00 | no / yes | no / yes |
| m | 50 | 25.8 | 52 | 1045.0 | 1045.0 | 6.8 | 6.8 | 1.00 | no / yes | no / yes |
| l | 95 | 49.1 | 39 | 780.5 | 780.5 | 11.1 | 11.1 | 1.00 | no / yes | no / yes |

Optimum for information: **none** (1352.5 features, 0.0 s tuning).
Optimum for information_in_focus: **s** (1133.0 features, 19.6 s tuning).

## large_error: C1 250 nm, |A1| 50 nm, drift 0 nm/s

| budget | tune images | tune s | tiles | found | found in focus | residual after tune nm | median focus error nm | in-focus frames | rule: tune? (found / in focus) | paid off? |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.0 | 67 | 264.5 | 0.0 | 254.9 | 254.9 | 0.00 | - | - |
| focus_only | 10 | 5.2 | 64 | 1259.0 | 0.0 | 60.7 | 60.7 | 0.00 | yes / no | yes / no |
| xs | 26 | 13.4 | 59 | 1181.0 | 0.0 | 38.8 | 38.8 | 0.00 | yes / no | yes / no |
| s | 38 | 19.6 | 56 | 1132.5 | 1132.5 | 4.9 | 4.9 | 1.00 | yes / yes | yes / yes |
| m | 50 | 25.8 | 52 | 1045.0 | 1045.0 | 8.6 | 8.6 | 1.00 | yes / yes | yes / yes |
| l | 95 | 49.1 | 39 | 781.0 | 781.0 | 6.0 | 6.0 | 1.00 | yes / yes | yes / yes |

Optimum for information: **focus_only** (1259.0 features, 5.2 s tuning).
Optimum for information_in_focus: **s** (1132.5 features, 19.6 s tuning).

## large_error_slow_drift: C1 250 nm, |A1| 50 nm, drift 0.3 nm/s

| budget | tune images | tune s | tiles | found | found in focus | residual after tune nm | median focus error nm | in-focus frames | rule: tune? (found / in focus) | paid off? |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.0 | 67 | 194.0 | 0.0 | 255.2 | 273.1 | 0.00 | - | - |
| focus_only | 10 | 5.2 | 64 | 1203.5 | 0.0 | 60.5 | 71.8 | 0.00 | yes / no | yes / no |
| xs | 26 | 13.4 | 59 | 1151.0 | 0.0 | 41.4 | 57.4 | 0.00 | yes / no | yes / no |
| s | 38 | 19.6 | 56 | 1132.5 | 603.0 | 4.8 | 18.6 | 0.54 | yes / yes | yes / yes |
| m | 50 | 25.8 | 52 | 1043.5 | 436.0 | 11.7 | 21.8 | 0.42 | yes / yes | yes / yes |
| l | 95 | 49.1 | 39 | 780.0 | 351.5 | 11.2 | 20.8 | 0.46 | yes / yes | yes / yes |

Optimum for information: **focus_only** (1203.5 features, 5.2 s tuning).
Optimum for information_in_focus: **s** (603.0 features, 19.6 s tuning).

## large_error_fast_drift: C1 250 nm, |A1| 50 nm, drift 2 nm/s

| budget | tune images | tune s | tiles | found | found in focus | residual after tune nm | median focus error nm | in-focus frames | rule: tune? (found / in focus) | paid off? |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.0 | 67 | 49.0 | 0.0 | 257.1 | 376.6 | 0.00 | - | - |
| focus_only | 10 | 5.2 | 64 | 713.0 | 0.0 | 66.0 | 166.4 | 0.00 | yes / no | yes / no |
| xs | 26 | 13.4 | 59 | 665.0 | 0.0 | 59.0 | 165.6 | 0.00 | yes / no | yes / no |
| s | 38 | 19.6 | 56 | 765.5 | 0.0 | 29.8 | 131.0 | 0.00 | yes / no | yes / no |
| m | 50 | 25.8 | 52 | 729.0 | 0.0 | 35.7 | 129.4 | 0.00 | yes / no | yes / no |
| l | 95 | 49.1 | 39 | 511.0 | 0.0 | 72.7 | 143.5 | 0.00 | yes / no | yes / no |

Optimum for information: **s** (765.5 features, 19.6 s tuning).
Optimum for information_in_focus: **none** (0.0 features, 0.0 s tuning).

