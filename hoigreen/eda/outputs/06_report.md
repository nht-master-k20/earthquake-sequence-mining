# EDA Report

## Scope

- Input: `data/dongdat.csv`
- Rows after basic cleaning: `3,119,538`
- Rows used for EDA: `3,119,538`
- Time range: `2000-01-01 00:02:46.200000+00:00` -> `2025-12-30 23:56:43.942000+00:00`
- Distinct analysis regions (`2.5` degree grid): `3,611`

## 1. Distribution Analysis

- Core variables are strongly skewed: many small magnitudes and shallow events, with a long tail of stronger and deeper events.
- `mmi`, `cdi`, `felt` are highly incomplete and should stay descriptive rather than become core modeling features.

### Highest missing ratios

| feature | missing_ratio |
| --- | --- |
| dmin | 0.3887 |
| nst | 0.2979 |
| gap | 0.2621 |
| rms | 0.0407 |
| sig | 0.0 |

## 2. Relationship Analysis

- The strongest numeric relationships are below.
- `mag` and `sig` are expected to move together very strongly, so using both in downstream modeling should be done consciously.
- Observation quality variables (`gap`, `nst`, `dmin`, `rms`) carry information about data reliability and station geometry, not just earthquake physics.

### Strongest correlations

| feature_a | feature_b | correlation |
| --- | --- | --- |
| mag | sig | 0.953 |
| mag | rms | 0.574 |
| sig | rms | 0.574 |
| sig | nst | 0.398 |
| sig | dmin | 0.369 |

## 3. Temporal Analysis

- Busiest year by event count: `2020` with `189,020` events
- Year with strongest maximum magnitude: `2004` with `mag_max = 9.10`
- Peak month-of-year by total count: `7` with `314,997` events

## 4. Spatial Analysis

- Activity is highly concentrated in a limited set of spatial cells.
- Top regions below are based on the same grid logic that will be reused in Pattern Discovering.

### Top regions

| region_code | event_count | mag_mean | mag_max | event_share |
| --- | --- | --- | --- | --- |
| G051_022 | 319782 | 0.878 | 5.09 | 0.1025 |
| G049_025 | 280718 | 1.116 | 5.71 | 0.09 |
| G051_024 | 203553 | 0.974 | 6.5 | 0.0653 |
| G050_024 | 163839 | 1.17 | 7.1 | 0.0525 |
| G043_009 | 155056 | 1.886 | 6.7 | 0.0497 |
| G050_023 | 112049 | 1.317 | 6.5 | 0.0359 |
| G060_011 | 109354 | 1.504 | 6.2 | 0.0351 |
| G061_011 | 84160 | 1.285 | 5.8 | 0.027 |
| G050_025 | 72555 | 0.881 | 5.51 | 0.0233 |
| G061_012 | 70740 | 1.19 | 6.6 | 0.0227 |

## Recommended Follow-up for Pattern Discovering

- Keep the core stable features: `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`, `tsunami`
- Apply log transform to skewed variables before scaling
- Use the same region grid to connect EDA and cluster interpretation
- Exclude `mmi`, `cdi`, `felt` from core clustering because missingness is too high
