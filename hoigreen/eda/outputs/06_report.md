# EDA Report

## Scope

- Input: `data/dongdat.csv`
- Rows after basic cleaning: `3,063,852`
- Rows used for EDA (`earthquake` only): `2,992,511`
- Time range: `2000-01-01 00:02:46.200000+00:00` -> `2024-12-30 23:56:29.977000+00:00`
- Distinct analysis regions (`2.5` degree grid): `3,565`

## 1. Distribution Analysis

- Raw data contains non-earthquake event types, so the main EDA focuses on `earthquake` only.
- Core variables are strongly skewed: many small magnitudes and shallow events, with a long tail of stronger and deeper events.
- `mmi`, `cdi`, `felt` are highly incomplete and should stay descriptive rather than become core modeling features.

### Highest missing ratios

| feature | missing_ratio |
| --- | --- |
| dmin | 0.3984 |
| nst | 0.3037 |
| gap | 0.2664 |
| rms | 0.0425 |
| sig | 0.0 |

## 2. Relationship Analysis

- The strongest numeric relationships are below.
- `mag` and `sig` are expected to move together very strongly, so using both in downstream modeling should be done consciously.
- Observation quality variables (`gap`, `nst`, `dmin`, `rms`) carry information about data reliability and station geometry, not just earthquake physics.

### Strongest correlations

| feature_a | feature_b | correlation |
| --- | --- | --- |
| mag | sig | 0.953 |
| sig | rms | 0.572 |
| mag | rms | 0.57 |
| sig | nst | 0.389 |
| sig | dmin | 0.36 |

## 3. Temporal Analysis

- Busiest year by event count: `2020` with `189,020` events
- Year with strongest maximum magnitude: `2004` with `mag_max = 9.10`
- Peak month-of-year by total count: `7` with `299,541` events

## 4. Spatial Analysis

- Activity is highly concentrated in a limited set of spatial cells.
- Top regions below are based on the same grid logic that will be reused in Pattern Discovering.

### Top regions

| region_code | event_count | mag_mean | mag_max | event_share |
| --- | --- | --- | --- | --- |
| G051_022 | 302540 | 0.88 | 5.09 | 0.1011 |
| G049_025 | 270102 | 1.121 | 5.71 | 0.0903 |
| G051_024 | 201480 | 0.972 | 6.5 | 0.0673 |
| G050_024 | 159868 | 1.169 | 7.1 | 0.0534 |
| G043_009 | 153099 | 1.885 | 6.7 | 0.0512 |
| G050_023 | 109177 | 1.317 | 6.5 | 0.0365 |
| G060_011 | 104829 | 1.506 | 6.2 | 0.035 |
| G061_011 | 82373 | 1.276 | 5.8 | 0.0275 |
| G050_025 | 71209 | 0.879 | 5.51 | 0.0238 |
| G061_012 | 69652 | 1.182 | 6.6 | 0.0233 |

## Recommended Follow-up for Pattern Discovering

- Keep the core stable features: `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`, `tsunami`
- Apply log transform to skewed variables before scaling
- Use the same region grid to connect EDA and cluster interpretation
- Exclude `mmi`, `cdi`, `felt` from core clustering because missingness is too high
