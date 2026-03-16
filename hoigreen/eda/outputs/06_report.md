# EDA Report

## Scope

- Input: `data/dongdat.csv`
- Rows after basic cleaning: `1,352,013`
- Rows used for EDA (`earthquake` only): `1,324,122`
- Time range: `2000-01-01 00:02:46.200000+00:00` -> `2024-12-30 23:56:29.977000+00:00`
- Distinct analysis regions (`2.5` degree grid): `3,208`

## 1. Distribution Analysis

- Raw data contains non-earthquake event types, so the main EDA focuses on `earthquake` only.
- Core variables are strongly skewed: many small magnitudes and shallow events, with a long tail of stronger and deeper events.
- `mmi`, `cdi`, `felt` are highly incomplete and should stay descriptive rather than become core modeling features.

### Highest missing ratios

| feature | missing_ratio |
| --- | --- |
| dmin | 0.3966 |
| nst | 0.2946 |
| gap | 0.2411 |
| rms | 0.0213 |
| sig | 0.0 |

## 2. Relationship Analysis

- The strongest numeric relationships are below.
- `mag` and `sig` are expected to move together very strongly, so using both in downstream modeling should be done consciously.
- Observation quality variables (`gap`, `nst`, `dmin`, `rms`) carry information about data reliability and station geometry, not just earthquake physics.

### Strongest correlations

| feature_a | feature_b | correlation |
| --- | --- | --- |
| mag | sig | 0.961 |
| mag | rms | 0.686 |
| sig | rms | 0.675 |
| sig | nst | 0.461 |
| mag | nst | 0.399 |

## 3. Temporal Analysis

- Busiest year by event count: `2020` with `189,020` events
- Year with strongest maximum magnitude: `2004` with `mag_max = 9.10`
- Peak month-of-year by total count: `7` with `133,503` events

## 4. Spatial Analysis

- Activity is highly concentrated in a limited set of spatial cells.
- Top regions below are based on the same grid logic that will be reused in Pattern Discovering.

### Top regions

| region_code | event_count | mag_mean | mag_max | event_share |
| --- | --- | --- | --- | --- |
| G051_022 | 130096 | 0.747 | 4.79 | 0.0983 |
| G049_025 | 118303 | 0.996 | 5.71 | 0.0893 |
| G051_024 | 110377 | 0.889 | 6.5 | 0.0834 |
| G043_009 | 77861 | 2.025 | 6.7 | 0.0588 |
| G050_024 | 60353 | 1.037 | 5.8 | 0.0456 |
| G060_011 | 43779 | 1.51 | 6.2 | 0.0331 |
| G061_011 | 34745 | 1.25 | 5.8 | 0.0262 |
| G050_023 | 28400 | 1.242 | 6.5 | 0.0214 |
| G061_012 | 27226 | 1.026 | 6.6 | 0.0206 |
| G043_045 | 26821 | 2.494 | 6.4 | 0.0203 |

## Recommended Follow-up for Pattern Discovering

- Keep the core stable features: `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`, `tsunami`
- Apply log transform to skewed variables before scaling
- Use the same region grid to connect EDA and cluster interpretation
- Exclude `mmi`, `cdi`, `felt` from core clustering because missingness is too high
