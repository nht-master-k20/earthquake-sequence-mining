# Visualization and Clustering Report

## Scope

- Input file: `data/dongdat.csv`
- Filtered event type: `earthquake`
- Rows after filtering: `1,324,122`
- Distinct regions with grid size `2.5` degree: `3,208`

## Feature Strategy

- Event clustering uses 8 core features: `mag, depth_log1p, sig_log1p, gap, rms_log1p, nst_log1p, dmin_log1p, tsunami`
- Added analytical fields: `year, month, hour, month_sin, month_cos, hour_sin, hour_cos, depth_log1p, sig_log1p, rms_log1p, nst_log1p, dmin_log1p, depth_band, mag_band, region_id, region_code, region_lat_center, region_lon_center`
- Excluded from clustering because missing ratio is too high: `mmi, cdi, felt`
- Scaling strategy: `log1p` for skewed numeric features, median imputation for `gap, rms, nst, dmin`, then `RobustScaler(10, 90)`
- Region strategy: split latitude/longitude into fixed grid cells and assign `region_id` plus `region_code`

### Missingness of retained numeric fields

| column | missing_ratio |
| --- | --- |
| dmin | 0.3966 |
| nst | 0.2946 |
| gap | 0.2411 |
| rms | 0.0213 |
| depth | 0.0 |
| mag | 0.0 |
| sig | 0.0 |

## Clustering Decision

- Best event-level `k`: `2`
- Best region-level `k`: `2`
- Sparse regions below `25` events keep `region_cluster = -1`

### Event cluster summary

| event_cluster | event_count | mag_mean | depth_mean | sig_mean | top_mag_type | top_depth_band |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1227143 | 1.786 | 24.305 | 78.565 | ml | shallow |
| 1 | 96979 | 4.388 | 92.407 | 302.396 | mb | shallow |

### Region cluster summary

| region_cluster | region_count | total_events | mag_mean | depth_mean | major_quake_ratio | tsunami_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1090 | 1296740 | 4.091 | 29.954 | 0.131 | 0.002 |
| 1 | 42 | 17289 | 4.548 | 410.172 | 0.144 | 0.0031 |

## Top Regions

| region_id | region_code | event_count | mag_mean | mag_max | region_cluster |
| --- | --- | --- | --- | --- | --- |
| 2145 | G051_022 | 130096 | 0.747 | 4.79 | 0 |
| 1965 | G049_025 | 118303 | 0.996 | 5.71 | 0 |
| 2147 | G051_024 | 110377 | 0.889 | 6.5 | 0 |
| 1614 | G043_009 | 77861 | 2.025 | 6.7 | 0 |
| 2053 | G050_024 | 60353 | 1.037 | 5.8 | 0 |
| 2814 | G060_011 | 43779 | 1.51 | 6.2 | 0 |
| 2870 | G061_011 | 34745 | 1.25 | 5.8 | 0 |
| 2052 | G050_023 | 28400 | 1.242 | 6.5 | 0 |
| 2871 | G061_012 | 27226 | 1.026 | 6.6 | 0 |
| 1635 | G043_045 | 26821 | 2.494 | 6.4 | 0 |

## Files Generated

- `00_feature_overview.csv`
- `01_event_cluster_assignments.csv`
- `02_event_cluster_k_eval.csv`
- `03_event_cluster_centroids.csv`
- `04_event_cluster_profile.csv`
- `05_region_lookup.csv`
- `06_region_summary.csv`
- `07_region_cluster_k_eval.csv`
- `08_region_cluster_profile.csv`
- `09_pipeline_metadata.json`
- `10_report.md`
- `11_missingness.png`
- `12_core_feature_correlation.png`
- `13_event_k_selection.png`
- `14_event_clusters_pca.png`
- `15_event_clusters_map.png`
- `16_region_k_selection.png`
- `17_region_clusters_map.png`
- `18_top_regions.png`
- `19_region_cluster_profile_heatmap.png`

## Notes

- `mmi`, `cdi`, `felt` are still preserved in the raw dataframe for descriptive analysis, but not used for clustering because their missing ratio is above 96%.
- The clustering task is separated from preprocessing, so this pipeline reads `dongdat.csv` directly and creates only analysis-ready outputs.
