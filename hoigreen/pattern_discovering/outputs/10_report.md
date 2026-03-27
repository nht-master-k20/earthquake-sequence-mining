# Pattern Discovering Report

## Scope

- Input file: `data/dongdat_full_with_region_code.csv`
- Filtered event type: `earthquake`
- Rows after filtering: `3,119,538`
- Distinct regions with grid size `2.5` degree: `3,611`

## Feature Strategy

- Event clustering uses 8 core features: `mag, depth_log1p, sig_log1p, gap, rms_log1p, nst_log1p, dmin_log1p, tsunami`
- Added analytical fields: `year, month, hour, month_sin, month_cos, hour_sin, hour_cos, depth_log1p, sig_log1p, rms_log1p, nst_log1p, dmin_log1p, depth_band, mag_band, region_id, region_code, region_lat_center, region_lon_center`
- Excluded from clustering because missing ratio is too high: `mmi, cdi, felt`
- Scaling strategy: `log1p` for skewed numeric features, median imputation for `gap, rms, nst, dmin`, then `RobustScaler(10, 90)`
- Region strategy: split latitude/longitude into fixed grid cells and assign `region_id` plus `region_code`

### Missingness of retained numeric fields

| column | missing_ratio |
| --- | --- |
| dmin | 0.3887 |
| nst | 0.2979 |
| gap | 0.2621 |
| rms | 0.0407 |
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
| 0 | 2957228 | 1.649 | 20.928 | 61.517 | ml | shallow |
| 1 | 162310 | 4.382 | 89.865 | 301.889 | mb | shallow |

### Region cluster summary

| region_cluster | region_count | total_events | mag_mean | depth_mean | major_quake_ratio | tsunami_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1313 | 3083567 | 3.828 | 27.424 | 0.1046 | 0.002 |
| 1 | 48 | 25108 | 4.351 | 398.831 | 0.1198 | 0.0028 |

## Top Regions

| region_id | region_code | event_count | mag_mean | mag_max | region_cluster |
| --- | --- | --- | --- | --- | --- |
| 2387 | G051_022 | 319782 | 0.878 | 5.09 | 0 |
| 2202 | G049_025 | 280718 | 1.116 | 5.71 | 0 |
| 2389 | G051_024 | 203553 | 0.974 | 6.5 | 0 |
| 2293 | G050_024 | 163839 | 1.17 | 7.1 | 0 |
| 1811 | G043_009 | 155056 | 1.886 | 6.7 | 0 |
| 2292 | G050_023 | 112049 | 1.317 | 6.5 | 0 |
| 3137 | G060_011 | 109354 | 1.504 | 6.2 | 0 |
| 3206 | G061_011 | 84160 | 1.285 | 5.8 | 0 |
| 2294 | G050_025 | 72555 | 0.881 | 5.51 | 0 |
| 3207 | G061_012 | 70740 | 1.19 | 6.6 | 0 |

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
