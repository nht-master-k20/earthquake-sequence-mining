# Raw Visualization and Clustering

Pipeline nay doc truc tiep `data/dongdat.csv` va chi tap trung vao phan:

- trich xuat dac trung de clustering
- chon so luong dac trung su dung
- bo sung field phuc vu phan tich
- chuan hoa du lieu
- truc quan hoa
- chia vung dia ly va gan `region_id`

## Cach chay

```bash
.venv/bin/python hoigreen/clustering_pattern_mining/run_raw_visualization_clustering.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/clustering_pattern_mining/raw_outputs
```

## Quyet dinh chinh

- Chi giu `type = earthquake` de tranh tron voi `quarry blast`, `explosion`, `ice quake`.
- Dung 8 feature event-level cho clustering:
  - `mag`
  - `depth_log1p`
  - `sig_log1p`
  - `gap`
  - `rms_log1p`
  - `nst_log1p`
  - `dmin_log1p`
  - `tsunami`
- Khong dua `mmi`, `cdi`, `felt` vao clustering vi missing ratio rat cao.
- Chuan hoa bang `RobustScaler(10, 90)` sau khi median-impute cho `gap`, `rms`, `nst`, `dmin`.
- Chia vung theo grid lat/lon, mac dinh `2.5` do, sinh `region_id`, `region_code`, tam vung.
- Chay hai tang clustering:
  - event-level clustering cho tung su kien
  - region-level clustering tren cac dac trung tong hop cua tung vung

## Field duoc them

- Thoi gian: `year`, `month`, `hour`, `month_sin`, `month_cos`, `hour_sin`, `hour_cos`
- Bien doi log: `depth_log1p`, `sig_log1p`, `rms_log1p`, `nst_log1p`, `dmin_log1p`
- Nhan phan tich: `depth_band`, `mag_band`
- Co missing hay khong: `has_mmi`, `has_cdi`, `has_felt`, `gap_missing`, `rms_missing`, `nst_missing`, `dmin_missing`
- Theo vung: `region_id`, `region_code`, `region_lat_center`, `region_lon_center`

## Output chinh

- `01_event_cluster_assignments.csv`
- `04_event_cluster_profile.csv`
- `05_region_lookup.csv`
- `06_region_summary.csv`
- `08_region_cluster_profile.csv`
- `10_report.md`
- Bo anh `.png` de dua vao slide / bao cao
