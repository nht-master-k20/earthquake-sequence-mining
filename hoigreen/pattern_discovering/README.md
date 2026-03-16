# Pattern Discovering Phase

Phase nay tap trung vao viec tim pattern an trong `dongdat.csv` sau khi da hieu du lieu qua EDA.

## Muc tieu

1. Chon feature de tim pattern
2. Chuan hoa du lieu
3. Ap dung clustering / pattern mining
4. Chon so cluster toi uu
5. Phan tich cluster
6. Visualization ket qua

## Cach chay

```bash
.venv/bin/python hoigreen/pattern_discovering/run_pattern_discovering.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/pattern_discovering/outputs
```

## Output chinh

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
- anh `.png` ve event clusters, region clusters, top regions, heatmap profile

## Ghi chu phuong phap

- Feature core duoc chon la cac bien co do phu du lieu tot va co y nghia vat ly / chat luong du lieu.
- Dung `log1p + median imputation + RobustScaler`.
- Chia khong gian thanh grid de cluster va giai thich ket qua theo region.
- Folder nay la phase-level entrypoint; no delegate sang pipeline clustering da duoc verify truoc do.
