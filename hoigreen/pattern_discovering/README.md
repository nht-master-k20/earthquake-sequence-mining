# Giai đoạn Khai phá Pattern

Giai đoạn này tập trung vào việc tìm các pattern ẩn trong `dongdat.csv` sau khi đã hiểu dữ liệu qua bước EDA.

## Mục tiêu

1. Chọn feature để tìm pattern
2. Chuẩn hóa dữ liệu
3. Áp dụng clustering / pattern mining
4. Chọn số cluster tối ưu
5. Phân tích cluster
6. Visualization kết quả

## Cách chạy

```bash
.venv/bin/python hoigreen/pattern_discovering/run_pattern_discovering.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/pattern_discovering/outputs
```

## Output chính

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
- ảnh `.png` về event clusters, region clusters, top regions, heatmap profile

## Ghi chú phương pháp

- Feature core được chọn là các biến có độ phủ dữ liệu tốt và có ý nghĩa vật lý / chất lượng dữ liệu.
- Dùng `log1p + median imputation + RobustScaler`.
- Chia không gian thành grid để cluster và giải thích kết quả theo region.
- Folder này là phase-level entrypoint; nó delegate sang pipeline clustering đã được verify trước đó.
