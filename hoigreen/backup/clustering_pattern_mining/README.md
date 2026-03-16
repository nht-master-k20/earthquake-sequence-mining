# Clustering & Pattern Mining Phase

Giai đoạn này xử lý các mục tiêu:

1. Phân nhóm động đất theo đặc điểm vật lý và vị trí địa lý
2. Phát hiện vùng có hoạt động động đất cao (hotspots)
3. Tìm pattern chuỗi động đất theo thời gian
4. Phân loại động đất thành mainshock / foreshock / aftershock
5. Phát hiện các trận/cụm động đất bất thường (outliers)
6. Khai phá luật kết hợp giữa đặc điểm động đất và vị trí

## Run

```bash
python3 hoigreen/clustering_pattern_mining/run_clustering_pattern_mining.py \
  --input-csv earthquake_cleaned.csv \
  --output-dir hoigreen/clustering_pattern_mining/outputs
```

Mặc định pipeline chạy trên toàn bộ dữ liệu (`--max-rows <= 0`, `--association-max-rows <= 0`).

## Tùy chọn hiệu năng

```bash
python3 hoigreen/clustering_pattern_mining/run_clustering_pattern_mining.py \
  --input-csv earthquake_cleaned.csv \
  --output-dir hoigreen/clustering_pattern_mining/outputs \
  --max-rows 300000 \
  --association-max-rows 150000 \
  --plot-sample-size 40000
```

## Input yêu cầu

CSV đã preprocess (ví dụ output từ `preprocess_usgs_quakes.py`) với các cột:

- `id`
- `time`
- `latitude`
- `longitude`
- `depth`
- `mag`
- `gap`
- `nst`
- `rms`

## Output

- `01_clustered_events.csv`
- `01_spatial_physical_clusters.png`
- `spatial_physical_cluster_meta.json`
- `02_hotspots.csv`
- `02_hotspots_map.html`
- `03_temporal_patterns.csv`
- `03_temporal_transition_heatmap.png`
- `04_shock_classification.csv`
- `05_outliers.csv`
- `05_outliers_map.png`
- `06_association_rules.csv`
- `06_frequent_itemsets.csv`
- `report.md`

## Ghi chú phương pháp

- `physical clustering` dùng K-Means NumPy với khoảng cách Euclidean trên các biến vật lý đã chuẩn hóa.
- `spatial + physical clustering` dùng metric hỗn hợp:
  - Euclidean cho `mag/depth/gap/nst/rms`
  - Haversine cho `latitude/longitude`
  - tâm cụm địa lý được cập nhật bằng spherical mean trên quả địa cầu
- Chọn `k` theo tỷ lệ `between_ss / within_ss`.
- Hotspot dùng lưới địa lý (grid) và chọn hotspot theo phân vị số lượng sự kiện.
- Temporal pattern mining dùng chuỗi token hóa theo `mag` và `depth` theo ngày.
- Foreshock/aftershock gán theo khoảng thời gian + bán kính không gian quanh mainshock.
- Outlier dùng điểm dị thường robust (MAD z-score) kết hợp độ hiếm theo vị trí.
- Association rules dùng frequent itemsets (max len=3), lọc luật có consequent thuộc location/hotspot.

## Raw Data Workflow

Neu can lam rieng phan visualization + clustering tren file raw `data/dongdat.csv`, dung:

```bash
.venv/bin/python hoigreen/clustering_pattern_mining/run_raw_visualization_clustering.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/clustering_pattern_mining/raw_outputs
```

Tai lieu tom tat quyet dinh feature, scaling, region id va output nam o:

- `hoigreen/clustering_pattern_mining/RAW_VISUALIZATION_CLUSTERING.md`
