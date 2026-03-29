# Giai đoạn EDA

Giai đoạn này tập trung vào 4 nhóm phân tích chính:

1. Phân tích phân bố các biến số chính (Distribution)
2. Mối quan hệ giữa các biến (Relationship between variables)
3. Phân tích theo thời gian (Temporal analysis)
4. Phân tích theo không gian (Spatial analysis)

## Input

- `data/dongdat.csv`

## Cách chạy

```bash
.venv/bin/python hoigreen/eda/run_eda_phase.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/eda/outputs
```

## Output chính

- `00_dataset_overview.json`
- `01_numeric_summary.csv`
- `02_correlation_matrix.csv`
- `03_yearly_summary.csv`
- `04_monthly_summary.csv`
- `05_region_summary.csv`
- `06_report.md`
- Bộ ảnh `.png` cho các phân tích numeric distribution / relationship / temporal / spatial

## Insight dự kiến

- Mức độ thiếu dữ liệu (missingness) và chất lượng dữ liệu
- Phân bố magnitude / depth / significance
- Tương quan giữa các biến vật lý và các biến chất lượng
- Các giai đoạn thời gian hoạt động mạnh
- Các vùng địa lý tập trung động đất
