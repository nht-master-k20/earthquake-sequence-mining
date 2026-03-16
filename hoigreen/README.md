# Các giai đoạn

Trong `hoigreen` có 2 giai đoạn phục vụ phân tích dữ liệu raw từ file `data/dongdat.csv`:

## 1. EDA

Folder: `hoigreen/eda`

Mục tiêu:

- Hiểu phân bố dữ liệu (Distribution)
- Hiểu quan hệ giữa các biến
- Hiểu xu hướng theo thời gian
- Hiểu phân bố theo không gian

Output:

- Các biểu đồ trực quan (visualization)
- Bảng thống kê
- File `report.md` tổng hợp insight

Cách chạy:

```bash
.venv/bin/python hoigreen/eda/run_eda_phase.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/eda/outputs
```

## 2. Pattern Discovering

Folder: `hoigreen/pattern_discovering`

Mục tiêu:

- Chọn feature cho bài toán pattern mining
- Chuẩn hóa dữ liệu
- Tìm cụm / pattern ẩn
- Chọn số cluster tối ưu
- Phân tích cụm
- Trực quan hóa kết quả (visualization)

Output:

- Cluster assignments
- Cluster profiles
- Region summaries
- Visualization clusters
- File `report.md` tổng hợp kết quả

Cách chạy:

```bash
.venv/bin/python hoigreen/pattern_discovering/run_pattern_discovering.py \
  --input-csv data/dongdat.csv \
  --output-dir hoigreen/pattern_discovering/outputs
```

## Ghi chú

- EDA và Pattern Discovering được tách riêng để dễ trình bày trong báo cáo.
- Pattern Discovering tái sử dụng pipeline clustering đã được verify trước đó, nhưng được expose trong một folder phase riêng để rõ task và deliverable.

