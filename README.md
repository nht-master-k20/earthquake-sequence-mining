# Earthquake Sequence Mining - Phase 1: Data Crawler

> Thu thập dữ liệu động đất từ USGS API

## Mục tiêu

Crawl dữ liệu động đất theo năm từ USGS Earthquake Hazards Program.

- **Data Source**: [USGS Earthquake Data](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data)
- **API**: [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)

> **Lưu ý**: Mặc định crawler sẽ lấy **tất cả độ lớn**. Số lượng data có thể rất lớn (~16,000 events/năm với M≥4.0). Nên dùng `--min-mag`/`--max-mag` để giới hạn nếu cần.

## Dữ liệu đầu ra

- **File JSON**: Chi tiết từng sự kiện (GeoJSON format)
  - Format: `event_<mag>_<id>.json` (ví dụ: `event_6.3_us70006vkq.json`)
- **File CSV**: Dữ liệu tổng hợp

## Môi trường

- **OS**: Ubuntu 24.04.3 LTS
- **Python**: 3.12

## Cài đặt

```bash
pip install -r requirements.txt
```

Hoặc dùng virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Sử dụng

### 1. Crawl dữ liệu

> **Lưu ý**: Crawler tự động skip các event đã có file JSON.

```bash
# Crawl 1 năm
python usgs_crawl.py 2023

# Crawl nhiều năm
python usgs_crawl.py --start-year 2020 --end-year 2023

# Crawl với độ lớn tối thiểu
python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 5.0

# Crawl với khoảng độ lớn
python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 5.0 --max-mag 6.5

# Crawl tất cả các năm
python usgs_crawl.py --all --start-year 2010
```

| Tham số | Mô tả |
|---------|-------|
| `year` | Năm cần crawl (single year) |
| `--start-year` | Năm bắt đầu |
| `--end-year` | Năm kết thúc |
| `--all` | Crawl đến năm hiện tại |
| `--min-mag` | Độ lớn tối thiểu |
| `--max-mag` | Độ lớn tối đa |
| `--limit` | Giới hạn số lượng events mỗi năm |
| `--output-dir` | Thư mục lưu file (default: data) |

### 2. Kiểm tra event thiếu JSON

```bash
# Kiểm tra tất cả các năm
python check_missing_events.py --all

# Kiểm tra 1 năm
python check_missing_events.py 1900

# Kiểm tra nhiều năm
python check_missing_events.py 1900 1910 1920
```

**Output format:**
```
year: csv=<số dòng CSV>, json=<số file JSON>, missing=<số thiếu>
year: event_id_1
year: event_id_2
```

### 3. Crawl lại các event bị fail

```bash
# Retry specific events
python retry_failed_events.py <year_dir> <event_id1> <event_id2> ...

# Ví dụ:
python retry_failed_events.py data/1969 iscgem811607 uw10835138
```

### 4. Tạo CSV từ JSON files (khi crawler bị interrupt)

```bash
# Tạo CSV cho 1 năm
python create_csv_from_json.py data/1974

# Tạo CSV cho tất cả các năm thiếu
python create_csv_from_json.py data/1974 --all
```

## Cấu trúc thư mục

```
data/
├── 1900/
│   ├── earthquakes_1900_all.csv
│   ├── event_7.0_cent19000105190000000.json
│   └── ...
├── 1901/
│   └── ...
└── earthquakes_1900-1962_all.csv  (file tổng hợp)
```

## Xử lý sự cố

- **Năm có >20k events**: Crawler tự động chia nhỏ theo tháng để tránh API limit
- **Thiếu CSV file**: Dùng `create_csv_from_json.py` để tạo từ JSON files
- **Event thiếu JSON**: Dùng `check_missing_events.py` để kiểm tra, sau đó dùng `retry_failed_events.py` để crawl lại
