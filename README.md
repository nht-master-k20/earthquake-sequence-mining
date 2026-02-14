# Earthquake Sequence Mining - Phase 1: Data Crawler

> Thu thập dữ liệu động đất từ USGS API

## Mục tiêu

Crawl dữ liệu động đất theo năm từ USGS Earthquake Hazards Program.

- **Data Source**: [USGS Earthquake Data](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data)
- **API**: [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)

> **Lưu ý**: Mặc định crawler sẽ lấy **tất cả độ lớn**. Số lượng data có thể rất lớn (~16,000 events/năm với M≥4.0). Nên dùng `--min-mag` để giới hạn nếu cần.

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

> **Lưu ý**: Có thể có lỗi phát sinh trong quá trình cài đặt thư viện do khác biệt môi trường.

## Sử dụng

### Crawl dữ liệu

> **Lưu ý**: Crawler sẽ tự động skip các event đã có file JSON (không crawl lại).

```bash
# Crawl 1 năm
python main.py 2023

# Crawl nhiều năm
python main.py --start-year 2020 --end-year 2023

# Crawl với giới hạn độ lớn (khuyên dùng)
python main.py --start-year 2020 --end-year 2023 --min-mag 6.5

# Crawl với khoảng độ lớn (ví dụ: chỉ M 5.0 - 6.5)
python main.py --start-year 2020 --end-year 2023 --min-mag 5.0 --max-mag 6.5

# Crawl tất cả các năm
python main.py --all --start-year 2010
```

### Tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `year` | - | Năm cần crawl (single year) |
| `--start-year` | `None` | Năm bắt đầu |
| `--end-year` | `None` | Năm kết thúc |
| `--all` | `False` | Crawl đến năm hiện tại |
| `--min-mag` | `None` | Độ lớn tối thiểu (None = tất cả) |
| `--max-mag` | `None` | Độ lớn tối đa (None = tất cả) |
| `--limit` | Không giới hạn | Giới hạn số lượng mỗi năm |
| `--output-dir` | `data` | Thư mục lưu file |

### Chế độ hoạt động

**Mode 1: Single year** - Crawl 1 năm

```bash
python main.py 2023              # Tất cả độ lớn
python main.py 2023 --min-mag 5.0  # Chỉ M ≥ 5.0
python main.py 2023 --min-mag 5.0 --max-mag 6.5  # Chỉ M 5.0 - 6.5
```

**Mode 2: Year range** - Crawl khoảng năm

```bash
python main.py --start-year 2020 --end-year 2023              # Tất cả độ lớn
python main.py --start-year 2020 --end-year 2023 --min-mag 5.0  # Chỉ M ≥ 5.0
python main.py --start-year 2020 --end-year 2023 --min-mag 5.0 --max-mag 6.5  # Chỉ M 5.0 - 6.5
```

**Mode 3: All years** - Crawl từ start-year đến hiện tại

```bash
python main.py --all --start-year 2010              # Tất cả độ lớn
python main.py --all --start-year 2010 --min-mag 5.0  # Chỉ M ≥ 5.0
python main.py --all --start-year 2010 --min-mag 5.0 --max-mag 6.5  # Chỉ M 5.0 - 6.5
```

### Kiểm tra event thiếu JSON

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
...
```

### Crawl lại các event bị fail

```bash
# Retry specific events
python retry_failed_events.py <year_dir> <event_id1> <event_id2> ...

# Ví dụ:
python retry_failed_events.py data/1969 iscgem811607 uw10835138 iscgem811616
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
