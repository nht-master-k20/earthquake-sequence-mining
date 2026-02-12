# Earthquake Sequence Mining - Phase 1: Data Crawler

> Thu thập dữ liệu động đất từ USGS API

## Mục tiêu

Crawl dữ liệu động đất theo năm từ USGS Earthquake Hazards Program.

- **Data Source**: [USGS Earthquake Data](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data)
- **API**: [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)

> **Lưu ý**: Mặc định crawler lấy **tất cả độ lớn**. Số lượng data rất lớn (~16,000 events/năm với M≥4.0). Nên dùng `--min-mag` để giới hạn.

## Dữ liệu đầu ra

- **File JSON**: Chi tiết từng sự kiện (GeoJSON)
  - Format: `event_<mag>_<id>.json` (ví dụ: `event_6.3_us70006vkq.json`)
- **File CSV**: Dữ liệu tổng hợp

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

### Crawl dữ liệu

```bash
# Crawl 1 năm (tất cả độ lớn)
python main.py 2023

# Crawl nhiều năm (tất cả độ lớn)
python main.py --start-year 2020 --end-year 2023

# Crawl với giới hạn độ lớn (khuyên dùng)
python main.py --start-year 2020 --end-year 2023 --min-mag 5.0

# Crawl từ năm X đến hiện tại
python main.py --all --start-year 2010

# Giới hạn số lượng (test)
python main.py 2023 --limit 10
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---------|-------|----------|
| `year` | - | Năm cần crawl (single year) |
| `--start-year` | `None` | Năm bắt đầu |
| `--end-year` | `None` | Năm kết thúc |
| `--all` | `False` | Crawl đến năm hiện tại |
| `--min-mag` | `None` | Độ lớn tối thiểu (None = tất cả) |
| `--limit` | Không giới hạn | Giới hạn số lượng mỗi năm |
| `--output-dir` | `data` | Thư mục lưu file |

**Tham số ẩn (set cố định):**
- `--save-json`: `True` (luôn lưu JSON)
- `--delay`: `0.5s` (delay giữa requests)
- `--max-retries`: `3` (số lần retry khi lỗi mạng)

### Crawl lại các event bị fail

```bash
python retry_failed_events.py <năm> <event_id1> <event_id2> ...
```

**Ví dụ:**
```bash
python retry_failed_events.py 1969 iscgem811607 uw10835138 iscgem811616 hv19690506 hv19690507 iscgemsup811630
```

**Tính năng:**
- Retry tối đa 5 lần với exponential backoff
- Lưu file JSON vào thư mục năm chỉ định

## Cấu trúc output

```
data/
├── 1969/
│   ├── event_5.4_iscgem811607.json
│   ├── event_6.8_iscgem811616.json
│   └── earthquakes_1969_all.csv
├── 1970/
│   └── ...
└── earthquakes_1969-1970_all.csv
```

**Lưu ý:** Thư mục năm chỉ được tạo khi có data.

## Retry logic

Khi gặp lỗi mạng (DNS, timeout, connection), crawler **tự động retry** với exponential backoff:
- Retry 1: chờ 2s
- Retry 2: chờ 4s
- Retry 3: chờ 8s

## USGS API

**Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query`

| Tham số | Mô tả | Ví dụ |
|---------|-------|-------|
| `format` | `geojson`, `csv`, `text` | `geojson` |
| `starttime` | Thời gian bắt đầu | `2023-01-01` |
| `endtime` | Thời gian kết thúc | `2023-12-31` |
| `minmagnitude` | Độ lớn tối thiểu | `6.0` |
| `eventid` | ID cụ thể | `us6000m0n6` |

## File

| File | Mô tả |
|------|-------|
| `main.py` | Entry point |
| `usgs_crawl.py` | Script crawl chính |
| `retry_failed_events.py` | Script crawl lại các event bị fail |
| `usgs_crawl.ipynb` | Notebook development |
| `requirements.txt` | Danh sách thư viện |

## Tài liệu

- [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)
- [GeoJSON Spec](https://geojson.org/)
