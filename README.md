# PHẦN 1: CRAWL DỮ LIỆU ĐỘNG ĐẤT TỪ USGS

> Đây là bước 1 trong đồ án: Thu thập dữ liệu động đất từ USGS API

## Mục tiêu

Crawl dữ liệu động đất theo năm từ USGS Earthquake Hazards Program.

- **Data Source**: [https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data)
- **API Documentation**: [https://earthquake.usgs.gov/fdsnws/event/1/](https://earthquake.usgs.gov/fdsnws/event/1/)

## Dữ liệu đầu ra

- **File JSON**: Chi tiết từng sự kiện (GeoJSON format)
- **File CSV**: Dữ liệu tổng hợp để xử lý tiếp

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

```bash
# Crawl 1 năm
python main.py 2023

# Crawl nhiều năm
python main.py --start-year 2020 --end-year 2023

# Crawl từ năm X đến hiện tại
python main.py --all --start-year 2010

# Kết hợp với các tùy chọn khác
python main.py --start-year 2020 --end-year 2023 --min-mag 6.5

# Test với giới hạn số lượng
python main.py 2023 --limit 10

# Không lưu JSON, chỉ CSV
python main.py --start-year 2020 --end-year 2023 --no-json
```

### Tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `year` | Năm cần crawl (single year) | *(tùy chọn)* |
| `--start-year` | Năm bắt đầu | `None` |
| `--end-year` | Năm kết thúc | `None` |
| `--all` | Crawl đến năm hiện tại | `False` |
| `--min-mag` | Độ lớn tối thiểu | `6.0` |
| `--limit` | Giới hạn số lượng mỗi năm | Không giới hạn |
| `--output-dir` | Thư mục lưu file | `data` |
| `--no-json` | Không lưu JSON | `False` |
| `--delay` | Delay giữa requests (giây) | `0.5` |

### Chế độ hoạt động

**Mode 1: Single year** - Crawl 1 năm
```bash
python main.py 2023
```

**Mode 2: Year range** - Crawl khoảng năm
```bash
python main.py --start-year 2020 --end-year 2023
```

**Mode 3: All years** - Crawl từ start-year đến hiện tại
```bash
python main.py --all --start-year 2010
```

## Cấu trúc output

```
data/
├── 2020/
│   ├── event_us6000m0n6.json         # Chi tiết từng event
│   ├── event_us6000m05c.json
│   ├── ...
│   └── earthquakes_2020_M6.0+.csv    # CSV riêng năm 2020
├── 2021/
│   ├── event_*.json
│   └── earthquakes_2021_M6.0+.csv
├── 2022/
│   └── ...
├── 2023/
│   └── ...
└── earthquakes_2020-2023_M6.0+.csv   # CSV tổng hợp tất cả năm
```

## USGS API

**Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query`

### Tham số API

| Tham số | Mô tả | Ví dụ |
|---------|-------|-------|
| `format` | `geojson`, `csv`, `text` | `geojson` |
| `starttime` | Thời gian bắt đầu | `2023-01-01` |
| `endtime` | Thời gian kết thúc | `2023-12-31` |
| `minmagnitude` | Độ lớn tối thiểu | `6.0` |
| `eventid` | ID cụ thể | `us6000m0n6` |

### Ví dụ

```bash
# Query theo event ID
curl "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=us6000m0n6&format=geojson"

# Query theo năm, độ lớn
curl "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2023-01-01&endtime=2023-12-31&minmagnitude=6.0"
```

## File

| File | Mô tả |
|------|-------|
| `main.py` | Entry point |
| `usgs_crawl.py` | Script crawl chính |
| `usgs_crawl.ipynb` | Notebook development |
| `requirements.txt` | Danh sách thư viện |

## Tài liệu

- [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)
- [GeoJSON Spec](https://geojson.org/)
