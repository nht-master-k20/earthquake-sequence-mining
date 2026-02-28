# Earthquake Sequence Mining

Crawl và phân tích dữ liệu động đất từ USGS API.

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

## Cách dùng

### Crawl dữ liệu

```bash
# Crawl tất cả các năm
python3 auto_crawl.py --all

# Crawl một năm cụ thể
python3 auto_crawl.py 1986

# Chỉ kiểm tra, không crawl
python3 auto_crawl.py --all --no-autofill

# Crawl với filter magnitude
python3 auto_crawl.py --all --min-mag 4.0
python3 auto_crawl.py --all --min-mag 4.0 --max-mag 6.0
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `year` | - | Năm cần crawl |
| `--all` | `False` | Crawl tất cả các năm |
| `--min-mag` | `None` | Độ lớn tối thiểu |
| `--max-mag` | `None` | Độ lớn tối đa |
| `--no-autofill` | `False` | Chỉ kiểm tra, không tự động crawl |
| `--output-dir` | `data` | Thư mục lưu dữ liệu |

### Cấu trúc dữ liệu

```
data/
├── 1900/
│   ├── event_7.0_cent19000105190000000.json
│   ├── event_6.5_cent19000112190000000.json
│   └── ...
├── 1986/
│   └── event_*.json
└── 2024/
    └── event_*.json
```

## Đặc điểm

- **Auto-crawl mặc định**: Tự động crawl các event bị thiếu
- **Chia nhỏ magnitude**: Split theo range 0.5 (M0.0-M0.5, M0.5-M1.0, ...) để tránh limit 20000 events/request
- **Bỏ qua event đã có**: Không crawl lại event đã tồn tại
- **Bỏ qua mag=None**: Không crawl events không có magnitude
- **Rate limiting**: Tự động retry khi gặp lỗi 429 (delay 15s)

## Output

```
============================================================
AUTO CRAWL
============================================================
Years: 1
Auto-crawl: ON
============================================================
  Fetching M0.0-M0.5... ✓ 1364 events
  Fetching M0.5-M1.0... ✓ 9665 events
  Fetching M1.0-M1.5... ✓ 12775 events
  ...
1986: api=55698, json=30507, missing=25191

  🔄 Auto-crawling 25191 missing events...
    [1] ✓ ci12345678 (M3.5): 2km W of Cobb, CA
    [2] ✓ ci12345679 (M4.1): 5km NE of Gilroy, CA
    [3] ⊗ ci12345680 - skipped (already exists)
    ...

============================================================
TOTAL CRAWLED: 25191 events
============================================================
```

## Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|-----------|
| HTTP 429 (Rate Limit) | Tự động retry sau 15s |
| HTTP 400 (Bad Request) | Quá nhiều events (>20000), đã split range 0.5 để giải quyết |
| Events bị thiếu | Chạy lại `auto_crawl.py` để crawl missing |
