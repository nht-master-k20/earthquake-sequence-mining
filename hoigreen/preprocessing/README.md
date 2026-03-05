# Earthquake Data Preprocessing Pipeline

Pipeline xử lý và làm sạch dữ liệu động đất từ USGS API cho mục đích **Earthquake Forecasting** và **Anomaly Detection**.

## Mục đích

Chuyển đổi dữ liệu thô từ USGS GeoJSON format (1,423,475 files) thành dataset sạch với 16 columns quan trọng phục vụ cho:
- Dự báo động đất (Earthquake Forecasting)
- Phát hiện bất thường (Anomaly Detection)
- Time series analysis
- Spatial analysis

## Input Data Structure

Dữ liệu từ USGS API có format GeoJSON:

```json
{
  "type": "Feature",
  "properties": {
    "mag": 5.8,
    "place": "10km NW of Tokyo",
    "time": 1234567890000,
    "magType": "mw",
    "felt": 1500,
    "tsunami": 0,
    ...
  },
  "geometry": {
    "type": "Point",
    "coordinates": [longitude, latitude, depth]
  },
  "id": "us1000abc"
}
```

## Output Schema (16 Columns)

| # | Column | Type | Source | Description |
|---|--------|------|--------|-------------|
| 1 | `id` | string | properties.id | Unique event identifier |
| 2 | `time` | datetime | properties.time | Event timestamp (ISO 8601) |
| 3 | `latitude` | float | geometry.coordinates[1] | Latitude [-90, 90] |
| 4 | `longitude` | float | geometry.coordinates[0] | Longitude [-180, 180] |
| 5 | `depth` | float | geometry.coordinates[2] | Depth in km (>= 0) |
| 6 | `mag` | float | properties.mag | Magnitude (>= 0) |
| 7 | `magType` | string | properties.magType | Magnitude type (md, ml, mb, mw...) |
| 8 | `mmi` | float | properties.mmi | Modified Mercalli Intensity |
| 9 | `cdi` | float | properties.cdi | Community Decimal Intensity |
| 10 | `felt` | int | properties.felt | Number of felt reports |
| 11 | `sig` | int | properties.sig | Significance score |
| 12 | `tsunami` | int | properties.tsunami | Tsunami flag (0/1) |
| 13 | `gap` | float | properties.gap | Azimuthal gap |
| 14 | `rms` | float | properties.rms | RMS travel time residual |
| 15 | `nst` | int | properties.nst | Number of stations |
| 16 | `dmin` | float | properties.dmin | Distance to nearest station |

## Preprocessing Steps

### 1. Data Extraction
- Parse GeoJSON (Feature/FeatureCollection)
- Extract 16 columns từ properties và geometry
- Support cả single Feature và FeatureCollection

### 2. Data Type Conversion
- `time`: milliseconds → datetime (ISO 8601)
- Numeric columns → float64/Int64
- `id`, `magType` → string

### 3. Missing Value Handling

| Column | Strategy | Reason |
|--------|----------|--------|
| `felt` | Fill 0 | No reports = 0 |
| `tsunami` | Fill 0 | No tsunami = 0 |
| `mmi`, `cdi`, `gap`, `rms`, `dmin` | Fill median | Numerical stability |
| `nst` | Fill median | Station count imputation |
| `magType` | Fill "unknown" | Categorical default |

### 4. Data Validation & Filtering

Loại bỏ records không hợp lệ:
- Null `id` hoặc `time`
- `latitude` ngoài [-90, 90]
- `longitude` ngoài [-180, 180]
- `depth` < 0
- `mag` < 0

### 5. Deduplication
- Remove duplicates dựa trên `id` (keep last)

### 6. Sorting
- Sort theo `time` (ascending)

## Cách sử dụng

### Mode 1: Single File Mode

Xử lý một file JSON đơn lẻ (Feature hoặc FeatureCollection):

```bash
python preprocess_usgs_quakes.py -i input.json -o output.csv
```

### Mode 2: Batch Mode (Recommended)

Xử lý toàn bộ thư mục data với nhiều năm:

```bash
# Với progress bar (mặc định)
python preprocess_usgs_quakes.py --batch --data-dir data -o earthquake_cleaned.csv

# Không có progress bar
python preprocess_usgs_quakes.py --batch --data-dir data -o output.csv --no-progress

# Custom data directory
python preprocess_usgs_quakes.py --batch --data-dir /path/to/data -o output.csv
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `-i, --input` | - | Input JSON file (single-file mode) |
| `-o, --output` | *required* | Output CSV file path |
| `--batch` | False | Batch mode: xử lý nhiều năm |
| `--data-dir` | `data` | Root directory chứa các thư mục năm |
| `--no-progress` | False | Tắt progress bar |

## Examples

### Example 1: Test trên 1 năm

```bash
# Tạo test directory
mkdir -p test_data
cp -r data/2020 test_data/

# Run preprocessing
python preprocess_usgs_quakes.py --batch --data-dir test_data -o test_2020.csv
```

### Example 2: Production Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Process all data (1980-2026)
python preprocess_usgs_quakes.py --batch -o earthquake_cleaned.csv

# Output:
# ======================================================================
# 🌍 USGS Earthquake Data Preprocessing Pipeline (Batch Mode)
# ======================================================================
# 📂 Found 47 year directories: 1980 - 2026
# 📅 Years: 100%|██████████| 47/47 [03:04<00:00,  1.43s/year]
# ...
# ✅ PREPROCESSING COMPLETED
```

## Output Format

File CSV với 16 columns, sorted by time:

```csv
id,time,latitude,longitude,depth,mag,magType,mmi,cdi,felt,sig,tsunami,gap,rms,nst,dmin
ci12277543,1980-01-01T00:05:01.210000Z,33.7228333,-118.854,6.01,1.8,mh,4.42,2.2,0,50,0,207.0,0.21,8,0.2836
hv19794818,1980-01-01T01:23:28.390000Z,19.509667,-155.2613333,17.35,0.17,md,4.42,2.2,0,0,0,139.0,0.08,24,0.07292
...
```

## Performance

Benchmark trên dataset đầy đủ (1980-2026):

| Metric | Value |
|--------|-------|
| Input files | 1,423,475 |
| Output rows | 1,338,745 |
| Duplicates removed | 2 |
| Invalid removed | 84,728 |
| Processing time | ~3-4 minutes |
| Output size | 137 MB |

## Data Quality

### Validation Results

✅ **Zero missing values** - Tất cả missing values đã được xử lý
✅ **Valid coordinates** - Latitude/longitude trong phạm vi hợp lệ
✅ **Physical constraints** - Depth và magnitude >= 0
✅ **Chronological order** - Sorted by time
✅ **No duplicates** - Unique event IDs

### Dataset Statistics

Từ dataset đã clean (1,338,745 records):

| Feature | Range/Count |
|---------|-------------|
| Time range | 1980-01-01 → 2026-02-17 |
| Latitude | -84.42° to 87.39° |
| Longitude | -180° to 180° |
| Depth | 0 - 735.8 km |
| Magnitude | 0.0 - 9.1 |
| MagTypes | 28 types (md, mb, ml, mc, mw...) |
| Tsunami events | 1,810 |
| Felt reports | 28,460 |

### Top Earthquakes

Top 5 highest magnitude events trong dataset:

| Date | Magnitude | Location | Depth | Type |
|------|-----------|----------|-------|------|
| 2004-12-26 | M9.1 | Indian Ocean (3.30°N, 95.98°E) | 30 km | mw |
| 2011-03-11 | M9.1 | Tohoku, Japan (38.30°N, 142.37°E) | 29 km | mww |
| 2010-02-27 | M8.8 | Chile (-36.12°S, -72.90°W) | 22.9 km | mww |
| 2025-07-29 | M8.8 | Kamchatka (52.49°N, 160.24°E) | 35 km | mww |
| 2005-03-28 | M8.6 | Northern Sumatra (2.09°N, 97.11°E) | 30 km | mww |

## Code Structure

### Core Functions

```python
parse_usgs_json(obj)          # Parse GeoJSON → list of dicts
coerce_types(df)              # Convert data types
fill_missing(df)              # Handle missing values
filter_invalid(df)            # Remove invalid records
preprocess(input_path)        # Full pipeline (single file)
process_year_batch(year_dir)  # Batch process one year
process_batch_mode(...)       # Full batch pipeline
```

### Module Usage

```python
from preprocess_usgs_quakes import preprocess

# Process single file
df = preprocess(Path("data.json"))
df.to_csv("output.csv", index=False)
```

## Next Steps

Dataset đã sẵn sàng cho:

1. **Feature Engineering**
   - Temporal features (hour, day, month, season)
   - Spatial clustering
   - Sequence features (b-value, inter-event time)

2. **Earthquake Forecasting**
   - Time series models (LSTM, Transformer)
   - Spatiotemporal prediction
   - Magnitude prediction

3. **Anomaly Detection**
   - Outlier detection
   - Unusual sequence patterns
   - Precursor identification

## Files

- `preprocess_usgs_quakes.py` - Main preprocessing script
- `earthquake_cleaned.csv` - Output dataset (137 MB, 1.3M rows)
- `run_full_preprocessing.py` - Legacy standalone script (deprecated)
- `test_*.py` - Test scripts for validation

## Dependencies

```bash
pip install pandas tqdm
```

Hoặc:

```bash
pip install -r ../requirements.txt
```

## Notes

- **Memory efficient**: Xử lý theo batch (year by year)
- **Progress tracking**: Real-time progress bars với tqdm
- **Robust error handling**: Continue on individual file errors
- **Flexible input**: Support cả Feature và FeatureCollection
- **Production ready**: Đã test trên 1.4M files

## Author

HoiGreen - Earthquake Sequence Mining Project

## License

MIT
