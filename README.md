# Earthquake Sequence Mining

## Project Overview

A comprehensive earthquake data analysis system with data crawler, web visualization, and sequence mining capabilities.

## Project Structure

```
earthquake-sequence-mining/
├── app_demo/          # Web visualization interface
├── data/               # Earthquake data directory
├── usgs_crawl.py        # Data crawler script
├── requirements.txt      # Python dependencies
└── README.md           # This file
```

## Phase 1: Data Crawler

### Objective

Crawl earthquake data from USGS Earthquake Hazards Program API.

- **Data Source**: [USGS Earthquake Data](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data)
- **API**: [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)

### Output Files

- **JSON Files**: Individual event details in GeoJSON format
  - Format: `event_<mag>_<id>.json` (e.g., `event_6.3_us70006vkq.json`)
- **CSV Files**: Aggregated earthquake data per year

### Environment

- **OS**: Ubuntu 24.04.3 LTS
- **Python**: 3.12

### Installation

```bash
pip install -r requirements.txt
```

Or using virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

#### Crawl Data

```bash
# Crawl 1 year
python usgs_crawl.py 2023

# Crawl multiple years
python usgs_crawl.py --start-year 2020 --end-year 2023

# Crawl with minimum magnitude
python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 5.0

# Crawl with magnitude range
python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 5.0 --max-mag 6.5

# Crawl all years
python usgs_crawl.py --all --start-year 2010
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `year` | - | Single year to crawl |
| `--start-year` | `None` | Start year |
| `--end-year` | `None` | End year |
| `--all` | `False` | Crawl until current year |
| `--min-mag` | `None` | Minimum magnitude |
| `--max-mag` | `None` | Maximum magnitude |
| `--limit` | No limit | Limit events per year |
| `--output-dir` | `data` | Output directory |

#### Check Missing Events

```bash
# Check all years
python check_missing_events.py --all

# Check specific year
python check_missing_events.py 1900

# Check multiple years
python check_missing_events.py 1900 1910 1920
```

#### Retry Failed Events

```bash
python retry_failed_events.py <year_dir> <event_id1> <event_id2> ...

# Example:
python retry_failed_events.py data/1969 iscgem811607 uw10835138
```

#### Create CSV from JSON

```bash
# Create CSV for one year
python create_csv_from_json.py data/1974

# Create CSV for all years
python create_csv_from_json.py data/1974 --all
```

### Directory Structure

```
data/
├── 1900/
│   ├── earthquakes_1900_all.csv
│   ├── event_7.0_cent19000105190000000.json
│   └── ...
├── 1901/
│   └── ...
└── earthquakes_1900-1962_all.csv  # Aggregated file
```

### Troubleshooting

- **Years with >20k events**: Crawler automatically splits by month to avoid API limits
- **Missing CSV file**: Use `create_csv_from_json.py` to generate from JSON files
- **Events missing JSON**: Use `check_missing_events.py` to check, then `retry_failed_events.py` to re-crawl

## Web Demo

See [app_demo/README.md](app_demo/README.md) for web visualization interface.

## License

MIT License
