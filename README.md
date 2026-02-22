# Earthquake Sequence Mining

## Project Overview

A comprehensive earthquake data analysis system with data crawler, web visualization, and sequence mining capabilities.

## Project Structure

```
earthquake-sequence-mining/
├── app_demo/              # Web visualization interface
├── data/                   # Earthquake data directory (JSON files)
├── usgs_crawl.py           # Data crawler script
├── check_missing_events.py # Check for missing event data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Phase 1: Data Crawler

### Objective

Crawl earthquake data from USGS Earthquake Hazards Program API.

- **Data Source**: [USGS Earthquake Data](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data)
- **API**: [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)

### Output Files

- **JSON Files**: Individual event details in GeoJSON format (PRIMARY DATA SOURCE)
  - Stored in: `data/{year}/event_<mag>_<id>.json`
  - Example: `data/1974/event_5.2_ci12321487.json`

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

#### Features

- **Incremental Crawling**: Skips events that already have JSON files
- **Rate Limiting**: Automatically handles HTTP 429 errors with retry logic (10s, 20s, 30s delays)
- **Month-by-Month**: For years with >20,000 events, automatically splits requests by month

#### Check Missing Events

```bash
# Check all years
python check_missing_events.py --all

# Check specific year
python check_missing_events.py 1900

# Check multiple years
python check_missing_events.py 1900 1910 1920

# Check with auto-fill missing events
python check_missing_events.py 1975 --autofill 1

# Check with magnitude filter and auto-fill
python check_missing_events.py --all --min-mag 4 --max-mag 6 --autofill 1
```

### Directory Structure

```
data/
├── 1900/
│   ├── event_7.0_cent19000105190000000.json
│   ├── event_6.5_cent19000112190000000.json
│   └── ...
├── 1901/
│   └── event_*.json
└── 1974/
    └── event_*.json                       # Individual event JSONs
```

### How It Works

1. **Fetch Event List**: Gets list of earthquake IDs for the year(s) from USGS API
2. **Crawl Individual Events**: For each ID, fetches detailed GeoJSON data
3. **Skip Existing**: If JSON already exists, skip re-downloading
4. **Check Missing**: `check_missing_events.py` calls USGS API directly to compare with local JSON files
5. **Auto-Fill**: Use `--autofill 1` to automatically crawl missing events

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **HTTP 429 Rate Limit** | Crawler auto-retries with 10s/20s/30s delays. Increase delay in code if needed |
| **Years with >20k events** | Crawler automatically splits by month to avoid API limits |
| **Events missing JSON** | Use `check_missing_events.py --autofill 1` to auto-crawl missing events |

### Notes

- JSON files are the PRIMARY data source
- Re-running the crawler will skip existing JSON files
- API server (app_demo/api.py) reads all JSON files in each year folder
- Delay between requests: 1 second (configurable in `usgs_crawl.py`)

## Web Demo

See [app_demo/README.md](app_demo/README.md) for web visualization interface.

### Quick Start

```bash
# 1. Start API server
cd app_demo
python api.py

# 2. Open web interface
xdg-open index.html
```

## License

MIT License
