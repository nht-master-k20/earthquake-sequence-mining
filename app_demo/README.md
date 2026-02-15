# Web Demo - Earthquake Visualization

A web-based visualization interface for earthquake data using Bootstrap 5, jQuery, DataTables, and ECharts.

## Pages

| Page | Description |
|------|-------------|
| `index.html` | Homepage with navigation buttons |
| `instruction.html` | User guide for Python scripts |
| `visualize.html` | Data visualization with DataTable and charts |

## Tech Stack

### Backend
- **FastAPI** - RESTful API server
- **pandas** - Data processing
- **uvicorn** - ASGI server

### Frontend
- **Bootstrap 5.3.3** - UI framework
- **jQuery 3.7.1** - DOM manipulation
- **DataTables 2.1.8** - Interactive table
- **ECharts 5.5.0** - Data visualization
- **Bootstrap Icons 1.11.3** - Icon library

All frontend libraries are loaded from CDN - no installation required.

## Installation

```bash
pip install fastapi uvicorn pandas
```

## Running the Application

### Step 1: Start API Server

```bash
cd app_demo
python api.py
```

The API server runs on `http://127.0.0.1:8386`

### Step 2: Open Web Interface

```bash
xdg-open index.html
# or
firefox index.html
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/years` | List of available years with data |
| `GET /api/data/{year}` | Earthquake data for specific year |
| `GET /api/stats` | Overall statistics |

## Data Format

### Year Data Response

```json
{
  "year": "2023",
  "count": 1234,
  "data": [
    {
      "time": "15/02/2023 14:30:45",
      "place": "10km W of Jakarta",
      "mag": 5.2,
      "depth": 15.3,
      "lat": -6.123,
      "lon": 106.456
    }
  ],
  "stats": {
    "total_events": 1234,
    "avg_mag": 4.5,
    "max_mag": 7.2,
    "avg_depth": 35.8
  },
  "charts": {
    "mag_ranges": {"0-3": 100, "3-5": 800, "5-7": 300, "7+": 34},
    "depth_ranges": {"0-50km": 900, "50-100km": 250, "100-300km": 70, "300km+": 14},
    "month_counts": [98, 102, 95, 110, 105, 120, 115, 130, 125, 108, 95, 100]
  }
}
```

## Features

### Visualize Page (`visualize.html`)

- **Year Selector**: Choose year to view data
- **Statistics Charts**:
  - Magnitude Distribution (bar chart)
  - Depth Distribution (bar chart)  
  - Monthly Distribution (line chart)
- **Quick Stats**: Total events, average/max magnitude, average depth
- **Event Table**: Full data table with:
  - Row index (auto-generated)
  - Time, Location, Magnitude, Depth, Latitude, Longitude
  - No pagination (show all records)

## Development

### File Structure

```
app_demo/
├── api.py              # FastAPI backend
├── index.html          # Homepage
├── instruction.html     # User guide (Vietnamese)
├── visualize.html      # Data visualization page
├── css/
│   └── style.css       # Custom styles
└── js/
    └── visualize.js      # Frontend logic
```

### Customization

- **API Port**: Change `port=8386` in `api.py`
- **Data Directory**: Change `DATA_DIR = "data"` in `api.py`
- **Chart Colors**: Modify `itemStyle.color` in `visualize.js`

## Notes

- API server must be running before using the web interface
- Data loads in real-time from `/data` directory - no static files needed
- All data processing (sorting, statistics, charts) is handled by the API
- Frontend only displays data - no client-side processing
