# Feature Extraction cho Dự án Dự đoán Động đất

## 📋 Mô tả
Script này thực hiện trích xuất và xây dựng features từ dữ liệu động đất USGS để phục vụ cho việc:
1. **Dự đoán độ lớn động đất (Magnitude Prediction)**
2. **Dự đoán thời gian và vị trí động đất theo khu vực**

## 🎯 Mục tiêu
- Trích xuất **40+ features** từ dữ liệu thô
- Xây dựng **Historical Features** dựa trên lịch sử 30 ngày
- Tạo **Temporal Features** để phân tích xu hướng theo thời gian
- Tạo **Spatial Features** để phân tích theo vùng địa lý

## 📊 Features được trích xuất

### 1. Temporal Features (9 features)
- `Year`, `Month`, `Day`, `DayOfWeek`, `DayOfYear`, `Hour`
- `Season` (Spring, Summer, Autumn, Winter)
- `TimeOfDay` (Morning, Afternoon, Evening, Night)
- `DaysSince2000` (số ngày kể từ 1/1/2000)

### 2. Spatial Features (9 features)
- `Latitude`, `Longitude`, `Depth`
- `Latitude_Normalized`, `Longitude_Normalized` (chuẩn hóa 0-1)
- `DepthCategory` (Shallow, Intermediate, Deep)
- `LatGrid`, `LonGrid`, `Region` (phân vùng theo lưới 10x10 độ)
- `TectonicRegion` (Pacific Ring of Fire, Mediterranean, etc.)

### 3. Historical Features (4 features)
- `HistoricalCount_30d`: Số động đất trong 30 ngày trước tại khu vực
- `HistoricalMagMean_30d`: Magnitude trung bình 30 ngày trước
- `HistoricalMagMax_30d`: Magnitude lớn nhất 30 ngày trước
- `DaysSinceLastEarthquake`: Số ngày kể từ động đất gần nhất

### 4. Statistical Features (10 features)
- `RMS`: Root Mean Square error (độ chính xác vị trí)
- `Gap`: Azimuthal gap (độ phủ của trạm quan sát)
- `Dmin`: Khoảng cách đến trạm gần nhất
- `Nst`: Số trạm sử dụng
- `Significance`: Chỉ số quan trọng của sự kiện
- `MagType`: Loại magnitude (mw, mb, ml, ms)
- `Status`: Trạng thái (reviewed, automatic, preliminary)
- `Network`: Mạng lưới quan sát
- `HasTsunami`: Cờ cảnh báo sóng thần
- `DataQuality`: Chất lượng dữ liệu (Low, Medium, High)

### 5. Regional Features (3 features)
- `RegionMagMean`: Magnitude trung bình của khu vực
- `RegionMagStd`: Độ lệch chuẩn magnitude của khu vực
- `RegionEventCount`: Tổng số sự kiện trong khu vực
- `YearlyRegionCount`: Số sự kiện/năm trong khu vực

### 6. Target Variables
- `Magnitude`: Độ lớn động đất (biến mục tiêu chính)
- `MagnitudeCategory`: Phân loại (Moderate, Strong, Major, Great)

## 🚀 Cách sử dụng

### Chạy Notebook
```bash
# Kích hoạt virtual environment
source ../.venv/bin/activate

# Mở Jupyter hoặc VSCode
jupyter notebook feature_extraction.ipynb
```

### Output Files
Sau khi chạy xong, bạn sẽ có:
1. **earthquake_features_extracted.csv** - Dữ liệu đã trích xuất features (~ 40 cột)
2. **features_analysis.png** - Biểu đồ phân tích 9 khía cạnh
3. **correlation_matrix.png** - Ma trận tương quan giữa các features

## 📈 Ứng dụng

### 1. Dự đoán Magnitude
```python
# Features quan trọng cho dự đoán Magnitude
X = df[['Depth', 'HistoricalMagMean_30d', 'HistoricalMagMax_30d', 
        'RegionMagMean', 'DaysSinceLastEarthquake', 'Latitude', 'Longitude']]
y = df['Magnitude']

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)
```

### 2. Dự đoán Thời gian xảy ra
```python
# Features temporal
X = df[['Month', 'Season', 'HistoricalCount_30d', 'YearlyRegionCount']]
# Predict: Số động đất trong tháng tới
```

### 3. Phân tích vùng nguy hiểm
```python
# Top regions by frequency and magnitude
high_risk_regions = df.groupby('Region').agg({
    'Magnitude': 'mean',
    'EventID': 'count'
}).sort_values('Magnitude', ascending=False)
```

## 🔬 Models đề xuất

### Regression Models (Dự đoán Magnitude)
- **Random Forest**: Tốt cho dữ liệu phức tạp, nhiều features
- **XGBoost**: Hiệu suất cao, xử lý tốt missing values
- **Neural Networks**: Cho dữ liệu lớn, pattern phức tạp

### Time Series Models (Dự đoán thời gian)
- **LSTM**: Cho sequential patterns
- **Prophet**: Cho seasonal patterns
- **ARIMA**: Cho trend analysis

### Classification Models (Phân loại độ nguy hiểm)
- **SVM**: Phân loại Magnitude Categories
- **Logistic Regression**: Dự đoán Tsunami risk

## 📊 Các bước tiếp theo

1. **Exploratory Data Analysis (EDA)**
   - Phân tích phân bố các features
   - Tìm patterns và correlations
   - Identify outliers

2. **Feature Selection**
   - Sử dụng feature importance từ tree-based models
   - Loại bỏ features có correlation cao
   - PCA cho dimensionality reduction

3. **Model Training**
   - Split data: train/validation/test
   - Cross-validation
   - Hyperparameter tuning

4. **Model Evaluation**
   - Metrics: MAE, RMSE, R²
   - Residual analysis
   - Feature importance analysis

5. **Deployment**
   - Early warning system
   - Real-time prediction API
   - Visualization dashboard

## 📝 Notes

### Về Historical Features
- Tính toán có thể mất 5-10 phút tùy vào kích thước dữ liệu
- Có thể điều chỉnh `time_window_days` và `spatial_window` trong hàm
- Thử nghiệm với nhiều window sizes: 7, 14, 30, 90 ngày

### Về Regional Features
- Phân vùng 10x10 độ có thể điều chỉnh (5x5 cho chi tiết hơn)
- TectonicRegion là phân loại đơn giản, có thể cải thiện với dữ liệu địa chất

### Data Quality
- Đã xử lý missing values bằng median
- DataQuality flag giúp filter dữ liệu không tin cậy
- Status='reviewed' có độ tin cậy cao nhất

## 🤝 Đóng góp
- Thiện LH - Feature Engineering & ML Pipeline
- Project: Earthquake Sequence Mining

## 📚 References
- USGS Earthquake Catalog: https://earthquake.usgs.gov/
- Richter Scale: https://en.wikipedia.org/wiki/Richter_magnitude_scale
- Seismic Analysis: Various geology papers

---
**Created**: March 2026  
**Last Updated**: March 2026  
**Version**: 1.0
