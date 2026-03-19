# Lộ trình: Dự báo Động đất với XGBoost

## Tổng quan dự án

**Mục tiêu:** Dự đoán (1) Thời gian và (2) Độ lớn của trận động đất tiếp theo

**Phương pháp:** XGBoost Regression với Advanced Features

**Dữ liệu:** Earthquake catalog (2000-2024, M ≥ 3.0)

---

## GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU (1-2 tuần)

### 1.1 Nạp Dữ liệu & Khám phá
```python
# Load data
df = pd.read_csv('haind/features_advanced.csv')

# Thông tin cơ bản
print(df.shape)      # (307751, 30)
print(df.columns)    # 30 cột
print(df.info())     # Kiểu dữ liệu, missing values

# Thiết lập target variables
# time_to_next: khoảng thời gian đến event tiếp theo (giờ/ngày)
# next_mag: độ lớn của event tiếp theo
```

### 1.2 Tạo Target Variables
```python
# Sắp xếp theo thời gian
df = df.sort_values('time').reset_index(drop=True)

# Tạo target variables
df['time_to_next'] = df['time'].shift(-1) - df['time']
df['next_mag'] = df['mag'].shift(-1)

# Xóa row cuối cùng (không có next)
df = df[:-1]

# Chuyển time_to_next sang số (ngày)
df['time_to_next_days'] = df['time_to_next'].dt.total_seconds() / 86400
```

### 1.3 Chia Train/Validation/Test
```python
# Theo thời gian (quan trọng cho time-series)
n = len(df)
train = df[:int(0.6*n)]      # 60% đầu
val = df[int(0.6*n):int(0.8*n)]   # 20% giữa
test = df[int(0.8*n):]        # 20% cuối

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

### 1.4 Chọn Features
```python
# Features sử dụng cho model
FEATURES = [
    # Basic features
    'latitude', 'longitude', 'depth', 'mag',

    # Aftershock features
    'is_aftershock', 'mainshock_id', 'mainshock_mag',

    # Fault proximity
    'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km',
    'dist_to_20th_neighbor_km', 'seismicity_density_100km',

    # Coulomb stress
    'coulomb_stress_proxy',

    # Regional features
    'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr'
]

TARGET_TIME = 'time_to_next_days'
TARGET_MAG = 'next_mag'
```

---

## GIAI ĐOẠN 2: FEATURE ENGINEERING (1 tuần)

### 2.1 Log Transform cho features skew
```python
# Log transform cho feature có phân phối skew
train['log_coulomb_stress'] = np.log1p(train['coulomb_stress_proxy'])
train['log_seismicity_density'] = np.log1p(train['seismicity_density_100km'])
```

### 2.2 Interaction Terms
```python
# Stress x Density
train['stress_x_density'] = train['coulomb_stress_proxy'] * train['seismicity_density_100km']

# Mag x Gap
train['mag_x_gap'] = train['mainshock_mag'] * train['seismic_gap_days']

# Distance ratio
train['dist_ratio'] = train['dist_to_20th_neighbor_km'] / (train['dist_to_5th_neighbor_km'] + 1)
```

### 2.3 Binning cho mối quan hệ phi tuyến
```python
# B-value bins
train['b_value_bin'] = pd.cut(train['regional_b_value'],
                                bins=[0, 0.8, 1.0, 1.2, 2.0],
                                labels=['rất_thấp', 'thấp', 'bình_thường', 'cao'])

# Distance bins
train['dist_bin'] = pd.cut(train['dist_to_5th_neighbor_km'],
                            bins=[0, 10, 50, 200, np.inf],
                            labels=['rất_gần', 'gần', 'trung_bình', 'xa'])
```

### 2.4 Time-based Features
```python
# Từ cột 'time'
train['year'] = pd.to_datetime(train['time']).dt.year
train['month'] = pd.to_datetime(train['time']).dt.month
train['day_of_year'] = pd.to_datetime(train['time']).dt.dayofyear
```

### 2.5 Xử lý Missing Values
```python
# Kiểm tra missing values
print(train.isnull().sum())

# Fill strategy
# - Numerical: median hoặc -1
# - Categorical: 'unknown' hoặc mode
train = train.fillna({
    'regional_b_value': train['regional_b_value'].median(),
    'seismic_gap_days': 3650,  # 10 năm
    'mainshock_mag': train['mag']  # Nếu NaN, dùng mag hiện tại
})
```

---

## GIAI ĐOẠN 3: TRAIN MODEL - DỰ ĐOÁN THỜI GIAN (1-2 tuần)

### 3.1 XGBoost Regression cho Time
```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# Datasets
X_train_time = train[FEATURES]
y_train_time = train[TARGET_TIME]

X_val_time = val[FEATURES]
y_val_time = val[TARGET_TIME]

# DMatrix cho XGBoost
dtrain_time = xgb.DMatrix(X_train_time, label=y_train_time)
dval_time = xgb.DMatrix(X_val_time, label=y_val_time)

# Parameters
params_time = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}

# Training
evals_result = {}
model_time = xgb.train(
    params_time,
    dtrain_time,
    num_boost_round=1000,
    evals=[(dtrain_time, 'train'), (dval_time, 'val')],
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=100
)

# Evaluation
y_pred_val = model_time.predict(dval_time)
r2 = r2_score(y_val_time, y_pred_val)
mae = mean_absolute_error(y_val_time, y_pred_val)

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f} ngày")
```

### 3.2 Hyperparameter Tuning cho Time
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    model = xgb.train(
        {**params_time, **params},
        dtrain_time,
        num_boost_round=500,
        evals=[(dval_time, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    score = model.best_score
    if score < best_score:
        best_score = score
        best_params = params

print(f"Best params: {best_params}")
```

---

## GIAI ĐOẠN 4: TRAIN MODEL - DỰ ĐOÁN ĐỘ LỚN (1-2 tuần)

### 4.1 XGBoost Regression cho Magnitude
```python
# Datasets
X_train_mag = train[FEATURES]
y_train_mag = train[TARGET_MAG]

X_val_mag = val[FEATURES]
y_val_mag = val[TARGET_MAG]

# DMatrix
dtrain_mag = xgb.DMatrix(X_train_mag, label=y_train_mag)
dval_mag = xgb.DMatrix(X_val_mag, label=y_val_mag)

# Parameters (khác với time model)
params_mag = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}

# Training
model_mag = xgb.train(
    params_mag,
    dtrain_mag,
    num_boost_round=1000,
    evals=[(dtrain_mag, 'train'), (dval_mag, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Evaluation
y_pred_val_mag = model_mag.predict(dval_mag)
r2_mag = r2_score(y_val_mag, y_pred_val_mag)
mae_mag = mean_absolute_error(y_val_mag, y_pred_val_mag)

print(f"R²: {r2_mag:.4f}")
print(f"MAE: {mae_mag:.2f} độ lớn")
```

### 4.2 Hyperparameter Tuning cho Magnitude
```python
# Tương tự Giai đoạn 3.2 nhưng với target là độ lớn
# Focus vào reducing MAE cho độ lớn
```

---

## GIAI ĐOẠN 5: ĐÁNH GIÁ MODEL (1 tuần)

### 5.1 Đánh giá trên Test Set
```python
# Chuẩn bị test data
X_test_time = test[FEATURES]
y_test_time = test[TARGET_TIME]
dtest_time = xgb.DMatrix(X_test_time, label=y_test_time)

X_test_mag = test[FEATURES]
y_test_mag = test[TARGET_MAG]
dtest_mag = xgb.DMatrix(X_test_mag, label=y_test_mag)

# Predictions
y_pred_test_time = model_time.predict(dtest_time)
y_pred_test_mag = model_mag.predict(dtest_mag)

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Time metrics
r2_time = r2_score(y_test_time, y_pred_test_time)
mae_time = mean_absolute_error(y_test_time, y_pred_test_time)
rmse_time = np.sqrt(mean_squared_error(y_test_time, y_pred_test_time))
mape_time = mean_absolute_percentage_error(y_test_time, y_pred_test_time) * 100

# Magnitude metrics
r2_mag = r2_score(y_test_mag, y_pred_test_mag)
mae_mag = mean_absolute_error(y_test_mag, y_pred_test_mag)
rmse_mag = np.sqrt(mean_squared_error(y_test_mag, y_pred_test_mag))

print("="*50)
print("KẾT QUẢ DỰ ĐOÁN THỜI GIAN")
print("="*50)
print(f"R²: {r2_time:.4f}")
print(f"MAE: {mae_time:.2f} ngày")
print(f"RMSE: {rmse_time:.2f} ngày")
print(f"MAPE: {mape_time:.2f}%")

print("\n" + "="*50)
print("KẾT QUẢ DỰ ĐOÁN ĐỘ LỚN")
print("="*50)
print(f"R²: {r2_mag:.4f}")
print(f"MAE: {mae_mag:.3f} độ lớn")
print(f"RMSE: {rmse_mag:.3f} độ lớn")
```

### 5.2 Phân tích Feature Importance
```python
# Time model feature importance
xgb.plot_importance(model_time, max_num_features=15)
plt.title('Feature Importance - Time Prediction')
plt.show()

# Magnitude model feature importance
xgb.plot_importance(model_mag, max_num_features=15)
plt.title('Feature Importance - Magnitude Prediction')
plt.show()

# SHAP values (khuyến nghị)
import shap

# Time model
explainer_time = shap.TreeExplainer(model_time)
shap_values_time = explainer_time.shap_values(X_test_time)
shap.summary_plot(shap_values_time, X_test_time, plot_type="bar")

# Magnitude model
explainer_mag = shap.TreeExplainer(model_mag)
shap_values_mag = explainer_mag.shap_values(X_test_mag)
shap.summary_plot(shap_values_mag, X_test_mag, plot_type="bar")
```

### 5.3 Phân tích Residual
```python
# Time residuals
residuals_time = y_test_time - y_pred_test_time

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_pred_test_time, residuals_time, alpha=0.3)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Thời gian dự đoán (ngày)')
axes[0].set_ylabel('Sai số')
axes[0].set_title('Residual Plot - Time')

# Magnitude residuals
residuals_mag = y_test_mag - y_pred_test_mag

axes[1].scatter(y_pred_test_mag, residuals_mag, alpha=0.3)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Độ lớn dự đoán')
axes[1].set_ylabel('Sai số')
axes[1].set_title('Residual Plot - Magnitude')

plt.tight_layout()
plt.show()

# Phân phối của residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(residuals_time, bins=50, edgecolor='black')
axes[0].set_xlabel('Sai số (ngày)')
axes[0].set_ylabel('Tần suất')
axes[0].set_title('Phân phối Sai số - Time')

axes[1].hist(residuals_mag, bins=50, edgecolor='black')
axes[1].set_xlabel('Sai số (độ lớn)')
axes[1].set_ylabel('Tần suất')
axes[1].set_title('Phân phối Sai số - Magnitude')

plt.tight_layout()
plt.show()
```

### 5.4 Actual vs Predicted Plots
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Time
axes[0].scatter(y_test_time, y_pred_test_time, alpha=0.3)
axes[0].plot([y_test_time.min(), y_test_time.max()],
             [y_test_time.min(), y_test_time.max()], 'r--', lw=2)
axes[0].set_xlabel('Thời gian thực (ngày)')
axes[0].set_ylabel('Thời gian dự đoán (ngày)')
axes[0].set_title(f'Thực vs Dự đoán - Time\nR² = {r2_time:.4f}')

# Magnitude
axes[1].scatter(y_test_mag, y_pred_test_mag, alpha=0.3)
axes[1].plot([y_test_mag.min(), y_test_mag.max()],
             [y_test_mag.min(), y_test_mag.max()], 'r--', lw=2)
axes[1].set_xlabel('Độ lớn thực')
axes[1].set_ylabel('Độ lớn dự đoán')
axes[1].set_title(f'Thực vs Dự đoán - Magnitude\nR² = {r2_mag:.4f}')

plt.tight_layout()
plt.show()
```

---

## GIAI ĐOẠN 6: DIỄN GIẢI MODEL & PHÂN TÍCH (1 tuần)

### 6.1 Partial Dependence Plots
```python
from sklearn.inspection import PartialDependenceDisplay

# Top features cho time prediction
top_features_time = ['coulomb_stress_proxy', 'seismicity_density_100km',
                     'is_aftershock', 'mainshock_mag']

PartialDependenceDisplay.from_estimator(
    model_time, X_test_time, top_features_time,
    kind='average', n_jobs=-1
)
plt.suptitle('Partial Dependence - Time Prediction')
plt.tight_layout()
plt.show()

# Top features cho magnitude prediction
top_features_mag = ['regional_b_value', 'mainshock_mag',
                    'regional_max_mag_5yr', 'mag']

PartialDependenceDisplay.from_estimator(
    model_mag, X_test_mag, top_features_mag,
    kind='average', n_jobs=-1
)
plt.suptitle('Partial Dependence - Magnitude Prediction')
plt.tight_layout()
plt.show()
```

### 6.2 Error Analysis theo độ lớn
```python
# Phân loại errors theo magnitude bins
test['mag_bin'] = pd.cut(test['mag'], bins=[3, 4, 5, 6, 7, 10],
                          labels=['3-4', '4-5', '5-6', '6-7', '7+'])

# MAE theo magnitude bin
mae_by_mag = test.groupby('mag_bin').apply(
    lambda x: mean_absolute_error(x[TARGET_MAG],
                                   model_mag.predict(xgb.DMatrix(x[FEATURES])))
)

print("MAE theo khoảng độ lớn:")
print(mae_by_mag)
```

---

## GIAI ĐOẠN 7: TRIỂN KHAI MODEL (1 tuần)

### 7.1 Lưu Models
```python
import joblib

# Lưu models
joblib.dump(model_time, 'models/xgboost_time_model.pkl')
joblib.dump(model_mag, 'models/xgboost_mag_model.pkl')

# Lưu feature names
joblib.dump(FEATURES, 'models/features.pkl')

# Lưu scalers (nếu có)
# joblib.dump(scaler, 'models/scaler.pkl')
```

### 7.2 Tạo Function Dự đoán
```python
def du_doan_dong_den_tiep_theo(event_hien_tai, model_time, model_mag, features):
    """
    Dự đoán thời gian và độ lớn của trận động đất tiếp theo

    Parameters:
    -----------
    event_hien_tai : dict hoặc pd.Series
        Thông tin event hiện tại
    model_time : xgboost.Booster
        Model đã train cho dự đoán thời gian
    model_mag : xgboost.Booster
        Model đã train cho dự đoán độ lớn
    features : list
        Danh sách features sử dụng

    Returns:
    --------
    dict : {'time_to_next': số ngày, 'next_magnitude': M}
    """
    # Chuyển thành DataFrame
    if isinstance(event_hien_tai, dict):
        event_hien_tai = pd.Series(event_hien_tai)

    # Tạo DMatrix
    X = xgb.DMatrix(event_hien_tai[features].values.reshape(1, -1))

    # Dự đoán
    time_to_next = model_time.predict(X)[0]
    next_mag = model_mag.predict(X)[0]

    return {
        'time_to_next_days': time_to_next,
        'next_magnitude': next_mag,
        'confidence': 'trung_binh'  # Có thể thêm logic tính confidence
    }

# Ví dụ sử dụng
result = du_doan_dong_den_tiep_theo(test.iloc[0], model_time, model_mag, FEATURES)
print(f"Động đất tiếp theo trong: {result['time_to_next_days']:.1f} ngày")
print(f"Độ lớn mong đợi: {result['next_magnitude']:.2f}")
```

### 7.3 Tạo API (tùy chọn)
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="API Dự báo Động đất")

@app.post("/du-doan")
def du_doan(event: dict):
    result = du_doan_dong_den_tiep_theo(event, model_time, model_mag, FEATURES)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## GIAI ĐOẠN 8: TÀI LIỆU & BÁO CÁO (1 tuần)

### 8.1 Tạo Báo cáo HTML
```python
# Tạo report tương tự ANALYSIS_REPORT.html
# Bao gồm:
# - Kiến trúc model
# - Kết quả training
# - Feature importance
# - Evaluation metrics
# - Ví dụ dự đoán
# - Hạn chế & hướng phát triển
```

### 8.2 Viết README
```markdown
# Dự báo Động đất với XGBoost

## Tổng quan
Dự đoán thời gian và độ lớn của trận động đất tiếp theo

## Models
- Time Prediction: R² = 0.XX, MAE = X.X ngày
- Magnitude Prediction: R² = 0.XX, MAE = 0.XX

## Sử dụng
```python
from models import du_doan_dong_den_tiep_theo

result = du_doan_dong_den_tiep_theo(event_hien_tai, ...)
```

## Features
- 11 advanced features
- Phát hiện dư chấn
- Proxy ứng suất Coulomb
- Features động đất khu vực
```

---

## GIAI ĐOẠN 9: CẢI TIẾN & HƯỚNG PHÁT TRIỂN (tiếp tục)

### 9.1 Cải thiện ngắn hạn
- [ ] Ensemble models (XGBoost + LightGBM + CatBoost)
- [ ] Cross-validation time-series (TimeSeriesSplit)
- [ ] Thêm features (dữ liệu GPS, tỷ lệ trượt đứt gãy)
- [ ] Tune hyperparameters (Optuna)

### 9.2 Cải thiện dài hạn
- [ ] Deep Learning (LSTM, Transformer)
- [ ] Spatial modeling (Graph Neural Networks)
- [ ] Probabilistic forecasting (khoảng dự đoán)
- [ ] Pipeline dự đoán real-time

---

## TÓM TẮT

| Giai đoạn | Thời gian | Kết quả chính |
|-----------|-----------|---------------|
| 1. Chuẩn bị dữ liệu | 1-2 tuần | Dataset với targets |
| 2. Feature engineering | 1 tuần | Enhanced features |
| 3. Model thời gian | 1-2 tuần | XGBoost time prediction |
| 4. Model độ lớn | 1-2 tuần | XGBoost magnitude prediction |
| 5. Đánh giá | 1 tuần | Metrics, plots |
| 6. Diễn giải | 1 tuần | SHAP, insights |
| 7. Triển khai | 1 tuần | Saved models, prediction function |
| 8. Tài liệu | 1 tuần | Reports, README |
| **TỔNG** | **8-12 tuần** | **Model production-ready** |

---

## CẤU TRÚC FILE

```
earthquake-sequence-mining/
├── data/
│   ├── dongdat.csv                    # Dữ liệu thô
│   └── features_advanced.csv          # Features + targets
├── models/
│   ├── xgboost_time_model.pkl         # Model time
│   ├── xgboost_mag_model.pkl          # Model magnitude
│   └── features.pkl                    # Danh sách features
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_time_model.ipynb
│   ├── 04_magnitude_model.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── train.py                       # Script training
│   ├── predict.py                     # Function dự đoán
│   └── evaluate.py                    # Script đánh giá
├── reports/
│   ├── MODEL_REPORT.html              # Kết quả model
│   └── FEATURE_IMPORTANCE.html        # Phân tích feature
├── haind/
│   ├── add_advanced_features.py       # Trích xuất feature
│   └── ADVANCED_FEATURES_GUIDE.html   # Tài liệu feature
└── ROADMAP.md                         # File này
```

---

## CÁC BƯỚC TIẾP THEO

1. **Bắt đầu với Giai đoạn 1**: Load data, tạo targets, chia train/val/test
2. **Xây dựng baseline model**: XGBoost đơn giản với minimal features
3. **Cải tiến**: Thêm features, tune hyperparameters
4. **Tài liệu**: Theo dõi experiments và kết quả
5. **Triển khai**: Tạo prediction function cho sử dụng thực tế

Chúc thành công! 🚀
