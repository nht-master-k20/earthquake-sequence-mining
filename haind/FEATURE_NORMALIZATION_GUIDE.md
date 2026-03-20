# Feature Normalization Guide - Earthquake Prediction

## 📊 Vấn Đề: Feature Scale Không Đồng Nhất

### Trước khi normalization:

| Feature | Min | Max | Mean | Std | Đơn vị |
|----------|-----|-----|------|-----|--------|
| latitude | -90 | 90 | ~0 | ~50 | degrees |
| longitude | -180 | 180 | ~0 | ~100 | degrees |
| depth | 0 | 700 | ~10 | ~50 | km |
| mag | 1 | 10 | ~3 | ~1.5 | Richter |
| stress_sigma_1_mpa | 30 | 150 | ~100 | ~20 | MPa |
| stress_tau_max_mpa | 10 | 60 | ~35 | ~10 | MPa |
| seismicity_density | 0 | 1000+ | ~50 | ~100 | count |
| coulomb_stress_proxy | 0 | 10000+ | ~500 | ~1000 | proxy |

⚠️ **LSTM gặp khó khăn khi features có scale khác biệt!**

## ✅ Giải Pháp: StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# Fit trên training data CHỈ một lần
scaler = StandardScaler()
scaler.fit(train_df[numeric_features])

# Transform cả 3 datasets
train_norm[numeric_features] = scaler.transform(train_df[numeric_features])
val_norm[numeric_features] = scaler.transform(val_df[numeric_features])
test_norm[numeric_features] = scaler.transform(test_df[numeric_features])
```

### Sau normalization:

- Mean ≈ 0
- Std ≈ 1
- Tất cả features trên cùng scale

## 📝 Features được Normalization

### Numeric features (37/40):
- **Basic**: latitude, longitude, depth, mag
- **Advanced**: toàn bộ 17 advanced features
- **Log transforms**: toàn bộ 9 log features
- **Rolling**: rolling_mag_mean, std, depth_mean
- **Cluster**: cluster_mag_mean, cluster_mag_std, cluster_count
- **Interaction**: mag_x_time_since_last, stress_x_density, mag_x_regional_max

### KHÔNG normalization (3/40):
- **is_aftershock**: Boolean (0/1)
- **spatial_cluster**: Categorical (cluster ID)

## 🎯 Kết Quả

| Trước | Sau |
|-------|-----|
| Features scale khác biệt | Tất cả ≈ N(0,1) |
| LSTM học chậm | LSTM học nhanh hơn |
| Gradient unstable | Gradient ổn định |
| Convergence kém | Convergence tốt hơn |

## 📁 Files Đã Cập Nhật

- `02_feature_engineering.py`: Thêm normalize_features() và scaler_params
- `scaler_params.json`: Lưu mean/std để inverse transform khi cần
- `features.json`: Thêm flag 'normalized': True

## 🔧 Khi Run

```bash
cd haind/predict
python 01_data_preparation.py  # Load features với 17 advanced features
python 02_feature_engineering.py  # Now WITH normalization!
python 03_sequence_data.py
python 04_lstm_model.py
```

**Lưu ý:** Scaler parameters được lưu vào `scaler_params.json` để có thể inverse transform predictions khi cần interpret results.
