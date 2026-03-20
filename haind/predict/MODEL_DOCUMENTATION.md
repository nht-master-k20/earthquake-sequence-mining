# Tài Liệu Kỹ Thuật Mô Hình Dự Báo Động Đất

## Mục Lục
1. [Định Nghĩa Bài Toán](#1-định-nghĩa-bài-toán)
2. [Xử Lý Dữ Liệu](#2-xử-lý-dữ-liệu)
3. [Feature Engineering](#3-feature-engineering)
4. [Kiến Trúc Mô Hình](#4-kiến-trúc-mô-hình)
5. [Chiến Lược Huấn Luyện](#5-chiến-lược-huấn-luyện)
6. [Đánh Giá Mô Hình](#6-đánh-giá-mô-hình)
7. [Phân Tích Kết Quả](#7-phân-tích-kết-quả)

---

## 1. Định Nghĩa Bài Toán

### 1.1 Mô Tả Nhiệm Vụ

**Input**: Chuỗi các sự kiện động đất trong quá khứ tại một khu vực địa lý

**Output**: Dự đoán 2 mục tiêu cho trận động đất tiếp theo:
- `time_to_next`: Thời gian (ngày) đến trận động đất tiếp theo
- `next_mag`: Độ lớn của trận động đất tiếp theo

### 1.2 Loại Bài Toán

```
Multi-task Learning + Sequence Modeling (Học đa nhiệm + Mô hình hóa chuỗi)
├── Sequential Data: Các sự kiện động đất tạo thành chuỗi thời gian
├── Spatial Dependency: Các sự kiện cluster theo khu vực địa lý
├── Multi-target: Dự đoán CẢ thời gian VÀ độ lớn đồng thời
└── Regression: Cả hai mục tiêu đều là giá trị liên tục
```

### 1.3 Các Thách Thuộc Chính

| Thách thức | Mô tả | Giải pháp |
|------------|-------|-----------|
| **Tính bất thường của thời gian** | Động đất xảy ra khoảng cách không đều | Log transform target |
| **Spatial Clustering** | Các sự kiện cluster theo đường đứt gãy | DBSCAN spatial clustering |
| **Sequence Dependency** | Sự kiện tiếp theo phụ thuộc các sự kiện trước | LSTM với sliding window |
| **Multi-scale Patterns** | Foreshock vs mainshock vs aftershock | Aftershock features |
| **Target Skewness** | Phân phối time bị lệch nặng | Log1p transform |

---

## 2. Xử Lý Dữ Liệu

### 2.1 Tổng Quan Pipeline Dữ Liệu

```
Raw Data (features_advanced.csv)
    ↓
┌─────────────────────────────────────┐
│  Bước 1: Chuẩn Bị Dữ Liệu            │
│  - Load 1.3M sự kiện                 │
│  - Spatial clustering (DBSCAN)       │
│  - Sắp xếp theo thời gian            │
│  - Tạo target variables              │
│  - Chia Train/Val/Test (60-20-20)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Bước 2: Feature Engineering         │
│  - Log transforms                    │
│  - Rolling windows                   │
│  - Cluster statistics                │
│  - Interaction terms                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Bước 3: Tạo Sequences               │
│  - Sliding window per cluster        │
│  - Fixed-length sequences (seq=10)   │
└─────────────────────────────────────┘
    ↓
Sẵn sàng cho LSTM Training
```

### 2.2 Spatial Clustering với DBSCAN

**Tại sao DBSCAN thay vì KMeans?**

```python
# DBSCAN: Density-Based Spatial Clustering
from sklearn.cluster import DBSCAN

# Tham số
eps_km = 50              # Bán kính 50km cho neighborhood
min_samples = 5          # Tối thiểu 5 sự kiện để tạo cluster
metric = 'haversine'     # Khoảng cách great-circle trên Trái Đất

# Chuyển đổi tọa độ sang radians cho haversine metric
coords_rad = np.radians(coords[['latitude', 'longitude']])
eps_rad = eps_km / 6371.0  # Bán kính Trái Đất tính bằng km

# DBSCAN với haversine metric
clustering = DBSCAN(
    eps=eps_rad,
    min_samples=min_samples,
    metric='haversine'
)
labels = clustering.fit_predict(coords_rad)
```

**Ưu điểm:**
- Tự động xác định số lượng clusters
- Xử lý noise (label = -1)
- Bắt được hình học phức tạp của đường đứt gãy
- Không cần chỉ định trước số lượng clusters

### 2.3 Target Variables

**Các targets chính:**
```python
# Thời gian đến trận động đất tiếp theo
df['time_to_next_days'] = (next_event_time - current_time).total_seconds() / 86400

# Độ lớn của trận động đất tiếp theo
df['next_mag'] = df['mag'].shift(-1)

# Log transform cho time (xử lý skewness)
df['log_time_to_next_days'] = np.log1p(df['time_to_next_days'])
```

**Tại sao Log Transform?**

```
Phân phối time_to_next ban đầu:
├── Mean: ~0.5 ngày
├── Median: ~0.01 ngày
├── Max: ~1000 ngày
└── Lệch phải (right-skewed) nặng

Sau log1p transform:
├── Phân phối đối xứng hơn
├── Tốt hơn cho MSE loss
└── Stabilize training
```

### 2.4 Chia Train/Validation/Test

**Time-based split (KHÔNG random):**

```python
# Sắp xếp theo thời gian trước
df = df.sort_values('time').reset_index(drop=True)

# Time-based split ngăn ngừa data leakage
n = len(df)
train = df.iloc[:int(0.6 * n)]      # 60% sớm nhất
val = df.iloc[int(0.6 * n):int(0.8 * n)]   # 20% giữa
test = df.iloc[int(0.8 * n):]       # 20% muộn nhất
```

**Tại sao Time-based split?**
- Ngăn chặn sử dụng dữ liệu tương lai để dự đoán quá khứ
- Mô phỏng kịch bản thực tế khi triển khai
- Kiểm tra khả năng tổng quát theo thời gian

---

## 3. Feature Engineering

### 3.1 Các Nhóm Features

| Nhóm | Features | Lý do |
|------|----------|-------|
| **Cơ bản** | `latitude, longitude, depth, mag` | Thuộc tính sự kiện thô |
| **Aftershock** | `is_aftershock, mainshock_mag` | Bắt pattern aftershock |
| **Khoảng cách đứt gãy** | `dist_to_5th_neighbor_km, ...` | Mật độ hoạt động địa chấn |
| **Stress** | `coulomb_stress_proxy` | Sự tích tụ stress |
| **Vùng** | `regional_b_value, seismic_gap_days` | Pattern dài hạn |
| **Cluster** | `spatial_cluster, cluster_mag_mean, ...` | Context cluster cục bộ |
| **Thời gian** | `time_since_last` | Hoạt động địa chấn gần đây |
| **Log Transform** | `log_coulomb_stress, log_dist_5th, ...` | Xử lý skewness |
| **Rolling** | `rolling_mag_mean, rolling_mag_std, ...` | Xu hướng ngắn hạn |
| **Interaction** | `mag_x_time_since_last, stress_x_density` | Tương tác features |

### 3.2 Giải Thích Các Feature Quan Trọng

#### 3.2.1 Time Since Last (Thời gian kể từ sự kiện gần nhất)

```python
# Thời gian (ngày) kể từ sự kiện gần nhất CÙNG cluster
def calculate_time_since_last(df):
    """
    Với mỗi sự kiện, tính hiệu thời gian từ
    sự kiện trước đó trong cùng spatial cluster.
    """
    df = df.sort_values('time').reset_index(drop=True)

    time_since_last = []
    for cluster_id in df['spatial_cluster'].unique():
        cluster_df = df[df['spatial_cluster'] == cluster_id]
        cluster_df = cluster_df.sort_values('time')

        for i in range(len(cluster_df)):
            if i == 0:
                time_since_last.append(0.0)
            else:
                diff = (cluster_df.iloc[i]['time'] -
                       cluster_df.iloc[i-1]['time']).total_seconds() / 86400
                time_since_last.append(diff)

    return time_since_last
```

**Ý nghĩa vật lý:**
- `time_since_last` ngắn → Giai đoạn địa chấn hoạt động
- `time_since_last` dài → Giai đoạn yên tĩnh, stress đang tích tụ

#### 3.2.2 Coulomb Stress Proxy

```python
# Proxy cho sự tích tụ stress trên đường đứt gãy
coulomb_stress_proxy = cumulative_magnitude_in_region / time_window
```

**Ý nghĩa vật lý:**
- Stress cao hơn → Khả năng động đất trong tương lai cao hơn
- Dựa trên lý thuyết Coulomb failure stress

#### 3.2.3 Regional B-Value

```python
# Giá trị b của vùng theo phân phối Gutenberg-Richter
# log10(N) = a - b*M
regional_b_value = frequency_magnitude_relationship_slope
```

**Ý nghĩa vật lý:**
- B-value thấp → Relatively nhiều động đất lớn hơn
- B-value cao → Relatively nhiều động đất nhỏ hơn

#### 3.2.4 Rolling Window Statistics

```python
# Thống kê qua N sự kiện trước đó
df['rolling_mag_mean_10'] = df['mag'].rolling(window=10).mean()
df['rolling_mag_std_10'] = df['mag'].rolling(window=10).std()
df['rolling_depth_mean'] = df['depth'].rolling(window=10).mean()
```

**Ý nghĩa vật lý:**
- Bắt xu hướng địa chấn cục bộ
- mag_mean tăng → Hoạt động địa chấn gia tăng
- mag_std cao → Hành vi địa chấn thất thường

### 3.3 Log Transform Features

```python
# Áp dụng log1p cho các features bị lệch (skewed)
features_to_log = [
    'coulomb_stress_proxy',
    'seismicity_density_100km',
    'dist_to_5th_neighbor_km',
    'dist_to_10th_neighbor_km',
    'time_since_last'
]

for feat in features_to_log:
    df[f'log_{feat}'] = np.log1p(np.maximum(0, df[feat]))
```

**Tại sao Log Transform?**
- Giảm tác động của các giá trị cực đoan
- Làm cho phân phối đối xứng hơn
- Cải thiện training neural network

### 3.4 Feature Set Cuối Cùng (UPDATED - 17 Advanced Features)

```python
TOTAL_FEATURES = 40  # Updated với new stress tensor & fault geometry features

# Phân loại:
Basic:              4   # lat, lon, depth, mag
Aftershock:         2   # is_aftershock, mainshock_mag
Fault:              3   # dist_to_5th, dist_to_10th, seismicity_density
Stress:             1   # coulomb_stress_proxy
Regional:           3   # b_value, gap_days, max_mag_5yr
Stress Tensor:      5   # stress_sigma_1, sigma_3, tau_max, rate, drop (NEW)
Fault Geometry:     4   # fault_depth, strike, dip, length (NEW)
Cluster:            4   # cluster, cluster_mag_mean, cluster_mag_std, count
Temporal:           1   # time_since_last
Log Transform:      9   # 5 cũ + 3 stress tensor + 1 fault geometry
Rolling:            3   # rolling_mag_mean, std, depth_mean
Interaction:        3   # mag_x_time, stress_x_density, mag_x_regional_max
```

---

## 4. Kiến Trúc Mô Hình

### 4.1 Lựa Chọn Mô Hình: LSTM

**Tại sao LSTM cho Earthquake Prediction?**

| Loại Mô Hình | Sequential Memory | Spatial Awareness | Phù hợp EQ? |
|--------------|-------------------|-------------------|-------------|
| **XGBoost** | ❌ Không | ⚠️ Gián tiếp | ❌ Không |
| **LSTM** | ✅ Có | ⚠️ Gián tiếp | ✅ Có |
| **Transformer** | ✅ Có | ⚠️ Gián tiếp | ✅ Có |
| **ST-GNN** | ✅ Có | ✅ Có | ✅ Có |

**Ưu điểm LSTM:**
- Tự nhiên cho sequential data
- Bắt được temporal dependencies
- Được chứng minh hiệu quả cho time series
- Hiệu quả tính toán

### 4.2 Tổng Quan Kiến Trúc

```
Input: (batch, seq_len=10, features=32)
    ↓
┌─────────────────────────────────────────┐
│  LSTM Layer (2 layers, hidden=128)      │
│  - Học temporal patterns                │
│  - Duy trì hidden state across sequence │
└─────────────────────────────────────────┘
    ↓ (batch, 128)
┌─────────────────────────────────────────┐
│  Dropout (p=0.3)                        │
│  - Ngăn overfitting                     │
└─────────────────────────────────────────┘
    ↓ (batch, 128)
    ├─────────────────┬─────────────────┐
    ↓                 ↓                 ↓
┌──────────────┐ ┌──────────────┐
│  Time Head   │ │  Mag Head    │
│  FC: 128→64  │ │  FC: 128→64  │
│  ReLU        │ │  ReLU        │
│  Dropout     │ │  Dropout     │
│  FC: 64→1    │ │  FC: 64→1    │
└──────────────┘ └──────────────┘
    ↓                 ↓
log_time_to_next   next_mag
```

### 4.3 Code Mô Hình

```python
class EarthquakeLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        # Time prediction head
        self.fc_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Magnitude prediction head
        self.fc_mag = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden)

        # Sử dụng hidden state cuối cùng
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        # Predictions
        pred_time = self.fc_time(last_hidden).squeeze(-1)
        pred_mag = self.fc_mag(last_hidden).squeeze(-1)

        return pred_time, pred_mag
```

### 4.4 Đặc Tả Input/Output

#### Input Format

```python
# Shape: (num_sequences, seq_len=10, num_features=32)

# Ví dụ sequence:
Sequence = [
    Event_0:  [lat, lon, depth, mag, ..., 32 features tổng],
    Event_1:  [lat, lon, depth, mag, ..., 32 features tổng],
    ...
    Event_9:  [lat, lon, depth, mag, ..., 32 features tổng]
]

# Target: Dự đoán thuộc tính của Event_10
```

#### Output Format

```python
# Hai outputs liên tục:
pred_time: (batch,)  # log(time_to_next_days)
pred_mag:  (batch,)  # next_magnitude

# Convert về scale gốc:
time_to_next_days = np.expm1(pred_time) - 1
next_mag = pred_mag
```

### 4.5 Chiến Lược Tạo Sequence

```python
def create_sequences(df, seq_len=10):
    """
    Tạo sequences dùng sliding window per spatial cluster.

    Với mỗi cluster có N events:
    - Sequence i: Events [i-seq_len : i]
    - Target: Event i

    Ví dụ với seq_len=3:
    Events: [E0, E1, E2, E3, E4, E5]

    Sequence 0: [E0, E1, E2] → Target: E3
    Sequence 1: [E1, E2, E3] → Target: E4
    Sequence 2: [E2, E3, E4] → Target: E5
    """
    sequences = []
    targets_time = []
    targets_mag = []

    for cluster_id in df['spatial_cluster'].unique():
        cluster_df = df[df['spatial_cluster'] == cluster_id]
        cluster_df = cluster_df.sort_values('time')

        for i in range(seq_len, len(cluster_df)):
            # Lấy seq_len events trước đó
            seq = cluster_df.iloc[i-seq_len:i]

            # Lấy target (event tiếp theo)
            target = cluster_df.iloc[i]

            sequences.append(seq[features].values)
            targets_time.append(target['log_time_to_next_days'])
            targets_mag.append(target['next_mag'])

    return np.array(sequences), np.array(targets_time), np.array(targets_mag)
```

---

## 5. Chiến Lược Huấn Luyện

### 5.1 Loss Function

**Multi-task Loss:**

```python
# Combined loss cho cả hai tasks
loss_time = MSELoss(pred_time, true_time)
loss_mag = MSELoss(pred_mag, true_mag)
loss = loss_time + loss_mag
```

**Tại sao MSE cho cả hai?**
- Gradient mượt cho regression
- Trừng phạt lỗi lớn nặng nề
- Phù hợp cho targets liên tục

### 5.2 Optimization

```python
# Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,           # Learning rate
    weight_decay=1e-5   # L2 regularization
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,         # Giảm LR một nửa
    patience=5          # Sau 5 epochs không cải thiện
)
```

### 5.3 Các Kỹ Thuật Regularization

| Kỹ thuật | Tham số | Mục đích |
|-----------|---------|----------|
| **Dropout** | p=0.3 | Ngăn co-adaptation |
| **Weight Decay** | 1e-5 | L2 regularization |
| **Gradient Clipping** | max_norm=1.0 | Ngăn exploding gradients |
| **Early Stopping** | patience=10 | Ngăn overfitting |

### 5.4 Cấu Hình Training

```python
# Hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
SEQ_LEN = 10

# Training loop
for epoch in range(NUM_EPOCHS):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer)

    # Validate
    val_metrics = evaluate(model, val_loader)

    # Learning rate scheduling
    scheduler.step(train_loss)

    # Early stopping
    if val_loss < best_val_loss:
        save_model()
        reset_patience()
    else:
        patience_counter += 1
```

---

## 6. Đánh Giá Mô Hình

### 6.1 Metrics Dự Báo Thời Gian

```python
# Metrics trên log scale (cho chất lượng prediction)
r2_time_log = r2_score(true_time_log, pred_time_log)

# Metrics trên scale gốc (cho interpretability)
pred_time_days = np.expm1(pred_time_log) - 1
true_time_days = np.exp(true_time_log) - 1

mae_time = mean_absolute_error(true_time_days, pred_time_days)
rmse_time = np.sqrt(mean_squared_error(true_time_days, pred_time_days))
```

**Giải thích Metrics:**

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **R² (log)** | 1 - SS_res/SS_tot | Tỷ lệ phương sai được giải thích (log scale) |
| **MAE (ngày)** | mean(\|pred - true\|) | Lỗi trung bình tính bằng ngày |
| **RMSE (ngày)** | sqrt(mean((pred - true)²)) | Trừng phạt lỗi nặng hơn |

### 6.2 Metrics Dự Báo Độ Lớn

```python
# Magnitude trên scale gốc
r2_mag = r2_score(true_mag, pred_mag)
mae_mag = mean_absolute_error(true_mag, pred_mag)
rmse_mag = np.sqrt(mean_squared_error(true_mag, pred_mag))
```

**Giải thích Metrics:**

| Metric | Giá trị tốt | Ý nghĩa |
|--------|-------------|---------|
| **R²** | > 0.5 | Mô hình giải thích >50% phương sai |
| **MAE** | < 0.5 | Lỗi trung bình < 0.5 đơn vị magnitude |
| **RMSE** | < 0.7 | Root mean squared error < 0.7 units |

### 6.3 So Sánh Baseline

```python
# Naive baseline 1: Dự đoán mean
pred_time_mean = np.full(len(true), train_mean_time)
mae_baseline = mean_absolute_error(true, pred_time_mean)

# Naive baseline 2: Dự đoán last value
pred_time_last = np.roll(true, 1)
mae_last = mean_absolute_error(true[1:], pred_time_last[1:])

# So sánh model với baselines
improvement_over_mean = (mae_baseline - mae_model) / mae_baseline * 100
improvement_over_last = (mae_last - mae_model) / mae_last * 100
```

### 6.4 Visualization Plots

```python
# 1. Predicted vs Actual Scatter
plt.scatter(true_time, pred_time, alpha=0.3)
plt.plot([0, max], [0, max], 'r--')  # Đường dự đoán hoàn hảo

# 2. Residual Plot
residuals = pred_time - true_time
plt.scatter(pred_time, residuals)
plt.axhline(y=0, color='r')

# 3. Error Distribution
plt.hist(residuals, bins=50)
plt.axvline(x=0, color='r')
```

---

## 7. Phân Tích Kết Quả

### 7.1 Hiệu Suất Mong Đợi

Dựa trên các nghiên cứu tương tự về dự báo động đất:

| Target | Metric | Baseline | LSTM Target |
|--------|--------|----------|-------------|
| **Time** | R² (log) | 0.1 - 0.3 | 0.4 - 0.6 |
| **Time** | MAE (ngày) | 2 - 5 | 0.5 - 2 |
| **Mag** | R² | 0.2 - 0.4 | 0.5 - 0.7 |
| **Mag** | MAE | 0.5 - 0.8 | 0.3 - 0.5 |

### 7.2 Phân Tích Lỗi

**Các lỗi thường gặp:**

1. **Lỗi thời gian lớn**
   - Nguyên nhân: Khoảng cách giữa các sự kiện không dự đoán được
   - Giải pháp: Sử dụng probabilistic forecasting

2. **Magnitude Saturation**
   - Nguyên nhân: Mô hình dự đoán mean magnitude
   - Giải pháp: Thêm categorical bins (nhỏ/vừa/lớn)

3. **Spatial Drift**
   - Nguyên nhân: Mô hình không tổng quát được vùng mới
   - Giải pháp: Thêm spatial features, dùng model-specific cho từng vùng

### 7.3 Di Giải Mô Hình

**LSTM học được gì?**

```python
# Attention weights (nếu dùng attention mechanism)
attention_weights = model.get_attention(sequence)

# High attention events cho thấy:
# - Các trận động đất lớn gần đây
# - Các sự kiện có depth bất thường
# - Các sự kiện trong active clusters

# Điều này có thể visualized để hiểu quyết định của model
```

---

## 8. Checklist Implementation

### 8.1 Chuẩn Bị Dữ Liệu
- [ ] Load full dataset (1.3M events)
- [ ] Áp dụng DBSCAN clustering (eps=50km)
- [ ] Tạo temporal order
- [ ] Tính time_to_next và next_mag targets
- [ ] Log transform time target
- [ ] Time-based train/val/test split

### 8.2 Feature Engineering
- [ ] Log transform skewed features
- [ ] Tính rolling window statistics
- [ ] Tính cluster-level statistics
- [ ] Tạo interaction features
- [ ] Xử lý missing values
- [ ] Normalize/scale features nếu cần

### 8.3 Tạo Sequences
- [ ] Tạo sliding windows per cluster
- [ ] Thiết lập sequence length (10 events)
- [ ] Validate sequence-target alignment
- [ ] Lưu trong efficient format (.npz)

### 8.4 Training Mô Hình
- [ ] Khởi tạo LSTM với architecture phù hợp
- [ ] Thiết lập dual output heads
- [ ] Configure optimizer và scheduler
- [ ] Implement early stopping
- [ ] Monitor training curves

### 8.5 Evaluation
- [ ] Tính R², MAE, RMSE cho cả hai targets
- [ ] Generate visualization plots
- [ ] So sánh với baselines
- [ ] Phân tích error patterns
- [ ] Document findings

---

## 9. Cải Tiến Tương Lai

### 9.1 Kiến Trúc Mô Hình

| Cải thiện | Mô tả | Gain mong đợi |
|-------------|-------------|---------------|
| **Bidirectional LSTM** | Xử lý sequence cả hai hướng | +5-10% R² |
| **Attention Mechanism** | Focus vào important events | +3-7% R² |
| **Transformer** | Self-attention cho long sequences | +10-15% R² |
| **Spatial-Temporal GNN** | Spatial modeling tường minh | +15-20% R² |

### 9.2 Cải Thiện Dữ Liệu

| Cải thiện | Mô tả | Gain mong đợi |
|-------------|-------------|---------------|
| **Thêm Features** | Stress tensor, fault geometry | +5% R² |
| **External Data** | GPS deformation, seismic waves | +10% R² |
| **Data Augmentation** | Synthetic sequences | +3% R² |

### 9.3 Cải Thiện Training

| Cải thiện | Mô tả | Gain mong đợi |
|-------------|-------------|---------------|
| **Curriculum Learning** | Bắt đầu với examples dễ | +2% R² |
| **Multi-task Learning** | Thêm auxiliary tasks | +5% R² |
| **Ensemble Methods** | Kết hợp nhiều models | +10% R² |

---

## 10. Tài Liệu Tham Khảo

1. **Lý thuyết Dự báo Động đất**
   - Omori's Law cho aftershock decay
   - Gutenberg-Richter relationship
   - Coulomb stress transfer theory

2. **Sequence Modeling**
   - Hochreiter & Schmidhuber (1997) - LSTM
   - Vaswani et al. (2017) - Transformer
   - Wu et al. (2020) - Temporal Fusion Transformer

3. **Ứng dụng Seismology**
   - DeVries et al. (2018) - Deep learning cho EQ prediction
   - Rouet-Leduc et al. (2019) - Laboratory earthquake prediction
   - Hulbert et al. (2019) - Earthquake patterns trong lab

---

**Phiên bản Tài liệu:** 1.0
**Cập nhật lần cuối:** 2025-03-20
**Tác giả:** Haind
