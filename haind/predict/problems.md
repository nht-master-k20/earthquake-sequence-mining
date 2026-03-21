# 🎯 3 Vấn đề của Baseline Earthquake Prediction

## 📊 Dữ liệu hiện tại

- **Số records**: ~1,300,000 events (full data, bao gồm cả mag < 3)
- **Advanced features**: spatial_cluster, coulomb_stress, seismicity_density, etc.
- **Lợi ích**: Có đầy đủ foreshocks (mag 1-3) giúp detect patterns tốt hơn

---

## Vấn đề 1: Data ordering & Spatial Clustering

**Mô tả:**
> Động đất thường xảy ra theo từng khu vực, từng thời điểm liên tiếp nhau. Trước và sau trận động đất chính thường có dấu hiệu xuất hiện các trận chấn động cường độ nhẹ. Hiện tại dữ liệu không được gom cụm theo từng khu vực xảy ra động đất.

### Các giải pháp

| Giải pháp | Mô tả | Độ phức tạp | Hiệu quả |
|-----------|-------|-------------|----------|
| A. Per-cluster time series | Chia data theo spatial_cluster, xử lý từng cluster như time series | Trung bình | Cao |
| **B. Sliding window per cluster** | Với mỗi event, lấy N events trước đó CÙNG cluster | Thấp | Cao |
| C. Spatial lag features | Thêm thống kê của events gần đó trong thời gian | Thấp | Trung bình |
| D. Graph-based clustering | Build graph từ fault lines, dùng GNN | Cao | Cao |

**✅ Khuyến nghị:** B (Sliding window per cluster) - đơn giản, hiệu quả

---

## Vấn đề 2: Multiple events as input

**Mô tả:**
> Input của mô hình đang là 1 record riêng lẻ. Trong thực tế, để dự báo về một trận động đất lớn sắp xảy ra, chúng ta nên kết hợp dấu hiệu nhận biết bằng nhiều trận động đất xảy ra trước đó tại thời điểm đó, tại khu vực đó.

### Các giải pháp

| Giải pháp | Mô tả | Độ phức tạp | Hiệu quả |
|-----------|-------|-------------|----------|
| **A. Fixed-length sequence** | Input: `[event_t, event_{t-1}, ..., event_{t-n}]` | Trung bình | Cao |
| B. Time-window aggregation | Tổng hợp thống kê trong T hours trước | Thấp | Trung bình |
| C. Attention-based selection | Model tự chọn relevant events | Cao | Cao |
| D. Hawkes process | Statistical model cho earthquake sequences | Cao | Cao |

**✅ Khuyến nghị:** A (Fixed-length sequence) - cân bằng giữa độ phức tạp và hiệu quả

---

## Vấn đề 3: Event relationships

**Mô tả:**
> Muốn biết trận động đất lớn tiếp theo có khả năng đến nhanh hay mau, cường độ mạnh hay yếu cần phải xem xét đến mối quan hệ giữa các trận ghi nhận trước đó tại thời điểm đó, ở cùng khu vực.

### Các giải pháp

| Giải pháp | Mô tả | Độ phức tạp | Hiệu quả |
|-----------|-------|-------------|----------|
| **A. Temporal difference features** | Δmag, Δtime, Δdistance giữa các events liên tiếp | Thấp | Trung bình |
| B. Sequence encoding | RNN/LSTM encode event sequence → embedding | Trung bình | Cao |
| C. Self-attention | Transformer learns relationships | Cao | Cao |
| D. Graph neural network | Model spatial-temporal graph | Cao | Cao |

**✅ Khuyến nghị:** A (Temporal difference features) - đơn giản, tương thích với XGBoost

---

## 🤔 XGBoost có phù hợp không?

### ❌ XGBoost KHÔNG phù hợp cho bài toán này

| Hạn chế | Tác động |
|---------|----------|
| **Independent samples** | Mỗi sample được xử lý riêng, không biết đến các events trước/sau |
| **No temporal memory** | Không thể capture sequences, patterns theo thời gian |
| **No spatial awareness** | Không hiểu distance/direction giữa events |
| **Tree-based structure** | Khó model continuous time dynamics |

**XGBoost chỉ phù hợp cho:**
- Dữ liệu tĩnh/bảng
- Các mẫu độc lập
- Feature-based prediction (không phải sequence-based)

---

## ✅ Mô hình phù hợp cho Earthquake Prediction

### 1. Hawkes Process (Tốt nhất cho lý thuyết)

Statistical model được thiết kế riêng cho earthquake prediction:
- Self-exciting point process
- Mỗi event tăng probability của events sau
- Có cơ sở lý thuyết từ seismology (Omori law)

**Ưu điểm:**
- ✅ Được thiết kế riêng cho earthquake sequences
- ✅ Tham số có thể giải thích được
- ✅ Được chứng minh trong literature

**Nhược điểm:**
- ❌ Phức tạp để implement
- ❌ Hạn chế về flexibility cho non-linear patterns

---

### 2. LSTM/GRU (Tốt nhất cho sequences)

Recurrent Neural Network với memory:
- Xử lý event sequences từng bước
- Duy trì hidden state encoding past context
- Có thể xử lý variable-length sequences

**Ưu điểm:**
- ✅ Phù hợp tự nhiên cho sequential data
- ✅ Có thể capture long-term dependencies
- ✅ Đã established, dễ implement

**Nhược điểm:**
- ❌ Khó kết hợp spatial info
- ❌ Vanishing gradient cho long sequences

---

### 3. Transformer (Tốt nhất cho attention)

Self-attention mechanism:
- Học relationships giữa TẤT CẢ events trong sequence
- Parallel training (khác RNN)
- Có thể attend cả temporal và spatial features

**Ưu điểm:**
- ✅ State-of-the-art cho sequences
- ✅ Attention weights có thể giải thích được
- ✅ Xử lý long-range dependencies

**Nhược điểm:**
- ❌ Cần nhiều data hơn
- ❌ Phức tạp hơn LSTM

---

### 4. Spatio-Temporal GNN (Tốt nhất cho spatial + temporal)

Graph Neural Network + RNN/Transformer:
- Nodes = earthquake events
- Edges = spatial proximity
- Message passing = spatial interactions
- Temporal component = sequence modeling

**Ưu điểm:**
- ✅ Spatial awareness tự nhiên
- ✅ Temporal modeling
- ✅ Về mặt lý thuyết là sound nhất

**Nhược điểm:**
- ❌ Phức tạp nhất
- ❌ Chi phí tính toán nặng

---

## 📊 Bảng so sánh

| Model | Seq. Modeling | Spatial | Độ phức tạp | Data Needed | Khuyến nghị |
|-------|---------------|---------|-------------|-------------|-------------|
| **XGBoost** | ❌ | ⚠️ | Thấp | Ít | ❌ Không |
| **Hawkes** | ✅ | ❌ | Cao | Ít | ⚠️ Có thể |
| **LSTM/GRU** | ✅ | ⚠️ | Trung bình | Trung bình | ✅ Có |
| **Transformer** | ✅ | ⚠️ | Trung bình | Nhiều | ✅ Có |
| **ST-GNN** | ✅ | ✅ | Cao | Nhiều | ⚠️ Tương lai |

---

## 🎬 Kế hoạch Implementation

### Phase 1: Quick Win (1-2 ngày)
> Giữ XGBoost, thêm sequence features

- Trích xuất sequences: `[event_t, event_{t-1}, ..., event_{t-4}]`
- Flatten thành features: `mag_t, mag_{t-1}, mag_{t-2}...`
- Thêm temporal diffs: `mag_t - mag_{t-1}`, `time_t - time_{t-1}`
- Thêm cluster stats: events trong last 24h per cluster

### Phase 2: Sequence Model chuẩn (1 tuần)
> LSTM với sequence input

```
Input: (batch, seq_len=10, features)
- seq_len = 10 events trước
- features = mag, depth, lat, lon, cluster, etc.

Model: LSTM → Dense → [time_to_next, next_mag]
```

### Phase 3: Nâng cao (2-4 tuần)
> Transformer hoặc Spatio-Temporal GNN

- Self-attention cho event relationships
- Graph-based spatial modeling
