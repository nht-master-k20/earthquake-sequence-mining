# Team Insights Report

## Scope

- Dataset phân tích: `2,992,511` sự kiện `earthquake`
- Giai đoạn dùng để tổng hợp: EDA + Pattern Discovering
- Spatial grid: `2.5` độ
- Tổng số region quan sát được: `3,611`
- Region đủ dữ liệu để cluster ổn định (`>= 25` events): `1,361`

## Executive Summary

1. Catalog bị chi phối bởi động đất nhỏ và nông.
  Median magnitude chỉ `1.44`, median depth `8.48 km`, trong khi `90%` sự kiện có magnitude không vượt `4.1`.
2. Activity tập trung rất mạnh theo không gian.
  Top `10` grid cells đã chiếm `50.94%` tổng số sự kiện, top `100` grid chiếm `86.22%`.
3. Bận rộn nhất không đồng nghĩa với mạnh nhất.
  Top `10` region đông sự kiện nhất có `mag_mean = 1.22`, thấp hơn đáng kể so với toàn dataset (`1.79`).
4. Event-level clustering tách dữ liệu thành 2 archetype rất rõ.
  `95.01%` là nhóm nhỏ, nông, cường độ thấp; `4.99%` là nhóm mạnh hơn nhiều, sâu hơn và có khả năng đi kèm tsunami cao hơn.
5. Region-level clustering lộ ra một nhóm vùng sâu rất khác biệt.
  Chỉ `48` region (`3.53%` eligible regions) thuộc cụm sâu, nhưng depth mean của nhóm này cao gấp `14.54x` nhóm còn lại.

## Key Insights

### 1. Dataset growth chủ yếu đến từ nhiều sự kiện nhỏ hơn, không phải từ sự kiện mạnh hơn

- Số sự kiện mỗi năm tăng từ `69,207` năm `2000` lên `137,979` năm `2024`, đỉnh là `189,020` năm `2020`.
- Trong cùng thời gian đó, mean magnitude giảm từ `2.05` xuống `1.75`.
- Diễn giải phù hợp nhất là catalog ngày càng ghi nhận tốt các sự kiện nhỏ, thay vì thế giới đang có nhiều động đất phá hủy hơn theo cùng tỷ lệ.

### 2. Spatial concentration rất cao, nhưng volume hotspot lại thiên về microseismicity

- Top `5` region chiếm `36.33%` tổng số sự kiện.
- Top `10` region chiếm `50.94%`.
- Top `20` region chiếm `64.61%`.
- Top `10` region đông nhất chỉ có `depth_mean = 18.38 km`, thấp hơn overall mean `24.52 km`.
- Nói cách khác, phần lớn volume đang đến từ một số ít vùng có động đất nhỏ, nông, lặp lại dày đặc.

### 3. Có 2 event archetype rất tách bạch

#### Event cluster 0: nền hoạt động thường xuyên, nhỏ và nông

- Quy mô: `2,843,224` events (`95.01%`)
- `mag_mean = 1.65`
- `depth_mean = 21.03 km`
- `sig_mean = 61.86`
- `tsunami_rate = 0.042%`
- `top_mag_type = ml`

#### Event cluster 1: nhóm mạnh hơn, sâu hơn, tác động lớn hơn

- Quy mô: `149,287` events (`4.99%`)
- `mag_mean = 4.37`
- `depth_mean = 90.90 km`
- `sig_mean = 300.05`
- `tsunami_rate = 0.380%`
- So với cluster 0:
`mag_mean` cao hơn `2.64x`, `depth_mean` cao hơn `4.32x`, `sig_mean` cao hơn `4.85x`, `tsunami_rate` cao hơn `9.06x`

### 4. Region clustering cho thấy một regime động đất sâu riêng biệt

#### Region cluster 0: regime phổ biến

- `1,313` regions
- `3,083,567` events
- `profile_depth_mean = 27.42 km`
- `profile_deep_ratio = 0.47%`
- `profile_major_quake_ratio = 10.46%`

#### Region cluster 1: regime sâu, hiếm nhưng khác biệt mạnh

- `48` regions
- `25,108` events
- `profile_depth_mean = 398.83 km`
- `profile_deep_ratio = 77.28%`
- `profile_major_quake_ratio = 11.98%`
- `profile_mag_mean = 4.35`

Insight chính ở đây không phải là cluster 1 có nhiều event hơn, mà là nó đại diện cho một cơ chế địa chấn khác hẳn: ít vùng, ít event, nhưng sâu hơn nhiều và magnitude trung bình cũng cao hơn.

### 5. Các vùng đông sự kiện nhất và các vùng "severity-like" không trùng nhau

- Các region đông sự kiện nhất như `G051_022`, `G049_025`, `G051_024`, `G050_024` đều thuộc region cluster `0`.
- Các region cluster `1` theo event count lại nằm ở những cell có depth cực lớn, ví dụ:
`G028_000` (`depth_mean = 560.05 km`), `G027_000` (`546.72 km`), `G026_000` (`495.02 km`), `G026_143` (`535.48 km`)
- Suy luận từ tọa độ cho thấy nhóm này nhiều khả năng rơi vào các dải subduction sâu ở Pacific.

## Data Quality Insights

- `mmi` thiếu `99.23%`, `cdi` và `felt` cùng thiếu `96.11%`.
Các trường này chỉ nên dùng mô tả, không nên coi là feature lõi.
- `mag` và `sig` có correlation `0.953`.
Nếu đưa cả hai vào downstream modeling thì cần hiểu là chúng đang mang tín hiệu gần nhau.
- `gap`, `nst`, `dmin`, `rms` không chỉ mô tả động đất, mà còn phản ánh chất lượng quan trắc và geometry của network.

## What To Say To The Team

1. Phần lớn catalog là động đất nhỏ, nông và lặp lại trong một số vùng rất hẹp.
2. Nếu nhìn theo số lượng event thì dễ kết luận sai về risk; volume hotspot và severity hotspot không phải là một.
3. Clustering đang cho thấy ít nhất 2 chế độ địa chấn rõ ràng:
  một chế độ background microseismic và một chế độ strong/deeper/high-impact hơn.
4. Ở cấp region, có một nhóm deep-focus regions rất khác biệt; nhóm này nên được theo dõi và mô hình hóa riêng nếu mục tiêu là hazard characterization.
5. Dữ liệu cảm nhận của con người (`mmi`, `cdi`, `felt`) quá thiếu để làm xương sống cho mô hình hiện tại.

## Caveats

- Đây là unsupervised clustering trên feature tĩnh của từng event và region, không phải sequential pattern mining theo chuỗi mainshock-aftershock.
- Peak count theo năm và theo tháng có thể chịu ảnh hưởng bởi thay đổi hệ thống ghi nhận, không nên diễn giải trực tiếp là trái đất "động đất nhiều hơn" theo nghĩa hazard tuyệt đối.
- `2,250` region có ít hơn `25` events nên chưa được gán region cluster ổn định; long-tail spatial coverage vẫn còn rất lớn.

## Source Files

- `hoigreen/eda/outputs/01_numeric_summary.csv`
- `hoigreen/eda/outputs/03_yearly_summary.csv`
- `hoigreen/eda/outputs/05_region_summary.csv`
- `hoigreen/pattern_discovering/outputs/04_event_cluster_profile.csv`
- `hoigreen/pattern_discovering/outputs/06_region_summary.csv`
- `hoigreen/pattern_discovering/outputs/08_region_cluster_profile.csv`
