## 1. Bức tranh end-to-end

Pipeline hiện tại đi theo 2 phase liên tiếp:

`EDA:`Dùng để hiểu dataset, đo missingness, xem phân bố, quan hệ biến, xu hướng thời gian và phân bố theo không gian.

`Pattern Discovering:`Không phải sequential pattern mining theo kiểu mainshock-aftershock chain. Phase này đang làm unsupervised clustering ở 2 cấp:

- cấp event: cluster từng trận động đất
- cấp region: cluster từng ô không gian sau khi đã aggregate event theo vùng

## 2. Các field gốc quan trọng của pipeline

Ngay khi load CSV, cả 2 phase đều yêu cầu tối thiểu các cột:

- `id`: định danh duy nhất của event
- `time`: thời điểm xảy ra
- `latitude`, `longitude`: vị trí địa lý
- `depth`: độ sâu
- `mag`: độ lớn động đất
- `type`: loại event

Nếu thiếu một trong các cột này thì pipeline dừng luôn vì:

- không có `time` thì không phân tích temporal được
- không có `latitude` / `longitude` thì không làm spatial và region grouping được
- không có `depth` / `mag` thì mất 2 tín hiệu vật lý cốt lõi
- không có `type` thì không thể tách riêng `earthquake`

Sau đó dữ liệu được chuẩn hóa kiểu dữ liệu:

- `time` được parse sang datetime UTC
- các numeric field được ép kiểu số
- các dòng thiếu `id`, `time`, `latitude`, `longitude`, `depth`, `mag` bị loại
- chỉ giữ `type == earthquake` cho phần phân tích chính

Điểm lưu ý:

- catalog gốc có thể chứa nhiều loại event khác ngoài động đất
- nhưng phần phân tích lõi của repo đang tập trung vào động đất tự nhiên

## 3. Ý nghĩa của các nhóm field chính

### 3.1. Nhóm vật lý của trận động đất

- `mag`: độ lớn động đất, là tín hiệu severity trực tiếp và luôn là feature lõi
- `depth`: độ sâu chấn tiêu, dùng cả ở EDA lẫn clustering
- `latitude`, `longitude`: vị trí tuyệt đối của event, chủ yếu phục vụ spatial analysis và quy đổi sang region grid
- `sig`: chỉ số significance tổng hợp của USGS, phản ánh mức độ đáng chú ý của event
- `tsunami`: cờ 0/1 cho nguy cơ sóng thần

Nhóm này mô tả trận động đất.

### 3.2. Nhóm chất lượng quan trắc / geometry của network

- `gap`: khoảng trống phương vị, phản ánh network coverage quanh event
- `rms`: sai số RMS trong định vị
- `nst`: số trạm được dùng
- `dmin`: khoảng cách tới trạm gần nhất

Nhóm này rất quan trọng vì nó không chỉ nói về event, mà còn nói về chất lượng quan sát. Vì vậy:

- trong EDA, nhóm này được xem như biến numeric lõi để hiểu dữ liệu
- trong pattern discovering, nhóm này vẫn được giữ vì nó giúp phân biệt event được quan trắc tốt hay kém, gần network hay xa network

Nói cách khác, clustering hiện tại không chỉ học "vật lý động đất thuần túy", mà còn học cả trạng thái quan trắc.

### 3.3. Nhóm cảm nhận/tác động từ con người

- `mmi`
- `cdi`
- `felt`

Ý nghĩa:

- `mmi`: mức rung ước tính theo thang Mercalli
- `cdi`: cường độ theo báo cáo cộng đồng
- `felt`: số báo cáo người dân cảm nhận được

Nhóm này có giá trị diễn giải rất tốt, nhưng dữ liệu thiếu quá nhiều. Vì vậy:

- vẫn được giữ trong dataframe để mô tả dữ liệu
- nhưng bị loại khỏi feature lõi dùng cho clustering

Đây là quyết định quan trọng của pipeline hiện tại: giữ để hiểu dữ liệu, không dùng làm core cho mô hình unsupervised.

### 3.4. Nhóm metadata / phân loại hỗ trợ

- `magType`: loại magnitude như `ml`, `mb`, `mw`
- `place`: mô tả textual về vị trí
- `type`: loại event

Nhóm này không được đưa vào event feature matrix, nhưng vẫn hữu ích cho diễn giải sau clustering. Ví dụ:

- `magType` được dùng để xác định `top_mag_type` trong từng event cluster
- `type` dùng để lọc `earthquake`

## 4. EDA phase

## 4.1. Field được dùng làm numeric core trong EDA

EDA chọn 7 biến numeric lõi:

- `mag`
- `depth`
- `sig`
- `gap`
- `rms`
- `nst`
- `dmin`

Lý do:

- đủ ý nghĩa vật lý hoặc chất lượng quan trắc
- độ phủ dữ liệu tốt hơn `mmi`, `cdi`, `felt`
- đủ để nhìn phân bố và tương quan chính của dataset

`tsunami` không nằm trong summary/correlation core, nhưng vẫn được vẽ trong distribution plot như một biến binary để xem mức độ hiếm.

## 4.2. EDA dùng field nào cho từng nhóm phân tích

### Distribution

- histogram / kde cho `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`
- count plot cho `tsunami`
- missingness plot cho `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`, `mmi`, `cdi`, `felt`

Ý nghĩa:

- so sánh biến lõi với biến bị thiếu nhiều
- giúp quyết định field nào chỉ nên để descriptive, field nào đủ tốt để sang phase tiếp theo

### Relationship

- correlation matrix trên `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`
- các scatter tiêu biểu:
  - `mag` vs `sig`
  - `depth` vs `mag`
  - `gap` vs `nst`
  - `dmin` vs `mag`

Ý nghĩa:

- xem biến nào gần như mang cùng tín hiệu, ví dụ `mag` và `sig`
- xem biến nào phản ánh observation quality

### Temporal

Từ `time`, EDA tạo:

- `year`
- `month`

Rồi aggregate để có:

- số event theo năm
- `mag_mean`, `mag_max`, `depth_mean` theo năm
- số event và `mag_mean` theo từng cặp `year-month`

Ý nghĩa:

- hiểu xu hướng ghi nhận theo năm
- tìm seasonality ở mức quan sát

### Spatial

Từ `latitude` và `longitude`, EDA tạo các field grid:

- `lat_cell`
- `lon_cell`
- `region_code`
- `region_lat_center`
- `region_lon_center`

Ý nghĩa:

- chuyển vị trí liên tục thành vùng rời rạc có thể group được
- tạo cầu nối trực tiếp sang phase pattern discovering

`region_code` là mã để đọc dễ hơn, ví dụ `G051_022`.

## 5. Pattern Discovering phase

## 5.1. Đây là phase clustering chứ chưa phải sequence mining

Tên phase là `pattern_discovering`, nhưng logic hiện tại là:

- tạo feature matrix cho từng event
- cluster event
- aggregate event theo region
- cluster region
- đây là pattern discovery theo feature/static profile

## 5.2. Scope giải thích của phase này

Ở mức phase `pattern_discovering`, tài liệu này chỉ nên giải thích:

- nhóm field lõi được dùng để tìm event pattern
- nhóm field tổng hợp xuất hiện trong output để giải thích region pattern
- các label và profile được sinh ra sau clustering

Không nên mô tả các field nội bộ của implementation như thể đó là public contract của phase.

## 5.3. Event clustering dùng field nào

Về mặt nghiệp vụ, event clustering đang xoay quanh 8 tín hiệu lõi:

- `mag`
- `depth`
- `sig`
- `gap`
- `rms`
- `nst`
- `dmin`
- `tsunami`

Ý nghĩa lựa chọn:

- `mag`, `depth`, `sig` là phần mô tả severity và kiểu event
- `gap`, `rms`, `nst`, `dmin` giữ lại tín hiệu observation quality
- `tsunami` giữ một dấu hiệu rare-but-important

Nhưng ở mức implementation, event feature matrix thực tế không dùng raw fields theo đúng tên gốc. Pipeline đang dùng:

- `mag`
- `depth_log1p`
- `sig_log1p`
- `gap`
- `rms_log1p`
- `nst_log1p`
- `dmin_log1p`
- `tsunami`

Điều này có nghĩa là:

- `depth`, `sig`, `rms`, `nst`, `dmin` được biến đổi trước khi đưa vào clustering
- `gap`, `rms`, `nst`, `dmin` được median imputation trước khi scale
- toàn bộ event matrix sau đó được đưa qua `RobustScaler`

Field bị loại khỏi event clustering:

- `mmi`, `cdi`, `felt`

Lý do:

- missing ratio quá cao
- nếu dùng sẽ làm matrix quá thưa hoặc làm việc imputation mất ý nghĩa

Ngoài ra, phase còn tạo một số analytical fields như:

- `year`, `month`, `hour`
- `month_sin`, `month_cos`, `hour_sin`, `hour_cos`
- `depth_band`, `mag_band`
- `region_id`, `region_code`, `region_lat_center`, `region_lon_center`

Các field này có thể xuất hiện trong `feature_overview` hoặc report để diễn giải, nhưng không phải là core clustering features của event matrix.

## 5.4. Event clustering trả ra field gì để diễn giải

Sau khi gán `event_cluster`, phase xuất ra profile để team đọc cluster như một archetype event:

- `event_count`
- `mag_mean`
- `mag_p90`
- `depth_mean`
- `sig_mean`
- `tsunami_rate`
- `region_count`
- `top_mag_type`
- `top_depth_band`

Ngoài các field tổng hợp trên, event profile hiện tại còn merge thêm centroid của chính 8 feature dùng để cluster, gồm:

- `mag`
- `depth_log1p`
- `sig_log1p`
- `gap`
- `rms_log1p`
- `nst_log1p`
- `dmin_log1p`
- `tsunami`
- cùng với các cột diễn giải ngược như `depth_centroid_km`, `sig_centroid`, `rms_centroid`, `nst_centroid`, `dmin_centroid`

Ý nghĩa:

- cluster không chỉ là số `0`, `1`, `2`
- mỗi cluster có thể đọc như một archetype event có profile vật lý và profile quan trắc riêng

Ngoài profile tổng hợp, output event assignment còn giữ các field gốc phục vụ trace-back như:

- `id`
- `time`
- `latitude`
- `longitude`
- `depth`
- `mag`
- `sig`
- `tsunami`
- `magType`
- `place`
- `depth_band`
- `mag_band`
- `region_id`
- `region_code`
- `region_lat_center`
- `region_lon_center`
- `event_cluster`
- `region_cluster`

## 5.5. Region-level output dùng field gì để diễn giải

Ở cấp region, phase này xuất ra một lớp summary để đọc "tính cách" của từng vùng:

- cường độ hoạt động:
  - `event_count`
- đặc trưng magnitude:
  - `mag_mean`
  - `mag_p90`
  - `mag_max`
- đặc trưng depth:
  - `depth_mean`
  - `depth_p90`
  - `shallow_ratio`
  - `intermediate_ratio`
  - `deep_ratio`
- đặc trưng impact/severity:
  - `sig_mean`
  - `tsunami_rate`
  - `major_quake_ratio`
- đặc trưng độ phủ dữ liệu và lịch sử:
  - `active_years`
  - `first_time`
  - `last_time`

Ý nghĩa:

- event cluster nói về từng trận
- region summary nói về "tính cách" dài hạn của một ô không gian

Ngoài summary theo region, phase còn xuất ra:

- `region_id`
- `region_code`
- `region_cluster`

để nối region summary với kết quả clustering và visualization.

Ngoài các field nghiệp vụ cốt lõi ở trên, `region_summary` hiện tại còn có thêm:

- `has_felt_ratio`
- `has_mmi_ratio`
- `dominant_event_cluster`
- `event_cluster_count`
- `dominant_cluster_share`
- `event_count_log1p`
- `region_rank_by_count`
- `lat_cell`
- `lon_cell`

Nhóm này chủ yếu phục vụ profiling, ranking và nối tiếp với visualization/output chi tiết.

## 5.6. Region clustering đang được hiểu ở mức nào

Về mặt diễn giải nghiệp vụ, region clustering đang dựa trên các nhóm tín hiệu sau:

- `event_count_log1p`
- `mag_mean`
- `mag_p90`
- `mag_max`
- `depth_mean`
- `depth_p90`
- `sig_mean`
- `major_quake_ratio`
- `shallow_ratio`
- `deep_ratio`
- `tsunami_rate`

Những field này đại diện cho 3 lớp thông tin:

- volume hoạt động
- độ mạnh và độ sâu đặc trưng
- tỷ lệ event đáng chú ý trong vùng

## 5.7. Region nào không được cluster

Pipeline chỉ cluster các region có đủ số event:

- nếu `event_count >= min_events_per_region` thì region được coi là `eligible`
- nếu không đủ ngưỡng, region được gán `region_cluster = -1`

Ý nghĩa:

- tránh cluster các vùng quá thưa, profile không ổn định
- giữ long-tail regions để hiển thị trên map nhưng không ép gán cluster thiếu tin cậy

## 6. Lifecycle của từng nhóm field

## 6.1. Field giữ xuyên suốt từ raw đến cluster interpretation

- `mag`
- `depth`
- `sig`
- `tsunami`
- `gap`
- `rms`
- `nst`
- `dmin`
- `latitude`
- `longitude`
- `time`

Đây là nhóm field quan trọng nhất vì:

- dùng trong EDA
- dùng trực tiếp hoặc được tổng hợp thành output của phase pattern discovering

## 6.2. Field giữ để mô tả nhưng không dùng làm feature lõi

- `mmi`
- `cdi`
- `felt`
- `magType`
- `place`

Lý do:

- hoặc thiếu quá nhiều
- hoặc mang tính text/categorical để diễn giải chứ không phù hợp cho feature matrix hiện tại

## 6.3. Field chỉ tồn tại để bridge giữa phase hoặc để aggregate

- `year`, `month`, `hour`
- `month_sin`, `month_cos`, `hour_sin`, `hour_cos`
- `depth_log1p`, `sig_log1p`, `rms_log1p`, `nst_log1p`, `dmin_log1p`
- `depth_band`, `mag_band`
- `region_code`, `region_id`
- `event_cluster`, `region_cluster`
- `event_count`, `mag_mean`, `mag_p90`, `mag_max`
- `depth_mean`, `depth_p90`, `sig_mean`, `tsunami_rate`
- `major_quake_ratio`, `active_years`, `first_time`, `last_time`
- `has_felt_ratio`, `has_mmi_ratio`
- `dominant_event_cluster`, `event_cluster_count`, `dominant_cluster_share`
- `event_count_log1p`, `region_rank_by_count`

Nhóm này không phải raw source fields, mà là các field tổng hợp hoặc field gán nhãn để đọc kết quả phase.

## 7. Những hiểu lầm dễ gặp

- `sig` không hoàn toàn độc lập với `mag`; nó là một tín hiệu severity tổng hợp nên có tương quan mạnh với `mag`.
- `gap`, `rms`, `nst`, `dmin` không phải thuần vật lý, mà phản ánh cả chất lượng network quan trắc.
- `mmi`, `cdi`, `felt` không bị bỏ vì vô nghĩa; chúng bị loại khỏi clustering chủ yếu vì quá thiếu.
- `pattern_discovering` hiện tại không phải sequence mining theo chuỗi động đất liên tiếp, mà là clustering trên feature tĩnh và aggregate theo region.
- `month_sin`, `month_cos`, `hour_sin`, `hour_cos` có thể xuất hiện trong output descriptive, nhưng hiện không nằm trong core event clustering features.

## 8. Kết luận

Thiết kế field hiện tại đang đi theo triết lý khá rõ:

- giữ một bộ field lõi đủ sạch, đủ ý nghĩa vật lý và đủ ổn định để clustering
- giữ riêng các field thiếu nhiều hoặc mang tính diễn giải để support analysis, không ép dùng trong feature matrix
- dùng region grid và aggregated region features để nối EDA với pattern discovering thành một câu chuyện thống nhất

Nếu sau này team muốn tiến sang sequence mining thật sự, phần có thể tái sử dụng nhiều nhất vẫn là:

- bước cleaning dữ liệu gốc
- bộ field vật lý lõi
- region mapping
- các output aggregate theo region và các label cluster đang có

