# Earthquake Data Preprocessing Pipeline

Pipeline này chuyển raw USGS GeoJSON thành dataset clean, gọn, và phù hợp hơn với bài toán hiện tại trong repo:

- Earthquake sequence mining
- Spatial and physical clustering
- Next-event prediction
- Hotspot and anomaly analysis

## Mục tiêu thiết kế

Schema output đã được thu gọn theo hướng modeling-focused:

- Giữ các cột lõi thật sự dùng cho pipeline hiện tại
- Giữ một ít metadata hữu ích (`magType`, `status`, `type`, `sig`)
- Loại các cột rất sparse hoặc thiên về post-event impact khỏi output mặc định (`mmi`, `cdi`, `felt`, `tsunami`, `dmin`)
- Chuẩn hóa string categories trước khi lưu
- Lọc bỏ các event không phải `earthquake`

## Output Schema (13 Columns)


| #   | Column      | Type            | Source                    | Vai trò                              |
| --- | ----------- | --------------- | ------------------------- | ------------------------------------ |
| 1   | `id`        | string          | root.id / properties.code | Khóa duy nhất                        |
| 2   | `time`      | datetime string | properties.time           | Trục thời gian cho sequence          |
| 3   | `latitude`  | float           | geometry.coordinates[1]   | Vị trí địa lý                        |
| 4   | `longitude` | float           | geometry.coordinates[0]   | Vị trí địa lý                        |
| 5   | `depth`     | float           | geometry.coordinates[2]   | Đặc trưng vật lý                     |
| 6   | `mag`       | float           | properties.mag            | Đặc trưng vật lý chính               |
| 7   | `magType`   | string          | properties.magType        | Metadata magnitude, đã normalize     |
| 8   | `sig`       | int             | properties.sig            | Chỉ số significance                  |
| 9   | `gap`       | float           | properties.gap            | Chất lượng geometry quan trắc        |
| 10  | `rms`       | float           | properties.rms            | Sai số định vị                       |
| 11  | `nst`       | int             | properties.nst            | Số trạm quan trắc                    |
| 12  | `status`    | string          | properties.status         | Trạng thái review, đã normalize      |
| 13  | `type`      | string          | properties.type           | Loại event, dùng để lọc `earthquake` |


## Vì sao chọn bộ cột này

### Cột quan trọng

- `id`, `time`, `latitude`, `longitude`, `depth`, `mag`, `gap`, `rms`, `nst`
- Đây là nhóm cột mà pipeline prediction và clustering đang dùng trực tiếp.

### Mở rộng

- `magType`: hữu ích sau khi normalize, vì các cách đo magnitude không hoàn toàn tương đương.
- `sig`: thêm một góc nhìn về mức độ quan trọng của event.
- `status`: giúp phân biệt `reviewed` và `automatic`.
- `type`: dùng để loại các event như `quarry blast`, `explosion`, `landslide`.

### Loại khỏi output mặc định

- `mmi`, `cdi`, `felt`: rất sparse và chủ yếu phản ánh tác động sau sự kiện.
- `tsunami`: hữu ích cho risk tagging, nhưng không phải feature nền cho pipeline hiện tại.
- `dmin`: có thể hữu ích, nhưng missing khá nhiều; chưa đưa vào output mặc định để tránh imputation không cần thiết.

## String Handling

### `magType`

- Chuyển về lowercase
- Strip khoảng trắng
- Xóa hậu tố dạng ngoặc như `ml(texnet)` -> `ml`
- Chuẩn hóa alias như `mb_lg` -> `mblg`
- Fill missing bằng `unknown`

### `status` và `type`

- Chuyển về lowercase
- Strip khoảng trắng
- Fill missing bằng `unknown`

## Validation và Filtering

Pipeline loại bỏ các record:

- Thiếu một trong các cột lõi: `id`, `time`, `latitude`, `longitude`, `depth`, `mag`
- `latitude` ngoài `[-90, 90]`
- `longitude` ngoài `[-180, 180]`
- `depth < 0`
- `mag < 0`
- `type != earthquake`

## Missing Value Strategy


| Column                                                         | Strategy                     |
| -------------------------------------------------------------- | ---------------------------- |
| `gap`, `rms`                                                   | Fill median                  |
| `nst`, `sig`                                                   | Fill median                  |
| `magType`, `status`, `type`                                    | Normalize + fill `unknown`   |
| Core columns (`time`, `latitude`, `longitude`, `depth`, `mag`) | Không impute, thiếu thì drop |


## Cách dùng

### Single File Mode

```bash
python preprocess_usgs_quakes.py -i input.json -o output.csv
```

### Batch Mode

```bash
python preprocess_usgs_quakes.py --batch --data-dir data -o earthquake_cleaned.csv
python preprocess_usgs_quakes.py --batch --data-dir data -o output.csv --no-progress
```

## Output Example

```csv
id,time,latitude,longitude,depth,mag,magType,sig,gap,rms,nst,status,type
ci12277543,1980-01-01T00:05:01.210000Z,33.7228333,-118.854,6.01,1.8,mh,50,207.0,0.21,8,reviewed,earthquake
hv19794818,1980-01-01T01:23:28.390000Z,19.5096667,-155.2496667,8.887,2.33,ml,84,104.0,0.08,24,reviewed,earthquake
```

## Tương thích với pipeline hiện tại

Hai pipeline downstream vẫn chạy bình thường vì chúng chỉ yêu cầu subset sau:

- `id`
- `time`
- `latitude`
- `longitude`
- `depth`
- `mag`
- `gap`
- `nst`
- `rms`

Các cột `magType`, `sig`, `status`, `type` được giữ lại để phục vụ feature engineering ở bước sau.

## Gợi ý bước tiếp theo

- Nếu muốn thử thêm categorical features vào model, dùng `magType` sau khi rare-bucket ở train split.
- Nếu muốn siết chất lượng dữ liệu hơn, có thể lọc riêng `status == reviewed`.
- Nếu cần risk-oriented dataset riêng, có thể tạo thêm một schema mở rộng chứa `tsunami` hoặc `dmin`.

