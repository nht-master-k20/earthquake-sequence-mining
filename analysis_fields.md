# Phân Tích Dữ Liệu Động Đất Từ API USGS

## Tổng Quan
Dữ liệu JSON từ API https://earthquake.usgs.gov/fdsnws/event/1/query được trả về dưới định dạng GeoJSON, chứa thông tin chi tiết về các sự kiện động đất. File `event_us6000jllz.json` là một ví dụ cho động đất Pazarcik (Thổ Nhĩ Kỳ) với cường độ 7.8 vào ngày 6/2/2023.

## Cấu Trúc Dữ Liệu Chính
Mỗi sự kiện động đất được biểu diễn dưới dạng một GeoJSON Feature với:
- `type`: "Feature"
- `properties`: Chứa tất cả thông tin chính về sự kiện
- `geometry`: Tọa độ địa lý (điểm với kinh độ, vĩ độ, độ sâu)
- `id`: Mã định danh duy nhất của sự kiện

## Các Trường Dữ Liệu Quan Trọng

### Thông Tin Cơ Bản
- **mag**: Cường độ động đất (magnitude) - số thực, ví dụ: 7.8
- **place**: Mô tả địa điểm bằng văn bản, ví dụ: "Pazarcik earthquake, Kahramanmaras earthquake sequence"
- **time**: Thời gian xảy ra (timestamp Unix milliseconds), ví dụ: 1675646254342
- **updated**: Thời gian cập nhật cuối cùng (timestamp)
- **url**: Liên kết đến trang chi tiết sự kiện trên USGS

### Thông Tin Vị Trí
- **geometry.coordinates**: Mảng [kinh độ, vĩ độ, độ sâu (km)]
  - Ví dụ: [37.0143, 37.2256, 10]
- **tz**: Múi giờ (có thể null)

### Thông Tin Tác Động và Cảnh Báo
- **felt**: Số lượng báo cáo cảm nhận từ cộng đồng
- **cdi**: Cường độ cảm nhận từ cộng đồng (Community Determined Intensity) - thang điểm 0-12
- **mmi**: Cường độ rung (Modified Mercalli Intensity) - thang điểm 0-12
- **alert**: Mức cảnh báo PAGER (green/yellow/orange/red)
- **tsunami**: Cờ sóng thần (0 = không, 1 = có)
- **sig**: Điểm quan trọng/độ nguy hiểm (significance) - số nguyên

### Thông Tin Kỹ Thuật
- **status**: Trạng thái đánh giá ("reviewed" hoặc "automatic")
- **net**: Mạng quan trắc (network code), ví dụ: "us"
- **code**: Mã sự kiện trong mạng, ví dụ: "6000jllz"
- **ids**: Danh sách các ID sự kiện từ các nguồn khác nhau
- **sources**: Danh sách các nguồn dữ liệu
- **types**: Danh sách các loại sản phẩm liên quan
- **nst**: Số lượng trạm quan trắc sử dụng
- **dmin**: Khoảng cách tối thiểu đến trạm quan trắc (độ)
- **rms**: Sai số gốc trung bình (root mean square) của vị trí
- **gap**: Khoảng trống phương vị (azimuthal gap) - độ
- **magType**: Loại cường độ ("mww", "mb", etc.)
- **type**: Loại sự kiện ("earthquake")

### Sản Phẩm Bổ Sung (products)
Chứa các sản phẩm phân tích chi tiết như:
- **dyfi**: Did You Feel It? - dữ liệu cảm nhận cộng đồng
- **shakemap**: Bản đồ rung động
- **losspager**: Đánh giá thiệt hại và cảnh báo
- **finite-fault**: Mô hình đứt gãy hữu hạn
- **moment-tensor**: Tensor moment
- **ground-failure**: Phân tích sụt lở đất
- Mỗi sản phẩm có các file dữ liệu (GeoJSON, XML, hình ảnh, etc.)

## Khai Thác Cho Cảnh Báo Động Đất

### Theo Thời Gian
- Sử dụng trường `time` để phân tích xu hướng động đất theo tháng/quý/năm
- Theo dõi `updated` để biết dữ liệu mới nhất
- Phân tích tần suất động đất trong các khoảng thời gian

### Theo Địa Lý
- Sử dụng `geometry.coordinates` để vẽ bản đồ phân bố động đất
- Phân tích `place` để nhóm theo khu vực địa lý
- Xác định các vùng có nguy cơ cao dựa trên mật độ sự kiện

### Các Chỉ Số Quan Trọng Cho Cảnh Báo
- **mag**: Xác định quy mô động đất
- **alert**: Mức cảnh báo sẵn có
- **sig**: Điểm quan trọng tổng hợp
- **mmi**: Cường độ rung tại khu vực
- **cdi**: Cảm nhận thực tế từ cộng đồng
- **felt**: Số lượng người bị ảnh hưởng
- **tsunami**: Nguy cơ sóng thần

## Ví Dụ Sử Dụng Dữ Liệu

### Cảnh Báo Theo Khu Vực
```python
# Lọc động đất mạnh trong khu vực
earthquakes = [event for event in data if event['properties']['mag'] >= 6.0]
# Nhóm theo khu vực từ place hoặc coordinates
```

### Phân Tích Theo Thời Gian
```python
# Chuyển timestamp sang datetime
from datetime import datetime
event_time = datetime.fromtimestamp(event['properties']['time'] / 1000)
# Phân tích theo tháng/năm
```

### Đánh Giá Nguy Cơ Địa Lý
- Sử dụng coordinates để tạo heatmap
- Phân tích mag và sig theo vùng
- Theo dõi các sequence động đất (như Kahramanmaras)

## Lưu Ý
- File JSON có thể rất lớn với nhiều sản phẩm bổ sung
- Tập trung vào các trường properties chính cho phân tích cơ bản
- Sử dụng products khi cần dữ liệu chi tiết hơn (shakemap, dyfi, etc.)
- API hỗ trợ lọc theo thời gian, vị trí, cường độ để lấy dữ liệu phù hợp

## Tài Liệu Tham Khảo
- [USGS Earthquake API Documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
- [GeoJSON Format Specification](https://geojson.org/)
- [USGS Real-time Feeds](https://earthquake.usgs.gov/earthquakes/feed/)</content>
<parameter name="filePath">/Users/thienlehoang/studyAnything/Data mining/earthquake-sequence-mining/analysis_fields.md