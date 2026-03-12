# Prediction Stage

Stage này tập trung vào xây dựng, train mô hình và dự báo:

1. Dự báo độ lớn động đất tiếp theo
2. Dự đoán thời gian xảy ra động đất tiếp theo
3. Phát hiện giai đoạn "yên tĩnh" trước động đất lớn
4. Xây dựng mô hình cảnh báo động đất sớm

## Run

```bash
python3 hoigreen/prediction/run_prediction_pipeline.py \
  --input-csv hoigreen/preprocessing/outputs/earthquake_cleaned.csv \
  --output-dir hoigreen/prediction/outputs
```

## Tùy chọn hiệu năng

```bash
python3 hoigreen/prediction/run_prediction_pipeline.py \
  --input-csv hoigreen/preprocessing/outputs/earthquake_cleaned.csv \
  --output-dir hoigreen/prediction/outputs \
  --max-rows 500000 \
  --lookback 10 \
  --train-ratio 0.85 \
  --xgb-n-estimators 700 \
  --xgb-max-depth 6 \
  --xgb-verbose-every 100
```

## Progress logging

- Mặc định pipeline in progress theo từng stage (`[1/4] ... [4/4]`) và thời gian chạy từng bước.
- Với XGBoost, có thể điều chỉnh tần suất in metric train bằng `--xgb-verbose-every N` (ví dụ `100`).
- Tắt toàn bộ progress log bằng `--no-progress`.

## Input yêu cầu

CSV đã preprocess với các cột:

- `id`
- `time`
- `latitude`
- `longitude`
- `depth`
- `mag`
- `gap`
- `nst`
- `rms`

## Output chính

- `01_next_magnitude_predictions.csv`
- `02_next_time_predictions.csv`
- `03_quiet_periods_detected.csv`
- `03_quiet_period_model_scores.csv`
- `04_early_warning_predictions.csv`
- `figures/01_next_magnitude_forecast.png`
- `figures/02_next_time_prediction.png`
- `figures/03_quiet_period_model_timeline.png`
- `figures/04_early_warning_timeline.png`
- `models/01_next_magnitude_model.json`
- `models/02_next_time_model.json`
- `models/03_quiet_period_model.json`
- `models/04_early_warning_model.json`
- `report.md`

## Ghi chú phương pháp

- Next magnitude và next time dùng XGBoost Regressor theo chuỗi event với lag features.
- Quiet-period detection dùng daily aggregation + Logistic Regression (gradient descent, NumPy).
- Early warning dùng Logistic Regression dự báo xác suất có động đất mạnh trong horizon ngắn.
- Tất cả model dùng time-based split để tránh data leakage theo thời gian.
