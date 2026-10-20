# Experimental Protocol

- Seed mặc định: 42.
- Closed classes và open classes không giao nhau.
- Encoding không được nhận label làm input.
- Threshold chỉ được calibration từ closed-validation.
- Mọi experiment phải lưu config, seed và metrics.

## Encoding (USTC)

- Trace thô: chuỗi 39–64 nhóm 4 ký tự hex cách nhau bằng dấu cách
  (vd. `"e742 421d 1dc8 ..."`); các nhóm chồng lấn đúng 1 byte, tương ứng 78–128 byte.
- Mã hoá cho GPT-2: mỗi ký tự tách bằng dấu cách — `encode_trace_gpt2(raw_trace)`
  trong `src/trafficllm/encoding/ustc.py`.
  Hàm nhận **đúng 1 tham số** (`raw_trace`), không bao giờ nhận nhãn.
  Khoá chữ ký bằng `tests/test_encoding_label_independence.py`.
- Công thức giống hệt upstream: `' '.join(raw.replace(" ", ""))`.
- Kết quả: `data/processed/ustc_seed42/encoded_gpt2.parquet` (97.115 dòng),
  cột `sample_id, partition, original_label, model_label, is_known, text, token_length`.

## Độ dài token (USTC, đo bằng GPT-2 tokenizer)

| partition | n | min | p50 | max | mean | % > 1024 token |
|---|---:|---:|---:|---:|---:|---:|
| closed_train | 39.980 | 176 | 256 | 256 | 253,85 | 0,0 |
| closed_valid | 8.567 | 176 | 256 | 256 | 253,79 | 0,0 |
| closed_test | 8.568 | 176 | 256 | 256 | 253,92 | 0,0 |
| open_valid | 20.000 | 144 | 256 | 256 | 244,59 | 0,0 |
| open_test | 20.000 | 144 | 256 | 256 | 244,22 | 0,0 |
| **ALL** | **97.115** | **144** | **256** | **256** | **249,96** | **0,0** |

- Không trace nào vượt `max_len=1024` → **không có trace nào bị truncate**, đầu vào model
  là toàn vẹn. `max_len` giữ ở 1024 để khớp cấu hình upstream.
- Kiểm tra rò rỉ qua độ dài: phân biệt closed/open chỉ bằng `token_length` cho
  AUROC = 0,449 (< 0,5) → độ dài **không** phải đường tắt để đoán lớp open.
  Chênh lệch mean giữa closed và open đến từ class 15 (open, mean 202) và
  class 3 (known, mean 235), không phải đặc điểm chung của nhóm known/open.
- Báo cáo: `outputs/metrics/token_length_report.csv`, hình `outputs/figures/token_length.png`.

## Nhãn model

- `model_label` 0..11 tương ứng 12 lớp known, **12 = unknown**
  (`metadata.json["unknown_model_label"]`).
- `model_label` ≠ `original_label` (lớp known được chọn ngẫu nhiên, seed 42).
  Luôn ánh xạ qua `metadata.json["class_to_model_label"]`.
