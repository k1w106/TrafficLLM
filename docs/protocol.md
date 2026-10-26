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

## Fine-tune GPT-2 (P3)

Cấu hình: `configs/ustc_gpt2_seed42.yaml`, seed 42, `GPT2ForSequenceClassification`
(124,4M tham số, `num_labels=12`), khởi tạo từ `gpt2`, lớp `score` (12 × 768) là mới.

| Tham số | Giá trị |
|---|---|
| `max_len` | 1024 (không trace nào bị cắt — xem bảng độ dài token) |
| Padding | động theo batch, bên phải, `pad_token` = eos (50256) |
| Optimizer | AdamW, lr 5e-5, weight decay 0,01, eps 1e-8 |
| Scheduler | giảm tuyến tính, warmup 0 |
| Gradient clip | 1,0 |
| Epoch / batch | 3 / 12 (train), 32 (eval) |
| Độ chính xác số học | bf16 autocast (`amp: bf16`) |
| Attention | `eager` — xem ghi chú bên dưới |
| Chọn checkpoint | `closed_valid` accuracy cao nhất, lưu ngay sau mỗi epoch |

**Vì sao `attn_implementation=eager`:** attention mặc định (flash / memory-efficient
SDPA) không tất định ở lượt backward — PyTorch cảnh báo rõ điều này. Giữ mặc định thì
"cùng seed ra cùng kết quả" sai, phá yêu cầu tái lập. Eager chậm hơn ~16%
(336 so với 289 ms/step) và tốn thêm ~0,5 GB VRAM, đổi lại tất định.
Kiểm chứng: `scripts/train_gpt2.py --determinism-check` chạy 2 lần cùng seed,
so loss + accuracy + tổng 124,4M trọng số → giống hệt từng chữ số.

**Ghi chú vận hành (GPU 6 GB):** sau khi train xong, allocator của PyTorch vẫn giữ
~6 GB cache; nếu đánh giá valid ngay thì tràn sang RAM hệ thống và chậm ~5 lần
(0,55 → 2,9 s/batch). Vì vậy `evaluate()` gọi `torch.cuda.empty_cache()` trước khi
chạy và `eval_batch_size` để 32. Đây là lý do một epoch mất ~23 phút chứ không phải
vài giờ.

### Kết quả

| Epoch | train_loss | train_acc | valid_loss | valid_acc | Thời gian |
|---|---:|---:|---:|---:|---:|
| 1 | 0,203433 | 0,941171 | 0,027017 | 0,996848 | 22,8 phút |
| 2 | 0,010403 | 0,998299 | 0,005974 | 0,998949 | 26,5 phút |
| 3 | 0,001806 | 0,999650 | 0,001986 | 0,999650 | 22,5 phút |

- Best epoch: **3**, `closed_valid` accuracy = **0,999650**. Tổng thời gian 71,8 phút.
- `closed_test` (không dùng khi train): **0,999767** (8.566/8.568).
- `closed_train` sau khi trích lại: 0,999925 (39.977/39.980).
- Không có dấu hiệu học vẹt: `valid_acc` ≥ `train_acc` ở cả 3 epoch.
- Checkpoint: `outputs/checkpoints/ustc_gpt2_seed42.pt` (475 MB) + `.config.yaml`.

**So sánh với paper:** 0,9996 cao hơn khoảng 0,98–0,996 mà paper báo cho closed-set.
Đây **không phải** bằng chứng tốt hơn paper — split khác nhau (ta 12/20 class known,
paper nói 40% = 8/20) và USTC là dataset "dễ" (trace cùng ứng dụng rất giống nhau).
Kết luận: bài closed-set gần bão hoà, phần đo lường có ý nghĩa của đồ án nằm ở
open-set (P4) và audit (P5).

## Feature cache (P3)

`outputs/features/ustc_seed42.npz` (602 MB), trích **một lần** cho toàn bộ 97.115 mẫu
bằng `scripts/extract_features.py` (15,6 phút). Từ P4 trở đi mọi detector chạy trên
cache, không chạy lại GPT-2.

| Khoá | Shape | Dtype | Nội dung |
|---|---|---|---|
| `sample_id` | (97115,) | int64 | Khớp thứ tự `manifest.csv`; nối bằng khoá này |
| `logits` | (97115, 12) | float32 | Điểm thô **trước** softmax cho 12 lớp known |
| `emb_last` | (97115, 768) | float32 | Hidden state lớp cuối tại **token cuối không phải pad** |
| `emb_mean` | (97115, 768) | float32 | Trung bình hidden state lớp cuối trên token không pad, có áp mask |

- `emb_last` là đúng vị trí classification head đọc. Lưu ý `transformers` 5.x áp
  `score` lên cả chuỗi rồi mới lấy vị trí token cuối (v4 lấy token cuối rồi mới `score`);
  vì `score` là Linear theo chiều cuối nên hai cách tương đương. Khoá bằng
  `tests/test_features.py` (so `score(emb_last)` với `model(...).logits`).
- Kiểm chứng cache không lệch: accuracy tính lại từ `logits` trên `closed_valid`
  = 0,9996498191, **khớp chính xác** con số lúc train. Không có NaN/Inf.
- `logits` là điểm thô, không phải xác suất: P4 tự áp softmax / nhiệt độ T, nhờ vậy
  chỉnh T cho Energy hay OpenMax **không cần** chạy lại model.
- Cache chứa cả 40.000 mẫu `open_*` (model chưa từng thấy) — cần cho unknown recall,
  AUROC, AUPR, OSCR và tỉ lệ phát hiện theo từng class open.

