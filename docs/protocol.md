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

## Open-set (P4): định nghĩa metric

Viết **trước khi** cài đặt, để mọi con số trong báo cáo có nghĩa xác định.

### Quy ước chung

- Mỗi detector trả về **anomaly score** `s`: **cao hơn = lạ hơn**. Quyết định:
  `unknown` (label 12) nếu `s > τ`, ngược lại giữ lớp `argmax` của logits.
- `τ` hiệu chỉnh **chỉ trên `closed_valid`**; không bao giờ dùng dữ liệu open (§6.3).
- Đánh giá trên `closed_test` (known) + `open_test` (unknown).
  `open_valid` để nguyên cho Background Class (P6), không dùng ở P4 (§6.2).
- Không gian nhãn khi đánh giá: `0..11` known, `12` unknown.

### Hai chế độ

| Khía cạnh | `reference` | `corrected` |
|---|---|---|
| Ngưỡng | phân vị 0,8 dạng nearest-rank như upstream (`TH_value=0.8`, loại ~20% known) | giữ 95% known (`keep_rate=0.95`) |
| Số láng giềng `K` | 5 như upstream USTC (`--K_number 5`), **gồm cả chính class đang xét** | 5, **loại chính class đang xét** |
| Mẫu dùng calibration K-LND | chỉ mẫu `closed_valid` **phân loại đúng**, score tính bằng nhánh láng giềng (giống upstream) | toàn bộ `closed_valid`, score cùng định nghĩa với lúc đánh giá |
| Tập láng giềng của K-LND | **gồm cả chính class đang xét** (lỗi upstream: `Values[i][i]` luôn = 0 nên luôn được chọn) | loại chính nó |
| K-LND 2/3 khi tính score | calibration/valid và `open_test` dùng K láng giềng; **`closed_test` cộng trên *tất cả* class** (bất đối xứng upstream) | cả `closed_test` lẫn `open_test` dùng **cùng một** tập K |
| Class rỗng | `TH = 10` cho class 0, class sau lấy TH class trước (magic number) | fallback về phân vị toàn cục |
| Mẫu số 0 (K-LND 3) | không chặn → NaN/inf (đã cài đúng: `guard_denominator=False`) | chặn bằng epsilon |
| Nhiệt độ `T` | 1 | 1 (xem ghi chú bên dưới) |

**Phạm vi `reference`:** `reference` tái hiện **encoding + detector + calibration**
của upstream, không tái hiện cách upstream chọn class known/open hay split.
Project dùng known ngẫu nhiên seed 42 (12/20 cho USTC, 4/10 cho DC) và split
stratified; upstream dùng known = `y < num_labels` (12 cho USTC, 4 cho DC theo
first-4) và split không stratified. Vì vậy chênh lệch tuyệt đối với số trong
`results_*.txt` của upstream là *discrepancy, chưa xác minh nguyên nhân*, không
phải "không tái lập được".

**Hệ quả của bất đối xứng `reference` (K-LND 2/3):** ngưỡng được học trên nhánh
láng giềng nhưng lại áp lên `closed_test` tính bằng nhánh *tất cả* class, nên
`FPR` trên `closed_test` gần 0 thay vì ~0,20; đồng thời score thô của known và
open không cùng thang đo, làm `AUROC`/`AUPR`/`FPR@95TPR` bị thổi phồng. Đây là
hành vi thật của upstream, không phải detector tốt hơn. Vì vậy khi đọc kết quả
`reference` K-LND 2/3 phải so cạnh `corrected` và ghi rõ nguyên nhân.

**MSP/Energy không có bản upstream.** Repo gốc chỉ cài K-LND; `reference` cho
MSP/Energy chỉ là *operating point kiểu upstream* (phân vị 0,8), không phải tái
lập code gốc. OpenMax cũng không có trong repo gốc, nên chỉ báo cáo `corrected`.

**Vì sao cả hai chế độ dùng `T = 1`:** dự định ban đầu là chọn `T` bằng NLL trên
`closed_valid`, nhưng với model này tiêu chí đó **suy biến**. Bằng chứng đo được trên
cache: top-probability trung vị trên `closed_valid` = **1,000000** (logit max 23,06,
margin trung vị 15,77), tức softmax đã bão hoà; giảm `T` chỉ làm NLL tiến sát 0, nên
nghiệm luôn rơi vào biên lưới. Không có tiêu chí nào chỉ dùng dữ liệu closed chọn được
`T > 1`. Vì vậy giữ `T = 1` ở cả hai chế độ và **báo cáo độ nhạy của MSP/Energy theo T**
như một *chẩn đoán*, ghi rõ không dùng để chọn ngưỡng hay chọn T.

### Định nghĩa từng metric

Ký hiệu: `K` = tập `closed_test`, `U` = tập `open_test`.

| Metric | Định nghĩa |
|---|---|
| `closed_argmax_accuracy` | Tỉ lệ `argmax` đúng trên `K` **trước** khi áp ngưỡng |
| `CCR` (closed-set correct classification rate) | Tỉ lệ mẫu `K` vừa **không bị từ chối** vừa **đúng lớp**. Đây là "Test accuracy Normal model_Closed_set" của upstream |
| `unknown_recall` (UCR) | Tỉ lệ mẫu `U` bị từ chối (`s > τ`). Đây là "Test accuracy Normal model_Open_set" của upstream |
| `FPR` | Tỉ lệ mẫu `K` bị từ chối nhầm = `1 − TNR` |
| `AUROC` | Diện tích dưới ROC với positive = unknown, dùng anomaly score thô (không phụ thuộc ngưỡng) |
| `AUPR` | Average precision, positive = unknown; kèm `prevalence = |U| / (|K|+|U|)` vì AUPR phụ thuộc tỉ lệ |
| `FPR@95TPR` | FPR nhỏ nhất đạt `unknown_recall ≥ 0,95`; quét toàn bộ ngưỡng trên score thô. Càng thấp càng tốt |
| `OSCR` | Trung bình điều hoà của `CCR` và `TNR` tại ngưỡng đã hiệu chỉnh: `2·CCR·TNR / (CCR + TNR)` |
| `balanced_F1` | macro-F1 của bài toán nhị phân known/unknown tại ngưỡng đã hiệu chỉnh. Báo ở **tỉ lệ tự nhiên** và **1:1** (lấy ngẫu nhiên `|K|` mẫu `U`, seed 42) |
| `confusion_matrix` | Ma trận 13 × 13 (12 known + unknown) trên `K ∪ U` |
| `open_class_detection` | Với từng class open (nhãn gốc), tỉ lệ bị từ chối. Giữ nhãn gốc xuyên suốt (§6.7) |
| `known_rejection_on_valid` | Tỉ lệ mẫu `closed_valid` bị từ chối. **Chẩn đoán chiều phân vị**: chế độ `reference` sẽ ra ~0,20, chế độ `corrected` ~0,05 |

**Ghi chú về ngưỡng theo lớp:** K-LND dùng ngưỡng riêng cho từng lớp dự đoán `d`
(giống upstream). Vì vậy `AUROC`, `AUPR`, `FPR@95TPR` tính trên score thô (không ngưỡng),
còn `CCR`, `unknown_recall`, `OSCR`, `balanced_F1`, `confusion_matrix` dùng ngưỡng
theo lớp. Khác biệt này được ghi rõ trong output.

**Ghi chú về AUPR:** vì `|U| = 20.000` lớn hơn `|K| = 8.568`, prevalence ≈ 0,70 nên
AUPR sẽ cao "tự nhiên"; luôn đọc kèm prevalence, không so sánh AUPR giữa hai tỉ lệ khác nhau.

### OpenMax (chỉ `corrected`)

Repo gốc không có code OpenMax, nên không tồn tại chế độ `reference` cho detector
này. Cài đặt ở `src/trafficllm/openset/openmax.py`, chỉ dùng `closed_train` để fit
và `closed_valid` để chọn ngưỡng; không nhìn dữ liệu open.

| Thành phần | Định nghĩa |
|---|---|
| Không gian khoảng cách | `emb_last` (hidden state lớp cuối tại token cuối không pad) |
| Mean mỗi class | Trung bình `emb_last` của mẫu `closed_train` **phân loại đúng** |
| Weibull | Fit `weibull_min` với `floc=0` trên khoảng cách Euclid từ mẫu `closed_train` đúng tới mean class đó |
| Hiệu chỉnh | Xếp hạng class theo logit giảm dần, lấy top `alpha = 3`; class hạng `r` bị phạt `log(1 − w_r)` với `w_r` là CDF Weibull của khoảng cách |
| Unknown logit | `logsumexp(logit_top + log(w))` trên top `alpha` |
| Score | `1 − max p_known` với `p = softmax([logits_revised, unknown_logit])`; lấy max **chỉ trên 12 lớp known** để xác suất unknown cao cho score cao |
| Ngưỡng | `keep_rate = 0,95`, hiệu chỉnh trên `closed_valid` như các detector khác |

Fallback: class không có mẫu `closed_train` đúng, hoặc fit Weibull thất bại/suy biến,
thì `w = 0` cho class đó (không hiệu chỉnh) và mean lấy centroid toàn cục. Cách này
tránh NaN và chia 0 nhưng được ghi rõ trong output (`openmax` chỉ có một bản `corrected`).



## Audit P5: `reference` vs `corrected` (USTC)

### Thiết kế 4 ô

`scripts/run_audit.py --config configs/openset_ustc_seed42.yaml` chạy cùng một
cache feature với 4 biến thể, để tách **sửa lỗi** khỏi **đổi operating point**:

| Biến thể | Mode | `keep_rate` | Ý nghĩa |
|---|---|---:|---|
| `reference_08` | reference | 0,8 | mặc định paper/repo |
| `reference_095` | reference | 0,95 | chỉ đổi operating point |
| `corrected_08` | corrected | 0,8 | chỉ sửa lỗi |
| `corrected_095` | corrected | 0,95 | cả hai |

Output: `outputs/metrics/audit_ustc_seed42.{json,csv}`. Chạy CPU ~15,6 s.

### Kết quả chính

| Detector | Ref08 CCR | Ref08 UCR | Ref08 AUROC | Ref08 FPR95 | Ref08 OSCR | Corr095 CCR | Corr095 UCR | Corr095 AUROC | Corr095 FPR95 | Corr095 OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSP | 0,8057 | 0,8277 | 0,8876 | 0,7443 | 0,8057 | 0,9545 | 0,7710 | 0,8876 | 0,7443 | 0,9545 |
| Energy | 0,8110 | 0,8183 | 0,9148 | 0,3979 | 0,8110 | 0,9503 | 0,7513 | 0,9148 | 0,3979 | 0,9503 |
| K-LND 1 | 0,8081 | 0,9284 | 0,9584 | 0,1912 | 0,8081 | 0,9437 | 0,7944 | 0,9584 | 0,1912 | 0,9437 |
| K-LND 2 | 0,9998 | 0,8276 | 1,0000 | 0,0000 | 0,9999 | 0,9497 | 0,6960 | 0,9291 | 0,3707 | 0,9497 |
| K-LND 3 | 0,9985 | 0,9090 | 0,9996 | 0,0006 | 0,9986 | 0,9447 | 0,7627 | 0,9600 | 0,2045 | 0,9447 |
| OpenMax | — | — | — | — | — | 0,9483 | 0,6674 | 0,8887 | 0,5158 | 0,9483 |

Tách riêng hai tác nhân (Δ = giá trị sau − giá trị trước):

| Detector | Δ bug-fix (ref08 → corr08): CCR / UCR / AUROC | Δ operating point (corr08 → corr095): CCR / UCR / AUROC |
|---|---|---|
| MSP | −0,0006 / +0,0001 / 0,0000 | +0,1494 / −0,0569 / 0,0000 |
| Energy | −0,0008 / 0,0000 / 0,0000 | +0,1401 / −0,0671 / 0,0000 |
| K-LND 1 | −0,0008 / +0,0001 / 0,0000 | +0,1364 / −0,1342 / 0,0000 |
| K-LND 2 | −0,1922 / +0,0425 / −0,0709 | +0,1422 / −0,1742 / 0,0000 |
| K-LND 3 | −0,1877 / +0,0002 / −0,0395 | +0,1339 / −0,1465 / 0,0000 |
| OpenMax | — | +0,1440 / −0,1200 / 0,0000 |

### Chẩn đoán và cách đọc

| Chẩn đoán | Kết quả |
|---|---|
| `same_branch` K-LND 2 | AUROC 0,9118 / FPR95 0,4664 / CCR 0,8088 / UCR 0,8276 (so với 1,0000 / 0,0000 / 0,9998 / 0,8276 khi `closed_test` cộng tất cả class) |
| `same_branch` K-LND 3 | AUROC 0,9585 / FPR95 0,2200 / CCR 0,8134 / UCR 0,9090 (so với 0,9996 / 0,0006 / 0,9985 / 0,9090) |
| `self_neighbor` K-LND 2 | without − with: ΔAUROC +0,0173, ΔFPR95 −0,0957, ΔCCR +0,0009, ΔUCR +0,0054 |
| `self_neighbor` K-LND 3 | without − with: ΔAUROC +0,0015, ΔFPR95 −0,0155, ΔCCR −0,0013, ΔUCR +0,0073 |
| `self_neighbor` K-LND 1 | 0 ở mọi metric (method 1 không dùng tập láng giềng) |
| `calibration_correct_only` | Δ ≈ 0; K-LND 2 chỉ lệch UCR −0,0007 vì correct-only chỉ dùng 8.564/8.567 mẫu valid phân loại đúng |
| `msp_saturation` | top-prob median 0,9999997; MSP median 2,79e−7; **80,35%** valid có MSP < 1e−6; ngưỡng median ~9,4e−7 (ref) / 1,65e−6 (corr) |

- **MSP/Energy và K-LND 1:** sửa lỗi gần như không đổi kết quả; chênh lệch
  `reference_08` → `corrected_095` chủ yếu là do đổi `keep_rate` 0,8 → 0,95.
  Với MSP, ngưỡng hiệu chỉnh rơi vào vùng bão hoà ~1e−6 nên `AUROC`/`FPR95`
  không đổi; `CCR` tăng và `UCR` giảm đúng như thiết kế operating point.
- **K-LND 2/3:** `reference_08` có `AUROC ≈ 1` và `FPR95 ≈ 0` **không phải vì
  detector tốt hơn**. Khi ép `closed_test` dùng cùng nhánh K láng giềng
  (`same_branch`), `AUROC` tụt còn 0,9118 / 0,9585 và `FPR95` tăng lên
  0,4664 / 0,2200. Phần "đẹp" còn lại là do bất đối xứng nhánh score của upstream.
- **Self-neighbor:** loại class đang xét khỏi tập láng giềng làm K-LND 2 tăng
  `AUROC` +0,0173 và giảm `FPR95` −0,0957; K-LND 3 nhỏ hơn. Đây là lỗi thật nhưng
  không phải nguyên nhân chính của `AUROC ≈ 1`.
- **Calibration correct-only:** gần như vô hại trên USTC (chỉ 3 mẫu khác), nên
  không thể quy chênh lệch lớn cho lỗi này.
- **Kết luận:** không so trực tiếp `AUROC` của `reference` K-LND 2/3 với
  `corrected`; dùng `same_branch` hoặc bảng `corrected` khi kết luận detector.

## DC (P5, run phụ)

DC dùng để audit **label leakage trong encoding**, không phải kết quả chính.

- Dữ liệu: 10 class video YouTube; mỗi trace 500 giá trị packet-count; 3.198 trace
  (2.509 train / 50 valid / 639 test gốc). Ta gộp rồi split lại stratified, seed 42:
  4 class known / 6 class open (40% closed như paper), `closed_train` 894,
  `closed_valid` 192, `closed_test` 192, `open_valid` 960, `open_test` 960.
- `reference`: mỗi giá trị `x` → pattern dài `floor(x/divisor)+1`, số bit 1 = độ
  dài pattern, bit 1/0 **phụ thuộc nhãn** (tái hiện upstream). Upstream **không cắt
  pattern ở 32 bit** (chỉ pad thêm 0 khi ngắn hơn) và đổi cả chunk cuối nếu không
  đủ 4 bit; ta đã sửa `_reference_pattern`/`_bits_to_hex` để khớp đúng hai chi tiết
  này. Code upstream dùng `divisor=50`, paper ghi 40; ta để `divisor` cấu hình
  được, mặc định 50 cho reference.
- `corrected`: cùng 32 bit / hex / space / `divisor`, nhưng pattern label-free:
  dồn đúng số bit 1 về bên trái rồi phần còn lại là 0. Đây là thay đổi tối thiểu
  để cô lập leakage, không thiết kế lại representation.
- Token length (GPT-2): min 500, p50 2.012, p95 2.390, max 3.923;
  **99,78%** trace vượt `max_len=1024` và bị truncate. Reference và corrected có
  cùng phân bố độ dài, nên so sánh vẫn công bằng nhưng chỉ trên phần prefix được
  model nhìn thấy. Đây là hạn chế phải nêu ở P7.
- Audit DC (`scripts/run_dc_audit.py`) so 3 evaluation:
  `reference_upstream` (keep 0,9, calibration correct-only, `magic_empty`),
  `reference_corrected_protocol` (cùng cache reference, keep 0,95, calibration
  toàn bộ valid, K-LND cùng nhánh), `corrected` (cache corrected + protocol corrected).
  `delta_encoding_isolated = corrected − reference_corrected_protocol` đo riêng
  tác động leakage, không lẫn với sửa protocol.
- Trạng thái: manifest + encoding + config + script + **kết quả thật** đã xong.
  Train reference: best epoch 2, `closed_valid` 0,9948, `closed_test` 1,0000;
  train corrected: best epoch 3, `closed_valid` 0,8229, `closed_test` 0,7812.
  Feature: `outputs/features/dc_seed42_{reference,corrected}.npz` (20 MB mỗi bản).

### Kết quả DC

| Evaluation | Detector | CCR | UCR | AUROC | FPR95 | OSCR |
|---|---|---:|---:|---:|---:|---:|
| reference_upstream | MSP | 0,9167 | 0,8198 | 0,5518 | 0,7917 | 0,9167 |
| reference_upstream | Energy | 0,9271 | 0,6094 | 0,6934 | 0,9323 | 0,9271 |
| reference_upstream | K-LND 1 | 0,9167 | 0,6833 | 0,9137 | 0,4219 | 0,9167 |
| reference_upstream | K-LND 2 | 0,9062 | 0,7198 | 0,8843 | 0,4896 | 0,9062 |
| reference_upstream | K-LND 3 | 0,9115 | 0,6865 | 0,9125 | 0,4219 | 0,9115 |
| reference_corrected_protocol | MSP | 0,9531 | 0,7833 | 0,5518 | 0,7917 | 0,9531 |
| reference_corrected_protocol | Energy | 0,9479 | 0,5979 | 0,6934 | 0,9323 | 0,9479 |
| reference_corrected_protocol | K-LND 1 | 0,9427 | 0,6542 | 0,9137 | 0,4219 | 0,9427 |
| reference_corrected_protocol | K-LND 2 | 0,9427 | 0,6594 | 0,8843 | 0,4896 | 0,9427 |
| reference_corrected_protocol | K-LND 3 | 0,9427 | 0,6677 | 0,9125 | 0,4219 | 0,9427 |
| reference_corrected_protocol | OpenMax | 0,9583 | 0,7427 | 0,9572 | 0,2396 | 0,9583 |
| corrected | MSP | 0,7604 | 0,1167 | 0,6817 | 0,8177 | 0,8355 |
| corrected | Energy | 0,7500 | 0,1271 | 0,6575 | 0,7031 | 0,8229 |
| corrected | K-LND 1 | 0,7604 | 0,1042 | 0,6429 | 0,9115 | 0,8418 |
| corrected | K-LND 2 | 0,7604 | 0,0917 | 0,7075 | 0,6927 | 0,8418 |
| corrected | K-LND 3 | 0,7604 | 0,0917 | 0,6590 | 0,8802 | 0,8418 |
| corrected | OpenMax | 0,7552 | 0,1094 | 0,6028 | 0,9115 | 0,8386 |

Tác động riêng của encoding (corrected − reference_corrected_protocol):

| Detector | ΔCCR | ΔUCR | ΔAUROC | ΔFPR95 | ΔOSCR |
|---|---:|---:|---:|---:|---:|
| MSP | −0,1927 | −0,6667 | **+0,1299** | +0,0260 | −0,1176 |
| Energy | −0,1979 | −0,4708 | −0,0359 | −0,2292 | −0,1250 |
| K-LND 1 | −0,1823 | −0,5500 | −0,2708 | +0,4896 | −0,1009 |
| K-LND 2 | −0,1823 | −0,5677 | −0,1768 | +0,2031 | −0,1009 |
| K-LND 3 | −0,1823 | −0,5760 | −0,2534 | +0,4583 | −0,1009 |
| OpenMax | −0,2031 | −0,6333 | −0,3544 | +0,6719 | −0,1197 |

### Đọc kết quả DC

- **Leakage làm open-set "dễ" hơn rất nhiều.** Cùng protocol, bỏ nhãn khỏi
  encoding làm UCR rơi từ 0,60–0,82 xuống 0,09–0,13 ở mọi detector; CCR cũng
  giảm ~0,18–0,20. Đây là bằng chứng định lượng cho lỗi label leakage: mẫu open
  mang pattern riêng của class chưa thấy, nên bị reject dễ.
- **Leakage cũng làm closed-set dễ hơn.** `closed_test` accuracy 1,0000 (reference)
  vs 0,7812 (corrected). Vì vậy không được đọc 0,99+ của reference là năng lực
  biểu diễn thật.
- **MSP là ngoại lệ về AUROC:** AUROC tăng +0,1299 khi bỏ leakage (0,5518 →
  0,6817) nhưng UCR vẫn sụp (−0,6667). Nghĩa là leakage không nhất thiết làm
  *ranking toàn cục* tốt hơn; nó làm *ngưỡng per-class* reject open dễ hơn.
  Đây là lý do phải đọc AUROC và UCR/OSCR cùng nhau, không thay thế nhau.
- **K-LND/OpenMax:** cả AUROC lẫn UCR đều giảm sau khi bỏ leakage (K-LND1
  AUROC 0,9137 → 0,6429; OpenMax 0,9572 → 0,6028), nên với các detector này
  leakage thổi phồng cả ranking lẫn operating point.
- **Phát hiện theo class open:** reference rất cao ở class 1, 2 (≈0,99–1,00),
  class 8 (0,71–0,97) nhưng thấp ở class 3, 9 (0,04–0,67); corrected gần như đều
  ~0,03–0,20. Tác động leakage phụ thuộc class, không đồng đều.
- **Kết luận:** DC chỉ là run phụ audit. Số reference không dùng làm kết quả
  chính; số corrected cho thấy open-set trên DC khó hơn nhiều khi không có nhãn
  trong encoding. Cần nêu truncation 99,78% >1024 token khi so với paper.

## P6 — Mở rộng

### Ablation: logits vs embedding (USTC, seed 42, corrected, keep_rate 0,95)

So sánh ba không gian feature cho K-LND 1/2/3. Lớp dự đoán luôn lấy từ logits
(argmax); chỉ feature tính khoảng cách/anomaly score thay đổi.

| Feature | Detector | UCR | AUROC | FPR@95 | OSCR | balanced F1 (1:1) |
|---|---|---:|---:|---:|---:|---:|
| logits | K-LND 1 | 0,7944 | 0,9584 | 0,1912 | 0,9437 | 0,8706 |
| logits | K-LND 2 | 0,6961 | 0,9291 | 0,3707 | 0,9497 | 0,8208 |
| logits | K-LND 3 | 0,7627 | 0,9600 | 0,2045 | 0,9447 | 0,8547 |
| emb_last | K-LND 1 | 0,6686 | 0,9081 | 0,4786 | 0,9483 | 0,8053 |
| emb_last | K-LND 2 | 0,6196 | 0,9106 | 0,4786 | 0,9518 | 0,7809 |
| emb_last | K-LND 3 | 0,6612 | 0,9224 | 0,4245 | 0,9512 | 0,8029 |
| emb_mean | K-LND 1 | 0,4182 | 0,7387 | 0,5175 | 0,9484 | 0,6594 |
| emb_mean | K-LND 2 | 0,4367 | 0,7459 | 0,8075 | 0,9523 | 0,6727 |
| emb_mean | K-LND 3 | 0,4377 | 0,7604 | 0,5292 | 0,9477 | 0,6704 |

**Đọc kết quả:**
- **Logits tốt nhất cho open-set.** K-LND 1 trên logits đạt AUROC 0,9584 và UCR
  0,7944; `emb_last` giảm ~5 pp AUROC và ~13 pp UCR; `emb_mean` mất ~20 pp AUROC
  và ~38 pp UCR.
- **CCR gần như không đổi** (~0,944–0,952) vì threshold giữ cùng keep_rate; sự khác
  biệt chủ yếu ở khả năng reject mẫu open.
- **FPR@95TPR** tăng mạnh khi dùng embedding, nghĩa là cần chấp nhận nhiều dương
  tính giả hơn để giữ 95% phát hiện open.
- Lý giải: logits (12-dim) là không gian compact, hướng trực tiếp theo class
  boundary; `emb_last` (768-dim) giàu thông tin nhưng nhiễu chiều cao hơn;
  `emb_mean` (768-dim) trung bình qua vị trí token nên mất tín hiệu cục bộ.

### Random-init GPT-2 (đối chứng, seed 42)

So sánh GPT-2 small pretrained vs cùng kiến trúc khởi tạo ngẫu nhiên
(`GPT2Config.from_pretrained` + `GPT2ForSequenceClassification` không load trọng số).
Mục đích: tách contribution của pretrained knowledge khỏi kiến trúc transformer.

| Detector | Pretrained (CCR / UCR / AUROC / FPR95) | Random-init (CCR / UCR / AUROC / FPR95) |
|---|---|---|
| MSP | 0,9545 / 0,7710 / 0,8876 / 0,7443 | 0,9471 / 0,8134 / 0,9376 / 0,3023 |
| Energy | 0,9503 / 0,7513 / 0,9148 / 0,3979 | 0,9482 / 0,8376 / 0,9370 / 0,3200 |
| K-LND 1 | 0,9437 / 0,7944 / 0,9584 / 0,1912 | 0,9458 / 0,8523 / 0,9598 / 0,2871 |
| K-LND 2 | 0,9497 / 0,6960 / 0,9291 / 0,3707 | 0,9475 / 0,8559 / **0,9707** / 0,1675 |
| K-LND 3 | 0,9447 / 0,7627 / 0,9600 / 0,2045 | 0,9448 / 0,8549 / 0,9622 / 0,2641 |
| OpenMax | 0,9483 / 0,6674 / 0,8887 / 0,5158 | 0,9450 / 0,8569 / **0,9611** / 0,2586 |

**Đọc kết quả:**
- **Closed-set:** random-init chỉ thấp hơn 0,0002 accuracy (0,9995 vs 0,9998) →
  kiến trúc transformer đủ mạnh để phân loại USTC ngay cả không có pretrained.
- **Open-set:** random-init **tốt hơn pretrained** ở hầu hết detector:
  UCR tăng +4–19 pp; AUROC tăng hoặc ngang; FPR@95 giảm (tốt hơn) ở MSP,
  Energy, K-LND2, OpenMax.
- **Giả thuyết:** pretrained knowledge về ngôn ngữ tự nhiên không khớp với
  phân phối lưu lượng mạng. Random-init học biểu diễn "thuần" cho traffic, không
  bị kéo bởi ngữ nghĩa sẵn có → khoảng cách giữa class rõ ràng hơn → phát hiện
  lạ tốt hơn. Đây là phát hiện quan trọng: paper claim pretrained giúp nhưng
  thực tế trên USTC thì không, ít nhất với encoding hiện tại.
### Random-init mean ± std (3 seed: 42, 43, 44)

| Detector | Pretrained AUROC | Random-init AUROC | Pretrained UCR | Random-init UCR |
|---|---|---|---|---|
| K-LND 1 | 0,9648 ± 0,0056 | 0,9635 ± 0,0119 | 0,8630 ± 0,0683 | 0,8451 ± 0,0406 |
| K-LND 2 | 0,9484 ± 0,0183 | **0,9736 ± 0,0044** | 0,8327 ± 0,1226 | 0,8465 ± 0,0397 |
| K-LND 3 | 0,9659 ± 0,0051 | 0,9659 ± 0,0105 | 0,8542 ± 0,0860 | 0,8474 ± 0,0409 |

Kết luận mean±std xác nhận: random-init **ngang hoặc tốt hơn** pretrained.
K-LND 2 random-init vượt rõ (+2,5 pp AUROC, std nhỏ hơn 4×).
Random-init có std thấp hơn pretrained ở UCR → ổn định hơn.

### Seed sweep pretrained (3 seed: 42, 43, 44)

| Detector | CCR (mean±std) | UCR (mean±std) | AUROC (mean±std) |
|---|---|---|---|
| K-LND 1 | 0,9481 ± 0,0038 | 0,8630 ± 0,0683 | 0,9648 ± 0,0056 |
| K-LND 2 | 0,9495 ± 0,0009 | 0,8327 ± 0,1226 | 0,9484 ± 0,0183 |
| K-LND 3 | 0,9491 ± 0,0038 | 0,8542 ± 0,0860 | 0,9659 ± 0,0051 |

**Đọc kết quả:**
- AUROC ổn định (std < 0,006 cho K-LND 1/3), CCR cũng ổn (std < 0,004).
- UCR dao động lớn hơn (std 0,07–0,12) → operating point nhạy cảm với seed hơn
  ranking toàn cục (AUROC).
- Closed-test accuracy: 0,9997 ± 0,0001 (gần như hoàn hảo ở mọi seed).

### AWF/DF adapter

Adapter streaming/mmap cho AWF và DF; không load toàn bộ vào RAM.
Encoding theo paper; reference tái hiện bug upstream (AWF bỏ pair cuối),
corrected giữ pair cuối.

- AWF encoding: `src/trafficllm/encoding/awf.py`; manifest + parquet đã có
  (`data/processed/awf_seed42/`); 280.000 trace, 200 known class + 100k open
  (label 200); **100% trace 2.500 token → truncate 1024**.
- DF encoding: `src/trafficllm/encoding/df.py` (cùng encoding AWF); manifest
  builder `scripts/build_df_manifest.py`; 95 class (60 known upstream / 35 open);
  **chưa encode + chưa train**.
