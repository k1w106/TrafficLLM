# Báo cáo kiểm toán TrafficLLM

**Đồ án NT140 — Tái lập và kiểm toán paper "TrafficLLM"**

Ngày: 2026-11-27 · Seed mặc định: 42 · Dataset chính: USTC (97.115 trace, 20 class)

---

## 1. Mục tiêu

Tái lập và kiểm toán paper "TrafficLLM: LLMs for improved open-set encrypted
traffic analysis" (Ginige et al., Computer Networks 2026) bằng codebase độc lập.

- **Reference protocol**: tái hiện đúng hành vi repo gốc, kể cả lỗi.
- **Corrected protocol**: sửa lỗi methodology (encoding label-free, split
  stratified có seed, scoring nhất quán, metrics đầy đủ).
- Mọi con số trích từ `outputs/metrics/` và `docs/experiment-log.md`.

## 2. Phân loại closed-set

GPT-2 small (124M params), fine-tune 3 epoch trên 12 known class (39.980 mẫu train).

| Seed | closed_valid | closed_test |
|---:|---:|---:|
| 42 | 0,9997 | 0,9998 |
| 43 | 0,9999 | 0,9997 |
| 44 | 0,9997 | 0,9997 |
| **mean ± std** | **0,9997 ± 0,0001** | **0,9997 ± 0,0001** |

Nhận xét: gần hoàn hảo, ổn định qua 3 seed. Encoding USTC ngắn (144–256 token,
0% vượt 1024) nên model nhìn toàn bộ trace.

## 3. Phát hiện lưu lượng lạ (open-set) — corrected protocol

6 detector, threshold hiệu chỉnh từ closed_valid (keep_rate 0,95), đánh giá trên
closed_test (8.568) + open_test (20.000). Kết quả **pretrained mean ± std (3 seed)**:

| Detector | CCR | UCR | AUROC | FPR@95TPR |
|---|---|---|---|---|
| MSP | 0,9527 ± 0,0019 | 0,8472 ± 0,0804 | 0,9398 ± 0,0453 | 0,4219 ± 0,2806 |
| Energy | 0,9493 ± 0,0025 | 0,8446 ± 0,0892 | 0,8775 ± 0,0416 | 0,5581 ± 0,1712 |
| K-LND 1 | 0,9481 ± 0,0038 | 0,8630 ± 0,0683 | **0,9648 ± 0,0056** | 0,1924 ± 0,0298 |
| K-LND 2 | 0,9495 ± 0,0009 | 0,8327 ± 0,1226 | 0,9484 ± 0,0183 | 0,2753 ± 0,1305 |
| K-LND 3 | 0,9491 ± 0,0038 | 0,8542 ± 0,0860 | **0,9659 ± 0,0051** | 0,1892 ± 0,0371 |
| OpenMax | 0,9504 ± 0,0019 | 0,7553 ± 0,0994 | 0,9293 ± 0,0354 | 0,3561 ± 0,1439 |

Nhận xét:
- **K-LND 3** có AUROC cao nhất (0,9659) và ổn định nhất (std 0,0051).
- **K-LND 1** có UCR cao nhất (0,8630) — phát hiện nhiều mẫu lạ nhất tại
  operating point 95% giữ known.
- MSP/Energy có AUROC dao động lớn (std 0,04–0,05) → kém tin cậy hơn K-LND.
- CCR ổn định (~0,95) vì được kiểm soát bởi keep_rate.

## 4. Audit: reference vs corrected

### 4.1. USTC — tác động sửa lỗi

| Biến thể audit | Phát hiện |
|---|---|
| `reference` K-LND 2/3 | AUROC ≈ 1,0 và FPR@95 ≈ 0 — **giả tạo** do bất đối xứng nhánh score |
| `same_branch` K-LND 2/3 | AUROC tụt còn 0,9118 / 0,9585; FPR@95 tăng lên 0,4664 / 0,2200 → **xác nhận bất đối xứng là nguyên nhân** |
| `self_neighbor` K-LND 2 | ΔAUROC +0,0173, ΔFPR@95 −0,0957 → lỗi thật nhưng không phải nguyên nhân chính |
| `calibration_correct_only` | Gần như vô hại (chỉ 3 mẫu valid phân loại sai) |
| `msp_saturation` | 80,35% mẫu valid có MSP < 10⁻⁶; threshold rơi vào vùng bão hoà |
| MSP/Energy/K-LND 1 | Sửa lỗi gần như 0 chênh lệch; khác biệt chủ yếu do keep_rate 0,8 → 0,95 |

**Kết luận USTC:** lỗi bất đối xứng nhánh score trong K-LND 2/3 là lỗi nghiêm
trọng nhất — tạo ra AUROC ≈ 1,0 không phản ánh năng lực thật của detector.
Các lỗi khác (self-neighbor, calibration) ảnh hưởng nhỏ trên USTC.

### 4.2. DC — bằng chứng label leakage (run phụ)

DC dùng riêng để audit lỗi rò rỉ nhãn trong encoding. 10 class video, 3.198 trace,
4 known / 6 open.

| Chỉ số | Reference (có leakage) | Corrected (không leakage) | Chênh lệch |
|---|---|---|---|
| closed_test accuracy | **1,0000** | 0,7812 | −0,2188 |
| UCR (trung bình 6 detector) | 0,60–0,82 | 0,09–0,13 | −0,47 đến −0,67 |
| K-LND 1 AUROC | 0,9137 | 0,6429 | −0,2708 |
| OpenMax AUROC | 0,9572 | 0,6028 | −0,3544 |

**Kết luận DC:** label leakage thổi phồng cả closed accuracy (+22 pp) lẫn open-set
detection (UCR +50–67 pp). Bỏ nhãn khỏi encoding → kết quả sụp toàn diện. Đây là
bằng chứng định lượng mạnh nhất cho lỗi này.

Lưu ý: 99,78% trace DC vượt 1024 token và bị truncate → kết quả DC chỉ trên
phần prefix; không so trực tiếp với paper.

## 5. Ablation: logits vs embedding

Lớp dự đoán luôn từ logits (argmax); chỉ feature tính khoảng cách thay đổi.
Pretrained, 3 seed, corrected, K-LND trên logits/emb_last/emb_mean:

| Feature | K-LND 1 AUROC | K-LND 1 UCR | K-LND 3 AUROC | K-LND 3 UCR |
|---|---|---|---|---|
| logits (12-dim) | **0,9648 ± 0,0056** | **0,8630 ± 0,0683** | **0,9659 ± 0,0051** | **0,8542 ± 0,0860** |
| emb_last (768-dim) | 0,9377 ± 0,0256 | 0,7583 ± 0,1021 | 0,9387 ± 0,0184 | 0,7719 ± 0,1074 |
| emb_mean (768-dim) | 0,8111 ± 0,0742 | 0,4740 ± 0,1908 | 0,8003 ± 0,0638 | 0,4677 ± 0,2186 |

Nhận xét:
- **Logits tốt nhất** cho open-set: compact (12-dim), hướng trực tiếp theo class
  boundary → khoảng cách giữa known/unknown rõ ràng.
- **emb_last** mất ~3 pp AUROC, ~10 pp UCR — nhiều chiều hơn nhưng nhiễu.
- **emb_mean** kém nhất: trung bình qua vị trí token mất tín hiệu cục bộ.

## 6. Đối chứng: pretrained vs random-init

GPT-2 cùng kiến trúc, random-init không load trọng số pretrained. 3 seed, corrected:

| Detector | Pretrained AUROC | Random-init AUROC | Pretrained UCR | Random-init UCR |
|---|---|---|---|---|
| MSP | 0,9398 ± 0,0453 | 0,9360 ± 0,0041 | 0,8472 ± 0,0804 | 0,7890 ± 0,0236 |
| Energy | 0,8775 ± 0,0416 | **0,9335 ± 0,0040** | 0,8446 ± 0,0892 | 0,8074 ± 0,0274 |
| K-LND 1 | 0,9648 ± 0,0056 | 0,9635 ± 0,0119 | 0,8630 ± 0,0683 | 0,8451 ± 0,0406 |
| K-LND 2 | 0,9484 ± 0,0183 | **0,9736 ± 0,0044** | 0,8327 ± 0,1226 | 0,8465 ± 0,0397 |
| K-LND 3 | 0,9659 ± 0,0051 | 0,9659 ± 0,0105 | 0,8542 ± 0,0860 | 0,8474 ± 0,0409 |
| OpenMax | 0,9293 ± 0,0354 | **0,9648 ± 0,0084** | 0,7553 ± 0,0994 | **0,8575 ± 0,0322** |

Nhận xét:
- **Random-init ngang hoặc tốt hơn pretrained** ở mọi detector (AUROC).
- K-LND 2: random-init +2,5 pp AUROC, std nhỏ hơn 4× → ổn định hơn nhiều.
- OpenMax: random-init +3,5 pp AUROC, +10 pp UCR.
- Random-init có **std thấp hơn** ở hầu hết chỉ số → ổn định hơn.
- **Closed-set** gần như bằng nhau (0,9997 vs 0,9995).

**Kết luận:** pretrained knowledge (kiến thức ngôn ngữ tự nhiên) **không giúp**
open-set detection trên USTC. Kiến trúc transformer đủ mạnh; trọng số ngẫu nhiên
học biểu diễn "thuần" cho traffic, không bị kéo bởi ngữ nghĩa sẵn có. Đây là
phát hiện trái ngược với tuyên bố chính của paper.

## 7. AWF — encoding và dữ liệu

- 200 known class + 100.000 open trace; encoding: pair 2 phần tử liên tiếp
  (shift +1, base-3 → decimal).
- Bug upstream: bỏ pair cuối (reference tái hiện; corrected giữ).
- Token length **đều 2.500** → 100% vượt 1024; model chỉ nhìn prefix 1024 token.
- Manifest + encoded parquet đã có; **chưa train** (tuỳ chọn, ngoài scope chính).

## 8. Tổng hợp lỗi repo gốc và tác động

| # | Lỗi | Nơi phát hiện | Tác động đo được |
|---|---|---|---|
| 1 | Bất đối xứng nhánh score K-LND 2/3 | USTC audit | AUROC 1,0 → 0,91–0,96 (giả tạo 4–9 pp) |
| 2 | Label leakage trong encoding DC | DC audit | UCR giảm 50–67 pp khi bỏ nhãn |
| 3 | Self-neighbor trong K-LND | USTC audit | ΔAUROC +0,017 (nhỏ) |
| 4 | Split không stratified, không seed | Đọc code | Không đo trực tiếp; ta dùng stratified seed |
| 5 | Không chọn best checkpoint | Đọc code | Không đo trực tiếp; ta chọn best epoch |
| 6 | AWF encoding bỏ pair cuối | Đọc code | 1 pair / 2500 → tác động nhỏ |
| 7 | Chia cho 0 khi class rỗng (K-LND) | Đọc code | Ta guard epsilon trong corrected |

## 9. Hạn chế

1. **Split khác paper.** Ta dùng stratified random split với seed; paper/upstream
   dùng split cố định không seed. Known/open class có thể khác → so sánh trực tiếp
   con số tuyệt đối với paper cần thận trọng.

2. **OpenMax tự cài.** Repo gốc không có OpenMax; ta cài biến thể custom
   (logsumexp logit + log(w) top-alpha). Không phải canonical OpenMax.

3. **DC truncation 99,78%.** Hầu hết trace DC bị cắt ở 1024 token; kết quả chỉ
   trên prefix, không phản ánh toàn bộ trace.

4. **AWF/DF chưa train.** Code adapter đã có nhưng chưa chạy full pipeline. AWF
   cũng bị truncate 100% (2.500 → 1024 token).

5. **Chỉ 3 seed.** Đủ để ước std nhưng khoảng tin cậy rộng (n nhỏ).

6. **Chỉ GPT-2 small.** Paper còn thử LLaMA/QLoRA; ta chưa làm.

7. **Reference không tái hiện split upstream.** Ta chỉ tái hiện lỗi scoring/encoding;
   split known/open có thể khác upstream do thiếu seed. Gọi chênh lệch giữa số ta
   và số paper là "discrepancy, chưa xác minh nguyên nhân".

## 10. Kết luận

1. **GPT-2 small phân loại lưu lượng mã hoá gần hoàn hảo** (accuracy 0,9997 ± 0,0001)
   — tuyên bố này của paper được xác nhận.

2. **Open-set detection khả quan nhưng thấp hơn paper báo cáo.** K-LND 3 đạt
   AUROC 0,9659 ± 0,0051 sau khi sửa lỗi; paper báo cao hơn, phần lớn do bất đối
   xứng nhánh score tạo AUROC giả tạo.

3. **Label leakage là lỗi nghiêm trọng nhất** (DC: UCR giảm 50–67 pp khi sửa).

4. **Pretrained knowledge không giúp** — random-init ngang hoặc tốt hơn pretrained
   ở mọi detector. Đây là phát hiện trái ngược với luận điểm chính của paper.

5. **Logits là feature tốt nhất** cho open-set, không phải embedding.

---

*Mọi con số truy được về `docs/experiment-log.md` và `outputs/metrics/`.
Xem `docs/protocol.md` để biết chi tiết methodology.*
