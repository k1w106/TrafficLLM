# TrafficLLM Audit

Đồ án phân loại encrypted traffic theo hướng open-set: tái lập **có kiểm toán** ý tưởng
của paper TrafficLLM (Ginige và cộng sự, *Computer Networks*, 2026) bằng codebase độc lập.

Mục tiêu:

- Tái lập protocol của paper/repo gốc (`mode="reference"`, kể cả lỗi) và một protocol đã
  sửa (`mode="corrected"`), rồi định lượng chênh lệch giữa hai bên.
- Pipeline tiền xử lý **không dùng nhãn**.
- So sánh logits và embedding cho open-set detection.
- Đánh giá bằng closed-set accuracy, unknown recall, AUROC, FPR@95TPR, AUPR, OSCR.

Phạm vi: dataset **USTC**, backbone **GPT-2 small**, detector **MSP / Energy / K-LND 1-2-3 /
OpenMax**. Chi tiết thiết kế: `AGENTS.md` và `docs/protocol.md`.

## Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| P0 | Môi trường, khung repo | xong |
| P1 | Manifest USTC (12 known / 8 open, seed 42) | xong |
| P2 | Encoding không dùng nhãn + báo cáo độ dài token | xong |
| P3 | Fine-tune GPT-2 + cache feature cho 97.115 mẫu | xong |
| P4 | Detector open-set + metrics trên cache | xong — MSP, Energy, K-LND 1/2/3, OpenMax; runner + test + docs |
| P5 | Audit `reference` vs `corrected` trên cùng cache, run DC riêng | xong — audit USTC + DC run phụ |
| P6 | Mở rộng: AWF/DF, GPT-2 random-init, logits vs embedding, ≥3 seed | chưa làm |
| P7 | Báo cáo `docs/audit-report.md`, hình, bảng, hạn chế | chưa làm |

Chi tiết tiêu chí "xong" của từng phase: `AGENTS.md` §8.

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate

# PyTorch cài riêng (bản +cuXXX chỉ có trên index của PyTorch, không có trên PyPI)
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126   # có GPU
# pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu    # chỉ CPU

pip install -r requirements.lock.txt      # phần còn lại, đã ghim version
```

Chạy test (lần đầu cần internet để tải tokenizer `gpt2`, ~1 MB):

```bash
PYTHONPATH=src pytest -q
```

## Dữ liệu và artifact (KHÔNG nằm trong git)

Repo này chỉ chứa code + metadata nhỏ (~3 MB). Ba thứ dưới đây không nằm trong git;
**chỉ 2 file artifact là bắt buộc** để chạy P4:

| Thứ cần | Kích thước | Lấy ở đâu |
|---|---:|---|
| `data/raw/.../USTC/` (chỉ khi dựng lại từ đầu) | ~118 MB | link Drive bên dưới |
| `outputs/checkpoints/ustc_gpt2_seed42.pt` | 475 MB | GitHub Release tag `p3-artifacts` |
| `outputs/features/ustc_seed42.npz` | 575 MB | GitHub Release tag `p3-artifacts` |

Dataset gốc (không bắt buộc — P4 chỉ chạy trên cache): zip **232 MB**, chứa cả 5 dataset,
giải nén hết là 11 GB, nên **chỉ lấy USTC**:

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1-MVfxyHdQeUguBmYrIIw1jhMVSqxXQgO" -O /tmp/trafficllm_data.zip
unzip -o /tmp/trafficllm_data.zip 'data/USTC/*' -d /tmp/ustc_extract
mkdir -p data/raw/trafficllm_reference/trafficllm_data
mv /tmp/ustc_extract/data/USTC data/raw/trafficllm_reference/trafficllm_data/
```

Tải artifact (bỏ qua bước train 72 phút):

```bash
mkdir -p outputs/checkpoints outputs/features
BASE=https://github.com/k1w106/TrafficLLM/releases/download/p3-artifacts
wget -P outputs/checkpoints $BASE/ustc_gpt2_seed42.pt
wget -P outputs/features    $BASE/ustc_seed42.npz
```

Kiểm tra file tải xong (phải khớp):

```text
6eae3fa539258b3f5b84e23ed05a7d1537daecb6a2e29fdf308d1bf4cd13dec4  ustc_gpt2_seed42.pt
14af43e95d57faa7f2b1e2bb9f158cda5ca28d11692e3205f7fc0b3fbe060a5f  ustc_seed42.npz
```

Chỉ cần checkpoint + feature là đủ để chạy P4 (mọi detector chạy trên cache).
`manifest.csv` + `metadata.json` đã có sẵn trong git; `traces.npy` và
`encoded_gpt2.parquet` chỉ cần nếu muốn dựng lại từ đầu.

Kiểm tra cache đúng là của model này (10 giây):

```bash
PYTHONPATH=src python3 -c "
import numpy as np, pandas as pd
m = pd.read_csv('data/processed/ustc_seed42/manifest.csv')
d = np.load('outputs/features/ustc_seed42.npz')
print('khớp sample_id:', np.array_equal(d['sample_id'], m.sample_id.values))
va = m.index[m.partition == 'closed_valid'].values
acc = (d['logits'][va].argmax(1) == m.model_label.values[va]).mean()
print('valid accuracy từ cache:', round(float(acc), 6), '(phải là 0.99965)')
"
```

> Ghi chú cho người upload: nếu GitHub Release không dùng được, đẩy 2 file lên
> HuggingFace (`hf upload <user>/trafficllm-ustc-seed42 <file>`) rồi sửa bảng trên.
> Không dùng Git LFS: free chỉ 1 GB và một lần pull đã gần hết quota.

## Tái lập từ số 0 (nếu không tải artifact)

Thứ tự bắt buộc — mỗi bước dùng output của bước trước:

```bash
# 1. Manifest: gán 12 class known / 8 class open, seed 42  (~10 s)
python3 scripts/build_ustc_manifest.py \
  --source-dir data/raw/trafficllm_reference/trafficllm_data/USTC \
  --output-dir data/processed/ustc_seed42 \
  --known-count 12 --known-mode random --seed 42

# 2. Encoding char-level cho GPT-2, không dùng nhãn  (~2 phút)
python3 scripts/build_ustc_encoding.py --data-dir data/processed/ustc_seed42

# 3. Fine-tune GPT-2 small, chọn checkpoint theo closed_valid  (~72 phút, GPU 6 GB)
python3 scripts/train_gpt2.py --config configs/ustc_gpt2_seed42.yaml

# 4. Trích và cache logits + embedding cho toàn bộ 97.115 mẫu  (~16 phút)
python3 scripts/extract_features.py --config configs/ustc_gpt2_seed42.yaml
```

Báo cáo độ dài token (P2):

```bash
python3 scripts/analyze_token_length.py
```

## Kết quả đã có

- Encoding: 97.115 trace, 144–256 token, **0%** vượt `max_len=1024` → không mẫu nào bị cắt.
- Closed-set: `closed_valid` 0,999650; `closed_test` **0,999767** (8.566/8.568).
- Cache feature: accuracy tính lại từ `logits` khớp chính xác con số lúc train.

### P4 — open-set trên cache

Chạy `python3 scripts/run_openset.py --config configs/openset_ustc_seed42.yaml`
(16,4 s trên CPU, không chạy lại GPT-2). K=5, `keep_rate` 0,8 (`reference`) / 0,95
(`corrected`), OpenMax fit trên `closed_train`.

| Detector | Mode | CCR | UCR | AUROC | FPR@95TPR | OSCR |
|---|---|---:|---:|---:|---:|---:|
| K-LND 1 | corrected | 0,9437 | 0,7944 | 0,9584 | 0,1912 | 0,9437 |
| K-LND 2 | corrected | 0,9497 | 0,6960 | 0,9291 | 0,3707 | 0,9497 |
| K-LND 3 | corrected | 0,9447 | 0,7627 | 0,9600 | 0,2045 | 0,9447 |
| OpenMax | corrected | 0,9483 | 0,6674 | 0,8887 | 0,5158 | 0,9483 |

- `reference` K-LND 1 (cùng định nghĩa score ở hai nhánh): CCR 0,8081 / UCR 0,9284.
- `reference` K-LND 2/3 có FPR≈0 và AUROC≈1,0 **do bất đối xứng nhánh score của
  upstream** (ngưỡng học trên nhánh láng giềng, `closed_test` lại cộng trên tất cả
  class) — không phải detector tốt hơn. Chi tiết: `docs/protocol.md`.
- Class open 2 bị phát hiện rất kém ở mọi detector (UCR 0,00–0,29); đây là điểm
  cần nêu trong audit P5/P7.
- Output: `outputs/metrics/openset_ustc_seed42.json` + `.csv`; 40 test pass.

### P5 — audit `reference` vs `corrected` (USTC)

Chạy `python3 scripts/run_audit.py --config configs/openset_ustc_seed42.yaml`
(15,6 s CPU). Script chạy 4 biến thể: `reference_08`, `reference_095`,
`corrected_08`, `corrected_095`, để tách **sửa lỗi** khỏi **đổi operating point**.

| Detector | Ref08 CCR | Ref08 UCR | Ref08 AUROC | Corr095 CCR | Corr095 UCR | Corr095 AUROC |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0,8057 | 0,8277 | 0,8876 | 0,9545 | 0,7710 | 0,8876 |
| Energy | 0,8110 | 0,8183 | 0,9148 | 0,9503 | 0,7513 | 0,9148 |
| K-LND 1 | 0,8081 | 0,9284 | 0,9584 | 0,9437 | 0,7944 | 0,9584 |
| K-LND 2 | 0,9998 | 0,8276 | 1,0000 | 0,9497 | 0,6960 | 0,9291 |
| K-LND 3 | 0,9985 | 0,9090 | 0,9996 | 0,9447 | 0,7627 | 0,9600 |
| OpenMax | — | — | — | 0,9483 | 0,6674 | 0,8887 |

- **MSP/Energy/K-LND 1:** sửa lỗi gần như không đổi kết quả; chênh lệch chủ yếu do
  `keep_rate` 0,8 → 0,95.
- **K-LND 2/3:** `reference` có AUROC≈1, FPR≈0 **không phải detector tốt hơn**.
  Khi ép `closed_test` dùng cùng nhánh K láng giềng (`same_branch`), AUROC còn
  0,9118 / 0,9585 và FPR@95TPR tăng lên 0,4664 / 0,2200. Nguyên nhân chính là bất
  đối xứng nhánh score của upstream.
- **Self-neighbor K-LND 2:** bỏ class đang xét khỏi tập láng giềng làm AUROC +0,0173,
  FPR@95TPR −0,0957. **Calibration correct-only** gần như vô hại (chỉ 3 mẫu khác).
- **MSP bão hoà:** top-prob median 0,9999997; 80,35% valid có MSP < 1e−6, nên ngưỡng
  rơi vào ~1e−6 và AUROC/FPR@95TPR không đổi khi đổi operating point.
- Output: `outputs/metrics/audit_ustc_seed42.json` + `.csv`.

### P5 — DC (run phụ, đã chạy)

DC chỉ dùng để audit **label leakage trong encoding**, không phải kết quả chính.
Đã chạy xong cả hai lần train + extract + audit:

```bash
# 1. Manifest DC: 4 known / 6 open, seed 42  (~1 s)
python3 scripts/build_dc_manifest.py \
  --source-dir data/raw/trafficllm_reference/trafficllm_data/DC \
  --output-dir data/processed/dc_seed42 \
  --known-count 4 --known-mode random --seed 42

# 2. Encoding reference (có nhãn) và corrected (label-free)  (~4 s mỗi bản)
python3 scripts/build_dc_encoding.py --mode reference --divisor 50
python3 scripts/build_dc_encoding.py --mode corrected --divisor 50

# 3. Train 2 lần riêng, batch 2 vì DC bị truncate ở 1024 token.
#    Đã chạy: reference 25,4 phút (best epoch 2), corrected 21,0 phút (best epoch 3).
python3 scripts/train_gpt2.py --config configs/dc_gpt2_seed42_reference.yaml
python3 scripts/train_gpt2.py --config configs/dc_gpt2_seed42_corrected.yaml

# 4. Extract feature: đã chạy 7,9 phút (reference) / 7,8 phút (corrected)
python3 scripts/extract_features.py --config configs/dc_gpt2_seed42_reference.yaml
python3 scripts/extract_features.py --config configs/dc_gpt2_seed42_corrected.yaml

# 5. Audit encoding  (~vài giây CPU)
python3 scripts/run_dc_audit.py --config configs/audit_dc_seed42.yaml
```

Kết quả train: reference best epoch 2, `closed_valid` 0,9948, `closed_test` 1,0000;
corrected best epoch 3, `closed_valid` 0,8229, `closed_test` 0,7812.

So sánh cùng protocol (`Ref*` = `reference_corrected_protocol` vs `corrected`):

| Detector | Ref* CCR | Ref* UCR | Ref* AUROC | Corr CCR | Corr UCR | Corr AUROC |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0,9531 | 0,7833 | 0,5518 | 0,7604 | 0,1167 | 0,6817 |
| Energy | 0,9479 | 0,5979 | 0,6934 | 0,7500 | 0,1271 | 0,6575 |
| K-LND 1 | 0,9427 | 0,6542 | 0,9137 | 0,7604 | 0,1042 | 0,6429 |
| K-LND 2 | 0,9427 | 0,6594 | 0,8843 | 0,7604 | 0,0917 | 0,7075 |
| K-LND 3 | 0,9427 | 0,6677 | 0,9125 | 0,7604 | 0,0917 | 0,6590 |
| OpenMax | 0,9583 | 0,7427 | 0,9572 | 0,7552 | 0,1094 | 0,6028 |

- **Leakage thổi phồng open-set rất mạnh:** bỏ nhãn khỏi encoding làm UCR rơi từ
  0,60–0,82 xuống 0,09–0,13; CCR giảm ~0,18–0,20.
- **Leakage cũng làm closed-set dễ hơn:** `closed_test` 1,0000 (reference) vs
  0,7812 (corrected), nên không đọc 0,99+ của reference là năng lực thật.
- **MSP là ngoại lệ:** AUROC tăng 0,5518 → 0,6817 khi bỏ leakage nhưng UCR vẫn sụp
  0,7833 → 0,1167. Leakage làm ngưỡng per-class reject open dễ hơn, không nhất thiết
  làm ranking toàn cục tốt hơn.
- **K-LND/OpenMax:** cả AUROC lẫn UCR đều giảm; ví dụ K-LND1 0,9137 → 0,6429,
  OpenMax 0,9572 → 0,6028.
- **Theo class open:** reference rất cao ở class 1, 2 (≈0,99–1,00), class 8
  (0,71–0,97); corrected gần như đều ~0,03–0,20.
- DC: 3.198 trace, 4 known / 6 open; `closed_train` 894, `closed_valid` 192,
  `closed_test` 192, `open_valid` 960, `open_test` 960.
- Token length GPT-2: p50 2.012, p95 2.390, max 3.923; **99,78%** vượt 1024 →
  bị truncate. Reference và corrected cùng phân phối độ dài nên so sánh vẫn công bằng
  trên phần prefix, nhưng đây là hạn chế phải nêu ở P7.

Số liệu chi tiết và cách đọc: `docs/protocol.md`, `docs/experiment-log.md`.

## Tiếp tục phát triển (P5 xong, sang P6)

Code P4/P5 đã có:

| File | Vai trò |
|---|---|
| `src/trafficllm/openset/scoring.py` | MSP, Energy, class means, neighbor sets, K-LND 1/2/3 |
| `src/trafficllm/openset/calibration.py` | ngưỡng phân vị per-class, hiệu chỉnh **chỉ** từ `closed_valid` |
| `src/trafficllm/openset/openmax.py` | OpenMax fit trên `closed_train`, chỉ báo cáo `corrected` |
| `src/trafficllm/openset/pipeline.py` | helper partition/metric/K-LND dùng chung cho P4/P5 |
| `src/trafficllm/evaluation/metrics.py` | AUROC, AUPR, FPR@95TPR, macro-F1, confusion matrix |
| `scripts/run_openset.py` + `configs/openset_ustc_seed42.yaml` | chạy 6 detector trên cache, xuất JSON + CSV |
| `scripts/run_audit.py` | audit 4 biến thể `reference`/`corrected` + chẩn đoán USTC |
| `scripts/build_dc_manifest.py` | manifest DC 4 known / 6 open, seed 42 |
| `src/trafficllm/encoding/dc.py` + `scripts/build_dc_encoding.py` | encoder DC reference (leaky) / corrected (label-free) |
| `scripts/run_dc_audit.py` + `configs/audit_dc_seed42.yaml` | so sánh encoding DC reference vs corrected |
| `tests/test_openset_detectors.py`, `tests/test_metrics.py`, `tests/test_dc_encoding_label_independence.py` | test Gaussian + metric + label independence |
| `docs/protocol.md` | định nghĩa metric, P5 USTC audit, DC protocol + kết quả |

Việc tiếp theo:

1. P5 đã xong: USTC audit + DC run phụ đều có kết quả, docs đã cập nhật.
2. Sang **P6**: adapter AWF/DF (streaming), GPT-2 random-init, ablation logits vs
   embedding, ≥3 seed → mean ± std. `open_valid` chỉ dùng từ P6.
3. Không trộn run phụ DC vào kết quả chính; luôn ghi rõ DC bị truncate 99,78% >1024.

Quy ước không được vi phạm khi làm tiếp (chi tiết: `AGENTS.md` §6):

- Mọi detector trả **anomaly score, cao hơn = lạ hơn**.
- Ngưỡng chỉ hiệu chỉnh từ `closed_valid`; **không** nhìn dữ liệu open. `open_valid` để nguyên cho P6.
- Hai chế độ `reference` / `corrected`; mọi bảng/hình phải ghi rõ chế độ nào sinh ra nó.
- Seed mọi thứ, lưu config + seed cùng kết quả.
- Không báo con số chưa tính.

Trước khi commit:

```bash
PYTHONPATH=src pytest -q
```

## Cấu trúc

```
configs/     cấu hình YAML từng thí nghiệm
data/raw/    dữ liệu gốc (không commit, chỉ đọc)
data/processed/  manifest + metadata (commit), trace/encoding (không commit)
docs/        protocol.md, experiment-log.md, setup.md
outputs/     checkpoints, features, figures, metrics (không commit, trừ metrics/figures nhỏ)
scripts/     CLI mỏng, chỉ parse tham số rồi gọi src/trafficllm
src/trafficllm/  data, splits, encoding, models, features, openset, evaluation
tests/       pytest
```
