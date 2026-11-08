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
| P4 | Detector open-set + metrics trên cache | **đang làm** — còn OpenMax + runner + test |
| P5 | Audit `reference` vs `corrected` trên cùng cache, run DC riêng | chưa làm |
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

Số liệu chi tiết và cách đọc: `docs/protocol.md`, `docs/experiment-log.md`.

## Tiếp tục phát triển (đang ở P4)

Code P4 hiện có:

| File | Vai trò |
|---|---|
| `src/trafficllm/openset/scoring.py` | MSP, Energy, class means, neighbor sets, K-LND 1/2/3 |
| `src/trafficllm/openset/calibration.py` | ngưỡng phân vị per-class, hiệu chỉnh **chỉ** từ `closed_valid` |
| `src/trafficllm/evaluation/metrics.py` | AUROC, AUPR, FPR@95TPR, macro-F1, confusion matrix |
| `docs/protocol.md` | định nghĩa metric, bảng khác biệt `reference` / `corrected` |

Việc còn lại, làm đúng thứ tự:

1. **OpenMax** → `src/trafficllm/openset/openmax.py`
   Mean embedding mỗi class từ `closed_train` (`emb_last`), fit Weibull trên `closed_valid`,
   hiệu chỉnh logits theo rank `alpha`, score = `1 − max p_revised`. Không có bản upstream nên
   chỉ báo cáo `corrected`.
2. **Runner** → `scripts/run_openset.py` + `configs/openset_ustc_seed42.yaml`
   Nạp cache + manifest, chạy 5 detector × 2 chế độ, xuất `outputs/metrics/openset_ustc_seed42.json`
   (đủ 9 metric, confusion 13×13, tỉ lệ theo 8 class open, độ nhạy T của MSP/Energy) và bản `.csv`.
3. **Test** → `tests/test_openset_detectors.py`
   Dữ liệu Gaussian giả lập: hướng score, chiều ngưỡng, self-neighbor, fallback class rỗng,
   mẫu số 0, ngưỡng per-class.
4. **Chạy thật + ghi log**: thêm một dòng vào `docs/experiment-log.md` (ngày, config, seed, lệnh, metric chính).
5. **Commit**: `P4: ...`, không commit `data/`, `outputs/checkpoints/`, `outputs/features/`.

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
