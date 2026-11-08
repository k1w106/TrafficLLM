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
| P4 | Detector open-set + metrics trên cache | đang làm |

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # bản đã chạy được, ghim version
```

Chạy test:

```bash
PYTHONPATH=src pytest -q
```

## Dữ liệu và artifact (KHÔNG nằm trong git)

Repo này chỉ chứa code + metadata nhỏ (~3 MB). Ba thứ dưới đây phải tải riêng vì quá nặng:

| Thứ cần | Kích thước | Lấy ở đâu |
|---|---:|---|
| `data/raw/.../USTC/` | ~118 MB | link gdown trong `docs/setup.md` |
| `outputs/checkpoints/ustc_gpt2_seed42.pt` | 475 MB | GitHub Release tag `p3-artifacts` |
| `outputs/features/ustc_seed42.npz` | 575 MB | GitHub Release tag `p3-artifacts` |

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
