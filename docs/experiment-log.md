# Experiment log

| Date | Experiment | Dataset | Seed | Result | Notes |
|---|---|---|---:|---|---|
| 2026-10-20 | P2 — mã hoá USTC (GPT-2 char-level) + báo cáo độ dài token | USTC | 42 | 97.115 trace mã hoá; token 144–256 (mean 249,96); 0% vượt 1024; AUROC phân biệt closed/open bằng độ dài = 0,449 | `scripts/build_ustc_encoding.py`, `scripts/analyze_token_length.py`; `data/processed/ustc_seed42/encoded_gpt2.parquet`; `tests/test_encoding_label_independence.py` 8/8 pass |
| 2026-10-26 | P3 — fine-tune GPT-2 small trên `closed_train` + cache feature | USTC | 42 | best epoch 3: `closed_valid` 0,999650; `closed_test` 0,999767 (8.566/8.568); 71,8 phút | `configs/ustc_gpt2_seed42.yaml`, `scripts/train_gpt2.py`, `scripts/extract_features.py`; bf16 + attention eager; determinism check PASS (2 lần cùng seed, checksum trọng số giống hệt); `outputs/features/ustc_seed42.npz` 602 MB, accuracy từ cache khớp lúc train |
