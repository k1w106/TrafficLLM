# Experiment log

| Date | Experiment | Dataset | Seed | Result | Notes |
|---|---|---|---:|---|---|
| 2026-10-20 | P2 — mã hoá USTC (GPT-2 char-level) + báo cáo độ dài token | USTC | 42 | 97.115 trace mã hoá; token 144–256 (mean 249,96); 0% vượt 1024; AUROC phân biệt closed/open bằng độ dài = 0,449 | `scripts/build_ustc_encoding.py`, `scripts/analyze_token_length.py`; `data/processed/ustc_seed42/encoded_gpt2.parquet`; `tests/test_encoding_label_independence.py` 8/8 pass |
