# AGENTS.md — Đồ án tái lập và kiểm toán TrafficLLM

## 1. Đây là dự án gì
Đồ án môn học (UIT, NT140) của HND. Ta tái lập và **kiểm toán** paper
"TrafficLLM: LLMs for improved open-set encrypted traffic analysis" (Ginige và cộng sự, Computer Networks, 2026)
bằng một **codebase độc lập**.

- Repo gốc của tác giả (github.com/YasodGinige/TrafficLLM) **chỉ để tham khảo**. Không copy code hay cấu trúc của họ.
  Đọc để đối chiếu hành vi thì được.
- Luận điểm: so sánh **reference protocol** (đúng như paper/repo làm, kể cả lỗi) với
  **corrected protocol** (encoding không dùng nhãn, split stratified có seed, open-set scoring nhất quán,
  metrics đầy đủ hơn) và định lượng sự chênh lệch.

## 2. Scope
- Trong scope: dataset **USTC**; backbone **GPT-2 small**; detector **MSP, Energy, K-LND 1/2/3**, sau đó **OpenMax**;
  feature = **logits** và **embedding**; đối chứng = **GPT-2 random-init** (cùng kiến trúc, không pretrain).
- Làm sau / tuỳ chọn: adapter AWF, DF, DC, CSTNet; baseline CNN; Background Class; LLaMA/QLoRA.
- DC **chỉ dùng để audit** lỗi label leakage, không dùng làm kết quả chính.
- Ngoài scope trừ khi user yêu cầu: PCAP parser, dashboard.

## 3. Môi trường
- Debian trên WSL2. Thư mục gốc: `/home/kiwi/UIT-DoAn/NT140/trafficllm`.
- Máy: RTX 4050 (6 GB VRAM), RAM 15 GB. Không bao giờ nạp toàn bộ AWF/DF vào RAM; cắt hoặc đọc streaming ngay lúc load.
- Kiểm tra có `.venv` không; nếu không có thì dùng `python3` hệ thống và `pip install --user`. Không dùng `sudo pip`.
- Kiểm tra version thật (`pip list`) trước khi viết code gọi API; `transformers` đổi API giữa các bản major.
  Ghim bản đã chạy được vào `requirements.lock.txt`.
- Chạy test: `PYTHONPATH=src pytest -q`.

## 4. Cấu trúc thư mục
```
configs/                 file YAML cấu hình thí nghiệm
data/raw/                chỉ đọc, không sửa
data/processed/ustc_seed42/   traces.npy, original_labels.npy, manifest.csv, metadata.json (+ encoded_gpt2.parquet)
docs/                    protocol.md, experiment-log.md (tiếng Việt)
outputs/{checkpoints,features,figures,metrics}/
scripts/                 điểm vào CLI mỏng
src/trafficllm/{data,splits,encoding,models,features,openset,evaluation}/
tests/
```
Nguyên tắc: **chạy model một lần, cache feature**; mọi detector open-set chạy trên cache,
không chạy lại model.

## 5. Dữ kiện về dữ liệu USTC
- 97.115 trace, 20 class (class 12 có 2.115 trace, các class còn lại 5.000).
- Trace thô = một chuỗi 319 ký tự: 64 token hex 4 ký tự chồng lấn, cách nhau bằng dấu cách (`"e742 421d 1dc8 ..."`).
- Cột của `manifest.csv`: `sample_id, partition, original_label, model_label, is_known`.
  `sample_id` = chỉ số dòng trong `traces.npy`.
- Các partition: `closed_train` 39.980 / `closed_valid` 8.567 / `closed_test` 8.568 / `open_valid` 20.000 / `open_test` 20.000.
- Class known (seed 42, ngẫu nhiên): {0,1,3,5,7,10,11,12,13,17,18,19}; class open: {2,4,6,8,9,14,15,16}.
- `model_label` 0..11 cho class known, **12 = unknown**. Luôn map qua
  `metadata.json["class_to_model_label"]`. **Không bao giờ giả định nhãn gốc = chỉ số của model.**
- Không sửa hay tạo lại các file trong `ustc_seed42/` trừ khi user yêu cầu.

## 6. Luật cứng (vi phạm bất kỳ luật nào là làm hỏng đồ án)
1. **Nhãn không bao giờ đi vào encoding/tiền xử lý.** `encode_trace_gpt2(raw_trace)` chỉ nhận đúng một tham số;
   có test kiểm tra điều này. Không thêm tham số.
2. **Class open không xuất hiện trong huấn luyện closed-set.** `open_valid` để dành cho baseline Background Class sau này;
   không dùng để tune MSP / Energy / K-LND.
3. **Ngưỡng chỉ hiệu chỉnh từ `closed_valid`.** Không nhìn dữ liệu open khi chọn ngưỡng,
   trừ một dòng "oracle" (cận trên) được ghi nhãn rõ ràng.
4. **Seed mọi thứ** (python, numpy, torch, cuda) và lưu config + seed cùng mọi checkpoint và kết quả.
5. **Hai chế độ, luôn ghi nhãn:** `mode="reference"` tái hiện hành vi của repo gốc (kể cả lỗi, phục vụ audit);
   `mode="corrected"` là mặc định. Mỗi bảng/hình phải nói rõ chế độ nào sinh ra nó.
6. **Không báo con số chưa tính.** Trích output lệnh thật. Không bịa metric.
7. Giữ danh tính class open (nhãn gốc) xuyên suốt đánh giá; không gộp lại trước khi tính tỉ lệ phát hiện theo từng class / OSCR.

## 7. Lỗi đã biết của repo gốc — KHÔNG tái hiện trong đường "corrected"
- Encoding của DC nhận nhãn thật làm đầu vào (label leakage) → mẫu open mang chính class của nó trong mẫu bit.
- K-LND: thống kê trên tập closed test cộng trên **tất cả** class còn lại, trong khi ngưỡng và tập open chỉ dùng
  **K class gần nhất**. Dùng một định nghĩa duy nhất cho cả hai. Kiểm tra cả chiều của phân vị ở mỗi ngưỡng
  (phân vị 0.8 với luật "reject nếu lớn hơn" sẽ loại ~20% mẫu known đúng ngay từ thiết kế).
- Chặn trường hợp class rỗng (chia cho 0 / NaN khi tính vector trung bình class) và mẫu số bằng 0 ở các thống kê dạng tỉ lệ.
- Split closed không shuffle và không stratified; huấn luyện không seed; không chọn checkpoint tốt nhất.
- Không trộn validation với một phần của tập train khi tính ngưỡng.
- Repo gốc chỉ có K-LND; OpenMax / Background Class không có trong repo.
- Cách diễn đạt: gọi chênh lệch giữa file kết quả của repo và README/config trong code là
  **"discrepancy, chưa xác minh nguyên nhân"** (file kết quả có thể đến từ phiên bản cũ hơn). Không gọi là
  "không tái lập được" trừ khi đã chứng minh.

## 8. Các phase và tiêu chí "xong"
Làm **từng phase một**. Không bắt đầu phase tiếp theo cho đến khi user xác nhận checklist của phase hiện tại.
Trước khi code, nêu kế hoạch trong ≤10 dòng. Chỉ đụng vào file thuộc phase hiện tại.

- **P0 ✅** môi trường, khung repo, smoke test.
- **P1 ✅** manifest USTC (xem §5).
- **P2 ✅** encoding không dùng nhãn + độ dài token.
  Xong khi: `encoded_gpt2.parquet` đủ 97.115 dòng; có `token_length_report.csv` và biết % vượt 1024 token
  theo từng partition; `tests/test_encoding_label_independence.py` pass; `docs/protocol.md` và
  `docs/experiment-log.md` đã cập nhật.
- **P3 ✅** fine-tune GPT-2 trên `closed_train`, chọn checkpoint theo accuracy `closed_valid`, cache feature.
  - Mặc định: `GPT2ForSequenceClassification(num_labels=12)`, pad_token = eos, padding bên phải, lr 5e-5, AdamW wd 0.01,
    lịch giảm tuyến tính, grad clip 1.0, 3 epoch, batch 12, max_len 1024 với **dynamic padding**.
    Được dùng mixed precision nếu cần cho 6 GB VRAM; ghi vào config.
  - Cache feature `outputs/features/ustc_seed42.npz` cho **toàn bộ 97.115 mẫu**, các khoá:
    `sample_id[N]`, `logits[N,12]`, `emb_last[N,768]`, `emb_mean[N,768]`.
    `emb_last` = hidden state lớp cuối tại token cuối không phải pad (đúng vị trí classification head đọc);
    `emb_mean` = trung bình trên các token không phải pad **có áp attention mask**. Join với manifest bằng `sample_id`.
  - Xong khi: cùng seed cho cùng kết quả hai lần; có accuracy trên closed_test; lưu checkpoint + config.
  - Không tự chạy job huấn luyện dài hơn ~10 phút: đưa cho user lệnh chính xác kèm thời gian ước lượng.
- **P4 (đang làm)** detector + metrics trên cache: MSP, Energy, K-LND 1/2/3 (cả hai chế độ), OpenMax; hiệu chỉnh ngưỡng từ
  `closed_valid` (ví dụ giữ lại 95% mẫu known); metrics: closed accuracy, unknown recall, AUROC,
  FPR@95%TPR, AUPR, OSCR, balanced F1 (tỉ lệ unknown tự nhiên và 1:1), confusion matrix, tỉ lệ phát hiện theo từng class open.
  Viết định nghĩa metric vào `docs/protocol.md` **trước khi** cài đặt. Unit-test detector trên dữ liệu Gaussian giả lập.
  Đã có: anomaly score MSP/Energy/K-LND 1-2-3, hiệu chỉnh ngưỡng, metric cơ bản trong
  `src/trafficllm/{openset,evaluation}/`.
  Còn thiếu: **OpenMax**, `scripts/run_openset.py`, unit-test Gaussian, mục OpenMax trong `docs/protocol.md`,
  dòng P4 trong `docs/experiment-log.md`.
- **P5 (chưa làm)** audit: chạy `reference` so với `corrected` trên cùng cache và giải thích mọi chênh lệch.
  DC cần **lần huấn luyện thứ hai** (leakage nằm ở encoding nên không dùng chung cache feature được).
  Thêm một run riêng, đặt tên rõ ràng, theo thiết lập của paper (paper nói 40% số class là closed = 8/20).
  Xong khi: có bảng so sánh hai chế độ cho cả 5 detector; mỗi chênh lệch truy được về nguyên nhân cụ thể
  (self-neighbor, bất đối xứng closed/open, chiều phân vị, operating point); run DC được đánh dấu rõ là run phụ.
- **P6 (chưa làm)** mở rộng: adapter AWF/DF (cắt/streaming lúc load), đối chứng GPT-2 random-init,
  ablation logits so với embedding, ≥3 seed → mean ± std.
  Xong khi: mỗi mục có bảng riêng, ghi rõ seed và cấu hình, kết quả báo mean ± std chứ không phải một lần chạy.
- **P7 (chưa làm)** báo cáo: `docs/audit-report.md`, hình, bảng, hạn chế.
  Xong khi: mọi con số trong báo cáo truy được về một dòng của `docs/experiment-log.md` và một file trong
  `outputs/metrics/`; phần hạn chế nêu rõ khác biệt split so với paper và những gì chưa làm.

## 9. Quy ước
- Python 3, `from __future__ import annotations`, type hint, hàm nhỏ thuần tuý, docstring ghi rõ
  mục/phương trình của paper khi liên quan.
- Tên biến và comment trong code bằng tiếng Anh; `docs/` bằng tiếng Việt (giữ phong cách hiện có).
- Config trong `configs/*.yaml`; `scripts/` chỉ parse tham số và gọi `src/trafficllm`.
- Mỗi thí nghiệm thêm một dòng vào `docs/experiment-log.md` (ngày, config, seed, lệnh, metrics chính).
- Ưu tiên **sửa có mục tiêu** hơn viết lại cả file.
- Commit: mỗi commit một thay đổi logic, message bắt đầu bằng phase (`P2: ...`).
  Không commit `data/`, `outputs/checkpoints/`, `outputs/features/`.

## 10. An toàn
- Không chạy lệnh phá hủy (`rm -rf`, `git reset --hard`, ghi đè cache, manifest hay checkpoint) khi chưa hỏi.
- Không sửa `data/raw/`.
- Không cài gói hệ thống hay dùng `sudo` khi chưa hỏi.

## 11. Cách nói chuyện với user
- Trả lời bằng **tiếng Việt**, ngắn gọn: kết quả trước, chi tiết sau.
- Chỉ hỏi những câu chặn tiến độ; còn lại nêu giả định của bạn rồi làm tiếp.
- Nếu thấy điều gì trong file này hoặc trong code repo gốc có vẻ sai, nói thẳng kèm bằng chứng thay vì lặng lẽ làm theo.
- Trả lời đơn giản, không cố lồng ghép các thuật ngữ phức tạp