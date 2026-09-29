# TrafficLLM Audit

Đồ án phân loại encrypted traffic theo open-set.

Mục tiêu:
- Tái lập có kiểm toán ý tưởng TrafficLLM.
- Xây pipeline preprocessing không dùng nhãn.
- So sánh logits và embedding cho open-set detection.
- Đánh giá bằng closed-set accuracy, AUROC, FPR@95TPR, AUPR, OSCR.

## Phase 0
- Thiết lập môi trường.
- Kiểm tra GPT-2 tokenizer và hidden states.
- Chưa tải dataset, chưa train model.