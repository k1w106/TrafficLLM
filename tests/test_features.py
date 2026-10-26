"""Khoá logic trích feature: ``emb_last`` phải đúng vị trí classification head đọc.

``GPT2ForSequenceClassification`` (transformers 5.x) lấy logits tại token cuối
cùng **không phải pad**. Nếu ``emb_last`` lệch một vị trí, logits cache trong
``ustc_seed42.npz`` sẽ khác logits lúc train và mọi detector ở P4 đều sai.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

from trafficllm.models.gpt2 import DynamicPadCollator

PAD_ID = 50256


def build_tiny_model() -> GPT2ForSequenceClassification:
    config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=32,
        n_positions=32,
        vocab_size=50257,
        num_labels=3,
        pad_token_id=PAD_ID,
        bos_token_id=PAD_ID,
        eos_token_id=PAD_ID,
    )
    return GPT2ForSequenceClassification(config).eval()


def test_score_of_emb_last_equals_model_logits() -> None:
    """``score(emb_last)`` phải bằng ``model(...).logits`` với mọi kiểu padding."""
    model = build_tiny_model()
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(15, 900, (4, 7), generator=generator)
    attention_mask = torch.ones_like(input_ids)

    # hai mẫu có padding bên phải, hai mẫu không padding
    input_ids[0, 5:] = PAD_ID
    attention_mask[0, 5:] = 0
    input_ids[1, 6:] = PAD_ID
    attention_mask[1, 6:] = 0

    with torch.inference_mode():
        reference = model(input_ids=input_ids, attention_mask=attention_mask).logits
        hidden = model.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        emb_last = hidden[torch.arange(hidden.size(0)), lengths]
        reproduced = model.score(emb_last)

    assert torch.allclose(reference, reproduced, atol=1e-5)


def test_emb_last_uses_last_non_pad_token() -> None:
    model = build_tiny_model()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(15, 900, (3, 6), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    input_ids[0, 3:] = PAD_ID
    attention_mask[0, 3:] = 0

    with torch.inference_mode():
        hidden = model.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        emb_last = hidden[torch.arange(3), lengths]

    assert torch.allclose(emb_last[0], hidden[0, 2])  # mẫu có 3 token thật
    assert torch.allclose(emb_last[1], hidden[1, 5])  # mẫu đầy đủ: token cuối


def test_masked_mean_ignores_padding() -> None:
    """``emb_mean`` chỉ lấy trung bình trên token thật, không tính pad."""
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    emb_mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    assert torch.allclose(emb_mean[0], hidden[0, :3].mean(dim=0))
    assert torch.allclose(emb_mean[1], hidden[1].mean(dim=0))


def test_collator_pads_to_batch_max_not_max_len() -> None:
    """Dynamic padding: độ rộng = mẫu dài nhất trong batch, không phải 1024."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    collator = DynamicPadCollator(tokenizer, max_len=1024)

    short = "e742 421d 1dc8"
    long = "e742 421d 1dc8 aaaa bbbb cccc dddd eeee ffff"
    batch = [
        {"sample_id": 0, "text": short, "label": 0},
        {"sample_id": 1, "text": long, "label": 1},
    ]
    encoded = collator(batch)

    expected_width = len(tokenizer(long)["input_ids"])
    assert encoded["input_ids"].shape == (2, expected_width)
    assert expected_width < 1024

    # mẫu ngắn được pad bên phải bằng eos, các vị trí pad có mask 0
    width = expected_width
    assert encoded["input_ids"][0, width - 1].item() == tokenizer.eos_token_id
    assert encoded["attention_mask"].sum(dim=1).tolist() == [
        len(tokenizer(short)["input_ids"]),
        expected_width,
    ]
    assert encoded["sample_id"].tolist() == [0, 1]
    assert encoded["labels"].tolist() == [0, 1]


def test_collator_truncates_at_max_len() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    collator = DynamicPadCollator(tokenizer, max_len=8)

    encoded = collator([{"sample_id": 0, "text": "a b c d e f g h i j k l", "label": 0}])

    assert encoded["input_ids"].shape == (1, 8)
    assert np.asarray(encoded["attention_mask"]).sum() == 8
