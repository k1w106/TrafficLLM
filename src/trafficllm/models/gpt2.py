"""Fine-tune GPT-2 cho phân loại traffic + trích feature.

Paper: Ginige et al., "TrafficLLM", Computer Networks 274:111847 (2026),
mục fine-tuning GPT-2 trên closed-set.

Ghi chú kỹ thuật (transformers 5.x): ``GPT2ForSequenceClassification.forward``
áp ``self.score`` lên **toàn bộ chuỗi** rồi mới lấy vị trí token cuối không
phải pad, khác với v4 (lấy token cuối rồi mới ``score``). Vì ``score`` là một
lớp Linear áp theo chiều cuối nên hai cách tương đương; ``extract_features``
dưới đây tái hiện đúng thứ tự của v5 và được ``tests/test_features.py`` khoá lại.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from trafficllm.utils import autocast_for


class TraceDataset(Dataset):
    """Tập trace đã mã hoá; trả về text thô để collator tokenize theo batch."""

    def __init__(
        self,
        frame: pd.DataFrame,
        text_column: str,
        label_column: str,
        with_labels: bool = True,
    ) -> None:
        self.texts = frame[text_column].tolist()
        self.sample_ids = frame["sample_id"].to_numpy()
        self.labels = frame[label_column].to_numpy() if with_labels else None
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "sample_id": int(self.sample_ids[index]),
            "text": self.texts[index],
        }
        if self.with_labels:
            item["label"] = int(self.labels[index])
        return item


class DynamicPadCollator:
    """Pad theo độ dài batch lớn nhất (không pad cứng ``max_len``)."""

    def __init__(self, tokenizer: GPT2Tokenizer, max_len: int, with_labels: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.with_labels = with_labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        output = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "sample_id": torch.tensor([item["sample_id"] for item in batch]),
        }
        if self.with_labels:
            output["labels"] = torch.tensor([item["label"] for item in batch])
        return output


def build_tokenizer(name: str) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(
    name: str,
    num_labels: int,
    device: torch.device,
    attn_implementation: str = "eager",
) -> GPT2ForSequenceClassification:
    model = GPT2ForSequenceClassification.from_pretrained(
        name, num_labels=num_labels, attn_implementation=attn_implementation
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model.to(device)


def build_loaders(
    config: dict[str, Any],
    tokenizer: GPT2Tokenizer,
    frames: dict[str, pd.DataFrame],
    seed: int,
) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["train"]
    max_len = config["model"]["max_len"]

    def make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool, generator=None):
        dataset = TraceDataset(frame, data_cfg["text_column"], data_cfg["label_column"])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=train_cfg["num_workers"],
            collate_fn=DynamicPadCollator(tokenizer, max_len),
            generator=generator,
            drop_last=False,
        )

    shuffle_generator = torch.Generator().manual_seed(seed)
    return {
        "train": make_loader(
            frames["train"], train_cfg["batch_size"], True, shuffle_generator
        ),
        "valid": make_loader(frames["valid"], train_cfg["eval_batch_size"], False),
        "test": make_loader(frames["test"], train_cfg["eval_batch_size"], False),
    }


def _lr_lambda(warmup_steps: int, total_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return schedule


def train_one_epoch(
    model: GPT2ForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_clip: float,
    amp: str,
    max_steps: int | None = None,
) -> tuple[float, float, int]:
    model.train()
    total_loss, correct, seen, steps = 0.0, 0, 0, 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch.pop("sample_id", None)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device, amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
        seen += labels.numel()
        steps += 1

    return total_loss / max(steps, 1), correct / max(seen, 1), steps


@torch.inference_mode()
def evaluate(
    model: GPT2ForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    amp: str = "fp32",
) -> tuple[float, float]:
    model.eval()
    if device.type == "cuda":
        # Train xong cache của allocator vẫn giữ ~6 GB; nếu không xả thì lượt
        # đánh giá valid bị tràn sang RAM hệ thống và chậm ~5 lần.
        torch.cuda.empty_cache()
    total_loss, correct, seen, steps = 0.0, 0, 0, 0

    for batch in loader:
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch.pop("sample_id", None)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with autocast_for(device, amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += outputs.loss.item()
        correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
        seen += labels.numel()
        steps += 1

    return total_loss / max(steps, 1), correct / max(seen, 1)


def fit(
    config: dict[str, Any],
    model: GPT2ForSequenceClassification,
    loaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int | None = None,
    max_steps: int | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Huấn luyện, giữ checkpoint có accuracy `closed_valid` cao nhất.

    Nếu ``checkpoint_path`` được truyền, checkpoint tốt nhất được ghi ngay sau
    mỗi epoch để một lần chạy dài bị ngắt vẫn không mất kết quả.
    """
    train_cfg = config["train"]
    epochs = int(epochs if epochs is not None else train_cfg["epochs"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        eps=float(train_cfg["adam_eps"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    steps_per_epoch = len(loaders["train"])
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(int(train_cfg["warmup_steps"]), total_steps)
    )

    best: dict[str, Any] = {"accuracy": -1.0, "epoch": -1, "state": None, "loss": float("nan")}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss, train_acc, steps = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scheduler,
            device,
            float(train_cfg["grad_clip"]),
            str(train_cfg["amp"]),
            max_steps=max_steps,
        )
        valid_loss, valid_acc = evaluate(model, loaders["valid"], device, str(train_cfg["amp"]))
        elapsed = time.time() - started

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "valid_loss": round(valid_loss, 6),
            "valid_accuracy": round(valid_acc, 6),
            "steps": steps,
            "seconds": round(elapsed, 1),
        }
        history.append(record)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"valid_loss={valid_loss:.4f}  valid_acc={valid_acc:.4f}  ({elapsed / 60:.1f} phut)",
            flush=True,
        )

        if valid_acc > best["accuracy"]:
            best = {
                "accuracy": valid_acc,
                "epoch": epoch,
                "loss": valid_loss,
                "state": copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }
            print(f"  -> checkpoint moi (valid_acc={valid_acc:.4f})", flush=True)
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state": best["state"],
                        "config": config,
                        "seed": int(config["seed"]),
                        "best_epoch": best["epoch"],
                        "best_valid_accuracy": best["accuracy"],
                        "history": history,
                    },
                    checkpoint_path,
                )
                print(f"  -> da luu -> {checkpoint_path}", flush=True)

    print(
        f"best epoch: {best['epoch']} | best valid acc: {best['accuracy']:.4f}",
        flush=True,
    )
    return best, history


@torch.inference_mode()
def extract_features(
    model: GPT2ForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    amp: str = "fp32",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (sample_id, logits, emb_last, emb_mean) cho toàn bộ loader.

    - ``emb_last``: hidden state lớp cuối tại token cuối **không phải pad**
      (đúng vị trí classification head đọc).
    - ``emb_mean``: trung bình có áp attention mask.
    """
    model.eval()
    sample_ids, logits_parts, last_parts, mean_parts = [], [], [], []

    for batch in loader:
        sample_ids.append(batch.pop("sample_id").numpy())
        batch.pop("labels", None)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with autocast_for(device, amp):
            hidden = model.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        hidden = hidden.float()
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(hidden.size(0), device=hidden.device)
        emb_last = hidden[rows, lengths]

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        emb_mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = model.score(emb_last)

        logits_parts.append(logits.float().cpu().numpy())
        last_parts.append(emb_last.cpu().numpy())
        mean_parts.append(emb_mean.cpu().numpy())

    return (
        np.concatenate(sample_ids),
        np.concatenate(logits_parts),
        np.concatenate(last_parts),
        np.concatenate(mean_parts),
    )


def load_checkpoint(
    model: GPT2ForSequenceClassification, checkpoint_path: str | Path, device: torch.device
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return payload
